"""
Reflective Learning CLI Toolkit

This script provides a command-line interface for working with ReflectiveCore models,
including training, preprocessing context data, and generating sequences.

Run with:
    python -m src.reflective_learning.tools.main <command> [options]

Commands:
    train         Train a ReflectiveCore model from token/state sequences
    preprocess    Convert raw JSON with text/images into numerical embeddings
    generate      Sample sequences from a trained model
"""

import argparse
import base64
import json
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from src.reflective_learning import train
from src.reflective_learning.inference import sample_multiple_sequences_batched
from src.reflective_learning.model import ReflectiveCore


# === Context Encoder ===
class ContextEncoder:
    def __init__(
        self, text_model, image_model, tokenizer, image_processor, device="cpu"
    ):
        assert text_model is not None, "text_model is required"
        assert image_model is not None, "image_model is required"
        assert tokenizer is not None, "tokenizer is required"
        assert image_processor is not None, "image_processor is required"

        assert text_model.config.hidden_size == image_model.config.hidden_size, (
            f"Text model (hidden={text_model.config.hidden_size}) and image model "
            f"(hidden={image_model.config.hidden_size}) must match."
        )

        self.text_model = text_model.to(device)
        self.image_model = image_model.to(device)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device

    @classmethod
    def from_pretrained(cls, context_dir, device="cpu"):
        with open(os.path.join(context_dir, "context_versions.json")) as f:
            versions = json.load(f)

        text_cfg = versions["pretrained_models"]["gpt2"]
        image_cfg = versions["pretrained_models"]["vit"]

        text_model = AutoModel.from_pretrained(
            text_cfg["model"], revision=text_cfg["revision"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_cfg["model"], revision=text_cfg["revision"]
        )
        image_model = AutoModel.from_pretrained(
            image_cfg["model"], revision=image_cfg["revision"]
        )
        image_processor = AutoImageProcessor.from_pretrained(
            image_cfg["model"], revision=image_cfg["revision"], use_fast=True
        )

        return cls(text_model, image_model, tokenizer, image_processor, device=device)

    @lru_cache(maxsize=1024)
    def encode_text_embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            output = self.text_model(input_ids=input_ids)
            return output.last_hidden_state.squeeze(0).detach().cpu()  # shape [T, D]

    @lru_cache(maxsize=1024)
    def encode_image_embed(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            output = self.image_model(pixel_values=pixel_values)
            return output.last_hidden_state.squeeze(0).detach().cpu()  # shape [I, D]

    def encode(self, text: list[str], image: list[str]) -> torch.Tensor:
        segments = []

        # Always create break embedding
        break_dim = self.text_model.config.hidden_size
        break_embed = torch.zeros((1, break_dim), dtype=torch.float32)

        # Add text embeddings
        for t in text:
            segments.append(self.encode_text_embed(t))
        segments.append(break_embed.clone())  # break between text and image

        # Add image embeddings
        for path in image:
            segments.append(self.encode_image_embed(path))
        segments.append(break_embed.clone())  # break between context and tokens

        return torch.cat(segments, dim=0)


# === Train ===
def run_train(args):
    with open(args.mapping) as f:
        mapping = json.load(f)

    vocab_size = len(mapping["vocab"])
    state_size = len(mapping["state"])

    print(f"📖 Loaded mapping from: {args.mapping}")
    print(f"🧠 Vocab size: {vocab_size}, State size: {state_size}")

    train.train(
        json_paths=args.input,
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        device=args.device,
        d_model=args.d_model,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
    )


# === Preprocess ===
def run_preprocess(args):
    with open(args.mapping) as f:
        mapping = json.load(f)

    vocab_map = mapping["vocab"]
    state_map = mapping["state"]

    context_encoder = None
    if args.context_dir:
        context_encoder = ContextEncoder.from_pretrained(
            args.context_dir, device=args.device
        )

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                example = json.loads(line)
                output = dict(example)  # preserve all user fields

                # Optionally remove token/state
                if args.prefix_only:
                    output.pop("token", None)
                    output.pop("state", None)
                else:
                    token_ids = [vocab_map[token] for token in example["token"]]
                    state_id = state_map[example["state"]]
                    output["token"] = token_ids
                    output["state"] = state_id

                # Add prefix
                if context_encoder:
                    if "text" not in example or "image" not in example:
                        raise ValueError(f"Line {line_num}: Missing 'text' or 'image'")
                    if not isinstance(example["text"], list):
                        raise ValueError(
                            f"Line {line_num}: 'text' must be list of strings."
                        )
                    if not isinstance(example["image"], list):
                        raise ValueError(
                            f"Line {line_num}: 'image' must be list of strings."
                        )

                    prefix = context_encoder.encode(example["text"], example["image"])
                    prefix_bytes = (
                        prefix.to(torch.float32).contiguous().numpy().tobytes()
                    )
                    output["prefix"] = base64.b64encode(prefix_bytes).decode("utf-8")

                json.dump(output, fout)
                fout.write("\n")
            except KeyError as e:
                raise KeyError(f"Line {line_num}: Missing mapping for {e}") from e
            except Exception as e:
                raise RuntimeError(
                    f"Line {line_num}: Failed to process line: {e}"
                ) from e


# === Generate ===
def run_generate(args):
    # Load state weights from file, JSON string, or CSV-style input
    if args.state_weights.endswith(".json"):
        with open(args.state_weights) as f:
            state_weights = json.load(f)
    elif "=" in args.state_weights:
        state_weights = {}
        for pair in args.state_weights.split(","):
            key, value = pair.split("=")
            state_weights[key.strip()] = float(value.strip())
    else:
        state_weights = json.loads(args.state_weights)

    # Load vocab and state mappings
    with open(args.mapping) as f:
        mapping = json.load(f)
    vocab_map = mapping["vocab"]
    state_map = mapping["state"]
    id_to_token = {v: k for k, v in vocab_map.items()}
    state_weights = {state_map[k]: v for k, v in state_weights.items()}

    # Load model
    model = ReflectiveCore(
        vocab_size=len(vocab_map),
        state_size=len(state_map),
        max_seq_len=args.max_seq_len,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Load input JSONL with prefix
    all_outputs = []
    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            try:
                base = json.loads(line)

                if "token" in base or "state" in base:
                    raise ValueError(
                        f"Line {line_num}: Input should not contain 'token' or 'state'"
                    )
                if "prefix" not in base:
                    raise ValueError("Missing 'prefix' field")

                prefix_bytes = base64.b64decode(base["prefix"])
                prefix_tensor = torch.from_numpy(
                    np.frombuffer(prefix_bytes, dtype=np.float32)
                ).to(device)
                prefix_tensor = prefix_tensor.view(-1, model.d_model)

                for _ in range(args.repeat):
                    tokens = sample_multiple_sequences_batched(
                        model=model,
                        state_weights=state_weights,
                        num_sequences=1,
                        max_seq_len=args.max_seq_len,
                        temperature=args.temperature,
                        prefix=prefix_tensor,
                        device=device,
                    )[0]

                    decoded = [id_to_token[tok] for tok in tokens]
                    result = dict(base)
                    result["token"] = decoded
                    all_outputs.append(result)
            except Exception as e:
                raise RuntimeError(f"Line {line_num}: Failed to process: {e}") from e

    if args.output:
        with open(args.output, "w") as f:
            for out in all_outputs:
                f.write(json.dumps(out) + "\n")
        print(f"✅ Saved {len(all_outputs)} sequences to {args.output}")
    else:
        for out in all_outputs:
            print(json.dumps(out))


# === Main CLI ===
def main():
    parser = argparse.ArgumentParser(
        description="Reflective Learning CLI Toolkit",
        epilog="""Example:
  python -m src.reflective_learning.tools.main train \\
      --input data/train.json \\
      --mapping checkpoints/context/mappings.json \\
      --epochs 5 --batch-size 16 --lr 1e-4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train a ReflectiveCore model")
    train_parser.add_argument("--input", nargs="+", required=True)
    train_parser.add_argument("--mapping", type=str, required=True)
    train_parser.add_argument("--max-seq-len", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--save-path", type=str)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--d-model", type=int, default=768)
    train_parser.add_argument("--num-layers", type=int, default=12)
    train_parser.add_argument("--nheads", type=int, default=12)
    train_parser.add_argument("--dim-feedforward", type=int, default=3072)
    train_parser.set_defaults(func=run_train)

    # --- Preprocess ---
    pre_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    pre_parser.add_argument("--input", type=str, required=True)
    pre_parser.add_argument("--output", type=str, required=True)
    pre_parser.add_argument("--mapping", type=str, required=True)
    pre_parser.add_argument("--context-dir", type=str)
    pre_parser.add_argument("--device", type=str, default="cpu")
    pre_parser.add_argument("--prefix-only", action="store_true")
    pre_parser.set_defaults(func=run_preprocess)

    # --- Generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate sequences")
    gen_parser.add_argument(
        "--input", type=str, required=True, help="Line-separated JSON input with prefix"
    )
    gen_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Trained model checkpoint"
    )
    gen_parser.add_argument(
        "--mapping", type=str, required=True, help="Path to mappings.json"
    )
    gen_parser.add_argument(
        "--state-weights",
        type=str,
        required=True,
        help="State weights (JSON string, file path, or CSV: 'a=0.5,b=0.5')",
    )
    gen_parser.add_argument("--max-seq-len", type=int, default=128)
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat each input N times"
    )
    gen_parser.add_argument("--device", type=str, default=None)
    gen_parser.add_argument("--output", type=str, help="Output path")
    gen_parser.set_defaults(func=run_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
