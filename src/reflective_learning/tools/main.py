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

import numpy as np
import torch

from src.reflective_learning import train
from src.reflective_learning.inference import sample_multiple_sequences_batched
from src.reflective_learning.model import ReflectiveCore
from src.reflective_learning.tools.encoder import ContextEncoder


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
        nhead=args.n_heads,
        dim_feedforward=args.dim_ff,
        num_layers=args.n_layers,
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
    if args.state_dist.endswith(".json"):
        with open(args.state_dist) as f:
            state_weights = json.load(f)
    elif "=" in args.state_dist:
        state_weights = {}
        for pair in args.state_dist.split(","):
            key, value = pair.split("=")
            state_weights[key.strip()] = float(value.strip())
    else:
        state_weights = json.loads(args.state_dist)

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
  python -m src.reflective_learning.tools.main train \
      --input data/train.json \
      --mapping checkpoints/context/mappings.json \
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
    train_parser.add_argument("--n-layers", type=int, default=12)
    train_parser.add_argument("--n-heads", type=int, default=12)
    train_parser.add_argument("--dim-ff", type=int, default=3072)
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
        "--state-dist",
        type=str,
        required=True,
        help="State distribution (JSON string, path, or CSV: 'a=0.5,b=0.5')",
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
