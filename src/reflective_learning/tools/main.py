"""
Reflective Learning CLI Toolkit

This script provides a command-line interface for working with ReflectiveCore models,
including training, preprocessing context data, generating sequences, and postprocessing outputs.

Run with:
    python -m src.reflective_learning.tools.main <command> [options]

Commands:
    train         Train a ReflectiveCore model from token/state sequences
    preprocess    Convert raw JSON with text/images into numerical embeddings
    generate      Sample sequences from a trained model
    postprocess   Convert numeric model output back to readable format
"""

import argparse
import base64
import json

import torch

from src.reflective_learning import train
from src.reflective_learning.tools import generate, postprocess
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
                output = dict(example)  # preserve all fields

                # Optionally strip token/state
                if args.prefix_only:
                    output.pop("token", None)
                    output.pop("state", None)
                else:
                    token_ids = [vocab_map[token] for token in example["token"]]
                    state_id = state_map[example["state"]]
                    output["token"] = token_ids
                    output["state"] = state_id

                # Add prefix if context encoder is provided
                if context_encoder:
                    if "text" not in example or "image" not in example:
                        raise ValueError(
                            f"Line {line_num}: Missing 'text' or 'image' for context encoding"
                        )
                    if not isinstance(example["text"], list):
                        raise ValueError(
                            f"Line {line_num}: 'text' must be a list of strings."
                        )
                    if not isinstance(example["image"], list):
                        raise ValueError(
                            f"Line {line_num}: 'image' must be a list of strings."
                        )

                    prefix = context_encoder.encode(example["text"], example["image"])
                    prefix_bytes = (
                        prefix.to(torch.float32).contiguous().numpy().tobytes()
                    )
                    output["prefix"] = base64.b64encode(prefix_bytes).decode("utf-8")

                json.dump(output, fout)
                fout.write("\n")
            except KeyError as e:
                raise KeyError(
                    f"Line {line_num}: Missing mapping for {e} in vocab/state."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Line {line_num}: Failed to process line: {e}"
                ) from e


def main():
    parser = argparse.ArgumentParser(
        description="Reflective Learning CLI Toolkit",
        epilog="""Example:
  python -m src.reflective_learning.tools.main train \\
      --input data/train.json data/val.json \\
      --mapping checkpoints/context/mappings.json \\
      --epochs 5 --batch-size 16 --lr 1e-4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === Train ===
    train_parser = subparsers.add_parser("train", help="Train a ReflectiveCore model")
    train_parser.add_argument(
        "--input", nargs="+", required=True, help="Input JSON files"
    )
    train_parser.add_argument(
        "--mapping", type=str, required=True, help="Path to mappings.json file"
    )
    train_parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Max sequence length"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument(
        "--save-path", type=str, help="Optional path to save the model"
    )
    train_parser.add_argument(
        "--device", type=str, default=None, help="Device to use (e.g. cuda, cpu)"
    )
    train_parser.add_argument(
        "--d-model", type=int, default=768, help="Transformer hidden size"
    )
    train_parser.add_argument(
        "--n-layers", type=int, default=12, help="Number of transformer layers"
    )
    train_parser.add_argument(
        "--n-heads", type=int, default=12, help="Number of attention heads"
    )
    train_parser.add_argument(
        "--dim-ff", type=int, default=3072, help="Feedforward network dimension"
    )
    train_parser.set_defaults(func=run_train)

    # === Preprocess ===
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess raw data into context embeddings"
    )
    preprocess_parser.add_argument(
        "--input", type=str, required=True, help="Path to raw JSON"
    )
    preprocess_parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file"
    )
    preprocess_parser.add_argument(
        "--mapping", type=str, required=True, help="Path to vocab/state mapping JSON"
    )
    preprocess_parser.add_argument(
        "--context-dir", type=str, help="Optional path to context encoder dir"
    )
    preprocess_parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu or cuda)"
    )
    preprocess_parser.add_argument(
        "--prefix-only",
        action="store_true",
        help="If set, strips 'token' and 'state' fields from output and keeps only context + metadata",
    )
    preprocess_parser.set_defaults(func=run_preprocess)

    # === Generate ===
    generate_parser = subparsers.add_parser(
        "generate", help="Generate sequences from a trained model"
    )
    generate_parser.set_defaults(func=generate.main)

    # === Postprocess ===
    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Convert output sequences to readable text"
    )
    postprocess_parser.set_defaults(func=postprocess.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
