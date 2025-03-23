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
import json

from src.reflective_learning import train
from src.reflective_learning.tools import generate, postprocess, preprocess


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
    preprocess_parser.set_defaults(func=preprocess.main)

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
