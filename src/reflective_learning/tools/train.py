import argparse

from src.reflective_learning.train import train


def main():
    parser = argparse.ArgumentParser(description="Train a ReflectiveCore model.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Path(s) to input JSONL dataset file(s)",
    )
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size")
    parser.add_argument(
        "--state-size", type=int, required=True, help="Number of states"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save-path", type=str, help="Optional path to save trained model"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to train on (cuda, cpu)"
    )

    # Optional overrides for testing/small models
    parser.add_argument(
        "--d-model", type=int, default=768, help="Transformer model dimension"
    )
    parser.add_argument(
        "--n-layers", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--dim-ff", type=int, default=3072, help="Feedforward dimension"
    )

    args = parser.parse_args()

    train(
        json_paths=args.input,
        vocab_size=args.vocab_size,
        state_size=args.state_size,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        device=args.device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_ff=args.dim_ff,
    )
