import argparse
import json
import torch
from src.reflective_learning.model import ReflectiveTransformer
from src.reflective_learning.inference import sample_multiple_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Generate sequences using a trained ReflectiveTransformer."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size")
    parser.add_argument(
        "--state-size", type=int, required=True, help="Number of states"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of sequences to sample"
    )
    parser.add_argument(
        "--state-dist",
        type=str,
        required=True,
        help="JSON string or path to file with state weights",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling (not yet implemented)"
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="Use argmax instead of sampling (not yet implemented)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument(
        "--output", type=str, help="Optional path to save generated sequences"
    )

    args = parser.parse_args()

    # Load state distribution from file or JSON string
    if args.state_dist.endswith(".json"):
        with open(args.state_dist) as f:
            state_weights = json.load(f)
    else:
        state_weights = json.loads(args.state_dist)
    state_weights = {int(k): float(v) for k, v in state_weights.items()}

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = ReflectiveTransformer(
        vocab_size=args.vocab_size,
        state_size=args.state_size,
        max_seq_len=args.max_seq_len,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    sequences = sample_multiple_sequences(
        model,
        state_weights=state_weights,
        num_sequences=args.num_samples,
        start_token=0,
        max_len=args.max_seq_len,
        temperature=args.temperature,
        stop_token=0,
        device=device,
    )

    if args.output:
        with open(args.output, "w") as f:
            for seq in sequences:
                f.write(json.dumps(seq) + "\n")
        print(f"✅ Saved {len(sequences)} sequences to {args.output}")
    else:
        for seq in sequences:
            print(seq)
