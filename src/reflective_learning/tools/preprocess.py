import json
import argparse
from src.reflective_learning import (
    build_mapping_from_list,
    load_mappings,
    save_mappings,
    validate_mapping,
    preprocess_textual_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess textual token/state JSON to numeric format.",
        epilog="""\
Example:
  # Provide vocab/state inline
  python3 preprocess.py --input in.json --output out.json \
    --vocab X1 X2 X3 --state S1 S2 --save-mappings mapping.json

  # Load from a pre-defined mapping file
  python3 preprocess.py --input in.json --output out.json \
    --load-mappings mapping.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", required=True, help="Path to input JSON file (line-separated)"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--vocab", nargs="*", help="Ordered token vocab (e.g. X1 X2 X3)"
    )
    parser.add_argument("--state", nargs="*", help="Ordered state names (e.g. S1 S2)")
    parser.add_argument(
        "--save-mappings", help="Optional path to save combined mapping JSON"
    )
    parser.add_argument(
        "--load-mappings", help="Optional path to load combined mapping JSON"
    )

    args = parser.parse_args()

    if args.load_mappings:
        vocab_map, state_map = load_mappings(args.load_mappings)
    elif args.vocab and args.state:
        vocab_map = build_mapping_from_list(args.vocab)
        state_map = build_mapping_from_list(args.state)
    else:
        raise ValueError(
            "You must either provide --load-mappings OR both --vocab and --state."
        )

    validate_mapping(vocab_map, "vocab")
    validate_mapping(state_map, "state")

    preprocess_textual_json(args.input, args.output, vocab_map, state_map)

    if args.save_mappings:
        save_mappings(vocab_map, state_map, args.save_mappings)
