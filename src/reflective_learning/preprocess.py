import json
import argparse


def build_mapping_from_list(items):
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate entries found in mapping list: {items}")
    return {k: i for i, k in enumerate(items)}


def load_mappings(path):
    with open(path) as f:
        data = json.load(f)

    if "vocab" not in data or "state" not in data:
        raise ValueError("Mapping file must contain 'vocab' and 'state' keys.")

    vocab_map = data["vocab"]
    state_map = data["state"]

    validate_mapping(vocab_map, "vocab")
    validate_mapping(state_map, "state")

    return vocab_map, state_map


def save_mappings(vocab_map, state_map, path):
    with open(path, "w") as f:
        json.dump({"vocab": vocab_map, "state": state_map}, f, indent=2)


def validate_mapping(mapping, name):
    values = list(mapping.values())
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate index values found in {name} mapping.")
    if sorted(values) != list(range(len(values))):
        raise ValueError(
            f"{name} mapping values must be a contiguous range from 0 to {len(values)-1}, got: {values}"
        )


def preprocess_textual_json(input_path, output_path, vocab_map, state_map):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                example = json.loads(line)
                token_ids = [vocab_map[token] for token in example["token"]]
                state_id = state_map[example["state"]]
                json.dump({"token": token_ids, "state": state_id}, fout)
                fout.write("\n")
            except KeyError as e:
                raise KeyError(
                    f"Line {line_num}: Missing mapping for {e} in vocab/state."
                ) from e


def main(argv=None):
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

    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    main()
