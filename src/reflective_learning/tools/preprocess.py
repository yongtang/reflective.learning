import argparse
import json
import os

import torch

from .encoder import ContextEncoder


def build_mapping_from_list(items):
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate entries found in mapping list: {items}")
    return {k: i for i, k in enumerate(items)}


def validate_mapping(mapping, name):
    values = list(mapping.values())
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate index values found in {name} mapping.")
    if sorted(values) != list(range(len(values))):
        raise ValueError(
            f"{name} mapping values must be a contiguous range from 0 to {len(values) - 1}, got: {values}"
        )


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


def preprocess_textual_json(
    input_path, output_path, vocab_map, state_map, context_encoder=None
):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                example = json.loads(line)
                token_ids = [vocab_map[token] for token in example["token"]]
                state_id = state_map[example["state"]]

                output = {"token": token_ids, "state": state_id}

                if context_encoder:
                    if "text" not in example or "image" not in example:
                        raise ValueError(
                            f"Line {line_num}: Missing 'text' or 'image' field for context encoding"
                        )
                    prefix = context_encoder.encode(
                        text=example["text"], image_path=example["image"]
                    )
                    output["prefix"] = prefix.tolist()

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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocess token/state JSON to numeric format with optional context embeddings.",
        epilog="""\
Examples:
  # Without context
  python tools/preprocess.py --input raw.json --output out.json \\
    --vocab X1 X2 X3 --state S1 S2 --save-mappings mappings.json

  # With context
  python tools/preprocess.py --input raw.json --output out.json \\
    --load-mappings mappings.json \\
    --context-dir checkpoints/context --device cuda
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")

    parser.add_argument("--vocab", nargs="*", help="Token vocab (e.g. X1 X2)")
    parser.add_argument("--state", nargs="*", help="State names (e.g. S1 S2)")

    parser.add_argument("--save-mappings", help="Path to save vocab/state mappings")
    parser.add_argument("--load-mappings", help="Path to load vocab/state mappings")

    parser.add_argument("--context-dir", help="Path to context encoder dir")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")

    args = parser.parse_args(argv)

    if args.load_mappings:
        vocab_map, state_map = load_mappings(args.load_mappings)
    elif args.vocab and args.state:
        vocab_map = build_mapping_from_list(args.vocab)
        state_map = build_mapping_from_list(args.state)
    else:
        raise ValueError("Provide either --load-mappings OR both --vocab and --state.")

    validate_mapping(vocab_map, "vocab")
    validate_mapping(state_map, "state")

    context_encoder = None
    if args.context_dir:
        context_encoder = ContextEncoder.from_pretrained(
            args.context_dir, device=args.device
        )

    preprocess_textual_json(
        args.input, args.output, vocab_map, state_map, context_encoder
    )

    if args.save_mappings:
        save_mappings(vocab_map, state_map, args.save_mappings)


if __name__ == "__main__":
    main()
