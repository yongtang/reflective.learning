import argparse
from src.reflective_learning.postprocess import postprocess


def main():
    parser = argparse.ArgumentParser(
        description="Convert numeric output back to original textual token/state format using mapping.json.",
        epilog="""\
Example:
  python -m src.reflective_learning.tools.main postprocess \\
    --input generated.json \\
    --mapping mapping.json \\
    --output readable.json

Each line of the input should be a JSON object with:
  {
    "token": [list of integers],
    "state": integer
  }

The output will be the reverse mapping:
  {
    "token": ["X1", "X2", ...],
    "state": "S1"
  }
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Path to generated JSON file")
    parser.add_argument(
        "--mapping",
        required=True,
        help="Path to mapping.json used during preprocessing",
    )
    parser.add_argument(
        "--output", required=True, help="Path to save converted readable JSON output"
    )

    args = parser.parse_args()
    postprocess(args.input, args.mapping, args.output)
    print(f"✅ Postprocessing complete. Output saved to {args.output}")
