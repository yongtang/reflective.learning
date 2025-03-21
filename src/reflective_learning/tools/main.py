import argparse
import sys

from src.reflective_learning.tools import train, preprocess, generate


def main():
    parser = argparse.ArgumentParser(
        description="Reflective Learning CLI Tools",
        usage="python -m src.reflective_learning.tools.main <command> [<args>]",
    )
    parser.add_argument(
        "command", help="Subcommand to run (train, preprocess, generate)"
    )

    args = parser.parse_args(sys.argv[1:2])
    if not hasattr(sys.modules[__name__], args.command):
        print(f"Unknown command: {args.command}")
        parser.print_help()
        exit(1)

    # Dispatch to the appropriate CLI tool
    if args.command == "train":
        train.main()
    elif args.command == "preprocess":
        preprocess.main()
    elif args.command == "generate":
        generate.main()


if __name__ == "__main__":
    main()
