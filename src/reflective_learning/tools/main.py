import argparse
from src.reflective_learning.tools import train, preprocess, generate, postprocess


def main():
    parser = argparse.ArgumentParser(description="Reflective Learning CLI Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train a ReflectiveCore model")
    train_parser.set_defaults(func=train.main)

    # Preprocess
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Convert textual tokens to numerical format"
    )
    preprocess_parser.set_defaults(func=preprocess.main)

    # Generate
    generate_parser = subparsers.add_parser(
        "generate", help="Sample sequences from a trained model"
    )
    generate_parser.set_defaults(func=generate.main)

    # Postprocess
    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Convert numeric output back to readable text"
    )
    postprocess_parser.set_defaults(func=postprocess.main)

    args = parser.parse_args()
    args.func()
