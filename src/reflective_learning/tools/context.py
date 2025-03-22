import argparse
import json
import os

import requests
import torch
import torchvision
from huggingface_hub import HfApi
from transformers import AutoTokenizer, GPT2Model, ViTModel


def resolve_model_revision(model_name):
    api = HfApi()
    model_info = api.model_info(repo_id=model_name)
    return model_info.sha


def convert_context(
    output_dir,
    gpt2_model="gpt2",
    vit_model="google/vit-base-patch16-224",
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️  Loading GPT-2 model: {gpt2_model}")
    AutoTokenizer.from_pretrained(gpt2_model)
    GPT2Model.from_pretrained(gpt2_model).eval()

    print(f"⚙️  Loading ViT model: {vit_model}")
    ViTModel.from_pretrained(vit_model).eval()

    metadata = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pretrained_models": {
            "gpt2": {
                "model": gpt2_model,
                "revision": resolve_model_revision(gpt2_model),
            },
            "vit": {
                "model": vit_model,
                "revision": resolve_model_revision(vit_model),
            },
        },
    }

    with open(os.path.join(output_dir, "context_versions.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Context conversion complete.")


def upload_context_to_hf(context_dir, repo_id, hf_token):
    api = HfApi()
    files = [
        f
        for f in os.listdir(context_dir)
        if os.path.isfile(os.path.join(context_dir, f))
    ]

    print(f"📤 Uploading context files to {repo_id}...")
    api.create_repo(repo_id, token=hf_token, exist_ok=True)

    for f in files:
        path = os.path.join(context_dir, f)
        print(f"  ⬆️  Uploading: {f}")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"context/{f}",
            repo_id=repo_id,
            token=hf_token,
        )

    print("✅ Upload complete.")


def validate_context_upload(context_dir, repo_id):
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    metadata_path = os.path.join(context_dir, "context_versions.json")
    with open(metadata_path) as f:
        meta = json.load(f)

    print("🔍 Validating uploaded files:")
    for fname in ["context_versions.json"]:
        url = f"{base_url}/context/{fname}"
        print(f"  🌐 Checking {url} ... ", end="")
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("✅ OK")
            else:
                print("❌ Not found")
        except Exception as e:
            print("❌ Error:", e)


def main():
    parser = argparse.ArgumentParser(
        description="Context encoder tools for version pinning and upload",
        epilog="""
Examples:

  # Convert and pin context encoder versions
  python -m src.reflective_learning.tools.context convert \
    --output-dir checkpoints/context

  # Upload to Hugging Face
  python -m src.reflective_learning.tools.context upload \
    --context-dir checkpoints/context \
    --repo-id yourname/reflective-context \
    --token $HF_TOKEN

  # Validate uploaded files
  python -m src.reflective_learning.tools.context validate \
    --context-dir checkpoints/context \
    --repo-id yourname/reflective-context
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert", help="Pin model and library versions"
    )
    convert_parser.add_argument("--output-dir", required=True)
    convert_parser.add_argument("--gpt2-model", default="gpt2")
    convert_parser.add_argument("--vit-model", default="google/vit-base-patch16-224")

    upload_parser = subparsers.add_parser(
        "upload", help="Upload pinned context info to Hugging Face"
    )
    upload_parser.add_argument("--context-dir", required=True)
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--token", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate uploaded context metadata"
    )
    validate_parser.add_argument("--context-dir", required=True)
    validate_parser.add_argument("--repo-id", required=True)

    args = parser.parse_args()

    if args.command == "convert":
        convert_context(
            output_dir=args.output_dir,
            gpt2_model=args.gpt2_model,
            vit_model=args.vit_model,
        )
    elif args.command == "upload":
        upload_context_to_hf(args.context_dir, args.repo_id, args.token)
    elif args.command == "validate":
        validate_context_upload(args.context_dir, args.repo_id)


if __name__ == "__main__":
    main()
