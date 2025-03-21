import argparse
import os
import json
import hashlib
import torch
import requests
from transformers import AutoTokenizer, GPT2Model, ViTModel
from huggingface_hub import HfApi


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def convert_context_encoders(
    output_dir, gpt2_model="gpt2", vit_model="google/vit-base-patch16-224"
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️  Loading GPT-2 from: {gpt2_model}")
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model)
    gpt2 = GPT2Model.from_pretrained(gpt2_model)
    gpt2.eval()
    torch.save(gpt2.state_dict(), os.path.join(output_dir, "text_weights.pt"))
    gpt2_tokenizer.save_pretrained(output_dir)

    print(f"⚙️  Loading ViT from: {vit_model}")
    vit = ViTModel.from_pretrained(vit_model)
    vit.eval()
    torch.save(vit.state_dict(), os.path.join(output_dir, "image_weights.pt"))

    metadata = {
        "transformers_version": torch.hub._get_torch_home(),  # optional version tracking
        "torch_version": torch.__version__,
        "text_encoder": {
            "model_name": gpt2_model,
            "weights_file": "text_weights.pt",
            "tokenizer_file": "text_tokenizer.json",
        },
        "image_encoder": {"model_name": vit_model, "weights_file": "image_weights.pt"},
    }

    # Add SHA256 for each file
    for name in ["text_weights.pt", "image_weights.pt", "tokenizer_config.json"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            metadata.setdefault("hashes", {})[name] = sha256sum(path)

    with open(os.path.join(output_dir, "context_versions.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Conversion complete. Files saved to:", output_dir)


def upload_context_to_hf(context_dir, repo_id, hf_token):
    api = HfApi()
    files = [
        f
        for f in os.listdir(context_dir)
        if os.path.isfile(os.path.join(context_dir, f))
    ]

    print(f"📤 Uploading to {repo_id}...")
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
    for fname, local_hash in meta.get("hashes", {}).items():
        url = f"{base_url}/context/{fname}"
        print(f"  🌐 Checking {url} ... ", end="")
        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                print("❌ Not found")
                continue
            remote_hash = hashlib.sha256(resp.content).hexdigest()
            if remote_hash == local_hash:
                print("✅ OK")
            else:
                print("⚠️ Hash mismatch")
        except Exception as e:
            print("❌ Error:", e)


def main():
    parser = argparse.ArgumentParser(
        description="Context encoder tools for exporting and uploading GPT-2 and ViT encoders.",
        epilog="""\
Example usage:

  # Convert and export GPT-2 and ViT encoders
  python -m src.reflective_learning.tools.context convert \\
    --output-dir checkpoints/context

  # Upload context folder to Hugging Face Hub
  python -m src.reflective_learning.tools.context upload \\
    --context-dir checkpoints/context \\
    --repo-id yourname/reflective-context \\
    --token $HF_TOKEN

  # Validate uploaded files against local SHA256
  python -m src.reflective_learning.tools.context validate \\
    --context-dir checkpoints/context \\
    --repo-id yourname/reflective-context
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert HF GPT-2 and ViT models into PyTorch .pt files under a context directory.",
    )
    convert_parser.add_argument(
        "--output-dir", required=True, help="Directory to save .pt and metadata files"
    )
    convert_parser.add_argument(
        "--gpt2-model", default="gpt2", help="Hugging Face GPT-2 model ID or path"
    )
    convert_parser.add_argument(
        "--vit-model",
        default="google/vit-base-patch16-224",
        help="Hugging Face ViT model ID or path",
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload the context directory to Hugging Face model repo",
    )
    upload_parser.add_argument(
        "--context-dir", required=True, help="Path to the context/ folder"
    )
    upload_parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repo ID (e.g., username/repo)",
    )
    upload_parser.add_argument(
        "--token", required=True, help="Hugging Face token for authentication"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate uploaded files on Hugging Face match local SHA256 hashes",
    )
    validate_parser.add_argument(
        "--context-dir", required=True, help="Local context/ folder path"
    )
    validate_parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repo ID (e.g., username/repo)",
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_context_encoders(args.output_dir, args.gpt2_model, args.vit_model)
    elif args.command == "upload":
        upload_context_to_hf(args.context_dir, args.repo_id, args.token)
    elif args.command == "validate":
        validate_context_upload(args.context_dir, args.repo_id)


if __name__ == "__main__":
    main()
