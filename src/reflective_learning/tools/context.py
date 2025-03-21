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
            path_or_fileobj=path, path_in_repo=f, repo_id=repo_id, token=hf_token
        )

    print("✅ Upload complete.")


def validate_context_upload(context_dir, repo_id):
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    metadata_path = os.path.join(context_dir, "context_versions.json")
    with open(metadata_path) as f:
        meta = json.load(f)

    print("🔍 Validating uploaded files:")
    for fname, local_hash in meta.get("hashes", {}).items():
        url = f"{base_url}/{fname}"
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
        description="Context encoder tools (convert, upload, validate)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert HF GPT-2 and ViT to PyTorch .pt"
    )
    convert_parser.add_argument("--output-dir", required=True)
    convert_parser.add_argument("--gpt2-model", default="gpt2")
    convert_parser.add_argument("--vit-model", default="google/vit-base-patch16-224")

    upload_parser = subparsers.add_parser(
        "upload", help="Upload context folder to Hugging Face Hub"
    )
    upload_parser.add_argument("--context-dir", required=True)
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--token", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate SHA256 of uploaded files"
    )
    validate_parser.add_argument("--context-dir", required=True)
    validate_parser.add_argument("--repo-id", required=True)

    args = parser.parse_args()

    if args.command == "convert":
        convert_context_encoders(args.output_dir, args.gpt2_model, args.vit_model)
    elif args.command == "upload":
        upload_context_to_hf(args.context_dir, args.repo_id, args.token)
    elif args.command == "validate":
        validate_context_upload(args.context_dir, args.repo_id)


if __name__ == "__main__":
    main()
