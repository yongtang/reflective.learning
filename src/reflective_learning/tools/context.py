import argparse
import os
import json
import hashlib
import torch
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Model, ViTModel
from huggingface_hub import HfApi


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def convert_context(
    output_dir,
    input_json=None,
    output_json=None,
    gpt2_model="gpt2",
    vit_model="google/vit-base-patch16-224",
    text_field="text",
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️  Loading GPT-2 model: {gpt2_model}")
    tokenizer = AutoTokenizer.from_pretrained(gpt2_model)
    gpt2 = GPT2Model.from_pretrained(gpt2_model)
    gpt2.eval()
    torch.save(gpt2.state_dict(), os.path.join(output_dir, "text_weights.pt"))

    print(f"⚙️  Saving GPT-2 vocab and merges for tokenizer-free usage")
    print(f"⚙️  Extracting tokenizer vocab/merges using backend_tokenizer")
    tokenizer.backend_tokenizer.model.save(output_dir)

    os.rename(
        os.path.join(output_dir, "vocab.json"),
        os.path.join(output_dir, "text_vocab.json"),
    )
    os.rename(
        os.path.join(output_dir, "merges.txt"),
        os.path.join(output_dir, "text_merges.txt"),
    )

    print(f"⚙️  Loading ViT model: {vit_model}")
    vit = ViTModel.from_pretrained(vit_model)
    vit.eval()
    torch.save(vit.state_dict(), os.path.join(output_dir, "image_weights.pt"))

    metadata = {
        "transformers_version": torch.__version__,
        "torch_version": torch.__version__,
        "text_encoder": {
            "model_name": gpt2_model,
            "weights_file": "text_weights.pt",
            "vocab_file": "text_vocab.json",
            "merges_file": "text_merges.txt",
        },
        "image_encoder": {
            "model_name": vit_model,
            "weights_file": "image_weights.pt",
        },
    }

    for name in [
        "text_weights.pt",
        "image_weights.pt",
        "text_vocab.json",
        "text_merges.txt",
    ]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            metadata.setdefault("hashes", {})[name] = sha256sum(path)

    with open(os.path.join(output_dir, "context_versions.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if input_json and output_json:
        print(f"📝 Tokenizing text from {input_json} → {output_json}")
        with open(input_json, "r") as f_in, open(output_json, "w") as f_out:
            for line in tqdm(f_in, desc="Tokenizing"):
                record = json.loads(line)
                text = record.get(text_field, "")
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                record["text_tokens"] = token_ids
                f_out.write(json.dumps(record) + "\n")

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
        description="Context encoder tools for exporting, tokenizing, and uploading GPT-2 and ViT encoders.",
        epilog="""
Examples:

  # Convert and tokenize context data
  python tools/context.py convert \
    --output-dir checkpoints/context \
    --input-json data/raw_context.json \
    --output-json data/tokenized_context.json

  # Upload to Hugging Face
  python tools/context.py upload \
    --context-dir checkpoints/context \
    --repo-id yourname/reflective-context \
    --token $HF_TOKEN

  # Validate uploaded files
  python tools/context.py validate \
    --context-dir checkpoints/context \
    --repo-id yourname/reflective-context
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert HF GPT-2 and ViT models to .pt and tokenize context JSON.",
    )
    convert_parser.add_argument(
        "--output-dir", required=True, help="Directory to save weights and metadata"
    )
    convert_parser.add_argument(
        "--input-json", help="Path to line-separated JSON with raw context data"
    )
    convert_parser.add_argument(
        "--output-json", help="Path to write updated context with text_tokens"
    )
    convert_parser.add_argument(
        "--gpt2-model", default="gpt2", help="Hugging Face GPT-2 model ID or path"
    )
    convert_parser.add_argument(
        "--vit-model",
        default="google/vit-base-patch16-224",
        help="Hugging Face ViT model ID or path",
    )
    convert_parser.add_argument(
        "--text-field", default="text", help="Name of the text field to tokenize"
    )

    upload_parser = subparsers.add_parser(
        "upload", help="Upload context directory to Hugging Face Hub"
    )
    upload_parser.add_argument("--context-dir", required=True)
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--token", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate uploaded files against SHA256 hash"
    )
    validate_parser.add_argument("--context-dir", required=True)
    validate_parser.add_argument("--repo-id", required=True)

    args = parser.parse_args()

    if args.command == "convert":
        convert_context(
            output_dir=args.output_dir,
            input_json=args.input_json,
            output_json=args.output_json,
            gpt2_model=args.gpt2_model,
            vit_model=args.vit_model,
            text_field=args.text_field,
        )
    elif args.command == "upload":
        upload_context_to_hf(args.context_dir, args.repo_id, args.token)
    elif args.command == "validate":
        validate_context_upload(args.context_dir, args.repo_id)


if __name__ == "__main__":
    main()
