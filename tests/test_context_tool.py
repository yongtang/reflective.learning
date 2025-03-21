import tempfile
import torch
import os
import json
import subprocess
import hashlib
from unittest import mock
from src.reflective_learning.tools import context
import pytest


def test_convert_context_encoders_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch(
            "transformers.GPT2Model.from_pretrained"
        ) as mock_gpt2, mock.patch(
            "transformers.ViTModel.from_pretrained"
        ) as mock_vit, mock.patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer:

            mock_gpt2.return_value = mock.MagicMock()
            mock_gpt2.return_value.eval = lambda: None
            mock_gpt2.return_value.state_dict.return_value = {
                "dummy": torch.tensor([1])
            }

            mock_vit.return_value = mock.MagicMock()
            mock_vit.return_value.eval = lambda: None
            mock_vit.return_value.state_dict.return_value = {"dummy": torch.tensor([1])}

            mock_tok = mock.MagicMock()
            mock_tok.get_vocab.return_value = {"hello": 42, "world": 43}
            mock_tok.backend_tokenizer.model.save = lambda path: None
            mock_tokenizer.return_value = mock_tok

            with open(os.path.join(tmpdir, "vocab.json"), "w") as f:
                json.dump({"hello": 1, "world": 2}, f)

            with open(os.path.join(tmpdir, "merges.txt"), "w") as f:
                f.write("#version: 0.2\nh e\nhe llo\n")

            context.convert_context(
                output_dir=tmpdir,
                input_json=None,
                output_json=None,
                gpt2_model="gpt2",
                vit_model="google/vit-base-patch16-224",
            )

        assert os.path.exists(f"{tmpdir}/text_weights.pt")
        assert os.path.exists(f"{tmpdir}/image_weights.pt")
        assert os.path.exists(f"{tmpdir}/context_versions.json")

        vocab_file = os.path.join(tmpdir, "text_vocab.json")
        merges_file = os.path.join(tmpdir, "text_merges.txt")

        assert os.path.exists(vocab_file)
        assert os.path.exists(merges_file)

        with open(vocab_file, "r") as f:
            vocab = json.load(f)
            assert isinstance(vocab, dict)
            assert all(isinstance(k, str) and isinstance(vocab[k], int) for k in vocab)

        with open(merges_file, "r") as f:
            lines = [line for line in f if not line.startswith("#")]
            assert len(lines) > 0
            for line in lines:
                parts = line.strip().split()
                assert len(parts) == 2


@mock.patch("huggingface_hub.HfApi.upload_file")
@mock.patch("huggingface_hub.HfApi.create_repo")
def test_upload_context_to_hf_mocks_upload(create_repo, upload_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in ["text_weights.pt", "image_weights.pt", "context_versions.json"]:
            with open(f"{tmpdir}/{fname}", "w") as f:
                f.write("dummy")

        context.upload_context_to_hf(
            context_dir=tmpdir, repo_id="mock/test-repo", hf_token="dummy_token"
        )

        create_repo.assert_called_once()
        assert upload_file.call_count == 3


def test_tokenization_output():
    sample_data = {"text": "Hello world!"}
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "raw.json")
        output_path = os.path.join(tmpdir, "tokenized.json")

        with open(input_path, "w") as f:
            f.write(json.dumps(sample_data) + "\n")

        with mock.patch(
            "transformers.GPT2Model.from_pretrained"
        ) as mock_gpt2, mock.patch(
            "transformers.ViTModel.from_pretrained"
        ) as mock_vit, mock.patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer:

            mock_gpt2.return_value = mock.MagicMock()
            mock_gpt2.return_value.eval = lambda: None
            mock_gpt2.return_value.state_dict.return_value = {
                "dummy": torch.tensor([1])
            }

            mock_vit.return_value = mock.MagicMock()
            mock_vit.return_value.eval = lambda: None
            mock_vit.return_value.state_dict.return_value = {"dummy": torch.tensor([1])}

            mock_tok = mock.MagicMock()
            mock_tok.encode.return_value = [1, 2, 3]
            mock_tok.get_vocab.return_value = {"hello": 1, "world": 2}
            mock_tok.backend_tokenizer.model.save = lambda path: None
            mock_tokenizer.return_value = mock_tok

            with open(os.path.join(tmpdir, "vocab.json"), "w") as f:
                json.dump({"hello": 1, "world": 2}, f)

            with open(os.path.join(tmpdir, "merges.txt"), "w") as f:
                f.write("#version: 0.2\nh e\nhe llo\n")

            context.convert_context(
                output_dir=tmpdir,
                input_json=input_path,
                output_json=output_path,
                gpt2_model="gpt2",
                vit_model="google/vit-base-patch16-224",
            )

        with open(output_path, "r") as f:
            result = json.loads(f.readline())
            assert "text_tokens" in result
            assert isinstance(result["text_tokens"], list)
            assert all(isinstance(t, int) for t in result["text_tokens"])


@mock.patch("requests.get")
def test_validate_cli_runs(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"dummy"

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_content = b"dummy"
        for fname in ["text_weights.pt", "image_weights.pt"]:
            with open(os.path.join(tmpdir, fname), "wb") as f:
                f.write(dummy_content)

        hashes = {
            fname: hashlib.sha256(dummy_content).hexdigest()
            for fname in ["text_weights.pt", "image_weights.pt"]
        }
        with open(os.path.join(tmpdir, "context_versions.json"), "w") as f:
            json.dump({"hashes": hashes}, f)

        script_path = os.path.abspath("src/reflective_learning/tools/context.py")

        result = subprocess.run(
            [
                "python3",
                script_path,
                "validate",
                "--context-dir",
                tmpdir,
                "--repo-id",
                "mock/test-repo",
            ],
            capture_output=True,
            text=True,
        )

        print("VALIDATE STDOUT:\n", result.stdout)
        print("VALIDATE STDERR:\n", result.stderr)
        assert result.returncode == 0
