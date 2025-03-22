import hashlib
import json
import os
import subprocess
import tempfile
from unittest import mock

import torch

from src.reflective_learning.tools import context


@mock.patch("huggingface_hub.HfApi.model_info")
def test_convert_context_versions_json_created(mock_model_info):
    mock_model_info.return_value.sha = "dummy_sha"

    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch(
            "transformers.GPT2Model.from_pretrained"
        ) as mock_gpt2, mock.patch("transformers.ViTModel.from_pretrained") as mock_vit:
            mock_gpt2.return_value = mock.MagicMock()
            mock_gpt2.return_value.eval = lambda: None
            mock_gpt2.return_value.state_dict.return_value = {
                "dummy": torch.tensor([1])
            }

            mock_vit.return_value = mock.MagicMock()
            mock_vit.return_value.eval = lambda: None
            mock_vit.return_value.state_dict.return_value = {"dummy": torch.tensor([1])}

            context.convert_context(
                output_dir=tmpdir,
                gpt2_model="gpt2",
                vit_model="google/vit-base-patch16-224",
            )

        versions_path = os.path.join(tmpdir, "context_versions.json")
        assert os.path.exists(versions_path)

        with open(versions_path, "r") as f:
            data = json.load(f)

        assert "torch_version" in data
        assert "transformers_version" in data
        assert "torchvision_version" in data
        assert "pretrained_models" in data
        assert data["pretrained_models"]["gpt2"]["model"] == "gpt2"
        assert data["pretrained_models"]["gpt2"]["revision"] == "dummy_sha"
        assert (
            data["pretrained_models"]["vit"]["model"] == "google/vit-base-patch16-224"
        )
        assert data["pretrained_models"]["vit"]["revision"] == "dummy_sha"


@mock.patch("requests.get")
def test_validate_cli_runs(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"dummy"

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_content = b"dummy"
        hashes = {
            "text_weights.pt": hashlib.sha256(dummy_content).hexdigest(),
            "image_weights.pt": hashlib.sha256(dummy_content).hexdigest(),
        }

        with open(os.path.join(tmpdir, "context_versions.json"), "w") as f:
            json.dump({"hashes": hashes}, f)

        for fname in hashes:
            with open(os.path.join(tmpdir, fname), "wb") as f:
                f.write(dummy_content)

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
