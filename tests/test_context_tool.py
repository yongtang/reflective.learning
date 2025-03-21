import tempfile
import torch
from unittest import mock
from src.reflective_learning.tools import context


def test_convert_context_encoders_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        context.convert_context(
            output_dir=tmpdir,
            input_json=None,
            output_json=None,
            gpt2_model="gpt2",
            vit_model="google/vit-base-patch16-224",
        )

        # Check core output files exist
        text_weights = f"{tmpdir}/text_weights.pt"
        image_weights = f"{tmpdir}/image_weights.pt"
        version_file = f"{tmpdir}/context_versions.json"

        assert torch.load(text_weights)
        assert torch.load(image_weights)
        assert open(version_file).read().startswith("{")


@mock.patch("huggingface_hub.HfApi.upload_file")
@mock.patch("huggingface_hub.HfApi.create_repo")
def test_upload_context_to_hf_mocks_upload(create_repo, upload_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy files
        for fname in ["text_weights.pt", "image_weights.pt", "context_versions.json"]:
            with open(f"{tmpdir}/{fname}", "w") as f:
                f.write("dummy")

        context.upload_context_to_hf(
            context_dir=tmpdir, repo_id="mock/test-repo", hf_token="dummy_token"
        )

        create_repo.assert_called_once()
        assert upload_file.call_count == 3
