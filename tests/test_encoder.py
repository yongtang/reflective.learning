import pytest
import torch

from src.reflective_learning.tools.encoder import ContextEncoder


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}


class DummyTextModel(torch.nn.Module):
    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        return type(
            "Output", (), {"last_hidden_state": torch.ones((batch_size, seq_len, 4))}
        )


class DummyImageModel(torch.nn.Module):
    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        return type(
            "Output", (), {"last_hidden_state": torch.ones((batch_size, 197, 6))}
        )


def test_combined_encoding(tmp_path):
    encoder = ContextEncoder(
        text_model=DummyTextModel(),
        image_model=DummyImageModel(),
        tokenizer=DummyTokenizer(),
        device="cpu",
    )

    from PIL import Image

    def dummy_open(path):
        return Image.new("RGB", (224, 224))

    Image.open = dummy_open

    output = encoder.encode("some text", "fake.jpg")
    assert isinstance(output, torch.Tensor)
    assert output.shape == (10,)


def test_missing_required_args_raises():
    with pytest.raises(TypeError, match="missing .* arguments"):
        ContextEncoder()


def test_text_only_encoding():
    encoder = ContextEncoder(
        text_model=DummyTextModel(),
        image_model=None,
        tokenizer=DummyTokenizer(),
        device="cpu",
    )
    with pytest.raises(ValueError, match="image_model must be set"):
        encoder.encode("text only", "fake.jpg")


def test_image_only_encoding():
    encoder = ContextEncoder(
        text_model=None, image_model=DummyImageModel(), tokenizer=None, device="cpu"
    )
    with pytest.raises(ValueError, match="text_model and tokenizer must be set"):
        encoder.encode("text goes here", "fake.jpg")
