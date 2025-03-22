import pytest
import torch

from src.reflective_learning.tools.encoder import ContextEncoder


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}


class DummyTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        return type(
            "Output", (), {"last_hidden_state": torch.ones((batch_size, seq_len, 4))}
        )


class DummyImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        return type(
            "Output", (), {"last_hidden_state": torch.ones((batch_size, 197, 4))}
        )


class DummyImageProcessor:
    def __call__(self, images, return_tensors):
        return type("Output", (), {"pixel_values": torch.randn(1, 3, 224, 224)})()


def test_combined_encoding(tmp_path):
    encoder = ContextEncoder(
        text_model=DummyTextModel(),
        image_model=DummyImageModel(),
        tokenizer=DummyTokenizer(),
        image_processor=DummyImageProcessor(),
        device="cpu",
    )

    from PIL import Image

    def dummy_open(path):
        return Image.new("RGB", (224, 224))

    Image.open = dummy_open

    output = encoder.encode(["some text"], ["fake.jpg"])
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2  # [seq_len, dim]
    assert output.shape[1] == 4  # hidden dim


def test_missing_required_args_raises():
    with pytest.raises(TypeError, match="missing .* arguments"):
        ContextEncoder()


def test_assert_missing_text_model():
    with pytest.raises(AssertionError, match="text_model is required"):
        ContextEncoder(
            text_model=None,
            image_model=DummyImageModel(),
            tokenizer=DummyTokenizer(),
            image_processor=DummyImageProcessor(),
            device="cpu",
        )
