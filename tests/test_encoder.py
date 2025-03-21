import torch
import pytest
from unittest.mock import patch
from src.reflective_learning.encoder import ContextEncoder


class DummyTextEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))  # avoid StopIteration

    def forward(self, x):
        return x.mean(dim=1, keepdim=True)

    def embed_text(self, text_ids):
        return torch.tensor([float(len(text_ids))] * self.embedding_dim)


class DummyImageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))  # avoid StopIteration

    def forward(self, x):
        return torch.ones((1, self.embedding_dim)) * 0.5

    def load_image(self, image_path):
        return torch.ones((1, 3, 224, 224))  # dummy tensor


def test_text_only_encoding():
    encoder = ContextEncoder(text_encoder=DummyTextEncoder(), image_encoder=None)
    result = encoder.encode([101, 102, 103], "")
    assert result.shape == (4,)
    assert torch.all(result == 3.0)


def test_image_only_encoding():
    encoder = ContextEncoder(text_encoder=None, image_encoder=DummyImageEncoder())
    with patch("os.path.exists", return_value=True):
        result = encoder.encode([], "image.jpg")
    assert result.shape == (6,)
    assert torch.allclose(result, torch.tensor([0.5] * 6))


def test_combined_encoding():
    encoder = ContextEncoder(
        text_encoder=DummyTextEncoder(embedding_dim=4),
        image_encoder=DummyImageEncoder(embedding_dim=6),
    )
    with patch("os.path.exists", return_value=True):
        result = encoder.encode([123, 456], "dummy.png")

    assert result.shape == (10,)
    assert torch.allclose(result[:4], torch.tensor([2.0] * 4))
    assert torch.allclose(result[4:], torch.tensor([0.5] * 6))


def test_missing_both_raises():
    encoder = ContextEncoder()
    with pytest.raises(ValueError, match="Both text and image encoders are missing"):
        encoder.encode([], "")


def test_text_cache_behavior():
    encoder = ContextEncoder(text_encoder=DummyTextEncoder(), image_encoder=None)

    result1 = encoder.encode([1, 2, 3], "")
    result2 = encoder.encode([1, 2, 3], "")
    assert torch.all(result1 == result2)

    result3 = encoder.encode([1, 2, 3, 4], "")
    assert not torch.all(result1 == result3)
