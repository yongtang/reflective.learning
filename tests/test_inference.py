import torch

from src.reflective_learning.inference import sequence
from src.reflective_learning.model import ReflectiveCore


def make_dummy_model():
    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            batch_first=True,
        ),
        num_layers=1,
    )
    return ReflectiveCore(
        vocab_size=5,
        state_size=2,
        max_seq_len=10,
        max_prefix_len=4,
        decoder=decoder,
    )


def test_sequence():
    model = make_dummy_model()
    prefix = torch.randn(4, 16)  # [C, d_model]
    weights = {0: 0.7, 1: 0.3}
    maximum = 10

    tokens = sequence(
        model=model,
        prefix=prefix,
        weights=weights,
        maximum=maximum,
        device="cpu",
    )

    # Updated assertions for torch.Tensor return type
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape[0] > 0
    if stop_token in tokens:
        assert (tokens == stop_token).nonzero(as_tuple=False)[0].item() < max_seq_len
    else:
        assert tokens.shape[0] <= max_seq_len
