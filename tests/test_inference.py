import torch

from src.reflective_learning.inference import sequence
from src.reflective_learning.model import ReflectiveCore


def make_dummy_model():
    decoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
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
    weight = torch.tensor([0.7, 0.3])  # [S]
    maximum = 10
    conditioned = False

    tokens = sequence(
        model=model,
        prefix=prefix,
        weight=weight,
        maximum=maximum,
        conditioned=conditioned,
        device="cpu",
    )

    # Updated assertions for torch.Tensor return type
    tokens = tokens
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape[0] > 0
    if 0 in tokens:
        assert (tokens == 0).nonzero(as_tuple=False)[0].item() < maximum
    else:
        assert tokens.shape[0] <= maximum
