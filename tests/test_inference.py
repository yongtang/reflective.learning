import torch

from src.reflective_learning.inference import sequence, sequence_batched
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
    state_weights = {0: 0.7, 1: 0.3}
    stop_token = 0
    max_seq_len = 10

    tokens = sequence(
        model=model,
        prefix=prefix,
        state_weights=state_weights,
        stop_token=stop_token,
        max_seq_len=max_seq_len,
        device="cpu",
    )

    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0
    if stop_token in tokens:
        assert tokens.index(stop_token) < max_seq_len
    else:
        assert len(tokens) <= max_seq_len


def _test_sequence_batched():
    model = make_dummy_model()
    prefixes = torch.randn(4, 3, 16)  # [B=4, C=3, d_model=16]
    state_weights = {0: 0.6, 1: 0.4}
    stop_token = 0
    max_seq_len = 10

    sequences = sequence_batched(
        model=model,
        prefixes=prefixes,
        state_weights=state_weights,
        stop_token=stop_token,
        max_seq_len=max_seq_len,
        device="cpu",
    )

    assert isinstance(sequences, list)
    assert len(sequences) == 4
    for seq in sequences:
        assert isinstance(seq, list)
        assert all(isinstance(t, int) for t in seq)
        assert len(seq) > 0
        if stop_token in seq:
            assert seq.index(stop_token) < max_seq_len
        else:
            assert len(seq) <= max_seq_len
