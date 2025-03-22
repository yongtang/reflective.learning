import pytest
import torch

from reflective_learning.model import ReflectiveCore


def test_reflective_transformer_forward_and_loss():
    vocab_size = 100
    state_size = 4
    seq_len = 8
    batch_size = 2

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dim_ff=256,
        max_seq_len=seq_len,
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    state_ids = torch.randint(0, state_size, (batch_size, seq_len))

    logits = model(token_ids, state_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size, state_size)

    loss = model.compute_loss(logits, token_ids, state_ids)
    assert loss.item() > 0
