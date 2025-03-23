import torch

from reflective_learning.model import ReflectiveCore


def test_reflective_transformer_forward_and_loss():
    vocab_size = 100
    state_size = 4
    seq_len = 8
    batch_size = 2
    d_model = 64

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        d_model=d_model,
        nhead=4,
        dim_feedforward=256,
        num_layers=2,
        max_seq_len=seq_len,
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    state_ids = torch.randint(0, state_size, (batch_size, seq_len))

    # Provide zero-length prefix to satisfy required input
    dummy_prefix = torch.zeros(batch_size, 0, d_model)

    logits = model(token_ids, state_ids, prefix=dummy_prefix)
    assert logits.shape == (batch_size, seq_len, vocab_size, state_size)

    loss = model.loss(logits, token_ids, state_ids)
    assert loss.item() > 0
