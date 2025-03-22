import json
from pathlib import Path

import torch

from src.reflective_learning.inference import sample_multiple_sequences
from src.reflective_learning.model import ReflectiveCore


def test_sample_multiple_sequences(tmp_path):
    # ---- Create and save a dummy model ----
    vocab_size = 5
    state_size = 2
    max_seq_len = 10

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=max_seq_len,
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_ff=32,
    )

    # Initialize randomly and save checkpoint
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Load it back
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # ---- Sample ----
    state_weights = {0: 0.5, 1: 0.5}
    sequences = sample_multiple_sequences(
        model,
        state_weights=state_weights,
        num_sequences=5,
        start_token=0,
        max_len=max_seq_len,
        temperature=1.0,
        stop_token=0,
        device="cpu",
    )

    assert len(sequences) == 5
    for seq in sequences:
        assert isinstance(seq, list)
        assert all(isinstance(t, int) for t in seq)
        # Ensure the STOP token ends the sequence, or we hit max_len with no STOP
        if seq[-1] == 0:
            assert len(seq) <= max_seq_len
        else:
            assert len(seq) == max_seq_len
