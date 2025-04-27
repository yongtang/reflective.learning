import torch

from src.reflective_learning.inference import (
    sample_multiple_sequences, sample_multiple_sequences_batched)
from src.reflective_learning.model import ReflectiveCore


def test_sample_multiple_sequences(tmp_path):
    vocab_size = 5
    state_size = 2
    max_seq_len = 10

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        max_seq_len=max_seq_len,
    )

    ckpt_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # ✅ Add a dummy prefix to satisfy the inference precondition
    dummy_prefix = torch.randn(1, model.d_model)

    state_weights = {0: 0.5, 1: 0.5}
    sequences = sample_multiple_sequences(
        model,
        state_weights=state_weights,
        num_sequences=5,
        max_seq_len=max_seq_len,
        temperature=1.0,
        prefix=dummy_prefix,
        device="cpu",
        stop_token=0,  # explicit stop_token
    )

    assert len(sequences) == 5
    for seq in sequences:
        assert isinstance(seq, list)
        assert all(isinstance(t, int) for t in seq)
        assert len(seq) > 0
        if 0 in seq:
            stop_index = seq.index(0)
            assert stop_index < max_seq_len
        else:
            assert len(seq) <= max_seq_len


def test_sample_multiple_sequences_no_stop_token(tmp_path):
    vocab_size = 5
    state_size = 2
    max_seq_len = 10

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        max_seq_len=max_seq_len,
    )

    ckpt_path = tmp_path / "no_stop_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    dummy_prefix = torch.randn(1, model.d_model)

    state_weights = {0: 0.5, 1: 0.5}
    sequences = sample_multiple_sequences(
        model,
        state_weights=state_weights,
        num_sequences=5,
        max_seq_len=max_seq_len,
        temperature=1.0,
        prefix=dummy_prefix,
        device="cpu",
        stop_token=None,  # no stop token
    )

    assert len(sequences) == 5
    for seq in sequences:
        assert isinstance(seq, list)
        assert all(isinstance(t, int) for t in seq)
        assert len(seq) > 0
        assert len(seq) <= max_seq_len
        # 0 might appear but shouldn't cause early stopping


def test_sample_multiple_sequences_batched(tmp_path):
    # ---- Create dummy model ----
    vocab_size = 5
    state_size = 2
    d_model = 16
    prefix_len = 4
    max_seq_len = 10
    num_sequences = 6

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        d_model=d_model,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        max_seq_len=max_seq_len,
    )

    # Save/load checkpoint
    ckpt_path = tmp_path / "batched_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # ---- Generate dummy prefix ----
    dummy_prefix = torch.randn(prefix_len, d_model)

    # ---- Run batched inference ----
    state_weights = {0: 0.6, 1: 0.4}
    sequences = sample_multiple_sequences_batched(
        model,
        state_weights=state_weights,
        num_sequences=num_sequences,
        max_seq_len=max_seq_len,
        temperature=1.0,
        prefix=dummy_prefix,
        device="cpu",
        stop_token=0,  # explicit stop_token
    )

    # ---- Assertions ----
    assert isinstance(sequences, list)
    assert len(sequences) == num_sequences

    for seq in sequences:
        assert isinstance(seq, list)
        assert all(isinstance(t, int) for t in seq)
        assert len(seq) > 0, "Sequence should not be empty"
        if 0 in seq:
            stop_index = seq.index(0)
            assert stop_index < max_seq_len
        else:
            assert len(seq) <= max_seq_len

    # Optional diversity check (very loose)
    unique_outputs = {tuple(seq) for seq in sequences}
    assert len(unique_outputs) > 1, "All sequences are identical — may indicate a bug"


def test_sample_multiple_sequences_batched_no_stop_token(tmp_path):
    vocab_size = 5
    state_size = 2
    d_model = 16
    prefix_len = 4
    max_seq_len = 10
    num_sequences = 6

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        d_model=d_model,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        max_seq_len=max_seq_len,
    )

    ckpt_path = tmp_path / "batched_model_no_stop.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    dummy_prefix = torch.randn(prefix_len, d_model)

    state_weights = {0: 0.6, 1: 0.4}
    sequences = sample_multiple_sequences_batched(
        model,
        state_weights=state_weights,
        num_sequences=num_sequences,
        max_seq_len=max_seq_len,
        temperature=1.0,
        prefix=dummy_prefix,
        device="cpu",
        stop_token=None,  # no stop token
    )

    assert isinstance(sequences, list)
    assert len(sequences) == num_sequences

    for seq in sequences:
        assert isinstance(seq, list)
        assert all(isinstance(t, int) for t in seq)
        assert len(seq) > 0
        assert len(seq) <= max_seq_len
