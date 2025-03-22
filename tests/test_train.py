import json
from pathlib import Path

import torch

from src.reflective_learning.train import train


def test_train_runs_end_to_end(tmp_path):
    # Prepare a small dummy dataset
    train_file = tmp_path / "train.json"
    data = [{"token": [1, 2, 0], "state": 0}, {"token": [2, 3, 0], "state": 1}]
    with train_file.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    # Define training parameters (with small model for testing)
    vocab_size = 4
    state_size = 2
    max_seq_len = 5
    save_path = tmp_path / "checkpoints" / "model.pt"

    # Run training
    train(
        json_paths=[str(train_file)],
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=max_seq_len,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        save_path=str(save_path),
        device="cpu",  # Force CPU for test environment
        d_model=32,  # Smaller model to avoid memory issues
        n_layers=1,
        n_heads=2,
        dim_ff=64,
    )

    # Checkpoint file should exist and be loadable
    assert save_path.exists()
    state_dict = torch.load(save_path)
    assert isinstance(state_dict, dict)
