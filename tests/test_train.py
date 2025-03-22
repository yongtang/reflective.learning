import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reflective_learning.train import train


def test_train_runs_end_to_end(tmp_path):

    def dummy_prefix(context_len=2, d_model=32):
        array = np.zeros((context_len, d_model), dtype=np.float32)
        return base64.b64encode(array.tobytes()).decode("utf-8")

    # Prepare a small dummy dataset
    train_file = tmp_path / "train.json"
    data = [
        {"token": [1, 2, 0], "state": 0, "prefix": dummy_prefix()},
        {"token": [2, 3, 0], "state": 1, "prefix": dummy_prefix()},
    ]
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
        nhead=2,
        dim_feedforward=64,
        num_layers=1,
    )

    # Checkpoint file should exist and be loadable
    assert save_path.exists()
    state_dict = torch.load(save_path)
    assert isinstance(state_dict, dict)
