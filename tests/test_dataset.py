import base64
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reflective_learning.dataset import ReflectiveDataset


def create_test_file(path, data):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def test_fixed_length_batching(tmp_path):
    dataset_path = tmp_path / "fixed.json"
    data = [{"token": [1, 2, 0], "state": 0}, {"token": [3, 4, 5, 0], "state": 1}]
    create_test_file(dataset_path, data)

    max_seq_len = 6
    dataset = ReflectiveDataset(str(dataset_path), max_seq_len=max_seq_len, d_model=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(dataloader))
    tokens = batch["token_ids"]
    states = batch["state_ids"]

    assert tokens.shape == (2, max_seq_len)
    assert states.shape == (2, max_seq_len)
    assert (states[0] == states[0][0]).all()
    assert (states[1] == states[1][0]).all()


def test_variable_length_mode(tmp_path):
    dataset_path = tmp_path / "variable.json"
    data = [{"token": [1, 2, 0], "state": 0}, {"token": [3, 4, 5, 0], "state": 1}]
    create_test_file(dataset_path, data)

    dataset = ReflectiveDataset(str(dataset_path), max_seq_len=None, d_model=32)
    dataloader = DataLoader(
        dataset, batch_size=2, collate_fn=lambda x: x, shuffle=False
    )

    batch = next(iter(dataloader))
    assert len(batch) == 2

    for item in batch:
        tokens = item["token_ids"]
        states = item["state_ids"]
        assert tokens.shape == states.shape
        assert (states == states[0]).all()


def test_multiple_files_combined(tmp_path):
    file1 = tmp_path / "a.json"
    file2 = tmp_path / "b.json"
    data1 = [{"token": [1, 0], "state": 0}]
    data2 = [{"token": [2, 3, 0], "state": 1}]
    create_test_file(file1, data1)
    create_test_file(file2, data2)

    dataset = ReflectiveDataset([str(file1), str(file2)], max_seq_len=4, d_model=32)
    assert len(dataset) == 2

    item = dataset[0]
    tokens = item["token_ids"]
    states = item["state_ids"]
    assert tokens.shape == states.shape == torch.Size([4])


def test_variable_length_prefix_decoding(tmp_path):
    def encode_prefix(array: np.ndarray) -> str:
        return "b64://" + base64.b64encode(array.astype(np.float32).tobytes()).decode(
            "utf-8"
        )

    d_model = 32
    context_len = 3  # variable length
    prefix_array = np.random.rand(context_len, d_model).astype(np.float32)

    data = [
        {
            "token": [1, 2, 3],
            "state": 0,
            "prefix": encode_prefix(prefix_array),
        }
    ]

    dataset_path = tmp_path / "prefix.json"
    create_test_file(dataset_path, data)

    dataset = ReflectiveDataset(str(dataset_path), max_seq_len=5, d_model=d_model)
    item = dataset[0]

    prefix = item["prefix"]
    assert isinstance(prefix, torch.Tensor)
    assert prefix.shape == (context_len, d_model)
    assert prefix.dtype == torch.float32
    assert torch.allclose(prefix, torch.tensor(prefix_array), atol=1e-6)
