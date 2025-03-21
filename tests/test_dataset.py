import json
import torch
from torch.utils.data import DataLoader
from src.reflective_learning.dataset import ReflectiveDataset


def create_test_file(path, data):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def test_fixed_length_batching(tmp_path):
    # Create one test file
    dataset_path = tmp_path / "fixed.json"
    data = [{"token": [1, 2, 0], "state": 0}, {"token": [3, 4, 5, 0], "state": 1}]
    create_test_file(dataset_path, data)

    max_seq_len = 6
    dataset = ReflectiveDataset(str(dataset_path), max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    tokens, states = next(iter(dataloader))

    assert tokens.shape == (2, max_seq_len)
    assert states.shape == (2, max_seq_len)
    assert (states[0] == states[0][0]).all()
    assert (states[1] == states[1][0]).all()


def test_variable_length_mode(tmp_path):
    # Create a dataset file
    dataset_path = tmp_path / "variable.json"
    data = [{"token": [1, 2, 0], "state": 0}, {"token": [3, 4, 5, 0], "state": 1}]
    create_test_file(dataset_path, data)

    dataset = ReflectiveDataset(str(dataset_path))  # No max_seq_len
    dataloader = DataLoader(
        dataset, batch_size=2, collate_fn=lambda x: x, shuffle=False
    )

    batch = next(iter(dataloader))
    assert len(batch) == 2

    for tokens, states in batch:
        assert tokens.shape == states.shape
        assert (states == states[0]).all()  # uniform state per row


def test_multiple_files_combined(tmp_path):
    file1 = tmp_path / "a.json"
    file2 = tmp_path / "b.json"
    data1 = [{"token": [1, 0], "state": 0}]
    data2 = [{"token": [2, 3, 0], "state": 1}]
    create_test_file(file1, data1)
    create_test_file(file2, data2)

    dataset = ReflectiveDataset([str(file1), str(file2)], max_seq_len=4)
    assert len(dataset) == 2

    tokens, states = dataset[0]
    assert tokens.shape == states.shape == torch.Size([4])
