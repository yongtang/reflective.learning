import json
import torch
from torch.utils.data import DataLoader
from src.reflective_learning.dataset import ReflectiveDataset


def test_reflective_dataset_fixed_length(tmp_path):
    # Create a temporary dataset file
    dataset_path = tmp_path / "test_dataset_fixed.json"
    data = [{"token": [2, 3, 0], "state": 0}, {"token": [8, 8, 7, 8, 0], "state": 1}]
    with dataset_path.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    max_seq_len = 6
    dataset = ReflectiveDataset(str(dataset_path), max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    tokens, states = next(iter(dataloader))

    assert tokens.shape == (2, max_seq_len)
    assert states.shape == (2, max_seq_len)

    # Verify padding applied
    for i in range(2):
        assert tokens[i, -1].item() == 0  # Expect STOP or padding token
        assert (states[i] == states[i][0]).all()  # All states in row are same


def test_reflective_dataset_variable_length(tmp_path):
    # Create a temporary dataset file
    dataset_path = tmp_path / "test_dataset_variable.json"
    data = [{"token": [2, 3, 0], "state": 0}, {"token": [8, 8, 7, 8, 0], "state": 1}]
    with dataset_path.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    dataset = ReflectiveDataset(str(dataset_path))  # no max_seq_len
    dataloader = DataLoader(
        dataset, batch_size=2, collate_fn=lambda x: x, shuffle=False
    )

    batch = next(iter(dataloader))
    assert len(batch) == 2

    for token_ids, state_ids in batch:
        assert token_ids.shape == state_ids.shape
        assert token_ids.dtype == torch.long
        assert state_ids.dtype == torch.long
        assert (state_ids == state_ids[0]).all()


def test_reflective_dataset_multiple_files(tmp_path):
    file1 = tmp_path / "file1.json"
    file2 = tmp_path / "file2.json"

    data1 = [{"token": [1, 2, 0], "state": 0}]
    data2 = [{"token": [3, 4, 0], "state": 1}]

    for file, data in [(file1, data1), (file2, data2)]:
        with open(file, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    dataset = ReflectiveDataset([str(file1), str(file2)], max_seq_len=5)

    assert len(dataset) == 2
    token_ids_0, state_ids_0 = dataset[0]
    token_ids_1, state_ids_1 = dataset[1]

    assert token_ids_0.shape == state_ids_0.shape == torch.Size([5])
    assert token_ids_1.shape == state_ids_1.shape == torch.Size([5])

    assert (state_ids_0 == 0).all()
    assert (state_ids_1 == 1).all()
