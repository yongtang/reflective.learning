import json
import torch
import pytest
from pathlib import Path
from src.reflective_learning import preprocess


class DummyContextEncoder:
    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def encode(self, text_ids, image_path):
        return torch.tensor([1.0, 2.0, 3.0, 4.0])


def test_preprocess_textual_json(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {"token": ["X1", "X2", "X1"], "state": "S1"},
        {"token": ["X2", "X1"], "state": "S2"},
    ]
    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    vocab_map = {"X1": 0, "X2": 1}
    state_map = {"S1": 0, "S2": 1}

    preprocess.preprocess_textual_json(input_path, output_path, vocab_map, state_map)

    with output_path.open() as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["token"] == [0, 1, 0]
    assert lines[0]["state"] == 0
    assert lines[1]["token"] == [1, 0]
    assert lines[1]["state"] == 1
    assert "prefix" not in lines[0]


def test_preprocess_with_context(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {
            "token": ["X1"],
            "state": "S1",
            "text": [101, 102],
            "image": "path/to/image1.png",
        },
        {"token": ["X2"], "state": "S2", "text": [], "image": ""},
    ]
    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    vocab_map = {"X1": 0, "X2": 1}
    state_map = {"S1": 0, "S2": 1}
    dummy_encoder = DummyContextEncoder()

    preprocess.preprocess_textual_json(
        input_path, output_path, vocab_map, state_map, dummy_encoder
    )

    with output_path.open() as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert "prefix" in lines[0]
    assert lines[0]["prefix"] == [1.0, 2.0, 3.0, 4.0]
    assert lines[1]["prefix"] == [1.0, 2.0, 3.0, 4.0]


def test_preprocess_context_missing_fields(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {"token": ["X1"], "state": "S1", "image": "some_image.jpg"},  # missing 'text'
        {"token": ["X2"], "state": "S2", "text": [123]},  # missing 'image'
    ]
    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    vocab_map = {"X1": 0, "X2": 1}
    state_map = {"S1": 0, "S2": 1}
    dummy_encoder = DummyContextEncoder()

    with pytest.raises(ValueError, match="Missing 'text' or 'image'"):
        preprocess.preprocess_textual_json(
            input_path, output_path, vocab_map, state_map, dummy_encoder
        )
