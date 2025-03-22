import json
from pathlib import Path

import pytest
import torch

from src.reflective_learning.tools.preprocess import preprocess_textual_json


class DummyContextEncoder:
    def encode(self, text, image_path):
        return torch.tensor([0.1, 0.2, 0.3])


def test_preprocess_with_context(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {
            "token": ["X1"],
            "state": "S1",
            "text": "Start location is A and goal is B.",
            "image": "some_image.jpg",
        },
    ]

    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    vocab_map = {"X1": 0}
    state_map = {"S1": 0}
    encoder = DummyContextEncoder()

    preprocess_textual_json(input_path, output_path, vocab_map, state_map, encoder)

    with output_path.open() as f:
        lines = list(f)
        assert len(lines) == 1
        example = json.loads(lines[0])
        assert example["token"] == [0]
        assert example["state"] == 0
        assert "prefix" in example
        assert example["prefix"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)


def test_preprocess_context_missing_fields(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {"token": ["X1"], "state": "S1", "image": "some_image.jpg"},
        {"token": ["X2"], "state": "S2", "text": "no image provided"},
    ]

    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    vocab_map = {"X1": 0, "X2": 1}
    state_map = {"S1": 0, "S2": 1}
    encoder = DummyContextEncoder()

    with pytest.raises(RuntimeError, match="Missing 'text' or 'image'"):
        preprocess_textual_json(input_path, output_path, vocab_map, state_map, encoder)
