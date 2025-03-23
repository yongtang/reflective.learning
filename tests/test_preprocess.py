import base64
import json

import numpy as np
import pytest
import torch

from src.reflective_learning.tools.preprocess import preprocess_textual_json


class DummyContextEncoder:
    def encode(self, text, image):
        return torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)


def test_preprocess_with_context(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {
            "token": ["X1"],
            "state": "S1",
            "text": ["Start location is A and goal is B."],
            "image": ["some_image.jpg"],
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
        example = json.loads(f.readline())

    assert example["token"] == [0]
    assert example["state"] == 0
    assert example["prefix"].startswith("b64://")

    tensor = torch.from_numpy(
        np.frombuffer(
            base64.b64decode(example["prefix"].removeprefix("b64://")), dtype=np.float32
        ).copy()
    )

    assert tensor.shape == (4,)
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0, 4.0]), rtol=1e-5)


def test_preprocess_context_missing_fields(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {"token": ["X1"], "state": "S1", "image": ["some_image.jpg"]},
        {"token": ["X2"], "state": "S2", "text": ["no image provided"]},
    ]

    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    vocab_map = {"X1": 0, "X2": 1}
    state_map = {"S1": 0, "S2": 1}
    encoder = DummyContextEncoder()

    with pytest.raises(RuntimeError, match="Missing 'text' or 'image'"):
        preprocess_textual_json(input_path, output_path, vocab_map, state_map, encoder)
