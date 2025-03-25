import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.reflective_learning.tools.main import run_preprocess


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

    mapping = {"vocab": {"X1": 0}, "state": {"S1": 0}}
    mapping_path = tmp_path / "mapping.json"
    with mapping_path.open("w") as f:
        json.dump(mapping, f)

    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        mapping=str(mapping_path),
        context_dir="mock",  # triggers mock below
        device="cpu",
        prefix_only=False,
    )

    # Inject dummy encoder
    from src.reflective_learning.tools import main

    main.ContextEncoder.from_pretrained = lambda *a, **kw: DummyContextEncoder()

    run_preprocess(args)

    with output_path.open() as f:
        example = json.loads(f.readline())

    assert example["token"] == [0]
    assert example["state"] == 0
    assert "prefix" in example
    assert example["prefix"].startswith("b64://")

    decoded = base64.b64decode(example["prefix"].removeprefix("b64://"))
    tensor = torch.from_numpy(np.frombuffer(decoded, dtype=np.float32).copy())
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

    mapping = {"vocab": {"X1": 0, "X2": 1}, "state": {"S1": 0, "S2": 1}}
    mapping_path = tmp_path / "mapping.json"
    with mapping_path.open("w") as f:
        json.dump(mapping, f)

    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        mapping=str(mapping_path),
        context_dir="mock",
        device="cpu",
        prefix_only=False,
    )

    from src.reflective_learning.tools import main

    main.ContextEncoder.from_pretrained = lambda *a, **kw: DummyContextEncoder()

    with pytest.raises(RuntimeError, match="Missing 'text' or 'image'"):
        run_preprocess(args)


def test_preprocess_prefix_only_mode(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_data = [
        {
            "token": ["X1"],
            "state": "S1",
            "text": ["a message"],
            "image": ["img.jpg"],
            "extra": "keep me",
        }
    ]

    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    mapping = {"vocab": {"X1": 0}, "state": {"S1": 0}}
    mapping_path = tmp_path / "mapping.json"
    with mapping_path.open("w") as f:
        json.dump(mapping, f)

    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        mapping=str(mapping_path),
        context_dir="mock",
        device="cpu",
        prefix_only=True,  # <- only prefix and original fields should remain
    )

    from src.reflective_learning.tools import main

    main.ContextEncoder.from_pretrained = lambda *a, **kw: DummyContextEncoder()

    run_preprocess(args)

    with output_path.open() as f:
        example = json.loads(f.readline())

    assert "prefix" in example
    assert "token" not in example
    assert "state" not in example
    assert "extra" in example and example["extra"] == "keep me"
