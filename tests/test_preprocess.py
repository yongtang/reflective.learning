import json
from pathlib import Path
from src.reflective_learning import preprocess


def test_preprocess_textual_json(tmp_path):
    # Setup input/output paths
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    # Sample input: two examples
    input_data = [
        {"token": ["X1", "X2", "X1"], "state": "S1"},
        {"token": ["X2", "X1"], "state": "S2"},
    ]
    with input_path.open("w") as f:
        for line in input_data:
            f.write(json.dumps(line) + "\n")

    # Define mappings
    vocab_map = {"X1": 0, "X2": 1}
    state_map = {"S1": 0, "S2": 1}

    # Run preprocessing
    preprocess.preprocess_textual_json(input_path, output_path, vocab_map, state_map)

    # Read and validate output
    with output_path.open() as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["token"] == [0, 1, 0]
    assert lines[0]["state"] == 0
    assert lines[1]["token"] == [1, 0]
    assert lines[1]["state"] == 1
