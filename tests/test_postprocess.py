import json
from pathlib import Path

from src.reflective_learning.postprocess import postprocess


def test_postprocess_roundtrip(tmp_path):
    # Simulate generated numeric output (after inference)
    input_file = tmp_path / "generated.json"
    with input_file.open("w") as f:
        f.write(json.dumps({"token": [0, 1, 0], "state": 1}) + "\n")

    # Simulate mapping used during preprocessing
    mapping_file = tmp_path / "mapping.json"
    vocab_map = {"A": 0, "B": 1}
    state_map = {"success": 0, "fail": 1}
    with mapping_file.open("w") as f:
        json.dump({"vocab": vocab_map, "state": state_map}, f)

    # Output file
    output_file = tmp_path / "readable.json"

    # Run postprocess
    postprocess(str(input_file), str(mapping_file), str(output_file))

    # Validate output
    with output_file.open() as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 1
    result = lines[0]
    assert result["token"] == ["A", "B", "A"]
    assert result["state"] == "fail"
