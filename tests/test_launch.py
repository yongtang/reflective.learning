import json
import pathlib

import torch

from reflective_learning.launch import launch


def test_launch_distributed(tmp_path):
    """
    Test the distributed DDP launch mechanism.

    This test verifies that:
      - The correct number of ranks are launched
      - Each rank receives the correct world_size and device
      - The callback is invoked on each rank with the correct arguments
    """
    file = tmp_path / "model.pt"
    data = tmp_path / "dataset"
    data.mkdir()

    file.write_text("dummy model content")

    # Callback invoked on each rank by launch()
    def f(
        model_file,
        dataset_file,
        callback_fn,
        datum_fn,
        choice,
        total,
        batch,
        interval,
        lr,
        device,
        rank,
        world_size,
        distributed,
    ):
        # Record a small JSON file per rank to verify distributed info
        rank_file = pathlib.Path(dataset_file) / f"rank_{rank}.json"
        rank_file.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "device": str(device),
                    "distributed": bool(distributed),
                }
            )
        )

    # Launch DDP simulation with 2 ranks on CPU
    launch(
        callback=f,
        model_file=str(file),
        dataset_file=str(data),
        callback_fn=None,
        datum_fn=None,
        choice="dummy",
        total=4,
        batch=2,
        interval=1,
        lr=1e-3,
        device=["cpu", "cpu"],  # simulate 2-rank CPU launch
    )

    # Verify the output JSON files exist and contain correct info
    for rank in range(2):
        rank_file = data / f"rank_{rank}.json"
        assert rank_file.exists()
        j = json.loads(rank_file.read_text())
        assert j["rank"] == rank
        assert j["world_size"] == 2
        assert j["distributed"] is True
        assert j["device"] == "cpu"
