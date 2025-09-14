import json
import pathlib

import pytest
import torch

from reflective_learning.launch import launch


def f(
    file,
    choice,
    data,
    image,
    total,
    batch,
    interval,
    lr,
    device,
    rank,
    world_size,
    distributed,
):
    one = torch.ones(1, device=device) * (rank + 1)
    pathlib.Path(data).joinpath(f"rank_{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
                "distributed": bool(distributed),
            }
        )
    )


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cpu",
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"),
        ),
    ],
)
def test_launch(tmp_path, device):

    data = tmp_path
    file = tmp_path.joinpath("ignored.index.npy")

    launch(
        callback=f,
        file=str(file),
        choice="dummy",
        data=str(data),
        image=None,
        total=4,
        batch=2,
        interval=1,
        lr=1e-3,
        device=[device],
    )

    r0 = data.joinpath("rank_0.json")
    r1 = data.joinpath("rank_1.json")
    assert r0.exists() and r1.exists()

    j0 = json.loads(r0.read_text())
    j1 = json.loads(r1.read_text())
    assert j0["world_size"] == 2 and j1["world_size"] == 2
    assert j0["distributed"] is True and j1["distributed"] is True
    assert j0["device"] == device and j1["device"] == device
