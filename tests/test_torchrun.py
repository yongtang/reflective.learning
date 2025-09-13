import os
import subprocess
import sys

import pytest
import torch


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
def test_torchrun(monkeypatch, tmp_path, device):
    monkeypatch.setenv("PYTHONPATH", "src")

    node = [
        sys.executable,
        "-m",
        "reflective_learning.tools.mini",
        "launch",
        "--data",
        str(tmp_path),
        "--device",
        str(device),
    ]
    proc = subprocess.run(
        node,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    print(f"\n==== {' '.join(node)} ====\n{proc.stdout}\n==== {' '.join(node)} ====")

    assert proc.returncode == 0, proc.stdout

    with open(os.path.join(tmp_path, "artifact.json"), "r") as f:
        data = f.read()
    print(
        f"\n==== {os.path.join(tmp_path, 'artifact.json')} ====\n{data}\n==== {os.path.join(tmp_path, 'artifact.json')} ===="
    )
    assert "loss_mean" in data, data
    assert '"world_size": 2, "device": "cpu"' in data, data
