import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize(
    "device, remote",
    [
        pytest.param(
            "cpu",
            [],
            id="cpu/local",
        ),
        pytest.param(
            "cuda",
            [],
            id="cuda/local",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"),
        ),
        pytest.param(
            "cuda",
            ["127.0.0.1:10061", "127.0.0.1:10062"],
            id="cuda/remote",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"),
        ),
    ],
)
def test_torchrun(device, remote):

    node = [sys.executable, "-m", "torch.distributed.run", "--help"]
    proc = subprocess.run(
        node,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    print(f"\n==== {' '.join(node)} ====\n{proc.stdout}\n==== {' '.join(node)} ====")
    assert proc.returncode == 0, proc.stdout
    assert ("usage" in proc.stdout) or (
        "torch.distributed.run" in proc.stdout
    ), proc.stdout
