import datetime
import json
import os

import torch
import torch.distributed.launcher


def f_entrypoint(data, kind):
    # env set by the elastic agent
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # single-node defaults
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    match kind:
        case "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            backend = "nccl"
        case "cpu":
            device = torch.device("cpu")
            backend = "gloo"
        case _:
            assert False, kind

    print(f"[rank={rank}] init process group", flush=True)
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )
    print(f"[rank={rank}] process group ok", flush=True)

    # tiny demo workload
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)
    ).to(device)
    x = torch.randn(16, 32, device=device)
    y = torch.randint(0, 4, (16,), device=device)
    opt = torch.optim.SGD(model.parameters(), 1e-2)
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()

    # reduce a metric across ranks
    loss = loss.detach().float().to(device)
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
    metric = loss.item() / world_size

    if rank == 0:
        with open(os.path.join(data, "artifact.json"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "loss_mean": metric,
                        "world_size": world_size,
                        "device": str(kind),
                    }
                )
            )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def launch(data, device):
    device = list(
        torch.device(e)
        for e in set(device or ["cuda"] if torch.cuda.is_available() else ["cpu"])
    )
    kind = {e.type for e in device}
    assert len(kind) == 1
    kind = next(iter(kind))

    match kind:
        case "cpu":
            nproc_per_node = 2
        case "cuda":
            assert torch.cuda.is_available(), device
            indices = {e.index for e in device}
            assert len(indices) > 0, device
            if len(indices) == 1 and next(iter(indices)) is None:
                assert torch.cuda.device_count() > 0, "no visible GPUs"
                nproc_per_node = torch.cuda.device_count()
            else:
                assert None not in indices, device
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in sorted(indices)
                )
                nproc_per_node = len(indices)
        case _:
            assert False, device

    endpoint = "127.0.0.1:29500"

    config = torch.distributed.launcher.api.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc_per_node,
        rdzv_backend="c10d",
        rdzv_endpoint=endpoint,
        run_id="launch",
        max_restarts=0,
        start_method="spawn",
        monitor_interval=1,
    )
    torch.distributed.launcher.api.elastic_launch(config, entrypoint=f_entrypoint)(
        data, kind
    )
