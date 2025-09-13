import datetime
import json
import os

import torch
import torch.distributed.launcher


def f_entrypoint(data, device):
    # env set by the elastic agent
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # single-node gloo defaults
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    print(f"[rank={rank}] init process group", flush=True)
    torch.distributed.init_process_group(
        backend="gloo",
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
    loss = loss.detach().float().cpu()
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
    metric = loss.item() / world_size

    if rank == 0:
        with open(os.path.join(data, "artifact.json"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "loss_mean": metric,
                        "world_size": world_size,
                        "device": str(device),
                    }
                )
            )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def launch(data, device):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    assert device == torch.device("cpu"), device
    endpoint = "127.0.0.1:29500"

    config = torch.distributed.launcher.api.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=2,
        rdzv_backend="c10d",
        rdzv_endpoint=endpoint,
        run_id="launch",
        max_restarts=0,
        start_method="spawn",
        monitor_interval=1,
    )
    torch.distributed.launcher.api.elastic_launch(config, entrypoint=f_entrypoint)(
        data, device
    )
