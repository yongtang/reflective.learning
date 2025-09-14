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

    # decide per-rank device/backend (kind is "cpu" or "cuda")
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

    # model
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)
    ).to(device)

    # minimal DDP
    if device.type == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # tiny dataset: [1,2,3,4,5] -> features 32, labels in 0..3
    vals = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    X = vals.view(-1, 1).repeat(1, 32)  # shape [5, 32]
    y = vals.long() % 4  # shape [5]
    dataset = torch.utils.data.TensorDataset(X, y)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, sampler=sampler, shuffle=False, drop_last=False
    )

    opt = torch.optim.SGD(model.parameters(), 1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    sampler.set_epoch(0)  # important: add more epochs later

    # accumulate sample-weighted loss for a true global mean
    sum_loss = torch.zeros((), device=device)
    n_samples = torch.zeros((), device=device)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)  # mean over batch
        loss.backward()
        opt.step()

        bs = xb.size(0)
        sum_loss += loss.detach() * bs
        n_samples += bs

    # global mean loss across all ranks
    values = torch.stack([sum_loss, n_samples], 0)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    metric = (values[0] / values[1]).item()

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
