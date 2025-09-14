import datetime
import os

import torch
import torch.distributed.launcher


def f_entrypoint(callback, file, choice, data, image, total, batch, interval, lr, kind):
    # single-node defaults
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # choose per-rank device/backend from requested kind
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

    # (optional) pre-init log using LOCAL_RANK since global RANK isn't set yet
    print(
        f"[local_rank={os.environ.get('LOCAL_RANK', 0)}] init process group", flush=True
    )

    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"[rank={rank}] process group ok", flush=True)

    callback(
        file=file,
        choice=choice,
        data=data,
        image=image,
        total=total,
        batch=batch,
        interval=interval,
        lr=lr,
        device=device,
        rank=rank,
        world_size=world_size,
        distributed=True,
    )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def launch(callback, file, choice, data, image, total, batch, interval, lr, device):
    device = list(torch.device(e) for e in set(device))
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
        callback, file, choice, data, image, total, batch, interval, lr, kind
    )
