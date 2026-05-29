import os
import shutil
import tempfile

import torch

from reflective_learning.model import ReflectiveCore


def f_model(info, state):
    vocab_indices = sorted(info["vocab"].values())
    assert vocab_indices == list(range(len(vocab_indices))), f"{info['vocab']}"
    vocab_size = len(vocab_indices)

    max_seq_len = info["max"]
    max_prefix_len = 512

    decoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            batch_first=True,
            **info["layer"],
        ),
        **info["decoder"],
    )

    model = ReflectiveCore(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
        decoder=decoder,
    )

    model.load_state_dict(state) if state else None

    return model


def callback(
    model_file,
    callback_fn,
    choice,
    interval,
    distributed,
    model,
    loss,
    progress,
    device,
    rank,
):
    if rank != 0:
        return
    model = (
        model.module
        if (
            distributed
            and isinstance(
                model,
                (
                    torch.nn.parallel.DataParallel,
                    torch.nn.parallel.DistributedDataParallel,
                    torch.distributed.fsdp.FullyShardedDataParallel,
                ),
            )
        )
        else model
    )

    if not hasattr(progress, "_meta_index_"):
        progress._meta_index_ = 0

    if not (
        progress.n > progress._meta_index_ + interval or progress.n == progress.total
    ):
        return

    # run callback_fn
    if callback_fn:
        callback_fn(model, loss, choice, progress, interval)

    # keep copy of max_version = 3
    max_version = 3
    for i in reversed(range(1, max_version)):
        src = f"{model_file}.{i:03d}"
        dst = f"{model_file}.{i+1:03d}"
        if os.path.exists(src):
            shutil.move(src, dst)

    # model.pt => model_1.pt
    shutil.move(
        f"{model_file}",
        f"{model_file}.{1:03d}",
    )

    save = torch.load(f"{model_file}.{1:03d}", map_location="cpu")
    save[choice] = model.state_dict()

    # save model
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(model_file), prefix="model.", suffix=".pt", delete=False
    ) as f:
        torch.save(save, f)
        f.flush()
        os.fsync(f.fileno())
        fname = f.name
    os.replace(fname, f"{model_file}")

    progress._meta_index_ += interval

    if progress.n == progress.total:
        return

    return


def load(file, *selection):
    model = torch.load(file, map_location="cpu", weights_only=False)
    info = model["info"]
    return (
        ([info] + [f_model(info, model[choice]) for choice in selection])
        if selection
        else info
    )


def save(file, max, vocab, state, param, meta):

    info = {
        "max": max,
        "vocab": vocab,
        "state": state,
        "layer": {
            "d_model": (param if param is not None else {}).get("d_model", 768),
            "nhead": (param if param is not None else {}).get("nhead", 12),
            "dim_feedforward": (param if param is not None else {}).get(
                "dim_feedforward", 3072
            ),
            "dropout": (param if param is not None else {}).get("dropout", 0.1),
        },
        "decoder": {
            "num_layers": (param if param is not None else {}).get("num_layers", 12),
        },
        "context": {
            "text": {
                "model": "openai-community/gpt2",
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            },
            "image": {
                "model": "google/siglip2-so400m-patch16-naflex",
                "revision": "cc24074f717b612951c2dead130904ab9b65a81e",
            },
            "transformers": "5.1.0",
        },
        "meta": ({} if meta is None else meta),
    }

    models = {choice: f_model(info, None) for choice in info["state"]}

    torch.save(
        {
            "info": info,
            **{choice: model.state_dict() for choice, model in models.items()},
        },
        file,
    )
    print(
        "Save model: {}".format(
            ", ".join(
                "{} ({:,} parameters)".format(
                    choice, sum(p.numel() for p in model.parameters())
                )
                for choice, model in models.items()
            )
        )
    )
