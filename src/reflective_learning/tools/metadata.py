import collections
import contextlib
import functools
import json
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from reflective_learning.encoder import ContextEncoder
from reflective_learning.model import ReflectiveCore
from reflective_learning.train import dpo, train


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


state_space = ["success", "failure"]


class LearnDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, datum_fn):
        super().__init__()
        self.dataset = dataset
        self.datum_fn = datum_fn

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        self.file = {
            file: self.stack.enter_context(open(file, "r"))
            for file in np.unique(self.dataset[:, 2])
        }
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)

    def __getitem__(self, index):
        offset, steps, file = self.dataset[index]
        self.file[file].seek(offset)
        line = self.file[file].readline()
        return self.datum_fn(entry=json.loads(line))

    def __len__(self):
        return len(self.dataset)


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, datum_fn):
        super().__init__()
        self.dataset = dataset
        self.datum_fn = datum_fn

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        self.file = {
            file: self.stack.enter_context(open(self.data, "r"))
            for file in np.unique(self.dataset[:, :, 2])
        }
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)

    def __getitem__(self, index):
        (offset_a, steps_a, file_a), (offset_b, steps_b, file_b) = self.dataset[index]
        self.file[file_a].seek(offset_a)
        line_a = self.file[file_a].readline()
        self.file[file_b].seek(offset_b)
        line_b = self.file[file_b].readline()

        return (
            self.datum_fn(entry=json.loads(line_a)),
            self.datum_fn(entry=json.loads(line_b)),
        )

    def __len__(self):
        return len(self.dataset)


def callback(
    model_file,
    choice,
    interval,
    distributed,
    model,
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


def datum(image, vocab_fn, state_fn, max_steps, encoder, entry):
    assert len(entry["token"]) <= max_steps, f"{max_steps} vs. {entry['token']}"

    token = torch.tensor(
        [vocab_fn(e) for e in entry["token"]],
        dtype=torch.long,
    )
    state = torch.tensor(
        state_fn(entry["state"]),
        dtype=torch.long,
    )

    prefix = encoder.encode(
        text=tuple(entry["text"]),
        image=tuple(os.path.join(image, e) for e in entry["image"]),
    )

    return {
        "token": token,
        "state": state,
        "prefix": prefix,
    }


def learn(
    model_file,
    dataset_file,
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
    print(f"Load model: {model_file} ({choice})")
    info, model = load(model_file, choice)
    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    dataset = np.load(dataset_file, allow_pickle=True)
    with LearnDataset(
        dataset=dataset,
        datum_fn=functools.partial(
            datum_fn,
            vocab_fn=lambda e: info["vocab"][e],
            state_fn=lambda e: state_space.index(e),
            max_steps=info["max"],
            encoder=encoder,
        ),
    ) as dataset:
        collate_fn = model.collate

        sampler = None
        model.to(device)
        if distributed:
            if device.type == "cuda":
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[device.index], output_device=device.index
                )
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)

            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            # no epoch loop so no sampler.set_epoch(0)

            # keep per-rank cap consistent with shard size (optional but safe)
            total = min(total, len(sampler))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch,
            collate_fn=collate_fn,  # use captured collate
            sampler=sampler,
            shuffle=False,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(
            model=model,
            loader=loader,
            optimizer=optimizer,
            total=total,
            callback=functools.partial(
                callback,
                model_file=model_file,
                choice=choice,
                interval=interval,
                distributed=distributed,
            ),
            device=device,
            rank=rank,
            desc=f"Learn {choice}",
        )


def finetune(
    model_file,
    dataset_file,
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
    print(f"Load model: {model_file}")
    info, baseline, finetune = load(model_file, [choice, choice])
    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    dataset = np.load(dataset_file, allow_pickle=True)
    with FinetuneDataset(
        dataset=dataset,
        datum_fn=functools.partial(
            datum_fn,
            vocab_fn=lambda e: info["vocab"][e],
            state_fn=lambda e: state_space.index(e),
            max_steps=info["max"],
            encoder=encoder,
        ),
    ) as dataset:
        collate_fn = finetune.collate

        sampler = None
        baseline.to(device)
        finetune.to(device)
        if distributed:
            if device.type == "cuda":
                finetune = torch.nn.parallel.DistributedDataParallel(
                    finetune, device_ids=[device.index], output_device=device.index
                )
            else:
                finetune = torch.nn.parallel.DistributedDataParallel(finetune)

            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            # no epoch loop so no sampler.set_epoch(0)

            # optional: cap per-rank total to shard size
            total = min(total, len(sampler))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch,
            collate_fn=lambda batch: tuple(
                map(collate_fn, zip(*batch))
            ),  # use captured collate
            sampler=sampler,
            shuffle=False,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(finetune.parameters(), lr=lr)
        dpo(
            baseline=baseline,
            finetune=finetune,
            loader=loader,
            optimizer=optimizer,
            total=total,
            callback=functools.partial(
                callback,
                model_file=model_file,
                choice=choice,
                interval=interval,
                distributed=distributed,
            ),
            device=device,
            rank=rank,
        )


def scan(desc, callback, file):
    if os.path.isfile(file):
        with open(file, "r") as f:
            with tqdm(
                total=os.path.getsize(file),
                desc=desc,
                unit="B",
                unit_scale=True,
                dynamic_ncols=True,
            ) as progress:

                def fn(line):
                    data = callback(progress.n, line) if line.strip() else None
                    progress.update(len(line.encode("utf-8")))
                    return data

                return list(filter(lambda e: e is not None, map(fn, f)))
    return list()


def dataset(file):
    entries = collections.defaultdict(list)

    def fn(offset, line):
        entry = json.loads(line)
        key = json.dumps(
            {
                "text": entry["text"],
                "image": entry["image"],
            },
            sort_keys=True,
        )
        entries[key].append((offset, len(entry["token"])))

    scan(f"Scan {file}", fn, file)

    entries = {k: np.array(v).reshape((-1, 2)) for k, v in entries.items()}

    entries = {
        k: np.concatenate(
            [
                np.array(v),
                np.full((len(v), 1), file, dtype=object),
            ],
            axis=1,
        )
        for k, v in entries.items()
    }

    return entries


def load(file, *selection):
    model = torch.load(file, map_location="cpu", weights_only=False)
    info = model["info"]
    return [info] + [f_model(info, model[choice]) for choice in selection]


def save(file, max, vocab, meta):
    info = {
        "max": max,
        "vocab": vocab,
        "layer": {
            "d_model": 768,
            "nhead": 12,
            "dim_feedforward": 3072,
            "dropout": 0.1,
        },
        "decoder": {
            "num_layers": 12,
        },
        "context": {
            "text": {
                "model": "gpt2",
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            },
            "image": {
                "model": "google/vit-base-patch16-224",
                "revision": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
            },
            "transformers": "4.50.0",
        },
        "meta": meta,
    }

    torch.save(
        {
            "info": info,
            **{choice: f_model(info, None).state_dict() for choice in state_space},
        },
        file,
    )
