import argparse
import collections
import contextlib
import functools
import glob
import itertools
import json
import operator
import os
import random
import shutil
import tempfile

import minigrid
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from reflective_learning.encoder import ContextEncoder
from reflective_learning.inference import explore, sequence
from reflective_learning.launch import launch
from reflective_learning.model import ReflectiveCore
from reflective_learning.train import dpo, train

state_space = ["success", "failure"]
action_space = [
    minigrid.core.actions.Actions.done,
    minigrid.core.actions.Actions.left,
    minigrid.core.actions.Actions.right,
    minigrid.core.actions.Actions.forward,
]
facing_space = ["right", "down", "left", "up"]


def f_step(step, max_steps):
    assert 0 < step, f"invalid step {step}"
    return f"success" if step <= max_steps else f"failure"


def f_observation(env_size, steps):
    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=steps, render_mode=None
    )  # disable truncation
    env.reset()

    # Replace the original (1, 1) start with a random position and direction
    env.agent_pos = (
        random.randint(1, env.width - 2),
        random.randint(1, env.height - 2),
    )
    env.agent_dir = random.randint(0, len(facing_space) - 1)
    start = tuple(int(e) for e in env.agent_pos)
    facing = facing_space[env.agent_dir]
    action = []
    visited = {(start, env.agent_dir)}  # track visited (pos, dir) to avoid loops

    try:
        choice = [
            minigrid.core.actions.Actions.left,
            minigrid.core.actions.Actions.right,
            minigrid.core.actions.Actions.forward,
        ]
        for step in range(steps - 1):
            # limit retries so we don't infinite loop
            for _ in range(4 * steps):
                selected = random.choice(choice)

                old_pos, old_dir = tuple(env.agent_pos), env.agent_dir
                env.step(selected)
                new_state = (tuple(env.agent_pos), env.agent_dir)

                if new_state not in visited:
                    visited.add(new_state)
                    action.append(selected.name)
                    break
                else:
                    # revert and try again
                    env.agent_pos, env.agent_dir = old_pos, old_dir
    finally:
        goal = tuple(int(v) for v in env.agent_pos)
        env.close()

    action.append(minigrid.core.actions.Actions.done.name)

    return goal, start, facing, action


def f_replay(env_size, max_steps, goal, start, facing, action):
    assert len(action) > 0

    done = minigrid.core.actions.Actions.done.name
    assert (
        done not in action or action.count(done) == 1 and action[-1] == done
    ), f"{action}"
    done = done in action

    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=max_steps, render_mode=None
    )  # disable truncation
    env.reset()

    env.agent_pos = list(start)
    env.agent_dir = facing_space.index(facing)

    # Track visited (pos, dir) to detect loops during replay
    visited = {(tuple(int(e) for e in env.agent_pos), env.agent_dir)}

    try:
        for name in action[:-1]:
            assert name != minigrid.core.actions.Actions.done.name, f"{action}"
            env.step(getattr(minigrid.core.actions.Actions, name))

            state_now = (tuple(int(e) for e in env.agent_pos), env.agent_dir)
            if state_now in visited:
                return max_steps + 1  # loop encountered -> failure
            visited.add(state_now)

        name = action[-1]
        if name == minigrid.core.actions.Actions.done.name:
            if tuple(int(e) for e in env.agent_pos) == tuple(goal):
                return len(action)

    finally:
        env.close()

    return max_steps + 1  # did not reach goal with done


def f_render(env_size, max_steps, goal, start, facing):
    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=max_steps, render_mode="rgb_array"
    )
    env.reset()

    # Remove the default goal placed by reset() (bottom-right)
    env.grid.set(env_size - 2, env_size - 2, None)

    # Set agent position and direction
    env.agent_pos = list(start)
    env.agent_dir = facing_space.index(facing)

    # Place a Goal tile at the goal position
    if tuple(goal) != tuple(start):
        env.grid.set(goal[0], goal[1], minigrid.core.world_object.Goal())

    try:
        img = env.render()
    finally:
        env.close()

    return img


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


def f_text(env_size, max_steps, goal, start, facing):
    return [
        f"env {env_size}",
        f"goal {goal[0]},{goal[1]}",
        f"start {start[0]},{start[1]}",
        f"facing {facing}",
        f"max {max_steps}",
    ]


def f_image(env_size, max_steps, goal, start, facing, image):
    filename = f"env_{env_size}_goal_{goal[0]}_{goal[1]}_start_{start[0]}_{start[1]}_facing_{facing}.png"
    if not os.path.exists(os.path.join(image, filename)):
        os.makedirs(image, exist_ok=True)
        img = f_render(env_size, max_steps, goal, start, facing)
        PIL.Image.fromarray(img).save(os.path.join(image, filename))
    return [filename]


def f_entry(goal, start, facing, image, env_size, max_steps, action, state):
    return {
        "text": f_text(
            env_size=env_size,
            max_steps=max_steps,
            goal=goal,
            start=start,
            facing=facing,
        ),
        "image": f_image(
            env_size=env_size,
            max_steps=max_steps,
            goal=goal,
            start=start,
            facing=facing,
            image=image,
        ),
        "token": action,
        "state": state,
    }


def f_sequence(
    goal,
    start,
    facing,
    image,
    encoder,
    info,
    model,
    device,
):
    env_size, max_steps, vocab = (
        info["env"],
        info["max"],
        info["vocab"],
    )

    entry_text = f_text(
        env_size=env_size,
        max_steps=max_steps,
        goal=goal,
        start=start,
        facing=facing,
    )
    entry_image = f_image(
        env_size=env_size,
        max_steps=max_steps,
        goal=goal,
        start=start,
        facing=facing,
        image=image,
    )
    prefix = f_prefix(
        entry_text=entry_text,
        entry_image=entry_image,
        encoder=encoder,
        image=image,
    )

    token = sequence(
        model=model,
        reduce=lambda logit: logit[0] - logit[1],  # success - failure
        prefix=prefix,
        maximum=max_steps,
        device=device,
    )
    action = token.tolist()
    action = action[: action.index(0) + 1] if 0 in action else action

    symbol = {v: k for k, v in vocab.items()}
    action = [symbol[e] for e in action]

    state = f_step(
        step=f_replay(env_size, max_steps, goal, start, facing, action),
        max_steps=max_steps,
    )

    return f_entry(
        goal=goal,
        start=start,
        facing=facing,
        image=image,
        env_size=env_size,
        max_steps=max_steps,
        action=action,
        state=state,
    )


def f_explore(
    goal,
    start,
    facing,
    image,
    encoder,
    info,
    model,
    device,
):
    env_size, max_steps, vocab = (
        info["env"],
        info["max"],
        info["vocab"],
    )

    prefix = f_prefix(
        entry_text=f_text(
            env_size=env_size,
            max_steps=max_steps,
            goal=goal,
            start=start,
            facing=facing,
        ),
        entry_image=f_image(
            env_size=env_size,
            max_steps=max_steps,
            goal=goal,
            start=start,
            facing=facing,
            image=image,
        ),
        encoder=encoder,
        image=image,
    )

    token = explore(
        model=model,
        prefix=prefix,
        maximum=max_steps,
        device=device,
    )
    action = token.tolist()
    action = action[: action.index(0) + 1] if 0 in action else action

    symbol = {v: k for k, v in vocab.items()}
    action = [symbol[e] for e in action]

    state = f_step(
        step=f_replay(env_size, max_steps, goal, start, facing, action),
        max_steps=max_steps,
    )

    return f_entry(
        goal=goal,
        start=start,
        facing=facing,
        image=image,
        env_size=env_size,
        max_steps=max_steps,
        action=action,
        state=state,
    )


def f_callback(
    data,
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
        src = os.path.join(data, f"model.{i:03d}.pt")
        dst = os.path.join(data, f"model.{i+1:03d}.pt")
        if os.path.exists(src):
            shutil.move(src, dst)

    # model.pt => model_1.pt
    shutil.move(
        os.path.join(data, f"model.pt"),
        os.path.join(data, f"model.{1:03d}.pt"),
    )

    save = torch.load(os.path.join(data, f"model.{1:03d}.pt"), map_location="cpu")
    save[choice] = model.state_dict()

    # save model
    with tempfile.NamedTemporaryFile(
        dir=data, prefix="model.", suffix=".pt", delete=False
    ) as f:
        torch.save(save, f)
        f.flush()
        os.fsync(f.fileno())
        fname = f.name
    os.replace(fname, os.path.join(data, "model.pt"))

    progress._meta_index_ += interval

    if progress.n == progress.total:
        return

    return


def f_prefix(entry_text, entry_image, encoder, image):
    text_embed = list(encoder.encode_text_embed(chunk) for chunk in entry_text)
    image_embed = list(
        encoder.encode_image_embed(os.path.join(image, chunk)) for chunk in entry_image
    )

    return encoder.encode_embed(text=text_embed, image=image_embed)


def f_datum(vocab_fn, state_fn, max_steps, image, encoder, entry):
    assert len(entry["token"]) <= max_steps, f"{max_steps} vs. {entry['token']}"

    token = torch.tensor(
        [vocab_fn(e) for e in entry["token"]],
        dtype=torch.long,
    )
    state = torch.tensor(
        state_fn(entry["state"]),
        dtype=torch.long,
    )

    prefix = f_prefix(
        entry_text=entry["text"],
        entry_image=entry["image"],
        encoder=encoder,
        image=image,
    )

    return {
        "token": token,
        "state": state,
        "prefix": prefix,
    }


def f_dataset(file, choice):
    if not os.path.isfile(file):
        return {}
    # Group by prefix
    entries = collections.defaultdict(list)
    with open(file, "r") as f:
        with tqdm(
            total=os.path.getsize(file),
            desc=f"Check {choice}",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = json.dumps(
                        {
                            "text": entry["text"],
                            "image": entry["image"],
                        },
                        sort_keys=True,
                    )
                    entries[key].append((progress.n, len(entry["token"])))
                progress.update(len(line.encode("utf-8")))
    return {k: np.array(v).reshape((-1, 2)) for k, v in entries.items()}


class LearnDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, datum_fn, data):
        super().__init__()
        self.dataset = dataset
        self.datum_fn = datum_fn
        self.data = data

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        self.file = {
            file: self.stack.enter_context(open(os.path.join(self.data, file), "r"))
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
    def __init__(self, dataset, datum_fn, data):
        super().__init__()
        self.dataset = dataset
        self.datum_fn = datum_fn
        self.data = data

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        self.file = {
            file: self.stack.enter_context(open(os.path.join(self.data, file), "r"))
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

        return self.datum_fn(entry=json.loads(line_a)), self.datum_fn(
            entry=json.loads(line_b)
        )

    def __len__(self):
        return len(self.dataset)


def f_learn(
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
    print(f"Load model: {os.path.join(data, f'model.pt')}")
    info, model = operator.itemgetter("info", choice)(
        torch.load(os.path.join(data, f"model.pt"), map_location="cpu")
    )
    model = f_model(info, model)
    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    dataset = np.load(file, allow_pickle=True)
    with LearnDataset(
        dataset=dataset,
        datum_fn=functools.partial(
            f_datum,
            vocab_fn=lambda e: info["vocab"][e],
            state_fn=lambda e: state_space.index(e),
            max_steps=info["max"],
            image=image,
            encoder=encoder,
        ),
        data=data,
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
                f_callback,
                data=data,
                choice=choice,
                interval=interval,
                distributed=distributed,
            ),
            device=device,
            rank=rank,
        )


def f_finetune(
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
    print(f"Load model: {os.path.join(data, f'model.pt')}")
    info, finetune = operator.itemgetter("info", choice)(
        torch.load(os.path.join(data, f"model.pt"), map_location="cpu")
    )
    baseline = f_model(info, finetune)
    finetune = f_model(info, finetune)
    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    dataset = np.load(file, allow_pickle=True)
    with FinetuneDataset(
        dataset=dataset,
        datum_fn=functools.partial(
            f_datum,
            vocab_fn=lambda e: info["vocab"][e],
            state_fn=lambda e: state_space.index(e),
            max_steps=info["max"],
            image=image,
            encoder=encoder,
        ),
        data=data,
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
                f_callback,
                data=data,
                choice=choice,
                interval=interval,
                distributed=distributed,
            ),
            device=device,
            rank=rank,
        )


def run_seed(env_size, max_steps, num_seeds, save_seed):
    step_width = len(str(max_steps))
    total_width = len(str(num_seeds))
    iteration_width = len(str(num_seeds * 2))  # allow room for retries
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{total_width}d}}/{{total:{total_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    iteration = 0
    with open(save_seed, "w") as f:
        with tqdm(
            total=num_seeds,
            desc="Seed",
            dynamic_ncols=True,
            unit="seed",
            bar_format=bar_format,
        ) as progress:
            count = 0
            while count < num_seeds:
                iteration += 1
                steps = random.randint(1, max_steps)
                goal, start, facing, action = f_observation(env_size, steps=steps)
                if count % len(state_space) != 0:
                    goal = (
                        random.randint(1, env_size - 2),
                        random.randint(1, env_size - 2),
                    )

                if list(goal) == list(start):
                    continue  # skip invalid seed
                seed = {
                    "env": env_size,
                    "goal": goal,
                    "start": start,
                    "facing": facing,
                    "action": action,
                }

                f.write(json.dumps(seed, sort_keys=True) + "\n")
                progress.set_postfix_str(
                    f"steps={steps:{step_width}d} saved={count+1:{total_width}d} iteration={iteration:{iteration_width}d}"
                )
                progress.update(1)
                count += 1


def run_spin(seed, data, image, max_steps):
    def f_fail(vocab, line):
        entry = json.loads(line)
        if not (0 < entry["goal"][0] and entry["goal"][0] < entry["env"] - 1):
            return True
        if not (0 < entry["goal"][1] and entry["goal"][1] < entry["env"] - 1):
            return True
        if not (0 < entry["start"][0] and entry["start"][0] < entry["env"] - 1):
            return True
        if not (0 < entry["start"][1] and entry["start"][1] < entry["env"] - 1):
            return True
        if not (entry["facing"] in facing_space):
            return True
        if not (all(e in [o.name for o in action_space] for e in entry["action"])):
            return True
        if not (all(e in vocab.keys() for e in entry["action"])):
            return True

        return False

    total = 0
    steps = set()
    env_size = set()
    vocab = {e.name: (action_space.index(e)) for e in action_space}
    with open(seed, "r") as f:
        with tqdm(
            total=os.path.getsize(seed),
            desc="Seed check",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            for line in f:
                progress.update(len(line.encode("utf-8")))
                if line.strip():
                    total += 1
                    if f_fail(vocab, line):
                        raise AssertionError(f"invalid seed:\n  {line.strip()}")
                    entry = json.loads(line)
                    steps.add(len(entry["action"]))
                    env_size.add(entry["env"])

    assert max(steps) <= max_steps, f"{sorted(steps)}"
    assert len(env_size) == 1, f"{env_size}"
    env_size = next(iter(env_size))

    total_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{total_width}d}}/{{total:{total_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    os.makedirs(data, exist_ok=True)

    with contextlib.ExitStack() as stack:
        f = {
            choice: stack.enter_context(
                open(os.path.join(data, f"seed.{choice}.data"), "w")
            )
            for choice in state_space
        }
        with open(seed, "r") as g:
            with tqdm(
                total=total,
                desc=f"Seed entry",
                dynamic_ncols=True,
                bar_format=bar_format,
                unit="seed",
            ) as progress:
                for line in g:
                    if line.strip():
                        progress.update(1)

                        entry = json.loads(line)

                        state = f_step(
                            step=f_replay(
                                env_size=env_size,
                                max_steps=max_steps,
                                goal=entry["goal"],
                                start=entry["start"],
                                facing=entry["facing"],
                                action=entry["action"],
                            ),
                            max_steps=max_steps,
                        )

                        f[state].write(
                            json.dumps(
                                f_entry(
                                    goal=entry["goal"],
                                    start=entry["start"],
                                    facing=entry["facing"],
                                    image=image,
                                    env_size=env_size,
                                    max_steps=max_steps,
                                    action=entry["action"],
                                    state=state,
                                ),
                                sort_keys=True,
                            )
                            + "\n"
                        )

    info = {
        "env": env_size,
        "max": max_steps,
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
    }

    torch.save(
        {
            "info": info,
            **{choice: f_model(info, None).state_dict() for choice in state_space},
        },
        os.path.join(data, f"model.pt"),
    )

    print(f"Save model: {os.path.join(data, f'model.pt')}")

    return


def run_learn(choice, data, image, total, batch, interval, lr, device, distributed):
    essential = f_dataset(os.path.join(data, f"seed.{choice}.data"), choice)
    reservoir = f_dataset(os.path.join(data, f"data.{choice}.data"), choice)

    essential_file = f"seed.{choice}.data"
    reservoir_file = f"data.{choice}.data"

    essential = {
        k: np.concatenate(
            [
                np.array(v),
                np.full((len(v), 1), essential_file, dtype=object),
            ],
            axis=1,
        )
        for k, v in essential.items()
    }
    reservoir = {
        k: np.concatenate(
            [
                np.array(v),
                np.full((len(v), 1), reservoir_file, dtype=object),
            ],
            axis=1,
        )
        for k, v in reservoir.items()
    }
    assert len(essential) or len(reservoir)

    essential = essential if len(essential) else reservoir
    reservoir = reservoir if len(reservoir) else essential

    essential = np.concatenate(list(essential.values()), axis=0)
    reservoir = np.concatenate(list(reservoir.values()), axis=0)

    random = np.random.default_rng()

    assert total % 2 == 0
    essential = essential[random.integers(0, len(essential), size=total // 2)]
    reservoir = reservoir[random.integers(0, len(reservoir), size=total // 2)]

    dataset = np.concatenate([essential, reservoir], axis=0)
    random.shuffle(dataset)

    with tempfile.TemporaryDirectory() as directory:
        file = os.path.join(directory, "dataset.npy")
        np.save(file, dataset)

        device = device or (["cuda"] if torch.cuda.is_available() else ["cpu"])

        if distributed:
            launch(
                callback=f_learn,
                file=file,
                choice=choice,
                data=data,
                image=image,
                total=total,
                batch=batch,
                interval=interval,
                lr=lr,
                device=device,
            )
        else:
            assert len(device) == 1, device
            device = next(iter(device))

            f_learn(
                file=file,
                choice=choice,
                data=data,
                image=image,
                total=total,
                batch=batch,
                interval=interval,
                lr=lr,
                device=device,
                rank=0,
                world_size=1,
                distributed=False,
            )


def run_explore(data, image, total, device):
    print(f"Load model: {os.path.join(data, f'model.pt')}")

    load = torch.load(os.path.join(data, f"model.pt"), map_location="cpu")
    info = load["info"]
    model = list(f_model(info, load[choice]) for choice in state_space)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    for choice in state_space:
        if os.path.isfile(os.path.join(data, f"data.{choice}.data")):
            entries = sorted(
                os.path.basename(e)
                for e in glob.glob(os.path.join(data, f"data.{choice}.*.data"))
            )
            assert entries == list(
                f"data.{choice}.{i:03d}.data" for i in range(1, len(entries) + 1)
            )
            for i in reversed(list(range(1, len(entries) + 1))):
                shutil.move(
                    os.path.join(data, f"data.{choice}.{i:03d}.data"),
                    os.path.join(data, f"data.{choice}.{i+1:03d}.data"),
                )

            shutil.move(
                os.path.join(data, f"data.{choice}.data"),
                os.path.join(data, f"data.{choice}.{1:03d}.data"),
            )

    total_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{total_width}d}}/{{total:{total_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    statistics = {state: 0 for state in state_space}
    with contextlib.ExitStack() as stack:
        f = {
            choice: stack.enter_context(
                open(os.path.join(data, f"data.{choice}.data"), "w")
            )
            for choice in state_space
        }
        with tqdm(
            total=total,
            desc=f"Data entry",
            dynamic_ncols=True,
            unit="data",
            bar_format=bar_format,
        ) as progress:
            for index in range(total):
                while True:
                    goal = (
                        random.randint(1, info["env"] - 2),
                        random.randint(1, info["env"] - 2),
                    )
                    start = (
                        random.randint(1, info["env"] - 2),
                        random.randint(1, info["env"] - 2),
                    )
                    if goal != start:
                        break
                facing = random.choice(facing_space)

                entry = f_explore(
                    goal=goal,
                    start=start,
                    facing=facing,
                    image=image,
                    encoder=encoder,
                    info=info,
                    model=model,
                    device=device,
                )

                f[entry["state"]].write(json.dumps(entry, sort_keys=True) + "\n")
                statistics[entry["state"]] += 1

                progress.update(1)
    print(
        "Statistics: "
        + "["
        + ", ".join(f"{k}:{statistics[k]}" for k in sorted(statistics))
        + "]"
    )

    return


def run_finetune(data, image, total, batch, interval, lr, device, distributed):

    choice = "success"

    info = torch.load(os.path.join(data, f"model.pt"), map_location="cpu")["info"]

    essential = f_dataset(os.path.join(data, f"seed.{choice}.data"), choice)
    reservoir = f_dataset(os.path.join(data, f"data.{choice}.data"), choice)

    essential_file = f"seed.{choice}.data"
    reservoir_file = f"data.{choice}.data"

    essential = {
        k: np.concatenate(
            [
                np.array(v),
                np.full((len(v), 1), essential_file, dtype=object),
            ],
            axis=1,
        )
        for k, v in essential.items()
    }
    reservoir = {
        k: np.concatenate(
            [
                np.array(v),
                np.full((len(v), 1), reservoir_file, dtype=object),
            ],
            axis=1,
        )
        for k, v in reservoir.items()
    }

    def f_pair(v):
        p = []
        for i in range(1, info["max"]):
            for j in range(i + 1, info["max"] + 1):
                v_i = v[v[:, 1] == i]
                v_j = v[v[:, 1] == j]
                if len(v_i) and len(v_j):
                    p.extend([(a, b) for a, b in itertools.product(v_i, v_j)])

        return np.array(p)

    entries = {
        k: np.concatenate(
            [
                essential.get(k, np.empty((0, 3), dtype=object)),
                reservoir.get(k, np.empty((0, 3), dtype=object)),
            ],
            axis=0,
        )
        for k in (essential.keys() | reservoir.keys())
    }

    pairs = np.concatenate(
        list(filter(lambda e: len(e) > 0, map(f_pair, entries.values()))), axis=0
    )

    random = np.random.default_rng()

    dataset = pairs[random.integers(0, len(pairs), size=total)]
    random.shuffle(dataset)

    with tempfile.TemporaryDirectory() as directory:
        file = os.path.join(directory, "dataset.npy")
        np.save(file, dataset)

        device = device or (["cuda"] if torch.cuda.is_available() else ["cpu"])

        if distributed:
            launch(
                callback=f_finetune,
                file=file,
                choice=choice,
                data=data,
                image=image,
                total=total,
                batch=batch,
                interval=interval,
                lr=lr,
                device=device,
            )
        else:
            assert len(device) == 1, device
            device = next(iter(device))

            f_finetune(
                file=file,
                choice=choice,
                data=data,
                image=image,
                total=total,
                batch=batch,
                interval=interval,
                lr=lr,
                device=device,
                rank=0,
                world_size=1,
                distributed=False,
            )


def run_play(goal, start, facing, model, device):
    print(f"Load model: {model}")

    load = torch.load(model, map_location="cpu")
    info = load["info"]
    model = list(f_model(info, load[choice]) for choice in state_space)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    with tempfile.TemporaryDirectory() as image:
        entry = f_sequence(
            goal=goal,
            start=start,
            facing=facing,
            image=image,
            encoder=encoder,
            info=info,
            model=model,
            device=device,
        )

        state, action = entry["state"], entry["token"]

    print(f"Play model: ({state}) {action}")


def main():
    def f_pair(value: str) -> tuple[int, int]:
        try:
            x, y = map(int, value.split(","))
            return x, y
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Value must be two integers (e.g. 3,4), got {value}"
            )

    parser = argparse.ArgumentParser(description="MiniGrid Reflective Model CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---- seed mode ----
    seed_parser = subparsers.add_parser("seed", help="Seed mode")
    seed_parser.add_argument("--env-size", type=int, required=True)
    seed_parser.add_argument("--max-steps", type=int, required=True)
    seed_parser.add_argument("--num-seeds", type=int, required=True)
    seed_parser.add_argument("--save-seed", required=True)

    # ---- spin mode ----
    spin_parser = subparsers.add_parser("spin", help="Spin mode")
    spin_parser.add_argument("--seed", required=True)
    spin_parser.add_argument("--data", required=True)
    spin_parser.add_argument("--image", required=True)
    spin_parser.add_argument("--max-steps", type=int, required=True)

    # ---- learn mode ----
    learn_parser = subparsers.add_parser("learn", help="Learn mode")
    learn_parser.add_argument("--choice", required=True)
    learn_parser.add_argument("--data", required=True)
    learn_parser.add_argument("--image", required=True)
    learn_parser.add_argument("--total", type=int, required=True)
    learn_parser.add_argument("--batch", type=int, required=True)
    learn_parser.add_argument("--interval", type=int, required=True)
    learn_parser.add_argument("--lr", type=float, required=True)
    learn_parser.add_argument("--device", nargs="+")

    # ---- explore mode ----
    explore_parser = subparsers.add_parser("explore", help="Explore mode")
    explore_parser.add_argument("--data", required=True)
    explore_parser.add_argument("--image", required=True)
    explore_parser.add_argument("--total", type=int, required=True)
    explore_parser.add_argument("--device")

    # ---- finetune mode ----
    finetune_parser = subparsers.add_parser("finetune", help="Finetune mode")
    finetune_parser.add_argument("--data", required=True)
    finetune_parser.add_argument("--image", required=True)
    finetune_parser.add_argument("--total", type=int, required=True)
    finetune_parser.add_argument("--batch", type=int, required=True)
    finetune_parser.add_argument("--interval", type=int, required=True)
    finetune_parser.add_argument("--lr", type=float, required=True)
    finetune_parser.add_argument("--device", nargs="+")

    # ---- play mode ----
    play_parser = subparsers.add_parser("play", help="Play mode")
    play_parser.add_argument("--model", required=True)
    play_parser.add_argument("--goal", type=f_pair, required=True)
    play_parser.add_argument("--start", type=f_pair, required=True)
    play_parser.add_argument("--facing", choices=facing_space, required=True)
    play_parser.add_argument("--device")

    args = parser.parse_args()
    print(f"Load args: {json.dumps(vars(args), sort_keys=True)}")

    if args.mode == "seed":
        run_seed(
            env_size=args.env_size,
            max_steps=args.max_steps,
            num_seeds=args.num_seeds,
            save_seed=args.save_seed,
        )

    elif args.mode == "spin":
        run_spin(
            seed=args.seed,
            data=args.data,
            image=args.image,
            max_steps=args.max_steps,
        )

    elif args.mode == "learn":
        run_learn(
            choice=args.choice,
            data=args.data,
            image=args.image,
            total=args.total,
            batch=args.batch,
            interval=args.interval,
            lr=args.lr,
            device=args.device,
            distributed=False,
        )

    elif args.mode == "explore":
        run_explore(
            data=args.data,
            image=args.image,
            total=args.total,
            device=args.device,
        )

    elif args.mode == "finetune":
        run_finetune(
            data=args.data,
            image=args.image,
            total=args.total,
            batch=args.batch,
            interval=args.interval,
            lr=args.lr,
            device=args.device,
            distributed=False,
        )

    elif args.mode == "play":
        run_play(
            goal=args.goal,
            start=args.start,
            facing=args.facing,
            model=args.model,
            device=args.device,
        )

    else:
        assert False, f"Unhandled mode: {args.mode}"


if __name__ == "__main__":
    main()
