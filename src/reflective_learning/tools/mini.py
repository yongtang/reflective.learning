import argparse
import functools
import io
import json
import operator
import os
import random
import shutil
import tempfile

import lmdb
import minigrid
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from reflective_learning.encoder import ContextEncoder
from reflective_learning.inference import sequence
from reflective_learning.model import ReflectiveCore
from reflective_learning.train import pretrain

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

    try:
        choice = [
            minigrid.core.actions.Actions.left,
            minigrid.core.actions.Actions.right,
            minigrid.core.actions.Actions.forward,
        ]
        for step in range(steps - 1):
            selected = random.choice(choice)
            env.step(selected)
            action.append(selected.name)
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

    try:
        for name in action[:-1]:
            assert name != minigrid.core.actions.Actions.done.name, f"{action}"
            env.step(getattr(minigrid.core.actions.Actions, name))

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


def f_model(info):
    vocab_indices = sorted(info["vocab"].values())
    assert vocab_indices == list(range(len(vocab_indices))), f"{info['vocab']}"
    vocab_size = len(vocab_indices)

    state_indices = sorted(info["state"].values())
    assert state_indices == list(range(len(state_indices))), f"{info['state']}"

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

    return model


def f_weight(info):
    assert list(sorted(info["state"].values())) == list(
        range(len(info["state"]))
    ), f"{info['state']}"

    weight = list(sorted((info["state"][k], v) for k, v in info["weight"].items()))
    assert list(i for i, v in weight) == list(
        range(len(info["state"]))
    ), f"{info['weight']} vs. {info['state']}"
    weight = list(v for i, v in weight)

    return weight


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


def f_inference(
    model,
    vocab,
    maximum,
    prefix,
    device,
):

    token = sequence(
        model=model,
        prefix=prefix,
        maximum=maximum,
        device=device,
    )
    action = token.tolist()
    action = action[: action.index(0) + 1] if 0 in action else action

    symbol = {v: k for k, v in vocab.items()}
    action = [symbol[e] for e in action]

    return action


def f_callback(
    info,
    interval,
    model,
):
    if not hasattr(progress, "_meta_index_"):
        progress._meta_index_ = 0

    if progress.n > progress._meta_index_ + interval or progress.n == progress.total:
        # keep copy of max_version = 3
        max_version = 3
        for i in reversed(range(1, max_version)):
            src = os.path.join(data, f"model_{i}.pt")
            dst = os.path.join(data, f"model_{i+1}.pt")
            if os.path.exists(src):
                shutil.move(src, dst)

        # model.pt => model_1.pt
        shutil.move(os.path.join(data, "model.pt"), os.path.join(data, "model_1.pt"))

        # save model
        torch.save(
            {"info": info, "weight": model.state_dict()}, os.path.join(data, "model.pt")
        )

        progress._meta_save_ += save_interval

    return


def f_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.cpu().numpy()
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def f_bytes_to_tensor(buffer: bytes) -> torch.Tensor:
    array = np.load(io.BytesIO(buffer), allow_pickle=False)
    return torch.from_numpy(array)


def f_encode_text_embed(chunk, encoder, database):
    key = f"text_{json.dumps(chunk, sort_keys=True)}".encode()

    if database:
        with database.begin() as transaction:
            value = transaction.get(key)
        if value is not None:
            return f_bytes_to_tensor(value)

    chunk = encoder.encode_text_embed(chunk)
    if database:
        value = f_tensor_to_bytes(chunk)
        with database.begin(write=True) as transaction:
            transaction.put(key, value)
    return chunk


def f_encode_image_embed(chunk, encoder, database, image):
    key = f"image_{json.dumps(chunk, sort_keys=True)}".encode()

    if database:
        with database.begin() as transaction:
            value = transaction.get(key)
        if value is not None:
            return f_bytes_to_tensor(value)

    chunk = encoder.encode_image_embed(os.path.join(image, chunk))
    if database:
        value = f_tensor_to_bytes(chunk)
        with database.begin(write=True) as transaction:
            transaction.put(key, value)
    return chunk


def f_prefix(entry_text, entry_image, encoder, database, image):
    text_embed = list(
        f_encode_text_embed(chunk, encoder, database) for chunk in entry_text
    )
    image_embed = list(
        f_encode_image_embed(chunk, encoder, database, image) for chunk in entry_image
    )

    return encoder.encode_embed(text=text_embed, image=image_embed)


@functools.lru_cache(maxsize=4096)
def f_line(vocab_fn, state_fn, max_steps, image, encoder, database, line):
    entry = json.loads(line)

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
        database=database,
        image=image,
    )

    return {
        "token": token,
        "state": state,
        "prefix": prefix,
    }


class PretrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, database, essential, reservoir, line_fn):
        super().__init__()
        self.database = database
        self.essential = essential
        self.reservoir = reservoir
        self.line_fn = line_fn

    def __iter__(self):
        while True:
            if random.random() < 0.5:
                selection = random.randint(0, self.essential - 1)
                selection = f"seed_{selection:08d}".encode()
            else:
                selection = random.randint(0, self.reservoir - 1)
                selection = f"more_{selection:08d}".encode()
            with self.database.begin() as transaction:
                selection = transaction.get(selection)
            if selection:
                yield self.line_fn(line=selection)


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
    def f_fail(line):
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

        return False

    total = 0
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
                    if f_fail(line):
                        raise AssertionError(f"invalid seed:\n  {line.strip()}")

    total_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{total_width}d}}/{{total:{total_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    env_size = set()
    with open(seed, "r") as f:
        with tqdm(
            total=total,
            desc="Seed check",
            dynamic_ncols=True,
            bar_format=bar_format,
            unit="seed",
        ) as progress:
            for line in f:
                if line.strip():
                    progress.update(1)

                    entry = json.loads(line)

                    env_size.add(entry["env"])

                    assert (
                        len(entry["action"]) <= max_steps
                    ), f"{max_steps} vs. {entry['action']}"
    assert len(env_size) == 1, f"{env_size}"
    env_size = next(iter(env_size))

    info = {
        "env": env_size,
        "max": max_steps,
        "vocab": {e.name: (action_space.index(e)) for e in action_space},
        "state": {"success": 0, "failure": 1},
        "weight": {"success": 1.0, "failure": 0.1},
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

    model = f_model(info)

    os.makedirs(data, exist_ok=True)

    with open(os.path.join(data, "seed.data"), "w") as f:
        with open(seed, "r") as g:
            with tqdm(
                total=total,
                desc="Seed entry",
                dynamic_ncols=True,
                bar_format=bar_format,
                unit="seed",
            ) as progress:

                for line in g:
                    if line.strip():
                        progress.update(1)

                        entry = json.loads(line)

                        assert entry["env"] == info["env"], f"{entry} vs. {info}"
                        assert (
                            len(entry["action"]) <= info["max"]
                        ), f"{entry} vs. {info}"
                        assert all(
                            e in info["vocab"].keys() for e in entry["action"]
                        ), f"{entry} vs. {info}"

                        f.write(
                            json.dumps(
                                {
                                    "text": f_text(
                                        env_size=info["env"],
                                        max_steps=info["max"],
                                        goal=entry["goal"],
                                        start=entry["start"],
                                        facing=entry["facing"],
                                    ),
                                    "image": f_image(
                                        env_size=info["env"],
                                        max_steps=info["max"],
                                        goal=entry["goal"],
                                        start=entry["start"],
                                        facing=entry["facing"],
                                        image=image,
                                    ),
                                    "token": entry["action"],
                                    "state": f_step(
                                        step=f_replay(
                                            env_size=info["env"],
                                            max_steps=info["max"],
                                            goal=entry["goal"],
                                            start=entry["start"],
                                            facing=entry["facing"],
                                            action=entry["action"],
                                        ),
                                        max_steps=max_steps,
                                    ),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
    with open(os.path.join(data, "data.data"), "w") as f:
        pass

    torch.save(
        {"info": info, "weight": model.state_dict()}, os.path.join(data, "model.pt")
    )

    print(f"Save model: {os.path.join(data, 'model.pt')}")


def run_learn(
    data,
    image,
    total,
    batch,
    reservoir,
    interval,
    lr,
    device,
):

    info, weight = operator.itemgetter("info", "weight")(
        torch.load(os.path.join(data, "model.pt"), map_location="cpu")
    )
    print(f"Load info: {json.dumps(info, sort_keys=True)}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = f_model(info).to(device)

    model.load_state_dict(weight)
    model.to(device)
    print(f"Load model: {os.path.join(data, 'model.pt')}")

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    weight = torch.tensor(f_weight(info), device=device)

    # 4GB = 1<<32
    database = lmdb.open(data, map_size=1 << 32, readonly=False, create=True)

    with database.begin(write=True) as transaction:
        with tqdm(
            total=transaction.stats["entries"],
            desc="Lmdb check",
            unit="key",
            dynamic_ncols=True,
        ) as progress:
            for key, _ in transaction.cursor():
                progress.update(1)
                if key.startswith(f"seed_".encode()) or key.startswith(
                    f"data_".encode()
                ):
                    transaction.delete(key)

        count = 0
        with open(os.path.join(data, "seed.data"), "r") as f:
            with tqdm(
                total=os.path.getsize(os.path.join(data, "seed.data")),
                desc="Seed index",
                unit="B",
                unit_scale=True,
                dynamic_ncols=True,
            ) as progress:
                for line in f:
                    progress.update(len(line.encode("utf-8")))
                    if line.strip():
                        entry = json.loads(line)
                        data_entry = json.dumps(entry, sort_keys=True)
                        transaction.put(
                            f"seed_{essential:08d}".encode(), data_entry.encode()
                        )
                        count += 1
        essential = count

    dataset = PretrainDataset(
        database,
        essential,
        reservoir,
        line_fn=functools.partial(
            f_line,
            vocab_fn=lambda e: info["vocab"][e],
            state_fn=lambda e: info["state"][e],
            max_steps=info["max"],
            image=image,
            encoder=encoder,
            database=database,
        ),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        collate_fn=model.collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pretrain(
        model=model,
        loader=loader,
        optimizer=optimizer,
        weight=weight,
        total=total,
        callback=functools.partial(
            f_callback,
            info=info,
            interval=interval,
        ),
        device=device,
    )


def run_play(goal, start, facing, model, device):

    info, weight = operator.itemgetter("info", "weight")(
        torch.load(model, map_location="cpu")
    )
    print(f"Load info: {json.dumps(info, sort_keys=True)}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Load model: {model}")
    model = f_model(info).to(device)

    model.load_state_dict(weight)
    model.to(device)

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    weight = torch.tensor(f_weight(info), device=device)

    env_size, max_steps, vocab = info["env"], info["max"], info["vocab"]

    with tempfile.TemporaryDirectory() as image:
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
            database=None,
            image=image,
        )

        action = f_inference(
            model=model,
            vocab=vocab,
            maximum=max_steps,
            prefix=prefix,
            device=device,
        )
    step = f_replay(env_size, max_steps, goal, start, facing, action)
    state = f_step(step=step, max_steps=max_steps)

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
    learn_parser = subparsers.add_parser("learn", help="Pretrain mode")
    learn_parser.add_argument("--data", required=True)
    learn_parser.add_argument("--image", required=True)
    learn_parser.add_argument("--total", type=int, required=True)
    learn_parser.add_argument("--batch", type=int, required=True)
    learn_parser.add_argument("--reservoir", type=int, required=True)
    learn_parser.add_argument("--interval", type=int, required=True)
    learn_parser.add_argument("--lr", type=float, required=True)
    learn_parser.add_argument("--device")

    # ---- play mode ----
    play_parser = subparsers.add_parser("play", help="Perform mode")
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
            data=args.data,
            image=args.image,
            total=args.total,
            batch=args.batch,
            reservoir=args.reservoir,
            interval=args.interval,
            lr=args.lr,
            device=args.device,
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
