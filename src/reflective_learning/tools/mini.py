import argparse
import functools
import json
import operator
import os
import random

import diskcache
import minigrid
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from reflective_learning.context import ContextEncoder
from reflective_learning.model import ReflectiveCore
from reflective_learning.train import train

action_space = [
    minigrid.core.actions.Actions.left,
    minigrid.core.actions.Actions.right,
    minigrid.core.actions.Actions.forward,
]
facing_space = ["right", "down", "left", "up"]


def f_observation(env_size, steps):
    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=steps, render_mode=None
    )  # disable truncation
    env.reset()

    start = tuple(int(v) for v in env.agent_pos)
    facing = facing_space[env.agent_dir]
    action = []

    try:
        for _ in range(steps):
            step = random.choice(action_space)
            env.step(step)
            action.append(step.name)
    finally:
        goal = tuple(int(v) for v in env.agent_pos)
        env.close()

    return goal, start, facing, action


def f_verify(env_size, goal, start, facing, action):
    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=0, render_mode=None
    )  # disable truncation
    env.reset()

    env.agent_pos = list(start)
    env.agent_dir = facing

    try:
        if tuple(env.agent_pos) == goal:
            return 1  # reached without any steps

        for i, step in enumerate(action):
            env.step(step)
            if tuple(env.agent_pos) == goal:
                return i + 1
    finally:
        env.close()

    return len(action) + 1  # did not reach goal


def f_render(env_size, goal, start, facing):
    env = minigrid.envs.EmptyEnv(size=env_size, max_steps=0, render_mode="rgb_array")
    env.reset()

    # Set agent position and direction
    env.agent_pos = list(start)
    env.agent_dir = facing  # 0=right, 1=down, 2=left, 3=up

    # Place a Goal tile at the goal position
    if tuple(goal) != tuple(start):
        env.grid.set(goal[0], goal[1], Goal())

    try:
        img = env.render()
    finally:
        env.close()

    return img


def f_model(info):
    vocab_size = len(action_space)
    state_size = info["max"] + 2
    max_seq_len = info["max"] + 2
    max_prefix_len = 512

    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(
            batch_first=True,
            **info["layer"],
        ),
        **info["decoder"],
    )

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
        decoder=decoder,
    )

    return model


def f_index(file):
    off = 0
    offset = []
    with open(file, "rb") as f:
        for line in f:
            if line.strip():
                offset.append(off)
            off += len(line)
    np.save(file + ".npy", np.array(offset, dtype=np.int64))


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, seed, stub, chance):
        super().__init__()
        self.seed = seed
        self.stub = stub
        self.chance = chance

    def __iter__(self):
        while True:
            if len(self.stub) == 0 or random.random() > self.chance:
                yield random.choice(self.seed)
            else:
                yield random.choice(self.stub)


def run_seed(env_size, max_steps, num_seeds, save_seed):
    iteration = 0
    with open(save_seed, "w") as f:
        with tqdm(
            total=num_seeds, desc="Seed", dynamic_ncols=True, unit=" seed"
        ) as progress:
            count = 0
            while count < num_seeds:
                iteration += 1
                steps = random.randint(1, max_steps)
                goal, start, facing, action = f_observation(env_size, steps)

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
                    f"steps={steps:3d} saved={count+1:6d} iteration={iteration:6d}"
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
        if not (all(e in [v.name for v in action_space] for e in entry["action"])):
            return True

        return False

    with open(seed, "r") as f:
        fail = list(filter(f_fail, f))
        assert len(fail) == 0, f"invalid seed:\n  {'  '.join(fail)}"

    with open(seed, "r") as f:
        env_size = {json.loads(line)["env"] for line in f if line.strip()}
        assert len(env_size) == 1
    env_size = next(iter(env_size))

    info = {
        "env": env_size,
        "max": max_steps,
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
    torch.save(model.state_dict(), os.path.join(data, "model.pt"))

    with open(os.path.join(data, "info.json"), "w") as f:
        f.write(json.dumps(info))

    assert False

    data_seed = (
        "\n".join(set(f_seed(entry, env_size, max_steps) for entry in data_sample))
        + "\n"
    )

    with open(os.path.join(save_data, "seed.json"), "w") as f:
        f.write(data_seed)
    with open(os.path.join(save_data, "stub.json"), "w") as f:
        pass


def run_learn(data, image, total, batch, save_interval, device):

    lr = 1e-3

    with open(os.path.join(data, "info.json"), "r") as f:
        info = json.loads(f.read())
    print(f"Load info: {json.dumps(info)}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = f_model(info).to(device)

    model.load_state_dict(
        torch.load(os.path.join(data, "model.pt"), map_location=device)
    )
    print(f"Load model: {os.path.join(save_data, 'model.pt')}")

    assert False

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    data_seed = diskcache.Deque(directory=os.path.join(save_data, "seed.data"))
    if len(data_seed) == 0:
        with open(os.path.join(save_data, "seed.json"), "r") as f:
            for i, line in enumerate(f):
                data_seed.append(
                    f_data(
                        json.loads(line),
                        save_image,
                        encoder,
                        model,
                        info["core"]["max_seq_len"],
                    )
                )
    print(f"Load seed: {len(data_seed)}")

    data_stub = diskcache.Deque(directory=os.path.join(save_data, "stub.data"))
    if len(data_stub) == 0:
        with open(os.path.join(save_data, "stub.json"), "r") as f:
            for i, line in enumerate(f):
                data_stub.append(
                    f_data(
                        json.loads(line),
                        save_image,
                        encoder,
                        model,
                        info["core"]["max_seq_len"],
                    )
                )
    print(f"Load stub: {len(data_stub)}")

    dataset = IterableDataset(data_seed, data_stub, chance=0.5)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=model.collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        total=total,
        save_data=save_data,
        save_interval=save_interval,
        callback_func=None,
        callback_interval=0,
        device=device,
    )


def f_seed(entry, env_size, max_steps):
    state = f_state(entry, env_size, max_steps)
    seed = {
        "text": entry["text"],
        "image": entry["image"],
        "token": entry["token"][: max_steps + 1],
        "state": (
            str(len(entry["token"]))
            if len(entry["token"]) <= max_steps
            else str(max_steps + 1)
        ),
    }
    assert (
        state == seed["state"]
    ), f'state mismatch {entry} vs. {seed} - {state} vs. {seed["state"]}'
    return json.dumps(seed, sort_keys=True)


def train_initial(save_sample, max_steps, save_data):
    with open(save_sample, "r") as f:
        data_sample = [json.loads(line) for line in f if line.strip()]
    unique_string = set([entry["env"] for entry in data_sample])
    assert len(unique_string) == 1
    unique_string = next(iter(unique_string))
    assert unique_string.startswith("MiniGrid-Empty-")
    unique_string = unique_string.removeprefix("MiniGrid-Empty-")
    if unique_string.startswith("Random-"):
        randomize = True
        unique_string = unique_string.removeprefix("Random-")
    else:
        randomize = False
    unique_string = set(unique_string.split("x"))
    assert len(unique_string) == 1
    unique_string = next(iter(unique_string))
    env_size = int(unique_string)

    info = {
        "env_size": env_size,
        "max_steps": max_steps,
        "randomize": randomize,
        "layer": {
            "d_model": 768,
            "nhead": 12,
            "dim_feedforward": 3072,
            "dropout": 0.1,
        },
        "decoder": {
            "num_layers": 12,
        },
        "core": {
            "vocab_size": len(action_space),
            "state_size": max_steps + 2,
            "max_seq_len": max_steps + 2,
            "max_prefix_len": 512,
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
            # "transformers": "4.50.0",
        },
    }
    data_info = json.dumps(info)

    data_seed = (
        "\n".join(set(f_seed(entry, env_size, max_steps) for entry in data_sample))
        + "\n"
    )

    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(
            batch_first=True,
            **info["layer"],
        ),
        **info["decoder"],
    )

    model = ReflectiveCore(
        **info["core"],
        decoder=decoder,
    )

    os.makedirs(save_data, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_data, "model.pt"))

    with open(os.path.join(save_data, "info.json"), "w") as f:
        f.write(data_info)
    with open(os.path.join(save_data, "seed.json"), "w") as f:
        f.write(data_seed)
    with open(os.path.join(save_data, "stub.json"), "w") as f:
        pass


def f_data(data, save_image, encoder, model, max_seq_len):
    if "text" not in data or "image" not in data:
        raise ValueError(f"'text' or 'image' does not exist: {data}")
    if not isinstance(data["text"], list) or not isinstance(data["image"], list):
        raise ValueError(f"'text' or 'image' is not a list: {data}")

    data_token = [f_action(token) for token in data["token"]]
    data_state = int(data["state"])

    images = [os.path.join(save_image, image) for image in data["image"]]
    prefix = encoder.encode(data["text"], images)

    # Pad or truncate token sequence
    padded = (data_token + [0] * max_seq_len)[:max_seq_len]
    token = torch.tensor(padded, dtype=torch.long)

    # Repeat state ID across sequence
    state = torch.tensor(data_state, dtype=torch.long)

    return {
        "token": token,
        "state": state,
        "prefix": prefix,
    }


def train_continue(save_data, save_image, total, batch_size, save_interval, device):

    lr = 1e-3

    with open(os.path.join(save_data, "info.json"), "r") as f:
        info = json.loads(f.read())
    print(f"Load info: {json.dumps(info)}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(
            batch_first=True,
            **info["layer"],
        ),
        **info["decoder"],
    )

    model = ReflectiveCore(
        **info["core"],
        decoder=decoder,
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(save_data, "model.pt"), map_location=device)
    )
    print(f"Load model: {os.path.join(save_data, 'model.pt')}")

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    data_seed = diskcache.Deque(directory=os.path.join(save_data, "seed.data"))
    if len(data_seed) == 0:
        with open(os.path.join(save_data, "seed.json"), "r") as f:
            for i, line in enumerate(f):
                data_seed.append(
                    f_data(
                        json.loads(line),
                        save_image,
                        encoder,
                        model,
                        info["core"]["max_seq_len"],
                    )
                )
    print(f"Load seed: {len(data_seed)}")

    data_stub = diskcache.Deque(directory=os.path.join(save_data, "stub.data"))
    if len(data_stub) == 0:
        with open(os.path.join(save_data, "stub.json"), "r") as f:
            for i, line in enumerate(f):
                data_stub.append(
                    f_data(
                        json.loads(line),
                        save_image,
                        encoder,
                        model,
                        info["core"]["max_seq_len"],
                    )
                )
    print(f"Load stub: {len(data_stub)}")

    dataset = IterableDataset(data_seed, data_stub, chance=0.5)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=model.collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        total=total,
        save_data=save_data,
        save_interval=save_interval,
        callback_func=None,
        callback_interval=0,
        device=device,
    )


def main():
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
    learn_parser.add_argument("--data", required=True)
    learn_parser.add_argument("--image", required=True)
    learn_parser.add_argument("--total", type=int, required=True)
    learn_parser.add_argument("--batch", type=int, required=True)
    learn_parser.add_argument("--save-interval", type=int, required=True)
    learn_parser.add_argument("--device")

    args = parser.parse_args()
    print(f"args: {vars(args)}")

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
            save_interval=args.save_interval,
            device=args.device,
        )

    else:
        assert False, f"Unhandled mode: {args.mode}"


if __name__ == "__main__":
    main()
