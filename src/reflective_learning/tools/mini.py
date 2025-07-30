import argparse
import functools
import json
import os
import random

import minigrid
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from reflective_learning.encoder import ContextEncoder
from reflective_learning.inference import sequence
from reflective_learning.model import ReflectiveCore
from reflective_learning.train import train

action_space = [
    minigrid.core.actions.Actions.left,
    minigrid.core.actions.Actions.right,
    minigrid.core.actions.Actions.forward,
]
facing_space = ["right", "down", "left", "up"]


def f_observation(env_size, max_steps):
    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=max_steps, render_mode=None
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
        for _ in range(max_steps):
            step = random.choice(action_space)
            env.step(step)
            action.append(step.name)
    finally:
        goal = tuple(int(v) for v in env.agent_pos)
        env.close()

    return goal, start, facing, action


def f_verify(env_size, max_steps, goal, start, facing, action):
    env = minigrid.envs.EmptyEnv(
        size=env_size, max_steps=max_steps, render_mode=None
    )  # disable truncation
    env.reset()

    env.agent_pos = list(start)
    env.agent_dir = facing_space.index(facing)

    try:
        if tuple(int(e) for e in env.agent_pos) == tuple(goal):
            return 0  # reached without any steps

        for i, step in enumerate(action):
            env.step(getattr(minigrid.core.actions.Actions, step))
            if tuple(int(e) for e in env.agent_pos) == tuple(goal):
                return i + 1
    finally:
        env.close()

    return max_steps + 1  # did not reach goal


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


def f_image(env_size, max_steps, goal, start, facing, image):
    filename = f"env_{env_size}_goal_{goal[0]}_{goal[1]}_start_{start[0]}_{start[1]}_facing_{facing}.png"
    if not os.path.exists(os.path.join(image, filename)):
        os.makedirs(image, exist_ok=True)
        img = f_render(env_size, max_steps, goal, start, facing)
        PIL.Image.fromarray(img).save(os.path.join(image, filename))
    return filename


def f_entry(env_size, max_steps, goal, start, facing, action, image):
    filename = f_image(env_size, max_steps, goal, start, facing, image)

    state = f_verify(env_size, max_steps, goal, start, facing, action[:max_steps])

    token = action[:state]

    return {
        "text": [
            f"goal {goal[0]},{goal[1]}",
            f"start {start[0]},{start[1]}",
            f"facing {facing}",
        ],
        "image": [filename],
        "token": token,
        "state": state,
    }


@functools.lru_cache
def f_line(encoder, image, line):
    entry = json.loads(line)
    token = torch.tensor(
        [getattr(minigrid.core.actions.Actions, e) for e in entry["token"]],
        dtype=torch.long,
    )
    state = torch.tensor(entry["state"], dtype=torch.long)
    prefix = encoder.encode(
        entry["text"], [os.path.join(image, e) for e in entry["image"]]
    )

    return {
        "token": token,
        "state": state,
        "prefix": prefix,
    }


def f_callback(
    encoder, image, data, stub_index, env_size, max_steps, device, model, count
):
    # goal, start, facing
    while True:
        goal = random.randint(1, env_size - 2), random.randint(1, env_size - 2)
        start = random.randint(1, env_size - 2), random.randint(1, env_size - 2)
        if goal != start:
            break
    facing = facing_space[random.randint(0, len(facing_space) - 1)]

    filename = f_image(env_size, max_steps, goal, start, facing, image)

    prefix = encoder.encode(
        [
            f"goal {goal[0]},{goal[1]}",
            f"start {start[0]},{start[1]}",
            f"facing {facing}",
        ],
        [os.path.join(image, filename)],
    )

    state_weights = {e: 1.0 for e in range(max_steps + 2)}
    token = sequence(
        model=model,
        prefix=prefix,
        state_weights=state_weights,
        stop_token=0,
        max_seq_len=max_steps,
        device=device,
    )
    action = [minigrid.core.actions.Actions(e.item()).name for e in token]
    print(f"Prediction: {token} {action}")

    stub = f_entry(env_size, max_steps, goal, start, facing, action, image)
    print(f"Stub: {stub}")

    with open(os.path.join(data, "stub.data"), "a") as f:
        f.seek(0, os.SEEK_END)
        offset = f.tell()
        f.write(json.dumps(stub) + "\n")

    selection = np.random.randint(0, len(stub_index))
    stub_index[selection] = offset


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, seed_file, seed_index, stub_file, stub_index, chance, line_fn):
        super().__init__()
        self.seed = seed_file, seed_index
        self.stub = stub_file, stub_index
        self.chance = chance
        self.line_fn = line_fn

    def __iter__(self):
        while True:
            if np.random.rand() > self.chance:
                selection = self.seed
            else:
                selection = self.stub
            file, index = selection
            offset = np.random.choice(index)
            file.seek(offset)
            line = file.readline()
            if line.strip():
                yield self.line_fn(line)


def run_seed(env_size, max_steps, num_seeds, save_seed):
    step_width = len(str(max_steps))
    count_width = len(str(num_seeds))
    iteration_width = len(str(num_seeds * 2))  # allow room for retries
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{count_width}d}}/{{total:{count_width}d}} "
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
                goal, start, facing, action = f_observation(env_size, max_steps=steps)

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
                    f"steps={steps:{step_width}d} saved={count+1:{count_width}d} iteration={iteration:{iteration_width}d}"
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
            desc="Seed pos",
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
            desc="Seed env",
            dynamic_ncols=True,
            bar_format=bar_format,
            unit="seed",
        ) as progress:
            for line in f:
                if line.strip():
                    progress.update(1)

                    env_size.add(json.loads(line)["env"])

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
    torch.save(
        {"info": info, "weight": model.state_dict()}, os.path.join(data, "model.pt")
    )

    print(f"Save model: {os.path.join(data, 'model.pt')}")

    with open(os.path.join(data, "seed.data"), "w") as f:
        with open(seed, "r") as g:
            with tqdm(
                total=total,
                desc="Seed spin",
                dynamic_ncols=True,
                bar_format=bar_format,
                unit="seed",
            ) as progress:

                for line in g:
                    if line.strip():
                        progress.update(1)

                        entry = json.loads(line)

                        f.write(
                            json.dumps(
                                f_entry(
                                    env_size,
                                    max_steps,
                                    entry["goal"],
                                    entry["start"],
                                    entry["facing"],
                                    entry["action"][:max_steps],
                                    image,
                                ),
                                sort_keys=True,
                            )
                            + "\n"
                        )


def run_learn(data, image, total, batch, reservoir, save_interval, device):

    lr = 1e-3

    info, weight = operator.itemgetter("info", "state")(
        torch.load(os.path.join(data, "model.pt"), map_location="cpu")
    )
    print(f"Load info: {json.dumps(info)}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = f_model(info).to(device)

    model.load_state_dict(weight).to(device)
    print(f"Load model: {os.path.join(data, 'model.pt')}")

    encoder = ContextEncoder.from_pretrained(info["context"], device=device)

    with open(os.path.join(data, "seed.data"), "r") as f:
        with tqdm(
            total=os.path.getsize(os.path.join(data, "seed.data")),
            desc="Seed index",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            off = 0
            offset = []
            for line in f:
                progress.update(len(line.encode("utf-8")))
                if line.strip():
                    offset.append(off)
                off += len(line)
    seed_index = np.array(offset, dtype=np.int64)

    with open(os.path.join(data, "stub.data"), "w") as f:
        pass
    stub_index = np.zeros(reservoir, dtype=np.int64)
    print(f"Stub index: {os.path.join(data, 'stub.data')}")

    with open(os.path.join(data, "seed.data"), "r") as seed_file:
        with open(os.path.join(data, "stub.data"), "r") as stub_file:

            dataset = IterableDataset(
                seed_file,
                seed_index,
                stub_file,
                stub_index,
                chance=0.5,
                line_fn=functools.partial(f_line, encoder, image),
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch,
                collate_fn=model.collate,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train(
                model=model,
                loader=loader,
                optimizer=optimizer,
                total=total,
                save=data,
                save_interval=save_interval,
                callback=functools.partial(
                    f_callback,
                    encoder,
                    image,
                    data,
                    stub_index,
                    info["env"],
                    info["max"],
                    device,
                ),
                callback_interval=1,
                device=device,
            )


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
    learn_parser.add_argument("--data", required=True)
    learn_parser.add_argument("--image", required=True)
    learn_parser.add_argument("--total", type=int, required=True)
    learn_parser.add_argument("--batch", type=int, required=True)
    learn_parser.add_argument("--reservoir", type=int, required=True)
    learn_parser.add_argument("--save-interval", type=int, required=True)
    learn_parser.add_argument("--device")

    # ---- play mode ----
    play_parser = subparsers.add_parser("play", help="Perform mode")
    play_parser.add_argument("--info", required=True)
    play_parser.add_argument("--model", required=True)
    play_parser.add_argument("--goal", type=f_pair, required=True)
    play_parser.add_argument("--start", type=f_pair, required=True)
    play_parser.add_argument("--facing", choices=facing_space, required=True)
    play_parser.add_argument("--device")

    args = parser.parse_args()
    print(f"Load args: {json.dumps(vars(args))}")

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
            save_interval=args.save_interval,
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
