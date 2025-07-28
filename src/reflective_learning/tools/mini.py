import argparse
import functools
import json
import operator
import os
import random

import diskcache
import minigrid
import PIL.Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from reflective_learning.model import ReflectiveCore

action_space = ["left", "right", "forward"]
facing_space = ["right", "down", "left", "up"]

f_action = functools.partial(list.index, action_space)
f_string_to_facing = functools.partial(list.index, facing_space)
f_facing_to_string = functools.partial(operator.getitem, facing_space)


def f_goal(grid):
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj and obj.type == "goal":
                return [x, y]
    raise ValueError("Goal not found in grid")


def f_shortest(goal, start, facing):
    path = []
    x, y = start
    gx, gy = goal

    def rotate_to(desired_facing, current_facing):
        turns = []
        while current_facing != desired_facing:
            turns.append("right")
            current_facing = (current_facing + 1) % 4
        return turns, current_facing

    if gx != x:
        desired = 0 if gx > x else 2
        turns, facing = rotate_to(desired, facing)
        path.extend(turns)
        for _ in range(abs(gx - x)):
            path.append("forward")
        x = gx

    if gy != y:
        desired = 1 if gy > y else 3
        turns, facing = rotate_to(desired, facing)
        path.extend(turns)
        for _ in range(abs(gy - y)):
            path.append("forward")

    return path


def f_variants(actions, max_extra_steps):
    """
    Generate variants of the shortest path by injecting small detours (e.g., right->forward->left).
    Each detour adds 2 steps.
    """
    variants = [actions]
    for extra in range(1, max_extra_steps + 1):
        path = list(actions)
        count = 0
        i = 0
        while i < len(path) and count < extra:
            if path[i] == "forward":
                # Replace one forward with right → forward → left
                path = path[:i] + ["right", "forward", "left"] + path[i + 1 :]
                i += 3
                count += 2
            else:
                i += 1
        if len(path) == len(actions) + extra:
            variants.append(path)
    return variants


class ContextEncoder:
    def __init__(
        self, text_model, image_model, tokenizer, image_processor, device="cpu"
    ):
        assert text_model is not None, "text_model is required"
        assert image_model is not None, "image_model is required"
        assert tokenizer is not None, "tokenizer is required"
        assert image_processor is not None, "image_processor is required"
        assert text_model.config.hidden_size == image_model.config.hidden_size

        self.text_model = text_model.to(device)
        self.image_model = image_model.to(device)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device

    @classmethod
    def from_pretrained(cls, context_dir, device="cpu"):
        with open(os.path.join(context_dir, "context_versions.json")) as f:
            versions = json.load(f)

        text_cfg = versions["pretrained_models"]["gpt2"]
        image_cfg = versions["pretrained_models"]["vit"]

        text_model = AutoModel.from_pretrained(
            text_cfg["model"], revision=text_cfg["revision"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_cfg["model"], revision=text_cfg["revision"]
        )
        image_model = AutoModel.from_pretrained(
            image_cfg["model"], revision=image_cfg["revision"]
        )
        image_processor = AutoImageProcessor.from_pretrained(
            image_cfg["model"], revision=image_cfg["revision"], use_fast=True
        )

        return cls(text_model, image_model, tokenizer, image_processor, device=device)

    @functools.lru_cache(maxsize=1024)
    def encode_text_embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            output = self.text_model(input_ids=input_ids)
            return output.last_hidden_state.squeeze(0).detach().cpu()

    @functools.lru_cache(maxsize=1024)
    def encode_image_embed(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        with torch.no_grad():
            output = self.image_model(pixel_values=pixel_values)
            return output.last_hidden_state.squeeze(0).detach().cpu()

    def encode(self, text: list[str], image: list[str]) -> torch.Tensor:
        segments = []
        break_embed = torch.zeros(
            (1, self.text_model.config.hidden_size), dtype=torch.float32
        )

        for t in text:
            segments.append(self.encode_text_embed(t))
        segments.append(break_embed.clone())

        for path in image:
            segments.append(self.encode_image_embed(path))
        segments.append(break_embed.clone())

        return torch.cat(segments, dim=0)


class EmptyEnv(minigrid.envs.EmptyEnv):
    def __init__(self, goal=None, randomize=False, **kwargs):
        self.goal = goal
        self.randomize = randomize
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        if self.goal:
            # Remove the default goal at bottom-right
            self.grid.set(width - 2, height - 2, None)
            # Place goal at goal
            self.grid.set(*self.goal, minigrid.core.world_object.Goal())
        elif self.randomize:
            # Remove the default goal at bottom-right
            self.grid.set(width - 2, height - 2, None)
            # Place goal randomly
            self.place_obj(minigrid.core.world_object.Goal())


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


def train_sample(env_size, max_steps, num_samples, save_sample, save_image, randomize):

    samples = []

    os.makedirs(save_image, exist_ok=True)

    with tqdm(total=num_samples, desc="Generating Samples") as progress:
        for index in range(num_samples):
            env = EmptyEnv(
                randomize=randomize,
                size=env_size,
                agent_start_pos=None,
                render_mode="rgb_array",
            )
            env.reset()

            goal = f_goal(env.unwrapped.grid)
            start = list(env.unwrapped.agent_pos)
            facing = f_facing_to_string(env.unwrapped.agent_dir)

            filename = f"goal_{goal[0]}_{goal[1]}_start_{start[0]}_{start[1]}_facing_{facing}.png"
            if not os.path.exists(os.path.join(save_image, filename)):

                img = env.render()
                image = PIL.Image.fromarray(img)
                image.save(os.path.join(save_image, filename))

            sample = {
                "text": [
                    f"goal {goal[0]},{goal[1]}",
                    f"start {start[0]},{start[1]}",
                    f"facing {facing}",
                ],
                "image": [filename],
                "env": (
                    f"MiniGrid-Empty-Random-{env_size}x{env_size}"
                    if randomize
                    else f"MiniGrid-Empty-{env_size}x{env_size}"
                ),
                "goal": [int(x) for x in goal],
                "start": [int(x) for x in start],
                "facing": facing,
            }

            shortest = f_shortest(
                goal=goal, start=start, facing=f_string_to_facing(facing)
            )
            max_extra_steps = max(0, max_steps - len(shortest))
            variants = f_variants(shortest, max_extra_steps)
            sample["token"] = random.choice(variants)

            samples.append(sample)
            env.close()
            progress.update(1)

    json_strings = [json.dumps(sample, sort_keys=True) for sample in samples]
    unique_json_strings = list(set(json_strings))

    with open(save_sample, "w") as f:
        f.write("\n".join(unique_json_strings) + "\n")

    print(f"Wrote {len(samples)} samples to {save_sample}")


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
    }
    data_info = json.dumps(info)

    def f_seed(sample):
        seed = {
            "text": sample["text"],
            "image": sample["image"],
            "token": sample["token"][: max_steps + 1],
            "state": (
                str(len(sample["token"]))
                if len(sample["token"]) <= max_steps
                else str(max_steps + 1)
            ),
        }
        return json.dumps(seed, sort_keys=True)

    data_seed = "\n".join(set(f_seed(sample) for sample in data_sample)) + "\n"

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
    torch.save(model.state_dict(), os.path.join(save_data, "model.pt"))

    os.makedirs(save_data, exist_ok=True)
    with open(os.path.join(save_data, "info.json"), "w") as f:
        f.write(data_info)
    with open(os.path.join(save_data, "seed.json"), "w") as f:
        f.write(data_seed)
    with open(os.path.join(save_data, "stub.json"), "w") as f:
        pass


def f_data(data, save_image, context_encoder, model):
    if "text" not in data or "image" not in data:
        raise ValueError(f"'text' or 'image' does not exist: {data}")
    if not isinstance(data["text"], list) or not isinstance(data["image"], list):
        raise ValueError(f"'text' or 'image' is not a list: {data}")

    max_seq_len = model.pos_embedding.num_embeddings - 512

    data_token = [f_action(token) for token in data["token"]]
    data_state = int(data["state"])

    images = [os.path.join(save_image, image) for image in data["image"]]
    prefix = context_encoder.encode(data["text"], images)

    # Pad or truncate token sequence
    padded = (data_token + [0] * max_seq_len)[:max_seq_len]
    token_ids = torch.tensor(padded, dtype=torch.long)

    # Repeat state ID across sequence
    state_ids = torch.full((max_seq_len,), data_state, dtype=torch.long)

    return {
        "token_ids": token_ids,
        "state_ids": state_ids,
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

    context_encoder = ContextEncoder.from_pretrained(save_data, device=device)

    data_seed = diskcache.Deque(directory=os.path.join(save_data, "seed.data"))
    if len(data_seed) == 0:
        with open(os.path.join(save_data, "seed.json"), "r") as f:
            for i, line in enumerate(f):
                data_seed.append(
                    f_data(json.loads(line), save_image, context_encoder, model)
                )
    print(f"Load seed: {len(data_seed)}")

    data_stub = diskcache.Deque(directory=os.path.join(save_data, "stub.data"))
    if len(data_stub) == 0:
        with open(os.path.join(save_data, "stub.json"), "r") as f:
            for i, line in enumerate(f):
                data_stub.append(
                    f_data(json.loads(line), save_image, context_encoder, model)
                )
    print(f"Load stub: {len(data_stub)}")

    dataset = IterableDataset(data_seed, data_stub, chance=0.5)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_with_prefix(batch, model),
    )
    entries = iter(dataloader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with tqdm(total=total, desc="Training", leave=True, ncols=100) as progress:
        interval = []
        total_loss = 0.0
        for step in range(batch_total):
            batch = next(entries)

            model.train()

            embed, mask = batch["embed"], batch["mask"]
            token_target = batch["token_target"]
            state_target = batch["state_target"]

            logits = model.call(embed, mask=mask)
            logits = logits[:, -token_target.size(1) - 1 : -1]

            loss = model.loss(logits, token_target, state_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

            progress.update(1)

            model.eval()
            with torch.no_grad():
                pass

            if (step + 1) % save_interval == 1:
                interval.append(os.path.join(save_data, f"model_{step:03d}.pt"))
                torch.save(model.state_dict(), interval[-1])
                if len(interval) > 3:
                    oldest = interval.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)

    torch.save(model.state_dict(), os.path.join(save_data, "model.pt"))
    print(f"Save model: {os.path.join(save_data, 'model.pt')}")


def main():
    parser = argparse.ArgumentParser(
        description="MiniGrid Model Pipeline CLI (Reflective + RL)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "sample",
            "initial",
            "continue",
        ],
    )

    parser.add_argument("--env-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--save-sample")
    parser.add_argument("--save-data")
    parser.add_argument("--save-image")
    parser.add_argument("--total", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--device")
    parser.add_argument("--randomize", type=bool, default=True)

    args = parser.parse_args()

    print(f"args: {vars(args)}")

    if args.mode == "sample":
        required = ["env_size", "max_steps", "num_samples", "save_sample", "save_image"]
        required = [name for name in required if not getattr(args, name)]
        assert not required, f"Missing required arguments: {', '.join(required)}"

        train_sample(
            env_size=args.env_size,
            max_steps=args.max_steps,
            num_samples=args.num_samples,
            save_sample=args.save_sample,
            save_image=args.save_image,
            randomize=args.randomize,
        )

    elif args.mode == "initial":
        required = ["save_sample", "max_steps", "save_data"]
        required = [name for name in required if not getattr(args, name)]
        assert not required, f"Missing required arguments: {', '.join(required)}"

        train_initial(
            save_sample=args.save_sample,
            max_steps=args.max_steps,
            save_data=args.save_data,
        )

    elif args.mode == "continue":
        required = ["save_data", "save_image", "total", "batch_size", "save_interval"]
        required = [name for name in required if not getattr(args, name)]
        assert not required, f"Missing required arguments: {', '.join(required)}"

        train_continue(
            save_data=args.save_data,
            save_image=args.save_image,
            total=args.total,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            device=args.device,
        )

    else:
        assert False, f"Unhandled mode: {args.mode}"


if __name__ == "__main__":
    main()
