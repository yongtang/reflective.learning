"""
MiniGrid Model Pipeline — Reflective Learning + RL Benchmarking

Modes:
------
--mode seed       : Generate labeled samples (text + image + facing + tokens + success state)
--mode stub       : Generate input-only samples (text + image + facing) for Reflective model
--mode predict    : Use model (default or Reflective) to predict action sequences
                    (optionally specify --state-weights like "0=0.99,1=0.01" and --checkpoint and --batch-size)
--mode verify     : Validate predicted tokens via environment replay (uses start_pos and facing)
--mode baseline   : Train PPO using stable-baselines3, then evaluate performance

Examples:
---------
python -m src.reflective_learning.tools.mini --mode seed --output seed.json --samples 50 --image image --max-success-steps 3
python -m src.reflective_learning.tools.mini --mode stub --output stub.json --samples 50 --image image --max-success-steps 3
python -m src.reflective_learning.tools.mini --mode predict --input stub.json --output predicted.json --model reflective --state-weights "0=0.9,1=0.1,2=0.0,3=0.0" --checkpoint path/to/model.pt --batch-size 32 --max-success-steps 3
python -m src.reflective_learning.tools.mini --mode verify --input predicted.json --output verified.json --image image --max-success-steps 3
python -m src.reflective_learning.tools.mini --mode baseline --timesteps 100000 --episodes 20 --save_model ppo_model.zip
"""

import argparse
import base64
import functools
import json
import os

import gymnasium
import minigrid  # pylint: disable=unused-import
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from reflective_learning.inference import sample_multiple_sequences_batched
from reflective_learning.model import ReflectiveCore

ACTION_MAP = {"left": 0, "right": 1, "forward": 2}
DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}
STR_TO_DIR = {v: k for k, v in DIR_TO_STR.items()}


def orientation_aware_planner(start, goal, start_dir=0):
    path = []
    x, y = start
    gx, gy = goal
    dir = start_dir

    def rotate_to(desired_dir, current_dir):
        turns = []
        while current_dir != desired_dir:
            turns.append("right")
            current_dir = (current_dir + 1) % 4
        return turns, current_dir

    if gx != x:
        desired = 0 if gx > x else 2
        turns, dir = rotate_to(desired, dir)
        path.extend(turns)
        for _ in range(abs(gx - x)):
            path.append("forward")
        x = gx

    if gy != y:
        desired = 1 if gy > y else 3
        turns, dir = rotate_to(desired, dir)
        path.extend(turns)
        for _ in range(abs(gy - y)):
            path.append("forward")

    return path


def render_env_image(env, output_dir, map_id):
    os.makedirs(output_dir, exist_ok=True)
    img = env.render()
    image = PIL.Image.fromarray(img)
    filename = f"map_{map_id}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return filename


def find_goal_pos(grid):
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj and obj.type == "goal":
                return [x, y]
    raise ValueError("Goal not found in grid")


@functools.lru_cache(maxsize=None)
def render_and_save_image(
    env_name, image_dir, start_x, start_y, goal_x, goal_y, agent_dir
):
    dir_str = DIR_TO_STR[agent_dir]
    filename = f"start_{start_x}_{start_y}_goal_{goal_x}_{goal_y}_facing_{dir_str}.png"
    path = os.path.join(image_dir, filename)

    env = gymnasium.make(env_name, render_mode="rgb_array")
    env.reset()
    img = env.render()
    image = PIL.Image.fromarray(img)
    image.save(path)
    env.close()

    return filename


def generate_samples(
    env_name,
    output_json,
    image_dir,
    num_samples,
    include_labels=True,
    max_success_steps=None,
):
    samples = []

    os.makedirs(image_dir, exist_ok=True)

    with tqdm(total=num_samples, desc="Generating Samples") as pbar:
        for _ in range(num_samples):
            env = gymnasium.make(env_name, render_mode="rgb_array")
            env.reset()

            start = list(env.unwrapped.agent_pos)
            agent_dir = env.unwrapped.agent_dir
            goal = find_goal_pos(env.unwrapped.grid)

            image_filename = render_and_save_image(
                env_name, image_dir, start[0], start[1], goal[0], goal[1], agent_dir
            )

            sample = {
                "text": [
                    f"start {start[0]},{start[1]}",
                    f"stop {goal[0]},{goal[1]}",
                    f"facing {DIR_TO_STR[agent_dir]}",
                ],
                "image": [image_filename],
                "start_pos": [int(x) for x in start],
                "goal_pos": [int(x) for x in goal],
                "facing": DIR_TO_STR[agent_dir],
            }

            if include_labels:
                actions = orientation_aware_planner(start, goal, agent_dir)
                sample["token"] = actions
                sample["state"] = (
                    str(len(actions))
                    if len(actions) <= max_success_steps
                    else str(max_success_steps + 1)
                )

            samples.append(sample)
            env.close()
            pbar.update(1)

    with open(output_json, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")

    print(
        f"✅ Wrote {len(samples)} {'labeled' if include_labels else 'stub'} samples to {output_json}"
    )


def predict_tokens(
    input_json,
    output_json,
    model_type="minigrid",
    state_weights_str="0=1.0",
    checkpoint_path=None,
    batch_size=32,
    max_success_steps=None,
    max_seq_len=128,
):
    with open(input_json, "r") as f:
        samples = [json.loads(line) for line in f]

    if model_type == "reflective":
        vocab_map = ACTION_MAP
        id_to_token = {v: k for k, v in vocab_map.items()}

        model = ReflectiveCore(
            vocab_size=len(vocab_map),
            state_size=max_success_steps + 2,
            max_seq_len=max_seq_len,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is None:
            base_dir = os.path.dirname(input_json)
            checkpoint_path = os.path.join(base_dir, "model.pt")

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        state_weights = {}
        for pair in state_weights_str.split(","):
            key, value = pair.split("=")
            state_weights[int(key.strip())] = float(value.strip())

        @functools.lru_cache(maxsize=4096)
        def decode_prefix(prefix_field: str) -> torch.Tensor:
            assert prefix_field.startswith("b64://"), "Invalid prefix format"
            prefix_bytes = base64.b64decode(prefix_field.removeprefix("b64://"))
            return torch.from_numpy(np.frombuffer(prefix_bytes, dtype=np.float32)).view(
                -1, model.d_model
            )

        predictions = []
        for i in tqdm(range(0, len(samples), batch_size), desc="Predicting Tokens"):
            batch_samples = samples[i : i + batch_size]

            prefix_tensors = []
            prefix_lens = []
            for sample in batch_samples:
                prefix_tensor = decode_prefix(sample["prefix"]).to(device)
                prefix_tensors.append(prefix_tensor)
                prefix_lens.append(prefix_tensor.size(0))

            max_prefix_len = max(prefix_lens)
            batch_prefixes = torch.zeros(
                len(prefix_tensors), max_prefix_len, model.d_model, device=device
            )

            for j, tensor in enumerate(prefix_tensors):
                batch_prefixes[j, : tensor.size(0)] = tensor

            tokens_batch = sample_multiple_sequences_batched(
                model=model,
                state_weights=state_weights,
                num_sequences=len(batch_samples),
                max_seq_len=max_seq_len,
                temperature=1.0,
                prefixes=batch_prefixes,
                device=device,
                stop_token=None,
            )

            for tokens in tokens_batch:
                decoded = [id_to_token[tok] for tok in tokens]
                predictions.append(decoded)
    else:
        predictions = [["forward"] * 5 for _ in samples]

    for sample, tokens in zip(samples, predictions):
        sample["token"] = tokens

    with open(output_json, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Wrote predictions using model '{model_type}' to {output_json}")


def validate_output(
    input_json, output_json, env_name, image_dir, max_success_steps=None
):
    def replay(env, token_seq, start_pos, facing_str):
        env.reset()
        env.unwrapped.agent_pos = list(start_pos)
        env.unwrapped.agent_dir = STR_TO_DIR[facing_str]

        step_count = 0
        for token in token_seq:
            assert token in ACTION_MAP, f"Invalid token: {token}"
            action = ACTION_MAP[token]
            _, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            if terminated or truncated:
                break

        if reward > 0 and step_count <= max_success_steps:
            return str(step_count)
        else:
            return str(max_success_steps + 1)

    with open(input_json, "r") as f:
        samples = [json.loads(line) for line in f]

    validated = []
    for sample in tqdm(samples, desc="Validating Outputs"):
        assert "token" in sample, f"Missing 'token' in sample: {sample}"
        assert (
            "start_pos" in sample and "facing" in sample
        ), "Missing start_pos or facing"

        env = gymnasium.make(env_name, render_mode="rgb_array")
        sample["state"] = replay(
            env, sample["token"], sample["start_pos"], sample["facing"]
        )
        validated.append(sample)

    with open(output_json, "w") as f:
        for item in validated:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Verified {len(validated)} samples and saved to {output_json}")


def train_and_evaluate_ppo(env_name, total_timesteps, eval_episodes, save_path=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    def make_env():
        return gymnasium.make(env_name, render_mode="rgb_array")

    env = make_vec_env(make_env, n_envs=4)
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    if save_path:
        model.save(save_path)
        print(f"✅ Saved PPO model to {save_path}")

    eval_env = gymnasium.make(env_name, render_mode="rgb_array")
    success, fail = 0, 0

    for _ in tqdm(range(eval_episodes), desc="Evaluating PPO"):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        if reward > 0:
            success += 1
        else:
            fail += 1

    print(
        f"PPO Evaluation: Success={success}/{eval_episodes}, Fail={fail}/{eval_episodes}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="MiniGrid Model Pipeline CLI (Reflective + RL)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["seed", "stub", "predict", "verify", "baseline"],
    )
    parser.add_argument("--env", default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--image", default="image")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save_model")
    parser.add_argument(
        "--model", default="minigrid", choices=["minigrid", "reflective"]
    )
    parser.add_argument("--state-weights", type=str, default="0=1.0")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--max-success-steps",
        type=int,
        required=True,
        help="Max number of steps to count as success. > this = failure (state = n + 1)",
    )

    args = parser.parse_args()

    if args.mode == "seed":
        assert args.output, "--output is required for --mode seed"
        generate_samples(
            args.env,
            args.output,
            args.image,
            args.samples,
            include_labels=True,
            max_success_steps=args.max_success_steps,
        )

    elif args.mode == "stub":
        assert args.output, "--output is required for --mode stub"
        generate_samples(
            args.env,
            args.output,
            args.image,
            args.samples,
            include_labels=False,
            max_success_steps=args.max_success_steps,
        )

    elif args.mode == "predict":
        assert (
            args.input and args.output
        ), "--input and --output are required for --mode predict"
        predict_tokens(
            args.input,
            args.output,
            model_type=args.model,
            state_weights_str=args.state_weights,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            max_success_steps=args.max_success_steps,
            max_seq_len=args.max_seq_len,
        )

    elif args.mode == "verify":
        assert (
            args.input and args.output
        ), "--input and --output are required for --mode verify"
        validate_output(
            args.input,
            args.output,
            args.env,
            args.image,
            max_success_steps=args.max_success_steps,
        )

    elif args.mode == "baseline":
        train_and_evaluate_ppo(
            args.env, args.timesteps, args.episodes, save_path=args.save_model
        )

    else:
        assert False, f"Unhandled mode: {args.mode}"


if __name__ == "__main__":
    main()
