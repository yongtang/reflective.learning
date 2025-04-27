"""
MiniGrid Model Pipeline — Reflective Learning + RL Benchmarking

Modes:
------
--mode seed       : Generate labeled samples (with tokens and success state)
--mode stub       : Generate input-only samples (text + image) for Reflective model
--mode predict    : Use model (default or Reflective) to predict action sequences
                    (optionally specify --state-weights like "success=0.99,fail=0.01" and --checkpoint)
--mode verify     : Validate predicted tokens via environment replay (adds "state")
--mode baseline   : Train PPO using stable-baselines3, then evaluate performance

Examples:
---------
python -m src.reflective_learning.tools.mini --mode seed --output seed.json --samples 50
python -m src.reflective_learning.tools.mini --mode stub --output stub.json --samples 50
python -m src.reflective_learning.tools.mini --mode predict --input stub.json --output predicted.json --model reflective --checkpoint path/to/model.pt --state-weights "success=0.99,fail=0.01"
python -m src.reflective_learning.tools.mini --mode verify --input predicted.json --output verified.json
python -m src.reflective_learning.tools.mini --mode baseline --timesteps 100000 --episodes 20 --save_model ppo_model.zip
"""

import argparse
import base64
import json
import os
import random

import gymnasium
import minigrid  # pylint: disable=unused-import
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm  # <-- added tqdm

from reflective_learning.inference import sample_multiple_sequences_batched
from reflective_learning.model import ReflectiveCore

# Action encoding for MiniGrid
ACTION_MAP = {"left": 0, "right": 1, "forward": 2}


# ----------------------------------------
# Orientation-aware planner using agent_dir
# ----------------------------------------
def orientation_aware_planner(start, goal, start_dir=0):
    """
    Simulates the MiniGrid agent's orientation and returns
    a list of "left", "right", "forward" actions to reach the goal.
    """
    path = []
    x, y = start
    gx, gy = goal
    dir = start_dir  # 0: right, 1: down, 2: left, 3: up

    def rotate_to(desired_dir, current_dir):
        turns = []
        while current_dir != desired_dir:
            turns.append("right")
            current_dir = (current_dir + 1) % 4
        return turns, current_dir

    # Move horizontally
    if gx != x:
        desired = 0 if gx > x else 2
        turns, dir = rotate_to(desired, dir)
        path.extend(turns)
        for _ in range(abs(gx - x)):
            path.append("forward")
        x = gx

    # Move vertically
    if gy != y:
        desired = 1 if gy > y else 3
        turns, dir = rotate_to(desired, dir)
        path.extend(turns)
        for _ in range(abs(gy - y)):
            path.append("forward")

    return path


# ----------------------------------------
# Get random valid (non-wall) grid position
# ----------------------------------------
def get_random_pos(env):
    return (random.randint(1, env.width - 2), random.randint(1, env.height - 2))


# ----------------------------------------
# Render and save current environment as image
# ----------------------------------------
def render_env_image(env, output_dir, map_id):
    img = env.render()
    image = PIL.Image.fromarray(img)
    filename = f"map_{map_id}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return filename


# ----------------------------------------
# Generate samples for seed or stub
# ----------------------------------------
def generate_samples(
    env_name, output_json, image_dir, num_samples, include_labels=True
):
    os.makedirs(image_dir, exist_ok=True)
    samples = []

    for i in tqdm(range(num_samples), desc="Generating Samples"):
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env.reset()

        start = get_random_pos(env.unwrapped)
        goal = get_random_pos(env.unwrapped)
        while goal == start:
            goal = get_random_pos(env.unwrapped)

        env.unwrapped.agent_pos = list(start)
        agent_dir = random.randint(0, 3)
        env.unwrapped.agent_dir = agent_dir

        image_filename = render_env_image(env, image_dir, i)

        sample = {
            "text": [f"start {start[0]},{start[1]}", f"stop {goal[0]},{goal[1]}"],
            "image": [image_filename],
            "start_pos": list(start),
            "goal_pos": list(goal),
            "map_id": i,
        }

        if include_labels:
            actions = orientation_aware_planner(start, goal, agent_dir)
            sample["token"] = actions
            sample["state"] = "success"

        samples.append(sample)

    with open(output_json, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")

    print(
        f"✅ Wrote {len(samples)} {'labeled' if include_labels else 'stub'} samples to {output_json}"
    )


# ----------------------------------------
# Predict token sequence using a model
# ----------------------------------------
def predict_tokens(
    input_json,
    output_json,
    model_type="minigrid",
    state_weights_str="success=0.99,fail=0.01",
    checkpoint_path=None,
):
    with open(input_json, "r") as f:
        samples = [json.loads(line) for line in f]

    if model_type == "reflective":
        vocab_map = ACTION_MAP
        id_to_token = {v: k for k, v in vocab_map.items()}

        model = ReflectiveCore(
            vocab_size=len(vocab_map),
            state_size=2,
            max_seq_len=128,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is None:
            base_dir = os.path.dirname(input_json)
            checkpoint_path = os.path.join(base_dir, "model.pt")

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        # Parse state weights
        state_weights = {}
        for pair in state_weights_str.split(","):
            key, value = pair.split("=")
            key = key.strip().lower()
            value = float(value.strip())
            state_id = 0 if key == "success" else 1
            state_weights[state_id] = value

        predictions = []
        for sample in tqdm(samples, desc="Predicting Tokens"):
            prefix_field = sample.get("prefix")
            if not prefix_field or not prefix_field.startswith("b64://"):
                raise ValueError(
                    "Missing or invalid 'prefix' field for reflective model."
                )

            prefix_bytes = base64.b64decode(prefix_field.removeprefix("b64://"))
            prefix_tensor = torch.from_numpy(
                np.frombuffer(prefix_bytes, dtype=np.float32)
            ).to(device)
            prefix_tensor = prefix_tensor.view(-1, model.d_model)

            tokens = sample_multiple_sequences_batched(
                model=model,
                state_weights=state_weights,
                num_sequences=1,
                max_seq_len=128,
                temperature=1.0,
                prefix=prefix_tensor,
                device=device,
                stop_token=None,
            )[0]

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


# ----------------------------------------
# Validate action sequence by replaying in MiniGrid
# ----------------------------------------
def validate_output(input_json, output_json, env_name):
    def replay(env, token_seq):
        env.reset()
        for token in token_seq:
            assert token in ACTION_MAP, f"Invalid token: {token}"
            action = ACTION_MAP[token]
            _, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                return "success" if reward > 0 else "fail"
        return "fail"

    with open(input_json, "r") as f:
        samples = [json.loads(line) for line in f]

    validated = []
    for sample in tqdm(samples, desc="Validating Outputs"):
        assert "token" in sample, f"Missing 'token' in sample: {sample}"
        env = gymnasium.make(env_name, render_mode="rgb_array")
        sample["state"] = replay(env, sample["token"])
        validated.append(sample)

    with open(output_json, "w") as f:
        for item in validated:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Verified {len(validated)} samples and saved to {output_json}")


# ----------------------------------------
# Train PPO with SB3 and evaluate it
# ----------------------------------------
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
        f"\n📊 PPO Evaluation: Success={success}/{eval_episodes}, Fail={fail}/{eval_episodes}"
    )


# ----------------------------------------
# CLI Entry Point
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MiniGrid Model Pipeline CLI (Reflective + RL)"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["seed", "stub", "predict", "verify", "baseline"],
        help="Which pipeline stage to run",
    )
    parser.add_argument("--env", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--input", help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--image_dir", default="images")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save_model")
    parser.add_argument(
        "--model", default="minigrid", choices=["minigrid", "reflective"]
    )
    parser.add_argument("--state-weights", type=str, default="success=0.99,fail=0.01")
    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint path for reflective model"
    )

    args = parser.parse_args()

    if args.mode == "seed":
        assert args.output, "--output is required for --mode seed"
        generate_samples(
            args.env, args.output, args.image_dir, args.samples, include_labels=True
        )

    elif args.mode == "stub":
        assert args.output, "--output is required for --mode stub"
        generate_samples(
            args.env, args.output, args.image_dir, args.samples, include_labels=False
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
        )

    elif args.mode == "verify":
        assert (
            args.input and args.output
        ), "--input and --output are required for --mode verify"
        validate_output(args.input, args.output, args.env)

    elif args.mode == "baseline":
        train_and_evaluate_ppo(
            args.env, args.timesteps, args.episodes, save_path=args.save_model
        )

    else:
        assert False, f"Unhandled mode: {args.mode}"


if __name__ == "__main__":
    main()
