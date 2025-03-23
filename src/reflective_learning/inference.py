import torch
import torch.nn.functional as F

from reflective_learning.train import collate_with_prefix


def normalize_distribution(distribution, epsilon=1e-4):
    """
    Normalize a dictionary of weights, ensuring no entry is zero.
    Adds epsilon to avoid degenerate distributions.

    Args:
        distribution (dict): mapping from state index → weight
        epsilon (float): minimum non-zero mass

    Returns:
        dict: normalized and smoothed distribution
    """
    adjusted = {k: max(v, epsilon) for k, v in distribution.items()}
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def sample_sequence(
    model,
    state_weights: dict,
    max_seq_len: int = 128,
    temperature: float = 1.0,
    prefix: torch.Tensor = None,
    device: str = "cpu",
    epsilon: float = 1e-6,
):
    """
    Generate a token sequence using state-weighted sampling.
    Stops at STOP token (0) or when max_seq_len is reached.

    Args:
        model: ReflectiveCore model
        state_weights: dict mapping state index → float
        max_seq_len: max length to generate
        temperature: sampling temperature
        prefix: context prefix embedding [C, d_model] (required)
        device: computation device
        epsilon: smoothing factor to avoid zero probs

    Returns:
        List[int]: sampled token sequence
    """
    assert prefix is not None, "prefix is required for generation"

    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    tokens = []
    states = []

    with torch.no_grad():
        for _ in range(max_seq_len):
            batch = [
                {
                    "token_ids": torch.tensor(tokens, dtype=torch.long),
                    "state_ids": torch.tensor(states, dtype=torch.long),
                    "prefix": prefix,
                }
            ]
            outputs = collate_with_prefix(batch, model)
            embed, mask = outputs["embed"], outputs["mask"]

            logits = model.call(embed.to(device), mask.to(device))  # [1, T, V, S]
            logits = logits[0, -1]  # [V, S]

            # Weighted state-conditioned token probabilities
            probs = sum(
                F.softmax(logits[:, s] / temperature, dim=0) * state_weights[s]
                for s in state_indices
            )
            probs += epsilon
            probs /= probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            states.append(state_indices[0])  # Use fixed final state for now

            if next_token == 0:
                break

    return tokens


def sample_multiple_sequences(
    model,
    state_weights: dict,
    num_sequences: int = 10,
    **kwargs,
):
    """
    Sample multiple sequences independently (non-batched).

    Args:
        model: ReflectiveCore
        state_weights: dict of state index → float
        num_sequences: number of sequences to sample

    Returns:
        List[List[int]]: token sequences
    """
    return [
        sample_sequence(model, state_weights, **kwargs) for _ in range(num_sequences)
    ]


def sample_multiple_sequences_batched(
    model,
    state_weights: dict,
    num_sequences: int = 10,
    max_seq_len: int = 128,
    temperature: float = 1.0,
    prefix: torch.Tensor = None,
    device: str = "cpu",
    epsilon: float = 1e-6,
):
    """
    Batched version of sampling. Each sequence is generated independently,
    but processed in parallel for speed.

    Args:
        model: ReflectiveCore
        state_weights: dict of state index → float
        num_sequences: number of sequences to sample
        max_seq_len: max generation length
        temperature: sampling temperature
        prefix: shared context prefix embedding [C, d_model] (required)
        device: computation device
        epsilon: smoothing factor

    Returns:
        List[List[int]]: generated token sequences
    """
    assert prefix is not None, "prefix is required for generation"

    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    sequences = [[] for _ in range(num_sequences)]
    states = [[] for _ in range(num_sequences)]
    finished = [False] * num_sequences

    with torch.no_grad():
        for _ in range(max_seq_len):
            active_indices = [i for i, done in enumerate(finished) if not done]
            if not active_indices:
                break

            max_len = max(len(sequences[i]) for i in active_indices)
            padded_tokens = torch.zeros(len(active_indices), max_len, dtype=torch.long)
            padded_states = torch.zeros(len(active_indices), max_len, dtype=torch.long)

            for i, idx in enumerate(active_indices):
                padded_tokens[i, : len(sequences[idx])] = torch.tensor(
                    sequences[idx], dtype=torch.long
                )
                padded_states[i, : len(states[idx])] = torch.tensor(
                    states[idx], dtype=torch.long
                )

            padded_tokens = padded_tokens.to(device)
            padded_states = padded_states.to(device)

            batch = [
                {
                    "token_ids": padded_tokens[i],
                    "state_ids": padded_states[i],
                    "prefix": prefix,
                }
                for i in range(len(active_indices))
            ]

            outputs = collate_with_prefix(batch, model)
            embed = outputs["embed"]
            mask = outputs["mask"]

            logits = model.call(embed.to(device), mask.to(device))  # [B, T, V, S]
            logits = logits[:, -1, :]  # [B, V, S]

            for i, global_idx in enumerate(active_indices):
                logit_i = logits[i]
                probs = sum(
                    F.softmax(logit_i[:, s] / temperature, dim=0) * state_weights[s]
                    for s in state_indices
                )
                probs += epsilon
                probs /= probs.sum()

                next_token = torch.multinomial(probs, num_samples=1).item()
                sequences[global_idx].append(next_token)
                states[global_idx].append(state_indices[0])  # fixed final state

                if next_token == 0:
                    finished[global_idx] = True

    return sequences
