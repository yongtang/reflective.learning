import torch
import torch.nn.functional as F

from reflective_learning.train import collate_with_prefix


def normalize_distribution(dist, epsilon=1e-4):
    """
    Normalize a dictionary of weights, ensuring no entry is zero.
    Adds epsilon to avoid degenerate distributions.

    Args:
        dist (dict): mapping state index -> float
        epsilon (float): minimum mass per state

    Returns:
        dict: normalized distribution
    """
    adjusted = {k: max(v, epsilon) for k, v in dist.items()}
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
    Generate a token sequence from the model using state-weighted sampling.

    Args:
        model: ReflectiveCore model
        state_weights: dict mapping state index → float
        max_seq_len: maximum number of tokens to generate
        temperature: softmax temperature
        prefix: optional prefix embedding [C, d_model]
        device: 'cpu' or 'cuda'
        epsilon: additive smoothing for token probabilities

    Returns:
        List[int]: sampled token sequence (including STOP token 0 if generated)
    """
    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    tokens = []
    states = []

    # Require either prefix or at least one starting token
    assert (
        prefix is not None or len(tokens) > 0
    ), "Cannot sample: no prefix and no starting tokens provided."

    with torch.no_grad():
        for _ in range(max_seq_len):
            # Create token/state input tensors (possibly empty at first step)
            if tokens:
                token_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                state_ids = torch.tensor([states], dtype=torch.long, device=device)
            else:
                token_ids = torch.empty((1, 0), dtype=torch.long, device=device)
                state_ids = torch.empty((1, 0), dtype=torch.long, device=device)

            # Prepare batch with optional prefix
            batch = [
                {
                    "token_ids": token_ids[0],
                    "state_ids": state_ids[0],
                    "prefix": (
                        prefix
                        if prefix is not None
                        else torch.zeros(0, model.d_model, device=device)
                    ),
                }
            ]

            # Project input and compute attention mask
            outputs = collate_with_prefix(batch, model)
            embed = outputs["embed"]
            mask = outputs["mask"]

            # Forward pass: logits of shape [1, T, V, S]
            logits = model.call(embed.to(device), mask.to(device))
            logits = logits[0, -1]  # [V, S] — logits at final position

            # Blend token distributions across states using state weights
            probs_per_state = []
            for s_id in state_indices:
                probs = F.softmax(logits[:, s_id] / temperature, dim=0)
                probs *= state_weights[s_id]
                probs_per_state.append(probs)

            final_probs = torch.stack(probs_per_state).sum(dim=0)
            final_probs += epsilon  # Smoothing
            final_probs /= final_probs.sum()

            # Sample next token
            next_token = torch.multinomial(final_probs, num_samples=1).item()
            tokens.append(next_token)
            states.append(state_indices[0])  # Final state is fixed for entire sequence

            if next_token == 0:  # STOP token
                break

    return tokens


def sample_multiple_sequences(
    model,
    state_weights,
    num_sequences=10,
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
    Batched version of sequence sampling for faster parallel generation.

    Args:
        model: ReflectiveCore
        state_weights: dict of state index → float
        num_sequences: number of sequences to sample
        max_seq_len: max sequence length
        temperature: sampling temperature
        prefix: shared context embedding [C, d_model]
        device: 'cpu' or 'cuda'
        epsilon: additive smoothing

    Returns:
        List[List[int]]: list of token sequences
    """
    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    sequences = [[] for _ in range(num_sequences)]
    states = [[] for _ in range(num_sequences)]
    finished = [False] * num_sequences

    assert prefix is not None or any(
        len(seq) > 0 for seq in sequences
    ), "Cannot sample: no prefix and no starting tokens provided."

    with torch.no_grad():
        for _ in range(max_seq_len):
            active_indices = [i for i, done in enumerate(finished) if not done]
            if not active_indices:
                break

            # Gather active token/state histories and pad
            token_batch, state_batch = [], []
            for i in active_indices:
                token_batch.append(sequences[i])
                state_batch.append(states[i])

            max_len = max(len(seq) for seq in token_batch)
            padded_tokens = torch.full(
                (len(active_indices), max_len), 0, dtype=torch.long
            )
            padded_states = torch.full(
                (len(active_indices), max_len), 0, dtype=torch.long
            )
            for i, (t_seq, s_seq) in enumerate(zip(token_batch, state_batch)):
                if t_seq:
                    padded_tokens[i, : len(t_seq)] = torch.tensor(
                        t_seq, dtype=torch.long
                    )
                    padded_states[i, : len(s_seq)] = torch.tensor(
                        s_seq, dtype=torch.long
                    )

            padded_tokens = padded_tokens.to(device)
            padded_states = padded_states.to(device)

            batch = [
                {
                    "token_ids": padded_tokens[i],
                    "state_ids": padded_states[i],
                    "prefix": (
                        prefix
                        if prefix is not None
                        else torch.zeros(0, model.d_model, device=device)
                    ),
                }
                for i in range(len(active_indices))
            ]

            outputs = collate_with_prefix(batch, model)
            embed = outputs["embed"]
            mask = outputs["mask"]

            logits = model.call(embed.to(device), mask.to(device))  # [B, T, V, S]
            logits = logits[:, -1, :]  # [B, V, S] — last position

            for i, global_idx in enumerate(active_indices):
                logits_i = logits[i]
                probs_per_state = []
                for s_id in state_indices:
                    probs = F.softmax(logits_i[:, s_id] / temperature, dim=0)
                    probs *= state_weights[s_id]
                    probs_per_state.append(probs)

                final_probs = torch.stack(probs_per_state).sum(dim=0)
                final_probs += epsilon
                final_probs /= final_probs.sum()

                next_token = torch.multinomial(final_probs, num_samples=1).item()
                sequences[global_idx].append(next_token)
                states[global_idx].append(
                    state_indices[0]
                )  # Fixed state for entire sequence

                if next_token == 0:
                    finished[global_idx] = True

    return sequences
