import torch
import torch.nn.functional as F

from reflective_learning.train import collate_with_prefix


def normalize_distribution(dist, epsilon=1e-4):
    """
    Normalize a dictionary of weights, ensuring no entry is zero.
    Adds epsilon to avoid degenerate distributions.
    """
    adjusted = {k: max(v, epsilon) for k, v in dist.items()}
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def truncate_at_stop(tokens, states):
    """
    Truncate tokens/states at the first STOP token (token == 0).
    """
    if 0 in tokens:
        stop_index = tokens.index(0)
        return tokens[: stop_index + 1], states[: stop_index + 1]
    return tokens, states


def sample_sequence(
    model,
    state_weights: dict,
    max_seq_len: int = 128,
    temperature: float = 1.0,
    prefix: torch.Tensor = None,
    token_ids: torch.LongTensor = None,
    state_ids: torch.LongTensor = None,
    device: str = "cpu",
    epsilon: float = 1e-6,
):
    """
    Generate a token sequence using state-weighted sampling.
    Stops at STOP token (0) or when max_seq_len is reached.
    """
    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    tokens = token_ids.tolist() if token_ids is not None else []
    states = state_ids.tolist() if state_ids is not None else []

    # If STOP token already appears, truncate and return early
    tokens, states = truncate_at_stop(tokens, states)
    if tokens and tokens[-1] == 0:
        return tokens

    assert (
        prefix is not None or len(tokens) > 0
    ), "Cannot sample: no prefix and no starting tokens provided."

    with torch.no_grad():
        for _ in range(max_seq_len - len(tokens)):
            # Prepare input batch of size 1
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            sta_tensor = torch.tensor(states, dtype=torch.long, device=device)
            batch = [
                {
                    "token_ids": tok_tensor,
                    "state_ids": sta_tensor,
                    "prefix": (
                        prefix
                        if prefix is not None
                        else torch.zeros(0, model.d_model, device=device)
                    ),
                }
            ]

            outputs = collate_with_prefix(batch, model)
            embed, mask = outputs["embed"], outputs["mask"]

            logits = model.call(embed.to(device), mask.to(device))  # [1, T, V, S]
            logits = logits[0, -1]  # [V, S]

            # Weighted combination of state-conditioned distributions
            probs = sum(
                F.softmax(logits[:, s] / temperature, dim=0) * state_weights[s]
                for s in state_indices
            )
            probs += epsilon
            probs /= probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            states.append(state_indices[0])  # Final state is fixed for now

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
    token_ids: torch.LongTensor = None,
    state_ids: torch.LongTensor = None,
    device: str = "cpu",
    epsilon: float = 1e-6,
):
    """
    Batched version of sampling. Each sequence is generated independently,
    but processed in parallel for speed.
    """
    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    # Initial tokens/states (can be empty)
    init_tokens = (
        token_ids.tolist()
        if token_ids is not None
        else [[] for _ in range(num_sequences)]
    )
    init_states = (
        state_ids.tolist()
        if state_ids is not None
        else [[] for _ in range(num_sequences)]
    )

    # Truncate at STOP if already present
    sequences, state_seqs, finished = [], [], []
    for t_seq, s_seq in zip(init_tokens, init_states):
        t_trunc, s_trunc = truncate_at_stop(t_seq, s_seq)
        sequences.append(t_trunc)
        state_seqs.append(s_trunc)
        finished.append(t_trunc and t_trunc[-1] == 0)

    assert prefix is not None or any(
        len(seq) > 0 for seq in sequences
    ), "Cannot sample: no prefix and no starting tokens provided."

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
                padded_states[i, : len(state_seqs[idx])] = torch.tensor(
                    state_seqs[idx], dtype=torch.long
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
                state_seqs[global_idx].append(state_indices[0])

                if next_token == 0:
                    finished[global_idx] = True

    return sequences
