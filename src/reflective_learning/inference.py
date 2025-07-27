import torch
import torch.nn.functional as F


def sequence(
    model,
    state_weights: dict,
    prefix: torch.Tensor,
    stop_token: int,
    max_seq_len: int = 128,
    device: str = "cpu",
):
    """
    Greedy sampling of a single token sequence from the model given a context prefix
    and state weights. Uses model.forward() for simplicity.

    Args:
        model: ReflectiveCore instance
        state_weights: dict of {state_index: weight}
        prefix: [C, d_model] tensor
        stop_token: token id that terminates generation (e.g. 0)
        max_seq_len: maximum tokens to generate (excluding prefix)
        device: execution device

    Returns:
        List[int]: sampled token sequence
    """
    model.eval()
    state_indices = list(state_weights.keys())
    state_weight_tensor = torch.zeros(model.state_size, device=device)
    for s, w in state_weights.items():
        state_weight_tensor[s] = w
    state_weight_tensor /= state_weight_tensor.sum()

    tokens, states = [], []

    with torch.no_grad():
        for _ in range(max_seq_len):
            token_tensor = torch.tensor(
                tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
            state_tensor = torch.tensor(
                [state_indices[0]] * len(tokens), dtype=torch.long, device=device
            ).unsqueeze(0)

            logits = model.forward(
                token_tensor, state_tensor, prefix.to(device)
            )  # [1, T, V, S]
            logits = logits[0, -1]  # [V, S]

            probs = (F.softmax(logits, dim=0) @ state_weight_tensor).clamp(min=1e-8)
            probs /= probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            states.append(state_indices[0])  # Fixed per-sequence state

            if next_token == stop_token:
                break

    return tokens


def sequence_batched(
    model,
    state_weights: dict,
    prefixes: torch.Tensor,
    stop_token: int,
    max_seq_len: int = 128,
    device: str = "cpu",
):
    """
    Batched sampling of multiple token sequences in parallel using model.call()
    for better performance.

    Args:
        model: ReflectiveCore instance
        state_weights: dict of {state_index: weight}
        prefixes: [B, C, d_model] tensor
        stop_token: token id that terminates generation (e.g. 0)
        max_seq_len: maximum tokens to generate (excluding prefix)
        device: execution device

    Returns:
        List[List[int]]: generated token sequences
    """
    model.eval()
    state_indices = list(state_weights.keys())
    state_weight_tensor = torch.zeros(model.state_size, device=device)
    for s, w in state_weights.items():
        state_weight_tensor[s] = w
    state_weight_tensor /= state_weight_tensor.sum()

    B = prefixes.size(0)
    sequences = [[] for _ in range(B)]
    states = [[] for _ in range(B)]
    finished = [False] * B

    with torch.no_grad():
        for _ in range(max_seq_len):
            active = [i for i, done in enumerate(finished) if not done]
            if not active:
                break

            max_len = max(len(sequences[i]) for i in active)
            token_tensor = torch.zeros(len(active), max_len, dtype=torch.long)
            state_tensor = torch.zeros(len(active), max_len, dtype=torch.long)

            for i, idx in enumerate(active):
                token_tensor[i, : len(sequences[idx])] = torch.tensor(
                    sequences[idx], dtype=torch.long
                )
                state_tensor[i, : len(states[idx])] = torch.tensor(
                    states[idx], dtype=torch.long
                )

            batch = [
                {
                    "token": token_tensor[i],
                    "state": state_tensor[i],
                    "prefix": prefixes[idx],
                }
                for i, idx in enumerate(active)
            ]
            batch = model.collate(batch)
            logits = model.call(
                batch["embed"].to(device),
                batch["mask"].to(device),
            )  # [B, L, V, S]
            logits = logits[:, -1]  # [B, V, S]

            for i, global_idx in enumerate(active):
                prob = (F.softmax(logits[i], dim=0) @ state_weight_tensor).clamp(
                    min=1e-8
                )
                prob /= prob.sum()

                next_token = torch.multinomial(prob, num_samples=1).item()
                sequences[global_idx].append(next_token)
                states[global_idx].append(state_indices[0])  # fixed per-sequence state

                if next_token == stop_token:
                    finished[global_idx] = True

    return sequences
