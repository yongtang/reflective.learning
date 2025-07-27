import torch
import torch.nn.functional as F


def sequence(
    model,
    prefix: torch.Tensor,
    state_weights: dict,
    stop_token: int,
    max_seq_len: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and state-weighted sampling.

    Args:
        model: Trained ReflectiveCore model.
        prefix (Tensor): [C, d_model] prefix embedding.
        state_weights (dict): Mapping from state index to probability.
        stop_token (int): Token that terminates generation.
        max_seq_len (int): Maximum number of tokens to generate.
        device (str): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    model.eval()

    V = model.vocab_size
    S = model.state_size
    prefix = prefix.unsqueeze(0).to(device)  # [1, C, d_model]

    # Prepare state indices and normalized weights
    state_indices = torch.arange(S, device=device)  # [S]
    weight_list = [state_weights[s.item()] for s in state_indices]
    state_weights_tensor = torch.tensor(weight_list, device=device)
    state_weights_tensor = state_weights_tensor / state_weights_tensor.sum()

    tokens = torch.empty(0, dtype=torch.long, device=device)  # [T]

    for _ in range(max_seq_len):
        T = tokens.shape[0]

        # Expand current sequence across all states — works even when T == 0
        token_input = tokens.unsqueeze(0).expand(S, -1)  # [S, T]
        state_input = state_indices  # [S]
        prefix_input = prefix.expand(S, -1, -1)  # [S, C, d_model]

        logit = model.forward(token_input, state_input, prefix_input)  # [S, V, S]

        # Select diagonal: P(token | state=s)
        diag_logits = logit.permute(1, 0, 2).diagonal(dim1=1, dim2=2).T  # [S, V]

        # Combine across states using provided weights
        probs = (diag_logits.softmax(dim=-1) * state_weights_tensor[:, None]).sum(
            dim=0
        )  # [V]

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens = torch.cat([tokens, torch.tensor([next_token], device=device)])

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
