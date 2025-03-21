import torch
import torch.nn.functional as F


def normalize_distribution(dist, epsilon=1e-4):
    """Ensure every state has non-zero mass and normalize the distribution."""
    adjusted = {k: max(v, epsilon) for k, v in dist.items()}
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def sample_sequence(
    model,
    state_weights,
    start_token=0,
    max_len=128,
    temperature=1.0,
    stop_token=0,
    device="cpu",
):
    """
    Generate a single sequence using state-weighted sampling.

    Args:
        model: ReflectiveTransformer
        state_weights: dict mapping state index -> float (weights must sum to 1)
        start_token: initial token (usually 0)
        max_len: max number of tokens to generate
        temperature: softmax temperature
        stop_token: token that ends the sequence
        device: device to run model on

    Returns:
        List[int]: the sampled token sequence (including stop_token)
    """
    model.eval()
    state_weights = normalize_distribution(state_weights)
    state_indices = list(state_weights.keys())

    tokens = [start_token]
    with torch.no_grad():
        for _ in range(max_len):
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            state_tensor = torch.tensor(
                [[s] * len(tokens) for s in state_indices],
                dtype=torch.long,
                device=device,
            )

            logits_all_states = []
            for s_id, s_tensor in zip(state_indices, state_tensor):
                logits = model(token_tensor, s_tensor.unsqueeze(0))  # (1, T, V, S)
                logits = logits[0, -1]  # (V, S)
                prob = F.softmax(logits[:, s_id] / temperature, dim=0)  # (V,)
                prob *= state_weights[s_id]
                logits_all_states.append(prob)

            final_prob = torch.stack(logits_all_states, dim=0).sum(dim=0)
            next_token = torch.multinomial(final_prob, num_samples=1).item()
            tokens.append(next_token)

            if next_token == stop_token:
                break

    return tokens


def sample_multiple_sequences(
    model,
    state_weights,
    num_sequences=10,
    **kwargs,
):
    """Sample multiple sequences using the same configuration."""
    return [
        sample_sequence(model, state_weights, **kwargs) for _ in range(num_sequences)
    ]
