import torch


def sequence(
    model,
    prefix: torch.Tensor,
    weights: dict,
    maximum: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and weighted sampling.

    Args:
        model: Trained ReflectiveCore model.
        prefix (Tensor): [B, C, D] prefix embedding.
        weights (dict): Mapping from state index to probability.
        maximum (int): Maximum number of tokens to generate.
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    model.eval()

    with torch.no_grad():

        V = model.vocab_size
        S = model.state_size
        prefix = prefix.reshape(-1, prefix.shape[-2], prefix.shape[-1])  # [B, C, D]

        # Prepare state indices and normalized weights
        indices = torch.arange(S, device=device)  # [S]
        weights = [weights[e.item()] for e in indices]
        weights = torch.tensor(weights, device=device)
        weights = weights / weights.sum()

        token = torch.empty(0, dtype=torch.long, device=device)  # []
        for length in range(maximum):
            logit = model.forward(token, prefix)  # [B, V, S]

            # Softmax over class dimension S to get prob: [B, V, S]
            probs = F.softmax(logit, dim=2)  # [B, V, S]

            # Compute expected utility of each action, weights: [S] -> [1, 1, S] to broadcast
            prob = torch.einsum("bvs,s->bv", probs, weights)  # shape [B, V]

            # Normalize over actions
            probs = probs / probs.sum(dim=1, keepdim=True)  # shape [B, V]

            # Sample one action per batch item
            prediction = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape [B]

            token = torch.cat([token, prediction])

            if (token == 0).any(dim=1).all().item():
                break

        return tokens
