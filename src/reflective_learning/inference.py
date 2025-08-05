import torch


def sequence(
    model,
    prefix: torch.Tensor,
    weight: torch.Tensor,
    maximum: int,
    conditioned: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and weighted sampling.

    Args:
        model: Trained ReflectiveCore model.
        prefix (Tensor): [B, C, D] prefix embedding.
        weight (Tensor): Mapping from state index to probability.
        maximum (int): Maximum number of tokens to generate.
        conditioned (bool): If True, sample state and use P(token | state) prediction.
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    model.eval()

    with torch.no_grad():

        V = model.vocab_size
        S = model.state_size

        prefix = torch.as_tensor(prefix, dtype=torch.float32, device=device)

        dim = prefix.dim()

        prefix = prefix.unsqueeze(0) if dim == 2 else prefix
        B = prefix.size(0)

        # Normalized weight
        weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
        weight = torch.clamp(weight, min=0)
        weight = weight / weight.sum()  # Ensures sum = 1.0
        token = torch.empty([B, 0], dtype=torch.long, device=device)  # [B, 0]
        for length in range(maximum):
            if conditioned:
                state = torch.multinomial(
                    weight, num_samples=B, replacement=True
                )  # [B]
                logit = model.forward(token, prefix, state=state)  # [B, V]
                probs = torch.nn.functional.softmax(logit, dim=1)  # [B, V]
            else:
                logit = model.forward(token, prefix)  # [B, V, S]

                # Softmax over class dimension S to get prob: [B, V, S]
                probs = torch.nn.functional.softmax(logit, dim=2)  # [B, V, S]

                # Compute expected utility of each action, weight: [S] -> [1, 1, S] to broadcast
                probs = torch.einsum("bvs,s->bv", probs, weight)  # shape [B, V]

                # Normalize over actions
                probs = probs / probs.sum(dim=1, keepdim=True)  # shape [B, V]

            # Sample one action per batch item
            prediction = torch.multinomial(probs, num_samples=1)  # shape [B]

            token = torch.cat([token, prediction], dim=1)

            if (token == 0).any(dim=1).all().item():
                break

        token = token.squeeze(0) if dim == 2 else token

        return token
