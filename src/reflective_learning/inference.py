import torch


def sequence(
    model,
    prefix: torch.Tensor,  # [C, D]
    maximum: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and greedy sampling.

    Args:
        model: Trained ReflectiveCore model.
        prefix (Tensor): [C, D] prefix embedding.
        maximum (int): Maximum number of tokens to generate.
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    model.eval()
    with torch.no_grad():
        prefix = prefix.to(dtype=torch.float32, device=device)
        token = torch.empty([0], dtype=torch.long, device=device)  # [T]

        for _ in range(maximum):
            logits = model.forward(token, prefix)  # [V]
            probs = torch.softmax(logits, dim=0)  # [V]
            next_token = torch.multinomial(probs, num_samples=1)  # [1]
            token = torch.cat([token, next_token], dim=0)  # [T+1]

            if next_token.item() == 0:
                break

        return token  # [T]


def explore(
    model,
    prefix: torch.Tensor,  # [C, D]
    maximum: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and greedy sampling.

    Args:
        model: Trained ReflectiveCore model.
        prefix (Tensor): [C, D] prefix embedding.
        maximum (int): Maximum number of tokens to generate.
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    for e in model:
        e.eval()
    with torch.no_grad():
        prefix = prefix.to(dtype=torch.float32, device=device)
        token = torch.empty([0], dtype=torch.long, device=device)  # [T]

        for _ in range(maximum):
            logits = torch.stack(list(e.forward(token, prefix) for e in model)).mean(
                dim=0
            )  # [V]

            logits = model.forward(token, prefix)  # [V]
            probs = torch.softmax(logits, dim=0)  # [V]
            next_token = torch.multinomial(probs, num_samples=1)  # [1]
            token = torch.cat([token, next_token], dim=0)  # [T+1]

            if next_token.item() == 0:
                break

        return token  # [T]
