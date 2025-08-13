import torch


def sequence(
    model,
    prefix: torch.Tensor,  # [C, D]
    maximum: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and sampling.

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
            prediction = torch.multinomial(probs, num_samples=1)  # [1]
            token = torch.cat([token, prediction], dim=0)  # [T+1]

            if prediction.item() == 0:
                break

        return token  # [T]


def explore(
    model,
    prefix: torch.Tensor,  # [C, D]
    maximum: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and stochastic sampling.

    This function mixes next-token logits from an iterable of per-label models
    (for example, success, failure, etc.) in a way that counteracts early domination
    by any one model, aiming to keep label usage balanced over the course of a sequence.

    Core idea (Bayes-balanced mixing, parameter-free):
      - Maintain a per-model cumulative log-likelihood S_k over the tokens already emitted.
        S_k is the sum of log-probabilities that model k assigned to the actually sampled tokens.
      - Before predicting the next token, compute per-model weights as:
            weight = softmax(-S)
        This is proportional to the inverse of each model's current posterior belief given
        the generated tokens so far (labels already ahead receive less weight; those
        behind receive more).
      - Predict next token from a single softmax over the convex combination of per-model logits:
            probs = softmax( sum_k weight_k * logit_k )
      - Update S_k by adding the log-probability that each model k assigned to the sampled token.
      - Stop early if the sampled token equals 0 (assumed STOP token).

    Notes:
      - Sampling is stochastic via torch.multinomial, not greedy.
      - We only ever query each model individually for next-token logits; the balancing happens
        purely in how we combine those logits each step.
      - The returned sequence will include the STOP token (0) if it is generated before `maximum`.

    Args:
        model: Iterable container of trained sub-models (one per label), each exposing
               forward(token, prefix) -> logits[V].
        prefix (Tensor): [C, D] fixed prefix embedding for this sequence.
        maximum (int): Maximum number of tokens to generate (upper bound).
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices (T <= maximum), possibly including STOP (0).
    """
    # Put each sub-model into eval mode (we iterate the container directly).
    for e in model:
        e.eval()

    with torch.no_grad():
        prefix = prefix.to(dtype=torch.float32, device=device)
        token = torch.empty([0], dtype=torch.long, device=device)  # [T]

        # Cumulative log-likelihood per model over tokens emitted so far.
        # Initialized to zeros -> uniform prior up to a constant (which cancels in softmax).
        S = torch.zeros(len(model), device=device)

        for _ in range(maximum):
            # Collect next-token logits from each model independently: shape [M, V].
            logit = torch.stack(
                [e.forward(token, prefix) for e in model], dim=0
            )  # [M, V]

            # Bayes-balanced mixing weights:
            #   weight_k proportional to exp(-S_k)  =>  weight = softmax(-S)
            weight = torch.softmax(-S, dim=0)  # [M]

            # Next-token distribution from Bayes-balanced mixed logits:
            # probs = softmax( sum_k weight_k * logit_k )
            mixed = (weight[:, None] * logit).sum(dim=0)  # [V]
            # Safety: if any sub-model produced NaN/Inf logits this step, avoid propagating
            # to softmax/multinomial; fall back to uniform for this token only.
            if not torch.isfinite(mixed).all():
                probs = torch.full_like(mixed, 1.0 / mixed.numel())
            else:
                probs = torch.softmax(mixed, dim=0)  # [V]

            prediction = torch.multinomial(probs, num_samples=1)  # [1]
            token = torch.cat([token, prediction], dim=0)  # [T+1]

            # Update S_k with the log-prob each model assigned to the single sampled token.
            # log_softmax(logit)[k, prediction] selects the same token column for all models.
            logp = torch.log_softmax(logit, dim=-1)  # [M, V]
            logp_k = logp[:, prediction].squeeze(1)  # [M]
            # Keep S finite even if a model hard-masks a token (e.g., -inf logit at sampled index).
            # In normal cases this clamp never activates (threshold is far below typical values).
            logp_k = torch.clamp(logp_k, min=-30.0)
            S = S + logp_k  # [M]

            # Early stop on STOP token (assumed id 0).
            if prediction.item() == 0:
                break

        return token  # [T]
