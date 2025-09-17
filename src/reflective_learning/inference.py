import torch

from reflective_learning.model import autocast


def sequence(
    model,
    reduce,
    prefix: torch.Tensor,  # [C, D]
    maximum: int,
    device: torch.device,
    generator: torch.Generator | None = None,  # optional reproducibility
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and sampling.

    Args:
        model: Iterable container of trained sub-models (one per label), each exposing
               call(token, prefix) -> logits[V].
        reduce: The function to summerize the logits from different models.
        prefix (Tensor): [C, D] prefix embedding.
        maximum (int): Maximum number of tokens to generate.
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    # materialize in case it's a generator; keep name 'model'
    model = list(model)
    for e in model:
        e.to(device)
        e.eval()

    with torch.inference_mode():
        prefix = prefix.to(dtype=torch.float32, device=device)
        # preallocate to avoid O(n^2) concatenation; track logical length
        token = torch.empty(maximum, dtype=torch.long, device=device)  # [maximum]

        for length in range(maximum):
            data = token[:length]  # current sequence view

            # run model forwards under autocast; keep softmax math in fp32
            with autocast():
                logit = tuple(e.call(data, prefix) for e in model)  # M x [V]
            logit = reduce(logit).float()  # [V]

            # guard against NaN/Inf for this step only
            if not torch.isfinite(logit).all():
                probs = torch.full_like(logit, 1.0 / logit.numel())
            else:
                probs = torch.softmax(logit, dim=0)  # [V]

            prediction = torch.multinomial(
                probs, num_samples=1, generator=generator
            )  # [1]
            token[length] = prediction.item()

            if prediction.item() == 0:
                break

        return token[: length + 1]  # [T]


def explore(
    model,
    prefix: torch.Tensor,  # [C, D]
    maximum: int,
    device: torch.device,
    generator: torch.Generator | None = None,  # optional reproducibility
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
               call(token, prefix) -> logits[V].
        prefix (Tensor): [C, D] fixed prefix embedding for this sequence.
        maximum (int): Maximum number of tokens to generate (upper bound).
        device (device): Device for computation.

    Returns:
        Tensor: [T] of generated token indices (T <= maximum), possibly including STOP (0).
    """
    # materialize in case it's a generator; keep name 'model'
    model = list(model)
    # Put each sub-model into eval mode (we iterate the container directly).
    for e in model:
        e.to(device)
        e.eval()

    with torch.inference_mode():
        prefix = prefix.to(dtype=torch.float32, device=device)
        # preallocate to avoid O(n^2) concatenation; track logical length
        token = torch.empty(maximum, dtype=torch.long, device=device)  # [maximum]

        # Cumulative log-likelihood per model over tokens emitted so far.
        # Initialized to zeros -> uniform prior up to a constant (which cancels in softmax).
        S = torch.zeros(len(model), device=device)

        for length in range(maximum):
            data = token[:length]

            # Collect next-token logits from each model independently: shape [M, V].
            with autocast():
                logit = torch.stack(
                    [e.call(data, prefix) for e in model], dim=0
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

            prediction = torch.multinomial(
                probs, num_samples=1, generator=generator
            )  # [1]
            token[length] = prediction.item()

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

        return token[: length + 1]
