import torch

from reflective_learning.model import autocast


@torch.inference_mode()
def sequence_step(
    model,
    reduce,
    prefix,
    token,
    length,
    device,
    generator,
):
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

    prediction = torch.multinomial(probs, num_samples=1, generator=generator)  # [1]

    return prediction


@torch.inference_mode()
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
            prediction = sequence_step(
                model, reduce, prefix, token, length, device, generator
            )
            token[length] = prediction.item()

            if prediction.item() == 0:
                break

        return token[: length + 1]  # [T]


@torch.inference_mode()
def sequence_batched(
    model,  # iterable of sub-models
    reduce,  # elementwise reducer: tuple(T_k) -> T_mixed  (works for [V] and [N, V])
    prefix: list[torch.Tensor],  # list length N, each [C_i, D]
    maximum: int,
    device: torch.device,
    generator: torch.Generator | None = None,  # optional reproducibility
):
    """
    Generate multiple token sequences in parallel using a user-provided `reduce` over sub-model logits.

    This is the batched analogue of `sequence(...)` and is intentionally kept symmetric with
    `explore_batched(...)` in structure, naming, and comments.

    Core behavior (matches non-batched `sequence(...)` semantics, but vectorized):
      - At each decoding step T, every sub-model is evaluated on the SAME preallocated per-model
        buffers (built once at T==0 via `collate()` with T==0 support). We take the logits at the
        "last visible" position of each row:
            pos = (index + length - 1).clamp_min(0)
        where `index[i] == C_i` (the first token slot after the prefix for row i) and `length[i]`
        is how many tokens row i has generated so far (0 at the beginning).
      - We then call the user-supplied `reduce` on the tuple of per-model logits. Crucially, each
        element we pass to `reduce` is shape **[N, V]**, not [V]; if `reduce` is written using
        elementwise tensor ops (e.g., `lambda L: L[0] - L[1]`, or `stack(...).mean(0)`), it
        naturally supports both the non-batched case ([V]) and the batched case ([N, V]) with no
        code changes. The result `mixed` is [N, V].
      - We softmax per-row to obtain [N, V] probabilities, then sample one token id per row.
      - STOP handling (id 0): once a row emits STOP at any step, all subsequent steps *force* that
        row to predict STOP deterministically by turning its probability row into a delta at 0:
            prob[stopped] = 0
            prob[stopped, 0] = 1
        This ensures the batched outputs are identical to running the non-batched `sequence(...)`
        per prefix, and prevents post-STOP drift. We do not skip computation for stopped rows; we
        simply constrain their probabilities.
      - Each sampled token is also projected to an embedding with `input_linear` and written into
        the per-model `embed` buffers at the just-filled absolute `position = index + T`, and the
        corresponding `mask` bit is set True so the token is visible next step.
      - Early termination: if *all* rows have STOPped at least once, we exit the loop.

    Efficiency / allocation:
      - `collate()` is called **exactly once per sub-model** at T==0 with empty tokens (T==0), which
        pads and returns prefix-only embeddings: `embed: [N, Cpad, D]`, `mask: [N, Cpad]`, and
        `index: [N]` where `index[i] == C_i`. We then **extend** those tensors on the RIGHT with
        zeros to allow writing up to `maximum` new tokens in-place: shapes become
            embed: [N, Cpad + maximum, D], mask: [N, Cpad + maximum].
        No further collate calls or buffer reallocations occur during decoding.

    Safety:
      - If any row in a step produces non-finite mixed logits, we replace that row’s probability
        with uniform for that step only so sampling remains safe.

    Args:
        model:   Iterable container of trained sub-models (one per label), each exposing
                 call(token, prefix) -> logits[V] and forward(mask, embed) -> [B, T, V].
        reduce:  Elementwise reducer over a tuple of per-model logits. Must work when each
                 element is [V] (non-batched) or [N, V] (batched), e.g.:
                     lambda L: L[0] - L[1]
                     lambda L: torch.stack(L, 0).mean(0)
                     lambda L: (torch.stack(L, 0) * weight[:, None]).sum(0)
        prefix:  Python list of N FloatTensor prefixes; item i is [C_i, D] for sequence i.
        maximum: Maximum number of tokens to generate for each sequence.
        device:  Computation device.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        final:   List of N LongTensors. Each entry i is [T_i] (T_i <= maximum) and includes STOP (0)
                 if STOP was sampled; sequences are truncated at the FIRST STOP.
    """
    # Materialize `model` in case it's a generator; keep naming consistent with non-batched code.
    model = list(model)
    # Put each sub-model into eval mode and move it to the device (mirrors non-batched setup).
    for e in model:
        e.to(device)
        e.eval()

    N = len(prefix)
    assert N > 0, "No prefix provided"
    M = len(model)
    assert M > 0, "No sub-model provided"

    # Per-sequence generation state:
    #   token[i, t] = sampled token id at time t for row i.
    #   length[i]   = number of tokens generated so far for row i.
    #   stopped[i]  = True once STOP has been sampled for row i (we then force STOP in later steps).
    token = torch.empty((N, maximum), dtype=torch.long, device=device)
    length = torch.zeros(N, dtype=torch.long, device=device)
    stopped = torch.zeros(N, dtype=torch.bool, device=device)

    # --- Step 0: call collate() ONCE per model with T==0 (inference-only), then extend buffers ---
    collected = []
    for k, e in enumerate(model):
        # Build prefix-only items (T==0) and tie `state` to the model index `k`.
        item = [
            {
                "prefix": prefix[i].to(device=device, dtype=torch.float32),  # [C_i, D]
                "token": torch.empty(
                    0, dtype=torch.long, device=device
                ),  # T==0 (empty token seq)
                "state": torch.tensor(
                    k, dtype=torch.long, device=device
                ),  # model index
            }
            for i in range(N)
        ]

        batch = e.collate(item)
        embed = batch["embed"]  # [N, Cpad, D]  (prefix-only visible at this point)
        mask = batch["mask"]  # [N, Cpad]     (True over prefix positions)
        index = batch[
            "index"
        ]  # [N]           (start of token area per row; equals C_i)

        # Extend to the RIGHT so we can write up to `maximum` tokens in-place.
        zeros_embed = torch.zeros(
            (N, maximum, embed.size(2)), dtype=embed.dtype, device=device
        )
        zeros_mask = torch.zeros((N, maximum), dtype=torch.bool, device=device)
        embed = torch.cat([embed, zeros_embed], dim=1)  # [N, Cpad + maximum, D]
        mask = torch.cat([mask, zeros_mask], dim=1)  # [N, Cpad + maximum]

        collected.append(
            {
                "model": e,
                "embed": embed,
                "mask": mask,
                "index": index.clone(),
                "V": e.vocab_size,
            }
        )

    rows = torch.arange(N, device=device)

    # --- Decoding loop over T = 0..maximum-1 (lockstep across rows) ---
    for T in range(maximum):
        # 1) Forward each model; extract last-position logits per row.
        #    `collected_logit[k]` has shape [N, V] for model k.
        collected_logit = []
        for k in range(M):
            e = collected[k]["model"]
            embed = collected[k]["embed"]
            mask = collected[k]["mask"]
            index = collected[k]["index"]

            # Last visible position per row i:
            #   if length[i] == 0: final prefix slot is `index[i] - 1`
            #   else:               last token slot is `index[i] + length[i] - 1`
            pos = (index + length - 1).clamp_min(0)  # [N]

            with autocast():
                all_logit = e.forward(mask=mask, embed=embed)  # [N, L, V]
            last_logit = all_logit[rows, pos, :]  # [N, V]
            collected_logit.append(last_logit)

        # 2) Vectorized reduce: pass the tuple of [N, V] logits directly to `reduce`.
        #    This requires `reduce` to be elementwise over the last dimension so it
        #    works for both [V] (non-batched) and [N, V] (batched) transparently.
        with autocast():
            mixed = reduce(tuple(collected_logit)).float()  # [N, V]

        # 3) Probabilities (row-wise softmax) with non-finite safety per row.
        prob = torch.softmax(mixed, dim=-1)  # [N, V]
        finite = torch.isfinite(mixed).all(dim=1)  # [N]
        if not finite.all():
            # Replace any problematic rows with uniform distribution for this step only.
            V = mixed.shape[-1]
            prob = prob.clone()
            prob[~finite] = torch.full((V,), 1.0 / V, device=device, dtype=prob.dtype)

        # 4) STOP forcing rule:
        #    Rows that have already produced STOP are forced to predict STOP again by
        #    turning their probability rows into a delta at index 0.
        if stopped.any():
            V = prob.size(1)
            prob = prob.clone()
            prob[stopped] = 0
            prob[stopped, 0] = 1

        # 5) Sample one token per row and record it at the current logical length.
        predict = torch.multinomial(prob, 1, generator=generator).squeeze(1)  # [N]
        token[rows, length] = predict
        length = length + 1

        # Mark rows that just emitted STOP; if all STOPped, terminate.
        stopped = stopped | (predict == 0)
        if stopped.all():
            break

        # 6) Write the new token embeddings into each model’s buffers at absolute `position = index + T`,
        #    and flip mask True so it becomes visible at the next step.
        for k in range(M):
            e = collected[k]["model"]
            embed = collected[k]["embed"]
            mask = collected[k]["mask"]
            index = collected[k]["index"]
            V = collected[k]["V"]

            one_hot = torch.nn.functional.one_hot(predict, V).to(embed.dtype)  # [N, V]
            projection = e.input_linear(one_hot)  # [N, D]
            position = index + T  # just-written slot
            embed[rows, position, :] = projection
            mask[rows, position] = True

    # --- Finalization: truncate each sequence at its FIRST STOP (including STOP) ---
    final = []
    for i in range(N):
        T = int(length[i].item())
        seq = token[i, :T]
        position = (seq == 0).nonzero(as_tuple=False)
        if position.numel() > 0:
            seq = seq[: position[0, 0] + 1]
        final.append(seq.clone())

    return final


@torch.inference_mode()
def explore_step(model, prefix, token, length, S, device, generator):
    data = token[:length]

    # Collect next-token logits from each model independently: shape [M, V].
    with autocast():
        logit = torch.stack([e.call(data, prefix) for e in model], dim=0)  # [M, V]

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

    prediction = torch.multinomial(probs, num_samples=1, generator=generator)  # [1]

    # Update S_k with the log-prob each model assigned to the single sampled token.
    # log_softmax(logit)[k, prediction] selects the same token column for all models.
    logp = torch.log_softmax(logit, dim=-1)  # [M, V]
    logp_k = logp[:, prediction].squeeze(1)  # [M]
    # Keep S finite even if a model hard-masks a token (e.g., -inf logit at sampled index).
    # In normal cases this clamp never activates (threshold is far below typical values).
    logp_k = torch.clamp(logp_k, min=-30.0)
    S = S + logp_k  # [M]

    return prediction, S


@torch.inference_mode()
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
            prediction, S = explore_step(
                model, prefix, token, length, S, device, generator
            )

            token[length] = prediction.item()

            # Early stop on STOP token (assumed id 0).
            if prediction.item() == 0:
                break

        return token[: length + 1]


@torch.inference_mode()
def explore_batched(
    model,  # iterable of sub-models
    prefix: list[torch.Tensor],  # list length N, each [C_i, D]
    maximum: int,
    device: torch.device,
    generator: torch.Generator | None = None,
):
    """
    Generate multiple token sequences in parallel using Bayes-balanced mixing.

    This batched variant is semantically equivalent to running `explore(...)`
    (the single-sequence version below) independently on each prefix, but does
    so in parallel for efficiency.

    Core idea (Bayes-balanced mixing, parameter-free):
      - Maintain, per sequence i and per model k, a cumulative log-likelihood S[i, k]
        over the tokens already emitted by that sequence: S[i, k] = sum_t log p_k(x_t).
      - Before predicting the next token for every sequence, compute per-model weights:
            weight[i] = softmax( -S[i] )          # shape [M], one set per sequence
        This downweights models that have explained the sequence well so far, and
        upweights models that have explained it less well (keeps label usage balanced).
      - For each sequence, mix per-model next-token logits with those weights and
        sample from a *single* softmax:
            probs[i] = softmax( sum_k weight[i,k] * logit[i,k] )
      - Update S[i, k] by adding the log-probability model k assigned to the actually
        sampled token for sequence i.
      - Stop early for any sequence that samples STOP (token id 0). In this batched
        version, once a sequence hits STOP, we *force* it to keep predicting STOP on
        later steps so outputs remain identical to independent decoding and to avoid
        post-STOP drift.

    Batched implementation details:
      - Variable-length prefixes: we call `collate()` once per sub-model at T==0 with
        empty token sequences (T==0) so that `collate()` pads only the prefix section
        (C_i) across the N sequences. This relies on the minimal change you made to
        support T==0 in collate (inference-only).
      - After that single call per model, we extend the returned [N, Cpad, D] embeddings
        and [N, Cpad] masks on the right with zeros to make room for up to `maximum`
        tokens per sequence. No further `collate()` calls or tensor reallocations are
        needed during decoding.
      - At each step T:
          1) For each sub-model e_k, run `e_k.forward(mask, embed)` on the *same*
             preallocated buffers to get [N, L, V] logits, and then index the “last
             visible” positions per sequence:
                 pos = (index + length - 1).clamp_min(0)
             where `index[i] == C_i` is the first token slot after prefix i and
             `length[i]` is how many tokens we’ve already generated for row i.
          2) Stack the per-model last-position logits to [N, M, V], Bayes-mix across
             models per row, softmax → [N, V] per-row distributions.
          3) **STOP forcing:** rows that have already produced STOP have their
             distribution forced to a delta at 0:
                 prob[stopped] = 0
                 prob[stopped, 0] = 1
             so they deterministically keep sampling STOP. We also freeze their S
             updates, keeping S identical to the single-sequence behavior.
          4) Sample one token per sequence, write it to the per-row token buffer
             and also project/write its embedding into every model’s preallocated
             `embed` at the just-written position (and flip the mask bit True).
          5) If *all* rows have produced STOP at least once, terminate.
      - Finally, each returned sequence is truncated at its first STOP (including STOP).

    Safety notes:
      - If any row’s mixed logits are non-finite in a step, we replace that row’s
        distribution with uniform for that step only, to avoid contaminating
        sampling or NaNs in `multinomial`.

    Args:
        model:   Iterable container of trained sub-models (one per label), each exposing
                 call(token, prefix) -> logits[V] and forward(mask, embed) -> [B, T, V].
        prefix:  Python list of N FloatTensor prefixes; item i is [C_i, D] for sequence i.
        maximum: Maximum number of tokens to generate (upper bound for each sequence).
        device:  Computation device.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        final:   List of N LongTensors. Each entry is [T_i] with tokens up to and
                 including the first STOP (0), or up to `maximum` if STOP never sampled.
    """
    # Materialize in case `model` is a generator; keep the name `model` (like single version).
    model = list(model)
    # Put each sub-model into eval mode and move to the target device.
    for e in model:
        e.to(device)
        e.eval()

    N = len(prefix)
    assert N > 0, "No prefix provided"
    M = len(model)
    assert M > 0, "No sub-model provided"

    # Per-sequence buffers:
    #   token[i, t] holds the t-th sampled token id for sequence i.
    #   length[i]   holds how many tokens have been generated for sequence i so far.
    #   S[i, k]     holds the cumulative log-likelihood assigned by model k to
    #               the tokens sampled for sequence i.
    #   stopped[i]  flags that sequence i has emitted STOP at least once.
    token = torch.empty((N, maximum), dtype=torch.long, device=device)
    length = torch.zeros(N, dtype=torch.long, device=device)
    S = torch.zeros((N, M), device=device)
    stopped = torch.zeros(N, dtype=torch.bool, device=device)

    # --- Step 0 (collate-once): build per-model prefix batches with T==0, then extend ---
    collected = []
    for k, e in enumerate(model):
        # Prepare N items with only prefix (token length == 0); state is tied to model index.
        item = [
            {
                "prefix": prefix[i].to(device=device, dtype=torch.float32),  # [C_i, D]
                "token": torch.empty(0, dtype=torch.long, device=device),  # [0] (T==0)
                "state": torch.tensor(
                    k, dtype=torch.long, device=device
                ),  # model index
            }
            for i in range(N)
        ]

        # collate pads the prefix section across sequences; it also returns:
        #   embed: [N, Cpad, D], mask: [N, Cpad], index: [N] where index[i] == C_i
        batch = e.collate(item)
        embed = batch["embed"]  # [N, Cpad, D]  (prefix-only at this point)
        mask = batch["mask"]  # [N, Cpad]     (True over actual prefix tokens)
        index = batch["index"]  # [N]           (start position for token embeddings)

        # Extend once with zeros on the right for up to `maximum` tokens.
        zeros_embed = torch.zeros(
            (N, maximum, embed.size(2)), dtype=embed.dtype, device=device
        )
        zeros_mask = torch.zeros((N, maximum), dtype=torch.bool, device=device)
        embed = torch.cat([embed, zeros_embed], dim=1)  # [N, Cpad+maximum, D]
        mask = torch.cat([mask, zeros_mask], dim=1)  # [N, Cpad+maximum]

        collected.append(
            {
                "model": e,
                "embed": embed,
                "mask": mask,
                "index": index.clone(),
                "V": e.vocab_size,
            }
        )

    rows = torch.arange(N, device=device)

    # --- Decoding loop (lockstep over T = 0..maximum-1) ---
    for T in range(maximum):
        # 1) Per-model forward pass to get last-position logits for every sequence.
        collected_logit = []
        for k in range(M):
            e = collected[k]["model"]
            embed = collected[k]["embed"]
            mask = collected[k]["mask"]
            index = collected[k]["index"]

            # For each row i:
            #  - if length[i] == 0, the last visible position is the final prefix slot: index[i] - 1
            #  - else it's the final generated token slot so far: index[i] + (length[i] - 1)
            pos = (index + length - 1).clamp_min(0)  # [N]

            with autocast():
                logit_all = e.forward(mask=mask, embed=embed)  # [N, L, V]
            logit_last = logit_all[rows, pos, :]  # [N, V] logits for this step
            collected_logit.append(logit_last)

        # 2) Bayes-balanced mixing across models, per sequence.
        #    Shape notes: logit is [N, M, V]; weight is [N, M]; mixed is [N, V].
        logit = torch.stack(collected_logit, dim=1)  # [N, M, V]
        weight = torch.softmax(-S, dim=1)  # [N, M]
        mixed = (weight.unsqueeze(-1) * logit).sum(dim=1)  # [N, V]

        # 3) Convert to probabilities (row-wise softmax), with non-finite safety.
        prob = torch.softmax(mixed, dim=-1)  # [N, V]
        finite = torch.isfinite(mixed).all(dim=1)  # [N]
        if not finite.all():
            # Replace any bad rows with uniform probabilities for this step only.
            V = mixed.shape[-1]
            prob = prob.clone()
            prob[~finite] = torch.full((V,), 1.0 / V, device=device, dtype=prob.dtype)

        # --- STOP forcing rule (post-STOP determinism) ---
        # Rows that have already STOPped are *forced* to predict STOP again by turning
        # their distribution into a delta at 0. This keeps behavior identical to
        # independent decoding and prevents post-STOP randomness:
        #   prob[stopped] = 0
        #   prob[stopped, 0] = 1
        if stopped.any():
            V = prob.size(1)
            prob = prob.clone()
            prob[stopped] = 0
            prob[stopped, 0] = 1

        # 4) Sample one token per sequence and append to the token buffer.
        predict = torch.multinomial(prob, 1, generator=generator).squeeze(1)  # [N]
        token[rows, length] = predict
        length = length + 1

        # 5) Update S with the per-model log-probability of the sampled token.
        #    Freeze S for rows that were already STOPped before this step.
        logp = torch.log_softmax(logit, dim=-1)  # [N, M, V]
        logp_index = predict.view(N, 1, 1).expand(N, M, 1)  # [N, M, 1]
        logp_k = logp.gather(2, logp_index).squeeze(2)  # [N, M]
        if stopped.any():
            logp_k[stopped] = 0
        S = S + torch.clamp(logp_k, min=-30.0)

        # 6) Mark rows that just emitted STOP; if all have STOPped, terminate.
        stopped = stopped | (predict == 0)
        if stopped.all():
            break

        # 7) Write the new token embeddings into each model's preallocated buffers:
        #    - Project sampled token IDs via that model's input_linear.
        #    - Write the projection at absolute slot (index + T).
        #    - Flip mask True at that slot so it becomes visible next step.
        for k in range(M):
            e = collected[k]["model"]
            embed = collected[k]["embed"]
            mask = collected[k]["mask"]
            index = collected[k]["index"]
            V = collected[k]["V"]

            one_hot = torch.nn.functional.one_hot(predict, V).to(embed.dtype)  # [N, V]
            projection = e.input_linear(one_hot)  # [N, D]
            position = index + T  # absolute token slot written this step
            embed[rows, position, :] = projection
            mask[rows, position] = True

    # --- Finalize: return each sequence truncated at its first STOP (including STOP) ---
    final = []
    for i in range(N):
        T = int(length[i].item())
        seq = token[i, :T]
        position = (seq == 0).nonzero(as_tuple=False)
        if position.numel() > 0:
            seq = seq[: position[0, 0] + 1]
        final.append(seq.clone())

    return final
