import torch
from tqdm import tqdm

from reflective_learning.model import autocast


def prob(model, mask, embed, token, index):
    """
    Compute sequence log-probabilities log P(y | x) in batch,
    leveraging collate() outputs.

    Args:
        mask:  [B, L]    from collate()
        embed: [B, L, D] from collate()
        token: [B, T]    from collate() (right-padded)
        index: [B]       from collate() (position of token[0] in input)

    Returns:
        logprob:  [B] sum log-probability of each sequence

    Equations (ASCII):
        For each sequence b and step t (target token y_{b,t} at position pos_{b,t}):
            logits_{b,t} = model(x_b)[pos_{b,t}, :]
            logZ_{b,t}   = logsumexp(logits_{b,t})
            logp_{b,t}   = logits_{b,t}[y_{b,t}] - logZ_{b,t]

        Let valid_{b,t} ∈ {0,1} indicate whether this step is inside the sequence.
        Sequence score (sum over valid steps):
            logP_b = sum_t valid_{b,t} * logp_{b,t}
    """
    # Run the transformer to get logits at every position
    logit = model(mask=mask, embed=embed)  # [B, L, V]
    B, L, V = logit.shape
    T = token.size(1)

    # Offsets for token positions: [0, 1, 2, ...]
    I = torch.arange(T, device=mask.device).view(1, T)  # [1, T]
    start = (index - 1).view(B, 1)  # position before the first token
    position = start + I  # [B, T] positions in logits

    # Mark valid positions: inside sequence length and not padding
    valid = (position >= 0) & ((position + 1) < L) & mask.gather(1, (position + 1))
    # position = position.clamp(0, L - 1)  # ensure positions are in range

    # Gather logits for each token prediction step
    step = logit.gather(1, position.unsqueeze(-1).expand(B, T, V))  # [B, T, V]

    # Normalize with log_softmax (fp32 for stability) vs. (log-softmax via logsumexp)
    with torch.autocast(device_type=embed.device.type, enabled=False):
        logp = torch.nn.functional.log_softmax(step, dim=-1)

    # Keep only the log-prob assigned to the reference token
    logp = logp.gather(-1, token.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # Zero-out invalid positions
    logp = logp.masked_fill(~valid, 0.0)

    # Return SUM of valid token log-probabilities per sequence
    return logp.sum(dim=1)  # [B]


def loss_fn(
    logit: torch.Tensor,  # [B, L, V] predicted logits for prefix + token
    token: torch.Tensor,  # [B, T] ground truth token indices
    index: torch.Tensor,  # [B] index where token sequence begins (first token position)
    mask: torch.Tensor,  # [B, L] boolean mask for valid positions in logits
) -> torch.Tensor:
    """
    Computes cross-entropy loss for next-token prediction.

    Args:
        logit: [B, L, V] predicted logits for prefix + token.
               Each position logit[b, t] predicts token[b, t + 1].
        token: [B, T] ground truth token indices (shifted right vs logits).
        index: [B] index where token sequence starts in logits (i.e. position of token[0]).
        mask:  [B, L] boolean mask for valid positions in logits.

    Returns:
        Scalar loss (cross entropy), averaged over valid tokens.
    """
    B, L, V = logit.shape
    T = token.size(1)

    # Offsets [0, 1, 2, ...]
    I = torch.arange(T, device=logit.device).view(1, T)  # [1,T]
    start = (index - 1).view(B, 1)  # [B,1]
    position = start + I  # [B,T] positions in logits

    # Number of valid positions per example
    N = mask.sum(dim=1)  # [B]
    count = N - index  # [B]
    # count excludes the final target position — there is no "next token" to predict
    M = int(count.max().item())  # process only needed columns

    I = I[:, :M]  # [1,M]
    position = position[:, :M].clamp_(min=0, max=L - 1)  # [B,M]
    valid = I < count.view(B, 1)  # [B,M]

    # Gather logits for each token prediction step
    step = logit.gather(1, position.unsqueeze(-1).expand(B, M, V))  # [B,M,V]

    # Cross-entropy for valid positions
    loss_flat = torch.nn.functional.cross_entropy(
        step.reshape(-1, V), token[:, :M].reshape(-1), reduction="none"
    ).view(
        B, M
    )  # [B,M]

    # Mask invalid positions
    loss_flat = loss_flat * valid.float()  # [B,M]

    # Average over valid tokens
    return loss_flat.sum() / valid.sum().clamp(min=1).float()


def train(
    model,
    loader,
    optimizer,
    total,
    callback,
    device,
    rank,
    desc,
):
    """
    Trains the model using a streaming loader

    Args:
        model: The ReflectiveCore model to train.
        loader: A torch DataLoader yielding training batches.
        optimizer: Optimizer for updating model parameters.
        total: Total number of training samples to process.
        callback: A function called periodically during training.
        device: Device for this rank (required). Model is assumed already on this device
                and may already be wrapped in DistributedDataParallel by the caller.
        rank: Global rank used only to disable tqdm on nonzero ranks.
        desc: Description for tqdm.
    """

    loss_width = 10
    sample_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{sample_width}d}}/{{total:{sample_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    count = 0

    with tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
        disable=(rank != 0),  # only rank 0 renders the bar; others are no-ops
    ) as progress:
        for batch in loader:
            model.train()

            # Move batch to device
            mask = batch["mask"].to(device, non_blocking=True)  # [B, L]
            embed = batch["embed"].to(device, non_blocking=True)  # [B, L, D]
            token = batch["token"].to(device, non_blocking=True)  # [B, T]
            state = batch["state"].to(device, non_blocking=True)  # [B]
            index = batch["index"].to(device, non_blocking=True)  # [B]

            batch_size = embed.size(0)
            if count + batch_size > total:
                batch_size = total - count
                mask = mask[:batch_size]
                embed = embed[:batch_size]
                token = token[:batch_size]
                state = state[:batch_size]
                index = index[:batch_size]

            # Forward pass (model returns [B, L, V])
            with autocast():
                logit = model(mask=mask, embed=embed)  # [B, L, V]
                loss = loss_fn(
                    logit=logit.float(),  # keep loss math in fp32 under autocast
                    token=token,
                    index=index,
                    mask=mask,
                )
            loss_value = loss.item()

            # Backpropagation (DDP grad sync happens automatically if model is wrapped)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Progress tracking (safe to call on all ranks; disabled bars ignore updates)
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.5e}  samples={count:{sample_width}d}"
            )

            if callback:
                # Let the callback decide how to handle DDP wrapping and rank in case.
                callback(model=model, progress=progress, device=device, rank=rank)

            if count >= total:
                break


def dpo(
    baseline,
    finetune,
    loader,
    optimizer,
    total,
    callback,
    device,
    rank: int = 0,
):

    loss_width = 10
    sample_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{sample_width}d}}/{{total:{sample_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    count = 0

    with tqdm(
        total=total,
        desc="Finetune",
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
        disable=(rank != 0),  # only rank 0 renders; others ignore updates
    ) as progress:
        for batch_pos, batch_neg in loader:
            baseline.eval()
            finetune.train()

            # Move batch to device
            mask_pos, mask_neg = (
                batch_pos["mask"].to(device, non_blocking=True),
                batch_neg["mask"].to(device, non_blocking=True),
            )  # [B, L]
            embed_pos, embed_neg = (
                batch_pos["embed"].to(device, non_blocking=True),
                batch_neg["embed"].to(device, non_blocking=True),
            )  # [B, L, D]
            token_pos, token_neg = (
                batch_pos["token"].to(device, non_blocking=True),
                batch_neg["token"].to(device, non_blocking=True),
            )  # [B, T]
            state_pos, state_neg = (
                batch_pos["state"].to(device, non_blocking=True),
                batch_neg["state"].to(device, non_blocking=True),
            )  # [B]
            index_pos, index_neg = (
                batch_pos["index"].to(device, non_blocking=True),
                batch_neg["index"].to(device, non_blocking=True),
            )  # [B]

            batch_size = embed_pos.size(0)
            if count + batch_size > total:
                batch_size = total - count
                mask_pos, mask_neg = mask_pos[:batch_size], mask_neg[:batch_size]
                embed_pos, embed_neg = embed_pos[:batch_size], embed_neg[:batch_size]
                token_pos, token_neg = token_pos[:batch_size], token_neg[:batch_size]
                state_pos, state_neg = state_pos[:batch_size], state_neg[:batch_size]
                index_pos, index_neg = index_pos[:batch_size], index_neg[:batch_size]

            # Forward pass
            with autocast():
                # Trainable model: grads ON
                logp_finetune_pos = prob(
                    finetune, mask_pos, embed_pos, token_pos, index_pos
                )
                logp_finetune_neg = prob(
                    finetune, mask_neg, embed_neg, token_neg, index_neg
                )

                # Reference model: grads OFF + eval()
                with torch.no_grad():
                    logp_baseline_pos = prob(
                        baseline, mask_pos, embed_pos, token_pos, index_pos
                    )
                    logp_baseline_neg = prob(
                        baseline, mask_neg, embed_neg, token_neg, index_neg
                    )

                # DPO margin (no temperature)
                s_pos = logp_finetune_pos - logp_baseline_pos
                s_neg = logp_finetune_neg - logp_baseline_neg
                margin = s_pos - s_neg
                loss = torch.nn.functional.softplus(-margin).mean()

            loss_value = loss.item()

            # Backpropagation (DDP grad sync happens automatically if finetune is wrapped)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Progress tracking
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.5e}  samples={count:{sample_width}d}"
            )

            if callback:
                # Let the callback decide DDP unwrapping and rank handling
                callback(model=finetune, progress=progress, device=device, rank=rank)

            if count >= total:
                break
