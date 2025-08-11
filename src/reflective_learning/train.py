from typing import Callable, Optional, List

import torch
from tqdm import tqdm

from reflective_learning.model import ReflectiveCore


def pretrain(
    model: ReflectiveCore,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    total: int,
    callback: Callable[[ReflectiveCore, tqdm, torch.device], None],
    device: Optional[torch.device] = None,
):
    """
    Trains the model using a streaming loader

    Args:
        model: The ReflectiveCore model to train.
        loader: A torch DataLoader yielding training batches.
        optimizer: Optimizer for updating model parameters.
        total: Total number of training samples to process.
        callback: A function called periodically during training.
        device: Optional device override (defaults to CUDA if available).
    """

    loss_width = 10
    sample_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{sample_width}d}}/{{total:{sample_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    count = 0

    with tqdm(
        total=total,
        desc="Pretrain",
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
    ) as progress:
        for batch in loader:
            model.train()

            # Move batch to device
            mask = batch["mask"].to(device)  # [B, L]
            embed = batch["embed"].to(device)  # [B, L, D]
            token_label = batch["token"].to(device)  # [B, T]
            state_label = batch["state"].to(device)  # [B]
            index = batch["index"].to(device)  # [B]

            if count + embed.size(0) > total:
                chunk = total - count
                mask = mask[:chunk]
                embed = embed[:chunk]
                token_label = token_label[:chunk]
                state_label = state_label[:chunk]
                index = index[:chunk]

            # Forward pass (model returns [B, L, V])
            logit = model.call(mask=mask, embed=embed)  # [B, L, V]
            loss = model.loss(
                logit=logit,
                token=token_label,
                index=index,
                mask=mask,
            )
            loss_value = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress tracking
            batch_size = embed.size(0)
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.2e}  samples={count:{sample_width}d}"
            )

            if callback:
                callback(model=model, progress=progress, device=device)

            if count >= total:
                break


def dpo(baseline, finetune, mask, embed, token, state, index, device):
    """
    Sequence-level loss (length-normalized) that always prefers the finetune model
    over the baseline model on the same sequence.

    Args:
        baseline: list of models, one per state (frozen reference models)
        finetune: list of models, one per state (trainable models)
        mask:     [B, L] bool attention mask
        embed:    [B, L, D] input embeddings
        token:    [B, T] target token indices
        state:    [B] int in 0..num_states-1 (model selection per sequence)
        index:    [B] start position of token[0] in logits
        device:   torch.device or str

    Returns:
        Scalar tensor loss

    Equation (ASCII form):
        margin[b] = logp_finetune[b] - logp_baseline[b]
        loss      = mean( log( 1 + exp( -margin[b] ) ) )

        This is equivalent to:
            loss = mean( log( 1 + exp( -(logp_finetune - logp_baseline) ) ) )
        which encourages logp_finetune > logp_baseline.

    Example:
        Suppose for one sequence:
            logp_finetune = -5.0   # finetune assigns prob = exp(-5)
            logp_baseline = -6.5   # baseline assigns prob = exp(-6.5)
        Then:
            margin = (-5.0) - (-6.5) = 1.5
            loss   = log( 1 + exp( -1.5 ) )
                   ≈ log( 1 + 0.2231 )
                   ≈ log( 1.2231 )
                   ≈ 0.201
        A larger margin (finetune better) → smaller loss.
        If margin were negative (baseline better), loss would be larger.
    """
    # Move inputs to the correct device
    mask = mask.to(device)
    embed = embed.to(device)
    token = token.to(device)
    state = state.to(device)
    index = index.to(device)

    B = embed.size(0)
    num_states = len(finetune)

    # Preallocate per-sequence log-prob tensors
    llr_finetune = torch.empty(B, device=device)
    llr_baseline = torch.empty(B, device=device)

    # Precompute indices for each state
    state_indices = [(state == s).nonzero(as_tuple=True)[0] for s in range(num_states)]

    # Loop per state, run models only for that state's subset
    for s, idx in enumerate(state_indices):
        if idx.numel() == 0:
            continue  # no sequences for this state in this batch

        mask_s = mask[idx]
        embed_s = embed[idx]
        token_s = token[idx]
        index_s = index[idx]

        # Each model's prob() returns length-normalized sequence log-prob
        logp_finetune = finetune[s].prob(mask_s, embed_s, token_s, index_s)
        logp_baseline = baseline[s].prob(mask_s, embed_s, token_s, index_s)

        # Scatter results back into batch order
        llr_finetune[idx] = logp_finetune
        llr_baseline[idx] = logp_baseline

    # Margin: how much finetune is better than baseline
    margin = llr_finetune - llr_baseline

    # Logistic loss (tau = 1): prefers margin to be large and positive
    loss = torch.log1p(torch.exp(-margin)).mean()
    return loss


def discover(
    baseline: List[ReflectiveCore],
    finetune: List[ReflectiveCore],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    total: int,
    epoch: int,
    device: Optional[torch.device] = None,
):
    """
    Trains the discover model using a streaming loader

    Args:
        baseline: The ReflectiveCore baseline models.
        finetune: The ReflectiveCore finetune models.
        loader: A torch DataLoader yielding training batches.
        optimizer: Optimizer for updating model parameters.
        total: Total number of training samples to process.
        epoch: Total number of training epoch rounds to process.
        device: Optional device override (defaults to CUDA if available).
    """

    loss_width = 10
    sample_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{sample_width}d}}/{{total:{sample_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in baseline:
        model.to(device)
        model.eval()
    for model in finetune:
        model.to(device)

    count = 0

    with tqdm(
        total=total,
        desc="Discover {epoch}",
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
    ) as progress:
        for batch in loader:
            # Move batch to device
            mask = batch["mask"].to(device)  # [B, L]
            embed = batch["embed"].to(device)  # [B, L, D]
            token_label = batch["token"].to(device)  # [B, T]
            state_label = batch["state"].to(device)  # [B]
            index = batch["index"].to(device)  # [B]

            if count + embed.size(0) > total:
                chunk = total - count
                mask = mask[:chunk]
                embed = embed[:chunk]
                token_label = token_label[:chunk]
                state_label = state_label[:chunk]
                index = index[:chunk]

            loss = dpo(
                baseline=baseline,
                finetune=finetune,
                mask=mask,
                embed=embed,
                token=token_label,
                state=state,
                index=index,
                device=device,
            )

            loss_value = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress tracking
            batch_size = embed.size(0)
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.2e}  samples={count:{sample_width}d}"
            )

            if callback:
                callback(model=model, progress=progress, device=device)

            if count >= total:
                break
