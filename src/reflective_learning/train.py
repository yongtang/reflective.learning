from typing import Callable, List, Optional

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
                f"loss={loss_value:{loss_width}.5e}  samples={count:{sample_width}d}"
            )

            if callback:
                callback(model=model, progress=progress, device=device)

            if count >= total:
                break


def dpo(baseline, finetune, mask, embed, token, state, index, device):
    """
    Sequence-level loss (length-normalized) with no favoritism.
    For each labeled state s in the batch, we compare finetune[s] against baseline[s]
    on the SAME sequences belonging to that state, then average over the batch.

    Notation (ASCII):
        For each sequence b with label s:
            logp_fine[b] = log p_theta_s(y_b | x_b)   # length-normalized seq log-prob
            logp_base[b] = log p_phi_s(y_b | x_b)
            margin[b]    = logp_fine[b] - logp_base[b]

        Stable logistic loss (tau = 1):
            loss = mean_b softplus( -margin[b] )
                 = mean_b log( 1 + exp( -margin[b] ) )

        Intuition:
            Encourages finetune[s] to assign higher sequence probability than
            its own frozen baseline on the same data.

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
    # Move inputs to the correct device (unchanged)
    mask = mask.to(device)
    embed = embed.to(device)
    token = token.to(device)
    state = state.to(device)
    index = index.to(device)

    B = embed.size(0)
    M = len(finetune)  # number of models/states

    # Per-sequence log-prob tensors (batch order)
    llr_finetune = torch.empty(B, device=device)
    llr_baseline = torch.empty(B, device=device)

    # Precompute indices for each state (unchanged batching semantics)
    state_indices = [(state == s).nonzero(as_tuple=True)[0] for s in range(M)]

    # Loop per state; run models only for that state's subset (unchanged)
    for s, idx in enumerate(state_indices):
        if idx.numel() == 0:
            continue  # no sequences for this state in this batch

        mask_s = mask[idx]
        embed_s = embed[idx]
        token_s = token[idx]
        index_s = index[idx]

        # Each model's prob() returns length-normalized sequence log-prob
        logp_finetune = finetune[s].prob(mask_s, embed_s, token_s, index_s)

        # Freeze baseline path for stability (no gradients, smaller graph)
        with torch.no_grad():
            logp_baseline = baseline[s].prob(mask_s, embed_s, token_s, index_s)

        # Scatter results back into batch order
        llr_finetune[idx] = logp_finetune
        llr_baseline[idx] = logp_baseline

    # Margin: how much finetune is better than baseline
    margin = llr_finetune - llr_baseline

    # Numerically stable logistic loss (prefers margin to be large and positive)
    loss = torch.nn.functional.softplus(-margin).mean()
    return loss


def discover(
    baseline: List[ReflectiveCore],
    finetune: List[ReflectiveCore],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    total: int,
    epoch: int,
    callback: Callable[[List[ReflectiveCore], tqdm, torch.device], None],
    device: Optional[torch.device] = None,
):
    """
    Trains the discover model using a streaming loader.

    Objective recap (ASCII):
        For each labeled sequence b with state s:
            margin[b] = log p_theta_s(y_b|x_b) - log p_phi_s(y_b|x_b)
            loss      = mean_b softplus( -margin[b] )

        => Pushes each finetune[s] to beat its OWN baseline on its OWN data,
           with the batching/labeling you already have.

    Numerical stability we actually use here:
        - Baseline prob is computed under no_grad() inside dpo() (frozen).
        - Non-finite guard: skip updates if loss is NaN/Inf.
        - Gradient clipping:
              Let g be the concatenated gradient and τ the threshold (e.g., 1.0).
              g_clipped = g * min(1, τ / ||g||_2)
              Ensures ||g_clipped||_2 ≤ τ, avoiding rare exploding updates.
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
        model.train()

    count = 0

    with tqdm(
        total=total,
        desc=f"Discover {epoch}",  # f-string fix
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
    ) as progress:
        for batch in loader:
            # Move batch to device (unchanged)
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
                state=state_label,
                index=index,
                device=device,
            )

            # Non-finite guard (prevents bad optimizer steps)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss_value = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (see equation in docstring)
            torch.nn.utils.clip_grad_norm_(
                (p for m in finetune for p in m.parameters() if p.requires_grad),
                max_norm=1.0,
            )

            optimizer.step()

            # Progress tracking
            batch_size = embed.size(0)
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.2e}  samples={count:{sample_width}d}"
            )

            if callback:
                callback(finetune, progress, device)  # pass the trainable models

            if count >= total:
                break
