from typing import Callable, Optional

import torch
from tqdm import tqdm

from reflective_learning.model import ReflectiveCore


def train(
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
        desc="Learn",
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
    ) as progress:
        for batch in loader:
            model.train()

            # Move batch to device
            mask = batch["mask"].to(device)  # [B, L]
            embed = batch["embed"].to(device)  # [B, L, D]
            token = batch["token"].to(device)  # [B, T]
            state = batch["state"].to(device)  # [B]
            index = batch["index"].to(device)  # [B]

            batch_size = embed.size(0)
            if count + batch_size > total:
                batch_size = total - count
                mask = mask[:batch_size]
                embed = embed[:batch_size]
                token = token[:batch_size]
                state = state[:batch_size]
                index = index[:batch_size]

            # Forward pass (model returns [B, L, V])
            logit = model.call(mask=mask, embed=embed)  # [B, L, V]
            loss = model.loss(
                logit=logit,
                token=token,
                index=index,
                mask=mask,
            )
            loss_value = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress tracking
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.5e}  samples={count:{sample_width}d}"
            )

            if callback:
                callback(model=model, progress=progress, device=device)

            if count >= total:
                break


def dpo(
    baseline: ReflectiveCore,
    finetune: ReflectiveCore,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    total: int,
    callback: Callable[[ReflectiveCore, tqdm, torch.device], None],
    device: Optional[torch.device] = None,
):

    loss_width = 10
    sample_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{sample_width}d}}/{{total:{sample_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline.to(device)
    finetune.to(device)

    count = 0

    with tqdm(
        total=total,
        desc="Finetune",
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
    ) as progress:
        for batch_pos, batch_neg in loader:
            baseline.eval()
            finetune.train()

            # Move batch to device
            mask_pos, mask_neg = (
                batch_pos["mask"].to(device),
                batch_neg["mask"].to(device),
            )  # [B, L]
            embed_pos, embed_neg = (
                batch_pos["embed"].to(device),
                batch_neg["embed"].to(device),
            )  # [B, L, D]
            token_pos, token_neg = (
                batch_pos["token"].to(device),
                batch_neg["token"].to(device),
            )  # [B, T]
            state_pos, state_neg = (
                batch_pos["state"].to(device),
                batch_neg["state"].to(device),
            )  # [B]
            index_pos, index_neg = (
                batch_pos["index"].to(device),
                batch_neg["index"].to(device),
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
            logp_finetune_pos = finetune.prob(mask_pos, embed_pos, token_pos, index_pos)
            logp_finetune_neg = finetune.prob(mask_neg, embed_neg, token_neg, index_neg)
            with torch.no_grad():
                logp_baseline_pos = baseline.prob(
                    mask_pos, embed_pos, token_pos, index_pos
                )
                logp_baseline_neg = baseline.prob(
                    mask_neg, embed_neg, token_neg, index_neg
                )
            s_pos = logp_finetune_pos - logp_baseline_pos
            s_neg = logp_finetune_neg - logp_baseline_neg
            margin = s_pos - s_neg
            loss = torch.nn.functional.softplus(-margin).mean()

            loss_value = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress tracking
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix_str(
                f"loss={loss_value:{loss_width}.5e}  samples={count:{sample_width}d}"
            )

            if callback:
                callback(model=finetune, progress=progress, device=device)

            if count >= total:
                break
