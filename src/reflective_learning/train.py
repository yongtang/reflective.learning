from typing import Callable, Optional

import torch
from tqdm import tqdm

from reflective_learning.model import ReflectiveCore


def train(
    model: ReflectiveCore,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    weight: torch.Tensor,
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
        weight: Weights of the desired states.
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
        desc="Train",
        dynamic_ncols=True,
        bar_format=bar_format,
        unit="sample",
    ) as progress:
        for batch in loader:
            model.train()

            # Move batch to device
            mask = batch["mask"].to(device)  # [B, L, L]
            embed = batch["embed"].to(device)  # [B, L, d_model]
            token_label = batch["token"].to(device)  # [B] — one token per example
            state_label = batch["state"].to(device)  # [B]

            if count + embed.size(0) > total:
                chunk = total - count

                mask = mask[:chunk]
                embed = embed[:chunk]
                token_label = token_label[:chunk]
                state_label = state_label[:chunk]

            # Forward pass (model returns logit at final position)
            logit = model.call(mask=mask, embed=embed)  # [B, V, S]
            loss = model.loss(logit, token_label, state_label, weight)
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
