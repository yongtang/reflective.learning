from typing import Callable, List, Optional

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
