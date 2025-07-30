import os
from typing import Callable, Optional

import torch
from tqdm import tqdm

from reflective_learning.model import ReflectiveCore


def train(
    model: ReflectiveCore,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    total: int,
    save: str,
    save_interval: int,
    callback: Callable[[ReflectiveCore, int], None],
    callback_interval: int,
    device: Optional[torch.device] = None,
):
    """
    Trains the model using a streaming loader and saves periodic checkpoints.

    Args:
        model: The ReflectiveCore model to train.
        loader: A torch DataLoader yielding training batches.
        optimizer: Optimizer for updating model parameters.
        total: Total number of training samples to process.
        save: Directory where model checkpoints will be saved.
        save_interval: Save model every N samples.
        callback: A function called periodically during training (e.g., for inference).
        callback_interval: Interval (in samples) at which to invoke the callback.
        device: Optional device override (defaults to CUDA if available).
    """

    loss_width = 7
    sample_width = len(str(total))
    bar_format = (
        f"{{desc}}: {{percentage:3.0f}}%|{{bar}}| "
        f"{{n:{sample_width}d}}/{{total:{sample_width}d}} "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(save, exist_ok=True)
    saved = []
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
            token_target = batch["token"].to(device)  # [B] — one token per example
            state_target = batch["state"].to(device)  # [B]

            if count + embed.size(0) > total:
                chunk = total - count

                mask = mask[:chunk]
                embed = embed[:chunk]
                token_target = token_target[:chunk]
                state_target = state_target[:chunk]

            # Forward pass (model returns logits at final position)
            logits = model.call(mask=mask, embed=embed)  # [B, V, S]
            loss = model.loss(logits, token_target, state_target)
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
                f"loss={loss_value:{loss_width}.4f}  samples={count:{sample_width}d}"
            )

            # Save checkpoint
            if count % save_interval < batch_size:
                filename = os.path.join(save, f"model_{count}.pt")
                torch.save(model.state_dict(), filename)
                saved.append(filename)
                progress.write(f"[Checkpoint] Saved to {filename}")
                if len(saved) > 3:
                    oldest = saved.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)

            # Run callback
            if callback and count % callback_interval < batch_size:
                callback(model, count)

            if count >= total:
                break

    # Final save
    final_filename = os.path.join(save, "model.pt")
    torch.save(model.state_dict(), final_filename)
    tqdm.write(f"[Final Save] Saved to {final_filename}")
