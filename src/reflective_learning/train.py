import os
from typing import Callable, Optional

import torch
from tqdm import tqdm

from reflective_learning.model import ReflectiveCore


def train(
    model: ReflectiveCore,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    total: int,
    save_data: str,
    save_interval: int,
    callback_func: Callable[[ReflectiveCore, int], None],
    callback_interval: int,
    device: Optional[torch.device] = None,
):
    """
    Trains the model using a streaming dataloader and saves periodic checkpoints.

    Args:
        model: The ReflectiveCore model to train.
        dataloader: A torch DataLoader yielding training batches.
        optimizer: Optimizer for updating model parameters.
        total: Total number of training samples to process.
        save_data: Directory where model checkpoints will be saved.
        save_interval: Save model every N samples.
        callback_func: A function called periodically during training (e.g., for inference).
        callback_interval: Interval (in samples) at which to invoke the callback.
        device: Optional device override (defaults to CUDA if available).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(save_data, exist_ok=True)
    saved = []
    count = 0

    with tqdm(total=total, desc="Training", leave=True, ncols=100) as progress:
        for batch in dataloader:
            model.train()

            # Move batch to device
            mask = batch["mask"].to(device)  # [B, L, L]
            embed = batch["embed"].to(device)  # [B, L, d_model]
            token_target = batch["token"].to(device)  # [B] — one token per example
            state_target = batch["state"].to(device)  # [B]

            # Forward pass (model returns logits at final position)
            logits = model.call(embed, mask=mask)  # [B, V, S]
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
            progress.set_postfix(loss=f"{loss_value:.4f}", samples=count)

            # Save checkpoint
            if count % save_interval < batch_size:
                filename = os.path.join(save_data, f"model_{count}.pt")
                torch.save(model.state_dict(), filename)
                saved.append(filename)
                progress.write(f"[Checkpoint] Saved to {filename}")
                if len(saved) > 3:
                    oldest = saved.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)

            # Run callback
            if callback_func and count % callback_interval < batch_size:
                callback_func(model, count)

            if count >= total:
                break

    # Final save
    final_filename = os.path.join(save_data, "model.pt")
    torch.save(model.state_dict(), final_filename)
    tqdm.write(f"[Final Save] Saved to {final_filename}")
