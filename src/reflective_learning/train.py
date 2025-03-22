import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.reflective_learning.dataset import ReflectiveDataset
from src.reflective_learning.model import ReflectiveCore


def train(
    json_paths,
    vocab_size,
    state_size,
    max_seq_len,
    epochs=1,
    batch_size=2,
    lr=1e-3,
    save_path=None,
    device=None,
    d_model=768,
    n_layers=12,
    n_heads=12,
    dim_ff=3072,
):
    """
    Trains a ReflectiveCore model on the provided dataset.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ReflectiveDataset(json_paths, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_ff=dim_ff,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            state_ids = batch["state_ids"].to(device)
            prefix_embed = batch.get("prefix_embed", None)
            prefix_embed = batch.get("prefix_embed", None)
            if prefix_embed is not None and prefix_embed.numel() > 0:
                prefix_embed = prefix_embed.to(device)
            else:
                prefix_embed = torch.zeros(
                    (token_ids.size(0), 0, model.d_model), device=device
                )

            logits = model(token_ids, state_ids, prefix_embed=prefix_embed)

            # Predict next token/state → shift target
            token_target = token_ids[:, 1:]
            state_target = state_ids[:, 1:]
            logits = logits[:, :-1, :, :]  # align prediction

            loss = model.compute_loss(logits, token_target, state_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to: {save_path}")
