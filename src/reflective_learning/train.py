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
    nhead=12,
    dim_feedforward=3072,
    dropout=0.1,
    num_layers=12,
):
    """
    Trains a ReflectiveCore model on the provided dataset.

    Args:
        json_paths: str or list of paths to JSON dataset files
        vocab_size: vocabulary size (V)
        state_size: number of possible states (S)
        max_seq_len: input sequence length
        epochs: number of training epochs
        batch_size: training batch size
        lr: learning rate
        save_path: optional path to save trained model
        device: torch device string (e.g., 'cuda', 'cpu')
        d_model: transformer embedding dimension
        nhead: number of attention heads
        dim_feedforward: dimension of FFN layers
        dropout: dropout rate
        num_layers: number of transformer decoder layers
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ReflectiveDataset(json_paths, max_seq_len=max_seq_len, d_model=d_model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            state_ids = batch["state_ids"].to(device)
            prefix = batch["prefix"].to(device)

            if prefix.ndim == 2:
                prefix = prefix.unsqueeze(0).expand(token_ids.size(0), -1, -1)

            logits = model(token_ids, state_ids, prefix=prefix)

            # Shift for autoregressive prediction
            token_target = token_ids[:, 1:]
            state_target = state_ids[:, 1:]
            logits = logits[:, :-1, :, :]

            loss = model.loss(logits, token_target, state_target)

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
