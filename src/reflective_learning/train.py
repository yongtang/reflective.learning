import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.reflective_learning.dataset import ReflectiveDataset
from src.reflective_learning.model import ReflectiveTransformer
import os


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
    Trains a ReflectiveTransformer model on the provided dataset.

    Args:
        json_paths (str or list): One or more paths to JSONL dataset files.
        vocab_size (int): Total number of token types.
        state_size (int): Total number of states.
        max_seq_len (int): Maximum sequence length.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        save_path (str, optional): If given, saves final model weights here.
        device (str, optional): 'cuda', 'cpu', or None (auto-detect).
        d_model (int): Transformer hidden dimension size.
        n_layers (int): Number of transformer decoder layers.
        n_heads (int): Number of attention heads.
        dim_ff (int): Feedforward network size.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ReflectiveDataset(json_paths, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ReflectiveTransformer(
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

        for token_ids, state_ids in dataloader:
            token_ids = token_ids.to(device)
            state_ids = state_ids.to(device)

            logits = model(token_ids, state_ids)

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
