import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.reflective_learning.dataset import ReflectiveDataset
from src.reflective_learning.model import ReflectiveCore


def collate_with_prefix(batch, model):
    """
    Collate function for variable-length prefix support.

    - Projects (token_id, state_id) pairs to embeddings
    - Concatenates prefix + projected token embeddings
    - Computes per-example attention masks
    - Pads [L, d_model] and [L, L] to max length across batch

    Returns:
        {
            "embed": FloatTensor [B, L, d_model],
            "mask": FloatTensor [B, L, L],
            "token_target": LongTensor [B, T-1],
            "state_target": LongTensor [B, T-1],
        }
    """
    B = len(batch)
    d_model = model.d_model
    V, S = model.vocab_size, model.state_size

    embeds = []  # list of [L_i, d_model]
    masks = []  # list of [L_i, L_i]
    token_targets = []  # list of [T-1]
    state_targets = []  # list of [T-1]
    max_len = 0

    for example in batch:
        token_ids = example["token_ids"]  # [T]
        state_ids = example["state_ids"]  # [T]
        prefix = example["prefix"]  # [C, d_model]

        T = token_ids.size(0)
        x = torch.zeros(T, V, S)  # [T, V, S]
        x.scatter_(
            1, token_ids.unsqueeze(-1).unsqueeze(-1), 1.0
        )  # one-hot in token dim
        x.scatter_(
            2, state_ids.unsqueeze(-1).unsqueeze(-2), 1.0
        )  # one-hot in state dim
        x = x.view(T, V * S)  # [T, V*S]

        projected = model.input_linear(x)  # [T, d_model]
        embed = torch.cat([prefix, projected], dim=0)  # [L, d_model] where L = C + T
        embeds.append(embed)

        L = embed.size(0)
        max_len = max(max_len, L)

        causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()  # [L, L]
        mask = torch.zeros(L, L).float()  # [L, L]
        mask.masked_fill_(causal_mask, float("-inf"))
        masks.append(mask)

        token_targets.append(token_ids[1:])  # [T-1]
        state_targets.append(state_ids[1:])  # [T-1]

    # Pad embeddings and masks to [B, L, d_model] and [B, L, L]
    padded_embed = torch.zeros(B, max_len, d_model)  # [B, L, d_model]
    padded_mask = torch.full((B, max_len, max_len), float("-inf"))  # [B, L, L]

    for i in range(B):
        L = embeds[i].size(0)
        padded_embed[i, :L] = embeds[i]  # [L, d_model]
        padded_mask[i, :L, :L] = masks[i]  # [L, L]

    # Pad targets to [B, T-1]
    max_tgt = max(t.size(0) for t in token_targets)
    padded_tokens = torch.zeros(B, max_tgt, dtype=torch.long)  # [B, T-1]
    padded_states = torch.zeros(B, max_tgt, dtype=torch.long)  # [B, T-1]

    for i in range(B):
        padded_tokens[i, : token_targets[i].size(0)] = token_targets[i]
        padded_states[i, : state_targets[i].size(0)] = state_targets[i]

    return {
        "embed": padded_embed,  # [B, L, d_model]
        "mask": padded_mask,  # [B, L, L]
        "token_target": padded_tokens,  # [B, T-1]
        "state_target": padded_states,  # [B, T-1]
    }


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
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ReflectiveDataset(json_paths, max_seq_len=max_seq_len, d_model=d_model)

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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_with_prefix(batch, model),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            embed = batch["embed"].to(device)  # [B, L, d_model]
            mask = batch["mask"].to(device)  # [B, L, L]
            token_target = batch["token_target"].to(device)  # [B, T-1]
            state_target = batch["state_target"].to(device)  # [B, T-1]

            logits = model.call(embed, mask=mask)  # [B, L, V, S]

            # Keep only the portion matching targets
            logits = logits[:, -token_target.size(1) - 1 : -1]  # [B, T-1, V, S]

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
