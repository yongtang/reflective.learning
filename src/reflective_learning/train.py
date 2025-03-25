import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    device = next(model.parameters()).device
    d_model = model.d_model
    V, S = model.vocab_size, model.state_size

    embeds, masks = [], []
    token_targets, state_targets = [], []
    max_len = 0

    for example in batch:
        token_ids = example["token_ids"].to(device)  # [T]
        state_ids = example["state_ids"].to(device)  # [T]
        prefix = example["prefix"].to(device)  # [C, d_model]

        T = token_ids.size(0)
        x = torch.zeros(T, V, S, device=device)  # [T, V, S]
        x.scatter_(1, token_ids.view(T, 1, 1), 1.0)  # one-hot over tokens
        x.scatter_(2, state_ids.view(T, 1, 1), 1.0)  # one-hot over states
        x = x.view(T, V * S)  # [T, V*S]

        projected = model.input_linear(x)  # [T, d_model]
        embed = torch.cat([prefix, projected], dim=0)  # [L, d_model]
        embeds.append(embed)

        L = embed.size(0)
        max_len = max(max_len, L)

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        mask = torch.zeros(L, L, device=device)
        mask.masked_fill_(causal_mask, float("-inf"))
        masks.append(mask)

        token_targets.append(token_ids[1:])  # [T-1]
        state_targets.append(state_ids[1:])  # [T-1]

    B = len(batch)
    padded_embed = torch.zeros(B, max_len, d_model, device=device)
    padded_mask = torch.full((B, max_len, max_len), float("-inf"), device=device)

    for i in range(B):
        L = embeds[i].size(0)
        padded_embed[i, :L] = embeds[i]
        padded_mask[i, :L, :L] = masks[i]

    max_tgt = max(t.size(0) for t in token_targets)
    padded_tokens = torch.zeros(B, max_tgt, dtype=torch.long, device=device)
    padded_states = torch.zeros(B, max_tgt, dtype=torch.long, device=device)

    for i in range(B):
        padded_tokens[i, : token_targets[i].size(0)] = token_targets[i]
        padded_states[i, : state_targets[i].size(0)] = state_targets[i]

    return {
        "embed": padded_embed,
        "mask": padded_mask,
        "token_target": padded_tokens,
        "state_target": padded_states,
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

    print("🚀 Starting training...")
    print(f"📁 Input files: {json_paths}")
    print(
        f"📐 Vocab size: {vocab_size}, State size: {state_size}, Max seq len: {max_seq_len}"
    )
    print(
        f"🧠 Model: d_model={d_model}, layers={num_layers}, heads={nhead}, ff={dim_feedforward}"
    )
    print(f"⚙️ Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"💻 Device: {device}")
    if save_path:
        print(f"💾 Will save model to: {save_path}")
    print()

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
        num_batches = len(dataloader)

        print(f"\n🌀 Epoch {epoch + 1}/{epochs}")

        with tqdm(dataloader, desc="Training", leave=True, ncols=100) as pbar:
            for step, batch in enumerate(pbar):
                embed = batch["embed"]
                mask = batch["mask"]
                token_target = batch["token_target"]
                state_target = batch["state_target"]

                logits = model.call(embed, mask=mask)
                logits = logits[:, -token_target.size(1) - 1 : -1]

                loss = model.loss(logits, token_target, state_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

        print(
            f"📉 Epoch {epoch + 1} complete. Avg loss: {total_loss / num_batches:.4f}"
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to: {save_path}")
