import os

import torch
from tqdm import tqdm

from reflective_learning.model import ReflectiveCore


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
    model: ReflectiveCore,
    dataloader: DataLoader,
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
    data_iter = iter(dataloader)

    with tqdm(total=total, desc="Training", leave=True, ncols=100) as progress:
        while count < total:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            model.train()
            embed = batch["embed"].to(device)
            mask = batch["mask"].to(device)
            token_target = batch["token_target"].to(device)
            state_target = batch["state_target"].to(device)

            logits = model.call(embed, mask=mask)
            logits = logits[:, -token_target.size(1) - 1 : -1]

            loss = model.loss(logits, token_target, state_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = embed.size(0)
            count += batch_size
            progress.update(batch_size)
            progress.set_postfix(loss=f"{loss.item():.4f}", samples=count)

            # Save model periodically
            if count % save_interval < batch_size:
                filename = os.path.join(save_data, f"model_{count}.pt")
                torch.save(model.state_dict(), filename)
                saved.append(filename)
                if len(saved) > 3:
                    oldest = saved.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)

            # Run callback periodically
            if callback_func and count % callback_interval < batch_size:
                callback_func(model, count)

    # Final model save
    final_filename = os.path.join(save_data, "model.pt")
    torch.save(model.state_dict(), final_filename)
