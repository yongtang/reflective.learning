def collate(batch, model):
    """
    Collate function for variable-length prefix support.

    Input format (per example):
        {
            "token": LongTensor [T],       # token sequence (no prefix)
            "state": LongTensor [1],       # scalar state per sequence
            "prefix": FloatTensor [C, d_model],  # prefix embedding
        }

    Output format (batched):
        {
            "mask":  FloatTensor [B, L, L],        # causal mask
            "embed": FloatTensor [B, L, d_model],  # model input (prefix + projected tokens)
            "token": LongTensor [B, T-1],          # token prediction target
            "state": LongTensor [B],               # per-sequence state
        }
    """
    device = next(model.parameters()).device
    d_model = model.d_model
    V, S = model.vocab_size, model.state_size

    masks, embeds = [], []
    token_targets, state_targets = [], []
    max_embed_len = 0  # max length after prefix + token projection
    max_token_len = 0  # max length of token prediction target (T-1)

    for entry in batch:
        token = entry["token"].to(device)  # [T]
        state = entry["state"].to(device)  # [1]
        prefix = entry["prefix"].to(device)  # [C, d_model]

        T = token.size(0)
        x = torch.zeros((T, V, S), device=device)
        x.scatter_(1, token.view(T, 1, 1), 1.0)
        x.scatter_(2, state.view(1, 1, 1).expand(T, 1, 1), 1.0)
        x = x.view(T, V * S)
        projected = model.input_linear(x)  # [T, d_model]

        embed = torch.cat([prefix, projected], dim=0)  # [L, d_model]
        embeds.append(embed)
        L = embed.size(0)
        max_embed_len = max(max_embed_len, L)

        causal_mask = torch.triu(torch.ones((L, L), device=device), diagonal=1).bool()
        mask = torch.zeros((L, L), device=device)
        mask.masked_fill_(causal_mask, float("-inf"))
        masks.append(mask)

        token_targets.append(token[1:])  # [T-1]
        state_targets.append(state.view(()))  # []

        max_token_len = max(max_token_len, T - 1)

    B = len(batch)
    padded_mask = torch.full(
        (B, max_embed_len, max_embed_len), float("-inf"), device=device
    )
    padded_embed = torch.zeros((B, max_embed_len, d_model), device=device)
    padded_tokens = torch.zeros((B, max_token_len), dtype=torch.long, device=device)
    padded_states = torch.stack(state_targets, dim=0)  # [B]

    for i in range(B):
        L = embeds[i].size(0)
        padded_embed[i, :L] = embeds[i]
        padded_mask[i, :L, :L] = masks[i]

        T1 = token_targets[i].size(0)
        padded_tokens[i, :T1] = token_targets[i]

    return {
        "mask": padded_mask,  # [B, L, L]
        "embed": padded_embed,  # [B, L, d_model]
        "token": padded_tokens,  # [B, T-1]
        "state": padded_states,  # [B]
    }


import base64
import json
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class ReflectiveDataset(Dataset):
    def __init__(
        self,
        json_paths: Union[str, List[str]],
        max_seq_len: int,
        d_model: int,
    ):
        """
        Loads a dataset of token/state sequences with real-valued context prefixes.

        Args:
            json_paths (str or list): Path(s) to JSON lines with "token", "state", and "prefix"
            max_seq_len (int): Max sequence length (for padding/truncation)
            d_model (int): Embedding dimension (used for prefix shape validation)
        """
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        self.data = []
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        for path in json_paths:
            with open(path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        entry = self.data[key]
        token_ids = entry["token"]
        state_id = entry["state"]

        # Pad or truncate token sequence
        assert self.max_seq_len is not None, "max_seq_len must be set"
        padded = (token_ids + [0] * self.max_seq_len)[: self.max_seq_len]
        token_ids = torch.tensor(padded, dtype=torch.long)

        # Repeat state ID across sequence
        state_ids = torch.full((self.max_seq_len,), state_id, dtype=torch.long)

        # Decode and reshape prefix (must be present)
        prefix_encoded = entry["prefix"]
        assert prefix_encoded.startswith("b64://")
        chunk = np.frombuffer(
            base64.b64decode(prefix_encoded.removeprefix("b64://")), dtype=np.float32
        )
        assert chunk.size % self.d_model == 0
        context_len = chunk.size // self.d_model
        prefix = torch.from_numpy(chunk.copy()).reshape(context_len, self.d_model)

        return {
            "token_ids": token_ids,
            "state_ids": state_ids,
            "prefix": prefix,
        }
