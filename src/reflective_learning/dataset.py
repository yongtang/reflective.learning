import base64
import json
import os
from typing import List, Union

import torch
from torch.utils.data import Dataset


class ReflectiveDataset(Dataset):
    def __init__(self, json_paths: Union[str, List[str]], max_seq_len=None):
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        self.data = []
        self.max_seq_len = max_seq_len

        for path in json_paths:
            with open(path, "r") as f:
                for line in f:
                    ex = json.loads(line)
                    self.data.append(ex)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        token_ids = ex["token"]
        state_id = ex["state"]

        if self.max_seq_len is not None:
            if len(token_ids) < self.max_seq_len:
                pad_length = self.max_seq_len - len(token_ids)
                token_ids = token_ids + [0] * pad_length  # pad with STOP token (0)
            else:
                token_ids = token_ids[: self.max_seq_len]

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        if self.max_seq_len is not None:
            state_ids = torch.full((self.max_seq_len,), state_id, dtype=torch.long)
        else:
            state_ids = torch.full((len(token_ids),), state_id, dtype=torch.long)

        # Decode b64://-encoded prefix if present
        prefix_embed = None
        if "prefix" in ex:
            prefix_b64 = ex["prefix"]
            if prefix_b64.startswith("b64://"):
                raw = base64.b64decode(prefix_b64[6:])
                prefix_embed = torch.frombuffer(raw, dtype=torch.float32)
            else:
                raise ValueError("Prefix field must start with 'b64://'")

        if prefix_embed is None:
            # Use consistent dummy tensor that will not break batching
            prefix_embed = torch.zeros((0,), dtype=torch.float32)

        return {
            "token_ids": token_ids,
            "state_ids": state_ids,
            "prefix_embed": prefix_embed,
        }
