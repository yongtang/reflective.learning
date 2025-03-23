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

        # Compute expected sequence length
        seq_len = self.max_seq_len or len(token_ids)

        # Pad or truncate to desired length
        padded = (token_ids + [0] * seq_len)[:seq_len]
        token_ids = torch.tensor(padded, dtype=torch.long)

        # Broadcast state across sequence
        state_ids = torch.full((seq_len,), entry["state"], dtype=torch.long)

        # Decode prefix if present (variable length)
        prefix = torch.zeros((0, self.d_model), dtype=torch.float32)
        if "prefix" in entry:
            assert entry["prefix"].startswith("b64://")
            chunk = np.frombuffer(
                base64.b64decode(entry["prefix"].removeprefix("b64://")),
                dtype=np.float32,
            )
            assert chunk.size % self.d_model == 0
            context_len = chunk.size // self.d_model
            prefix = torch.from_numpy(chunk.copy()).reshape(context_len, self.d_model)

        return {
            "token_ids": token_ids,
            "state_ids": state_ids,
            "prefix": prefix,
        }
