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
