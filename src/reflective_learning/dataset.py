import json
import os
from typing import List, Union

import torch
from torch.utils.data import Dataset


class ReflectiveDataset(Dataset):
    def __init__(self, json_paths: Union[str, List[str]], max_seq_len=None):
        """
        Args:
            json_paths (str or List[str]): Path(s) to JSONL file(s).
            max_seq_len (int, optional): If set, pad or truncate token sequences to this length.
        """
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        self.data = []
        self.max_seq_len = max_seq_len

        for path in json_paths:
            with open(path, "r") as f:
                for line in f:
                    ex = json.loads(line)
                    self.data.append((ex["token"], ex["state"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids, state_id = self.data[idx]

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

        return token_ids, state_ids
