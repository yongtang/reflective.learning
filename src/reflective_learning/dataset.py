import torch
from torch.utils.data import Dataset
import json


class ReflectiveDataset(Dataset):
    def __init__(self, json_path):
        self.data = []
        with open(json_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.data.append((ex["token"], ex["state"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids, state_id = self.data[idx]
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        state_ids = torch.full_like(token_ids, fill_value=state_id)
        return token_ids, state_ids
