import os
import shutil
import tempfile

import torch

from reflective_learning.model import ReflectiveCore
from reflective_learning.train import train


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=10, state_size=2, d_model=16):
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.d_model = d_model

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        T = 5
        return {
            "token": torch.randint(1, self.vocab_size, (T,)),  # [T]
            "state": torch.randint(0, self.state_size, []),  # []
            "prefix": torch.randn(3, self.d_model),  # [C, d_model]
        }


def test_train_sanity():
    decoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            batch_first=True,
        ),
        num_layers=1,
    )
    model = ReflectiveCore(
        vocab_size=10,
        state_size=2,
        max_seq_len=16,
        max_prefix_len=8,
        decoder=decoder,
    )

    dataset = DummyDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=model.collate
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tmpdir = tempfile.mkdtemp()

    def f_callback(model, progress, device):
        with open(os.path.join(tmpdir, "model.pt"), "w") as f:
            pass

    try:
        train(
            model=model,
            loader=loader,
            optimizer=optimizer,
            total=20,
            callback=f_callback,
            device=torch.device("cpu"),
        )
        assert os.path.exists(os.path.join(tmpdir, "model.pt"))
    finally:
        shutil.rmtree(tmpdir)
