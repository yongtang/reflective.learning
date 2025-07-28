import torch
import tempfile
import shutil
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
            "state": torch.randint(0, self.state_size, (1,)),  # [1]
            "prefix": torch.randn(3, self.d_model),  # [C, d_model]
        }


def test_train_sanity():
    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=1
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
    try:
        train(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            total=20,
            save_data=tmpdir,
            save_interval=10,
            callback_func=None,
            callback_interval=20,
            device=torch.device("cpu"),
        )
        assert os.path.exists(os.path.join(tmpdir, "model.pt"))
    finally:
        shutil.rmtree(tmpdir)


def test_checkpoint_rotation():
    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=1
    )
    model = ReflectiveCore(
        vocab_size=10,
        state_size=2,
        max_seq_len=16,
        max_prefix_len=8,
        decoder=decoder,
    )

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                "token": torch.randint(1, 10, (5,)),
                "state": torch.randint(0, 2, (1,)),
                "prefix": torch.randn(3, 16),
            }

    loader = torch.utils.data.DataLoader(
        DummyDataset(), batch_size=2, collate_fn=model.collate
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tmpdir = tempfile.mkdtemp()
    try:
        train(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            total=40,
            save_data=tmpdir,
            save_interval=8,  # Should save at 8, 16, 24, 32
            callback_func=None,
            callback_interval=999,
            device=torch.device("cpu"),
        )
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.startswith("model_")]
        assert len(checkpoint_files) <= 3, f"Too many checkpoints: {checkpoint_files}"
    finally:
        shutil.rmtree(tmpdir)


def test_callback_invoked():
    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=1
    )
    model = ReflectiveCore(
        vocab_size=10,
        state_size=2,
        max_seq_len=16,
        max_prefix_len=8,
        decoder=decoder,
    )

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 12

        def __getitem__(self, idx):
            return {
                "token": torch.randint(1, 10, (5,)),
                "state": torch.randint(0, 2, (1,)),
                "prefix": torch.randn(3, 16),
            }

    loader = torch.utils.data.DataLoader(
        DummyDataset(), batch_size=2, collate_fn=model.collate
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    calls = []

    def callback_fn(model, count):
        calls.append(count)

    tmpdir = tempfile.mkdtemp()
    try:
        train(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            total=10,
            save_data=tmpdir,
            save_interval=100,
            callback_func=callback_fn,
            callback_interval=4,  # Should trigger at 4 and 8
            device=torch.device("cpu"),
        )
        assert calls == [4, 8], f"Expected callback at [4, 8], got {calls}"
    finally:
        shutil.rmtree(tmpdir)
