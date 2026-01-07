import torch

from reflective_learning.model import ReflectiveCore


def f_model(info, state):
    vocab_indices = sorted(info["vocab"].values())
    assert vocab_indices == list(range(len(vocab_indices))), f"{info['vocab']}"
    vocab_size = len(vocab_indices)

    max_seq_len = info["max"]
    max_prefix_len = 512

    decoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            batch_first=True,
            **info["layer"],
        ),
        **info["decoder"],
    )

    model = ReflectiveCore(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
        decoder=decoder,
    )

    model.load_state_dict(state) if state else None

    return model


state_space = ["success", "failure"]


def load(file, *selection):
    model = torch.load(file, map_location="cpu", weights_only=False)
    info = model["info"]
    return [info] + [f_model(info, model[choice]) for choice in selection]


def save(file, max, vocab, meta):
    info = {
        "max": max,
        "vocab": vocab,
        "layer": {
            "d_model": 768,
            "nhead": 12,
            "dim_feedforward": 3072,
            "dropout": 0.1,
        },
        "decoder": {
            "num_layers": 12,
        },
        "context": {
            "text": {
                "model": "gpt2",
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            },
            "image": {
                "model": "google/vit-base-patch16-224",
                "revision": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
            },
            "transformers": "4.50.0",
        },
        "meta": meta,
    }

    torch.save(
        {
            "info": info,
            **{choice: f_model(info, None).state_dict() for choice in state_space},
        },
        file,
    )
