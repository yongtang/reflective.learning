import collections
import contextlib
import functools
import json
import os
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from reflective_learning.encoder import ContextEncoder
from reflective_learning.model import ReflectiveCore
from reflective_learning.train import dpo, train


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


def load(file, *selection):
    model = torch.load(file, map_location="cpu", weights_only=False)
    info = model["info"]
    return (
        ([info] + [f_model(info, model[choice]) for choice in selection])
        if selection
        else info
    )


def save(file, max, vocab, state, param, meta):

    info = {
        "max": max,
        "vocab": vocab,
        "state": state,
        "layer": {
            "d_model": (param if param is not None else {}).get("d_model", 768),
            "nhead": (param if param is not None else {}).get("nhead", 12),
            "dim_feedforward": (param if param is not None else {}).get(
                "dim_feedforward", 3072
            ),
            "dropout": (param if param is not None else {}).get("dropout", 0.1),
        },
        "decoder": {
            "num_layers": (param if param is not None else {}).get("num_layers", 12),
        },
        "context": {
            "text": {
                "model": "openai-community/gpt2",
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            },
            "image": {
                "model": "google/siglip2-so400m-patch16-naflex",
                "revision": "cc24074f717b612951c2dead130904ab9b65a81e",
            },
            "transformers": "5.1.0",
        },
        "meta": ({} if meta is None else meta),
    }

    models = {choice: f_model(info, None) for choice in info["state"]}

    torch.save(
        {
            "info": info,
            **{choice: model.state_dict() for choice, model in models.items()},
        },
        file,
    )
    print(
        "Save model: {}".format(
            ", ".join(
                "{} ({:,} parameters)".format(
                    choice, sum(p.numel() for p in model.parameters())
                )
                for choice, model in models.items()
            )
        )
    )
