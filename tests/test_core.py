import torch

from reflective_learning.model import ReflectiveCore


def test_reflective_transformer_forward_and_loss():
    vocab_size = 100
    max_seq_len = 2
    max_prefix_len = 6
    batch_size = 2

    decoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        ),
        num_layers=2,
    )

    model = ReflectiveCore(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
        decoder=decoder,
    )

    token = torch.randint(0, vocab_size, (batch_size, max_seq_len))  # [B, T]

    d_model = decoder.layers[0].linear1.in_features
    prefix = torch.zeros(batch_size, max_prefix_len, d_model)  # [B, C, D]

    input = token[:, :-1]  # [B, T-1]
    label = token[:, 1:]  # [B, T-1]

    # One-hot → project → concat
    x = torch.nn.functional.one_hot(
        input, num_classes=vocab_size
    ).float()  # [B, T-1, V]
    x = model.input_linear(x)  # [B, T-1, D]
    embed = torch.cat([prefix, x], dim=1)  # [B, C+T-1, D]
    mask = torch.ones(embed.shape[:2], dtype=torch.bool)  # [B, C+T-1]

    logit = model.forward(mask=mask, embed=embed)  # [B, C+T-1, V]
    assert logit.shape == (batch_size, max_prefix_len + input.size(1), vocab_size)

    # Pad label to match logit (prepend dummy for prefix positions)
    pad = torch.zeros(batch_size, max_prefix_len, dtype=torch.long)
    full_label = torch.cat([pad, label], dim=1)  # [B, C+T-1]

    index = torch.full((batch_size,), max_prefix_len, dtype=torch.long)  # [B]
    loss = model.loss(logit, full_label, index, mask)
    assert loss.item() > 0


def test_reflective_transformer_forward_single():
    vocab_size = 100
    max_seq_len = 2
    max_prefix_len = 6

    decoder = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        ),
        num_layers=2,
    )

    model = ReflectiveCore(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
        decoder=decoder,
    )

    T = max_seq_len
    C = max_prefix_len
    D = decoder.layers[0].linear1.in_features

    token = torch.randint(0, vocab_size, (T,))  # [T]
    prefix = torch.randn(C, D)  # [C, D]

    logit = model.call(token=token, prefix=prefix)  # [V]
    assert logit.shape == (vocab_size,)  # Single-step prediction
