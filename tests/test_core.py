import torch

from reflective_learning.model import ReflectiveCore


def test_reflective_transformer_forward_and_loss():
    vocab_size = 100
    state_size = 4
    max_seq_len = 2
    max_prefix_len = 6
    batch_size = 2

    decoder = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        ),
        num_layers=2,
    )

    model = ReflectiveCore(
        vocab_size=vocab_size,
        state_size=state_size,
        max_seq_len=max_seq_len,
        max_prefix_len=max_prefix_len,
        decoder=decoder,
    )

    token = torch.randint(0, vocab_size, (batch_size, max_seq_len))  # [B, T]
    state = torch.randint(0, state_size, (batch_size,))  # [B]

    d_model = decoder.layers[0].linear1.in_features
    prefix = torch.zeros(batch_size, max_prefix_len, d_model)  # [B, C, d_model]

    input = token[:, :-1]  # [B, T-1]
    label = token[:, -1]  # [B]

    logit = model(input, state, prefix=prefix)  # [B, 1, V, S]
    assert logit.shape == (batch_size, vocab_size, state_size)

    loss = model.loss(logit, label, state)  # [B] target
    assert loss.item() > 0
