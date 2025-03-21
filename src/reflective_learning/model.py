import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        dim_ff=3072,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.state_size = state_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.state_embedding = nn.Embedding(state_size, d_model)
        self.pos_embedding = nn.Embedding(
            max_seq_len, d_model
        )  # Parameterized max sequence length

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, vocab_size * state_size)

        self.d_model = d_model

    def forward(self, token_ids, state_ids, mask=None):
        batch_size, seq_len = token_ids.shape

        token_embed = self.token_embedding(token_ids)
        state_embed = self.state_embedding(state_ids)
        pos_embed = self.pos_embedding(
            torch.arange(seq_len, device=token_ids.device).expand(batch_size, -1)
        )

        x = token_embed + state_embed + pos_embed

        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=token_ids.device), diagonal=1
            )
            mask = mask.masked_fill(mask == 1, float("-inf"))

        x = self.decoder(x, x, tgt_mask=mask)  # Decoder-only structure

        logits = self.output_proj(
            x
        )  # Shape: (batch_size, seq_len, vocab_size * state_size)
        logits = logits.view(batch_size, seq_len, self.vocab_size, self.state_size)

        return logits  # Raw logits used for training

    def compute_loss(self, logits, token_targets, state_targets):
        """
        Computes cross-entropy loss on joint (token, state) prediction.

        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size, state_size)
            token_targets: Tensor of shape (batch_size, seq_len)
            state_targets: Tensor of shape (batch_size, seq_len)

        Returns:
            loss: scalar tensor
        """
        batch_size, seq_len = token_targets.shape

        # Flatten inputs
        logits = logits.reshape(
            batch_size * seq_len, -1
        )  # (B * L, vocab_size * state_size)
        joint_targets = (
            token_targets * self.state_size + state_targets
        )  # (B, L) → joint index
        joint_targets = joint_targets.view(-1)  # (B * L)

        return F.cross_entropy(logits, joint_targets)
