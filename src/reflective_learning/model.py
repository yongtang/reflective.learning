import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveCore(nn.Module):
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
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.state_embedding = nn.Embedding(state_size, d_model)

        # Allow extra positions to accommodate context prefix
        self.pos_embedding = nn.Embedding(max_seq_len + 512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, vocab_size * state_size)

    def forward(self, token_ids, state_ids, mask=None, prefix_embed=None):
        """
        Args:
            token_ids: Tensor of shape (B, T) – token IDs for the input sequence.
            state_ids: Tensor of shape (B, T) – state IDs corresponding to each token.
            prefix_embed: Optional tensor of shape (B, C, d_model), prepended to token embeddings before transformer input.
            mask: Optional tensor of shape (B, T), where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            logits: Tensor of shape (B, T, vocab_size, state_size)
        """

        B, T = token_ids.shape

        if prefix_embed is not None:
            C = prefix_embed.shape[1]
            pos = torch.arange(C + T, device=token_ids.device).expand(B, -1)
            pos_embed = self.pos_embedding(pos)

            token_embed = self.token_embedding(token_ids)
            state_embed = self.state_embedding(state_ids)
            x = torch.cat([prefix_embed, token_embed + state_embed], dim=1)
        else:
            C = 0
            pos = torch.arange(T, device=token_ids.device).expand(B, -1)
            pos_embed = self.pos_embedding(pos)

            token_embed = self.token_embedding(token_ids)
            state_embed = self.state_embedding(state_ids)
            x = token_embed + state_embed

        x = x + pos_embed

        total_len = C + T
        if mask is None:
            mask = torch.triu(
                torch.ones(total_len, total_len, device=token_ids.device), diagonal=1
            )
            mask = mask.masked_fill(mask == 1, float("-inf"))

        x = self.decoder(x, x, tgt_mask=mask)
        x = x[:, C:]  # remove context tokens before output projection

        logits = self.output_proj(x)
        logits = logits.view(B, T, self.vocab_size, self.state_size)
        return logits

    def compute_loss(self, logits, token_targets, state_targets):
        """
        Computes cross-entropy loss on joint (token, state) prediction.

        Args:
            logits: Tensor of shape (B, T, V, S)
            token_targets: Tensor of shape (B, T)
            state_targets: Tensor of shape (B, T)

        Returns:
            loss: scalar tensor
        """
        B, T = token_targets.shape
        logits = logits.reshape(B * T, -1)
        joint_targets = (token_targets * self.state_size + state_targets).view(-1)
        return F.cross_entropy(logits, joint_targets)
