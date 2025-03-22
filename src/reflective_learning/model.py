import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveCore(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_size,
        d_model=768,
        nhead=12,
        dim_feedforward=3072,
        dropout=0.1,
        num_layers=12,
        max_seq_len=1024,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.state_size = state_size
        self.d_model = d_model

        self.input_linear = nn.Linear(vocab_size * state_size, d_model)
        self.output_linear = nn.Linear(d_model, vocab_size * state_size)

        self.pos_embedding = nn.Embedding(max_seq_len + 512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, token_ids, state_ids, prefix=None, mask=None):
        """
        Args:
            token_ids: [B, T] LongTensor of token indices
            state_ids: [B, T] LongTensor of state indices
            prefix: Optional [B, C, d_model] float tensor of context prefix embeddings
            mask: Optional [T+C, T+C] causal mask

        Returns:
            logits: [B, T, V, S] output logits over (token, state) pairs
        """
        B, T = token_ids.shape
        V, S = self.vocab_size, self.state_size

        # Build one-hot input distribution over (token, state) pairs
        x = torch.zeros(B, T, V, S, device=token_ids.device)
        x.scatter_(2, token_ids.unsqueeze(-1).unsqueeze(-1), 1.0)
        x.scatter_(3, state_ids.unsqueeze(-1).unsqueeze(-2), 1.0)

        # Flatten (V * S) → project into d_model
        x = x.view(B, T, V * S)
        x = self.input_linear(x)  # [B, T, d_model]

        # Prepend prefix if available
        if prefix is not None:
            C = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)  # [B, C+T, d_model]
        else:
            C = 0

        total_len = C + T
        pos = torch.arange(total_len, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(pos)

        if mask is None:
            mask = torch.triu(
                torch.ones(total_len, total_len, device=x.device), diagonal=1
            )
            mask = mask.masked_fill(mask == 1, float("-inf"))

        x = self.decoder(x, x, tgt_mask=mask)  # [B, C+T, d_model]
        x = x[:, C:]  # remove prefix portion

        logits = self.output_linear(x)  # [B, T, V*S]
        logits = logits.view(B, T, V, S)
        return logits

    def loss(self, logits, token_ids, state_ids):
        """
        Computes training loss using one-hot supervision.

        Args:
            logits: Tensor [B, T, V, S] – model outputs
            token_ids: Tensor [B, T] – ground truth token indices
            state_ids: Tensor [B, T] – ground truth state indices

        Returns:
            Scalar loss
        """
        B, T, V, S = logits.shape
        logits_flat = logits.reshape(B * T, V * S)

        # Linearize (token, state) index
        linear_target = state_ids * V + token_ids  # [B, T]
        loss = F.cross_entropy(logits_flat, linear_target.view(-1))
        return loss
