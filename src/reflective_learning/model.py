import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveCore(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        state_size: int,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        num_layers: int = 12,
        max_seq_len: int = 1024,
    ):
        """
        Initializes the ReflectiveCore Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary (number of tokens).
            state_size (int): Number of mutually exclusive state classes.
            d_model (int): Transformer embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            num_layers (int): Number of transformer decoder layers.
            max_seq_len (int): Maximum sequence length for positional embeddings.
        """
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

    def forward(
        self,
        token_ids: torch.Tensor,
        state_ids: torch.Tensor,
        prefix: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Standard forward interface that takes token and state IDs and optional prefix embeddings.

        Args:
            token_ids (Tensor): [B, T] LongTensor of token indices.
            state_ids (Tensor): [B, T] LongTensor of state indices.
            prefix (Tensor, optional): [B, C, d_model] context prefix embeddings.
            mask (Tensor, optional): [L, L] or [B, L, L] causal attention mask.

        Returns:
            Tensor: [B, T, vocab_size, state_size] output logits.
        """
        B, T = token_ids.shape
        V, S = self.vocab_size, self.state_size

        x = torch.zeros(B, T, V, S, device=token_ids.device)
        x.scatter_(2, token_ids.unsqueeze(-1).unsqueeze(-1), 1.0)
        x.scatter_(3, state_ids.unsqueeze(-1).unsqueeze(-2), 1.0)

        x = x.view(B, T, V * S)
        x = self.input_linear(x)  # [B, T, d_model]

        if prefix is not None:
            x = torch.cat([prefix, x], dim=1)  # [B, C+T, d_model]

        return self.call(x, mask=mask)

    def call(self, embed: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for pre-embedded inputs.

        Args:
            embed (Tensor): [B, L, d_model] input embeddings (prefix + token sequence).
            mask (Tensor, optional): [B, L, L] or [L, L] attention mask. Defaults to None.

        Returns:
            Tensor: [B, L, vocab_size, state_size] output logits.
        """
        B, L, _ = embed.shape
        pos_ids = (
            torch.arange(L, device=embed.device).unsqueeze(0).expand(B, -1)
        )  # [B, L]
        x = embed + self.pos_embedding(pos_ids)  # [B, L, d_model]

        if mask is not None and mask.ndim == 3:  # [B, L, L] -> [B * nhead, L, L]
            mask = mask.repeat_interleave(
                self.decoder.layers[0].self_attn.num_heads, dim=0
            )

        x = self.decoder(x, x, tgt_mask=mask)  # [B, L, d_model]
        logits = self.output_linear(x)  # [B, L, V * S]
        return logits.view(B, L, self.vocab_size, self.state_size)

    def loss(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes training loss using one-hot supervision.

        Args:
            logits (Tensor): [B, T, vocab_size, state_size] model outputs.
            token_ids (Tensor): [B, T] ground truth token indices.
            state_ids (Tensor): [B, T] ground truth state indices.

        Returns:
            Tensor: Scalar loss value (cross entropy).
        """
        B, T, V, S = logits.shape
        logits_flat = logits.reshape(B * T, V * S)
        linear_target = state_ids * V + token_ids
        return F.cross_entropy(logits_flat, linear_target.view(-1))
