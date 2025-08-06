import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveCore(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        max_prefix_len: int,
        decoder: nn.TransformerEncoder,
    ):
        """
        ReflectiveCore Transformer model that predicts P(token | context).

        Args:
            vocab_size: Number of token classes.
            max_seq_len: Max token sequence length (for position embeddings).
            max_prefix_len: Max prefix length (prepended to each sequence).
            decoder: Standard nn.TransformerEncoder module.
        """
        super().__init__()

        d_model = decoder.layers[0].linear1.in_features
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Project one-hot token to d_model
        self.input_linear = nn.Linear(vocab_size, d_model)

        # Output logit over token vocabulary
        self.output_linear = nn.Linear(d_model, vocab_size)

        # Positional embeddings for prefix + sequence
        self.pos_embedding = nn.Embedding(max_seq_len + max_prefix_len, d_model)

        self.decoder = decoder

    def forward(
        self,
        token: torch.Tensor,  # [T]
        prefix: torch.Tensor,  # [C, D]
    ) -> torch.Tensor:
        """
        Forward pass using token indices and prefix.

        Args:
            token: [T] LongTensor of token indices.
            prefix: [C, D] prefix embeddings to prepend.

        Returns:
            [V] logits for the final token position.
        """
        assert prefix is not None, "prefix is required"
        T = token.shape[0]
        V = self.vocab_size

        # One-hot encode token
        x = F.one_hot(token, num_classes=V).float()  # [T, V]

        # Input projection
        x = self.input_linear(x)  # [T, D]

        # Prepend prefix
        x = torch.cat([prefix, x], dim=0)  # [C+T, D]

        # Add batch dimension
        x = x.unsqueeze(0)  # [1, C+T, D]

        # Call transformer
        logit = self.call(mask=None, embed=x)  # [1, C+T, V]

        # Return logits for final token position
        return logit[0, -1]  # [V]

    def call(self, mask: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for precomputed embeddings.
        Args:
            mask:  [B, T]
            embed: [B, T, D]
        Returns:
            [B, T, V] logits at each position
        """
        B, T, D = embed.shape

        # Default mask: all positions valid
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.bool, device=embed.device)

        # Positional embedding
        pos = torch.arange(T, device=embed.device).unsqueeze(0).expand(B, T)
        x = embed + self.pos_embedding(pos)  # [B, T, D]

        # Padding mask: [B, T], True = PAD — so invert `mask`
        src_key_padding_mask = ~mask

        # Causal mask: [T, T], True = masked
        mask = torch.triu(torch.ones(T, T, device=embed.device), diagonal=1).bool()

        # Transformer: decoder-only via TransformerEncoder + causal mask
        x = self.decoder(
            src=x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # shape [B, T, D]

        # Output projection to token logits
        logit = self.output_linear(x)  # [B, T, V]

        return logit

    def loss(
        self,
        logit: torch.Tensor,  # [B, T, V] predicted logits
        token: torch.Tensor,  # [B, T] ground truth token indices
        state: torch.Tensor,  # [B] ground truth state indices
        weight: torch.Tensor,  # [S] weight per state
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss for next-token prediction, with per-state weighting.

        Args:
            logit: [B, T, V] predicted logits at each position.
            token: [B, T] ground truth token indices.
            state: [B] ground truth state per example.
            weight: [S] weight per state (e.g. success=1.0, failure=0.1)

        Returns:
            Scalar loss (cross entropy).
        """
        B, T, V = logit.shape

        # Use all logits to predict all tokens (including token[0])
        pred = logit[:, -token.size(1) :, :].reshape(-1, V)  # [B*T, V]
        target = token.reshape(-1)  # [B*T]

        loss = F.cross_entropy(pred, target, reduction="none")  # [B*T]

        # Convert state → per-example weight → per-token weight
        weight = torch.clamp(weight, min=0.0)  # [S]
        weight = weight[state]  # [B]
        weight = weight.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]

        loss = loss * weight
        return loss.mean()

    def collate(self, batch):
        """
        Collate function for training the ReflectiveCore model using next-token prediction.

        Each example in the input batch consists of:
            - prefix: FloatTensor [C, D], real-valued context embedding
            - token: LongTensor [T], where T > 1
            - state: LongTensor [] — categorical state label

        Returns:
            A dict with:
                - "mask":  BoolTensor [B, L] — attention mask for valid input embeddings
                - "embed": FloatTensor [B, L, D] — input embeddings (prefix + tokens[:-1])
                - "token": LongTensor [B, T] — full token sequence
                - "state": LongTensor [B] — state per example
        """
        device = next(self.parameters()).device
        D = self.d_model
        V = self.vocab_size

        embed, token_list, mask, state_list = [], [], [], []

        for entry in batch:
            token = entry["token"].to(device)  # [T]
            prefix = entry["prefix"].to(device)  # [C, D]
            state = entry["state"].to(device)  # scalar

            T = token.size(0)
            assert T > 0, "Token sequence must have at least 1 token"

            input_token = token[:-1]  # [T - 1]

            x = F.one_hot(input_token, num_classes=V).float()  # [T - 1, V]
            x = self.input_linear(x)  # [T - 1, D]
            full = torch.cat([prefix, x], dim=0)  # [C + T - 1, D]

            embed.append(full)
            token_list.append(token)  # target tokens: predict token[t] from prefix + token[:t]
            mask.append(torch.ones(full.shape[0], dtype=torch.bool, device=device))
            state_list.append(state)

        embed = torch.nn.utils.rnn.pad_sequence(embed, batch_first=True)  # [B, L, D]
        token = torch.nn.utils.rnn.pad_sequence(
            token_list, batch_first=True, padding_value=0
        )  # [B, T]
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)  # [B, L]
        state = torch.stack(state_list)  # [B]

        return {
            "mask": mask,  # [B, L] — for prefix + input tokens
            "embed": embed,  # [B, L, D]
            "token": token,  # [B, T] — full token sequence, including token[0]
            "state": state,  # [B]
        }
