import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveCore(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        state_size: int,
        max_seq_len: int,
        max_prefix_len: int,
        decoder: nn.TransformerDecoder,
    ):
        """
        ReflectiveCore Transformer model that predicts P(token | context, state).

        Args:
            vocab_size: Number of token classes.
            state_size: Number of mutually exclusive state classes.
            max_seq_len: Max token sequence length (for position embeddings).
            max_prefix_len: Max prefix length (prepended to each sequence).
            decoder: Standard nn.TransformerDecoder module.
        """
        super().__init__()

        d_model = decoder.layers[0].linear1.in_features
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.d_model = d_model

        # Project one-hot token × state to d_model
        self.input_linear = nn.Linear(vocab_size * state_size, d_model)

        # Output logits over joint token-state space
        self.output_linear = nn.Linear(d_model, vocab_size * state_size)

        # Positional embeddings for prefix + sequence
        self.pos_embedding = nn.Embedding(max_seq_len + max_prefix_len, d_model)

        self.decoder = decoder

    def forward(
        self,
        token: torch.Tensor,  # [B, T]
        state: torch.Tensor,  # [B]
        prefix: torch.Tensor,  # [B, C, d_model]
        mask: torch.Tensor = None,  # Optional [L, L] or [B, L, L] attention mask
    ) -> torch.Tensor:
        """
        Forward pass using token/state indices and prefix.

        Args:
            token: [B, T] LongTensor of token indices.
            state: [B] LongTensor, single state index per sequence.
            prefix: [B, C, d_model] prefix embeddings to prepend.
            mask: Optional causal attention mask.

        Returns:
            [B, T, vocab_size, state_size] output logits.
        """
        assert prefix is not None, "prefix is required"
        B, T = token.shape
        V, S = self.vocab_size, self.state_size

        # One-hot encode inputs
        token_onehot = F.one_hot(token, num_classes=V).float()  # [B, T, V]
        state_onehot = F.one_hot(state, num_classes=S).float()  # [B, S]
        state_onehot = state_onehot.unsqueeze(1)  # [B, 1, S]

        # Outer product to get [B, T, V, S]
        joint_input = torch.einsum("btv,bks->btvs", token_onehot, state_onehot)
        joint_flat = joint_input.view(B, T, V * S)  # [B, T, V*S]

        x = self.input_linear(joint_flat)  # [B, T, d_model]
        x = torch.cat([prefix, x], dim=1)  # [B, C+T, d_model]

        logits = self.call(x, mask=mask)  # [B, C+T, V, S]
        return logits[:, prefix.shape[1] :]  # Drop prefix outputs

    def call(self, embed: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for precomputed embeddings.

        Args:
            embed: [B, L, d_model] real-valued embeddings (prefix + tokens).
            mask: Optional [L, L] or [B, L, L] attention mask.

        Returns:
            [B, L, vocab_size, state_size] logits.
        """
        assert embed is not None, "embed (with prefix) is required"
        B, L, _ = embed.shape

        # Add positional embeddings
        pos_ids = (
            torch.arange(L, device=embed.device).unsqueeze(0).expand(B, L)
        )  # [B, L]
        x = embed + self.pos_embedding(pos_ids)  # [B, L, d_model]

        x = self.decoder(x, x, tgt_mask=mask)  # [B, L, d_model]
        logits = self.output_linear(x)  # [B, L, V*S]
        return logits.view(B, L, self.vocab_size, self.state_size)  # [B, L, V, S]

    def loss(
        self,
        logits: torch.Tensor,  # [B, T, V, S]
        token: torch.Tensor,  # [B, T]
        state: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss against (token, state) targets.

        Args:
            logits: [B, T, vocab_size, state_size] output from the model.
            token: [B, T] ground truth token indices.
            state: [B] ground truth single state per sequence.

        Returns:
            Scalar loss (cross entropy).
        """
        B, T, V, S = logits.shape
        assert state.shape == (B,), "Each sequence must have a single state"

        # Broadcast state across sequence
        state_expanded = state.unsqueeze(1).expand(-1, T)  # [B, T]
        target = state_expanded * V + token  # [B, T] as flat index

        logits_flat = logits.view(B * T, V * S)
        target_flat = target.view(-1)
        return F.cross_entropy(logits_flat, target_flat)
