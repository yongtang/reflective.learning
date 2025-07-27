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
        Initializes the ReflectiveCore Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary (number of token classes).
            state_size (int): Number of mutually exclusive state classes.
            max_seq_len (int): Maximum token sequence length.
            max_prefix_len (int): Maximum prefix (context) length.
            decoder (nn.TransformerDecoder): Transformer decoder module.
        """
        super().__init__()

        d_model = decoder.layers[0].linear1.in_features

        self.vocab_size = vocab_size
        self.state_size = state_size
        self.d_model = d_model

        self.input_linear = nn.Linear(vocab_size * state_size, d_model)
        self.output_linear = nn.Linear(d_model, vocab_size * state_size)
        self.pos_embedding = nn.Embedding(max_seq_len + max_prefix_len, d_model)

        self.decoder = decoder

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, T]
        state_ids: torch.Tensor,  # [B]
        prefix: torch.Tensor,  # [B, C, d_model]
        mask: torch.Tensor = None,  # Optional [L, L] or [B, L, L] boolean mask
    ) -> torch.Tensor:
        """
        Forward interface for token/state ID inputs.

        Args:
            token_ids (Tensor): [B, T] LongTensor of token indices.
            state_ids (Tensor): [B] LongTensor of fixed state index per sequence.
            prefix (Tensor): [B, C, d_model] context prefix embeddings.
            mask (Tensor, optional): [L, L] or [B, L, L] boolean causal attention mask.

        Returns:
            Tensor: [B, T, vocab_size, state_size] output logits (excluding prefix).
        """
        assert prefix is not None, "prefix is required"
        B, T = token_ids.shape
        V, S = self.vocab_size, self.state_size

        # One-hot encode tokens and states
        token_onehot = F.one_hot(token_ids, num_classes=V).float()  # [B, T, V]
        state_onehot = F.one_hot(state_ids, num_classes=S).float()  # [B, S]
        state_onehot = state_onehot.unsqueeze(1)  # [B, 1, S]

        # Outer product over tokens and state
        joint_input = torch.einsum("btv, bks -> btvs", token_onehot, state_onehot)
        x = joint_input.view(B, T, V * S)  # [B, T, V*S]
        x = self.input_linear(x)  # [B, T, d_model]

        # Concatenate prefix
        x = torch.cat([prefix, x], dim=1)  # [B, C+T, d_model]
        logits = self.call(x, mask=mask)  # [B, C+T, V, S]
        return logits[:, prefix.shape[1] :]  # Return token-only logits

    def call(self, embed: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for pre-embedded inputs.

        Args:
            embed (Tensor): [B, L, d_model] input embeddings (prefix + token sequence).
            mask (Tensor, optional): [L, L] or [B, L, L] boolean attention mask.

        Returns:
            Tensor: [B, L, vocab_size, state_size] output logits.
        """
        assert embed is not None, "prefix/embed is required"

        B, L, _ = embed.shape
        pos_ids = (
            torch.arange(L, device=embed.device).unsqueeze(0).expand(B, L)
        )  # [B, L]
        x = embed + self.pos_embedding(pos_ids)  # [B, L, d_model]

        # PyTorch >= 2.0 supports both 2D and 3D masks directly
        x = self.decoder(x, x, tgt_mask=mask)  # [B, L, d_model]
        logits = self.output_linear(x)  # [B, L, V * S]
        return logits.view(B, L, self.vocab_size, self.state_size)  # [B, L, V, S]

    def loss(
        self,
        logits: torch.Tensor,  # [B, T, V, S]
        token_ids: torch.Tensor,  # [B, T]
        state_ids: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Computes training loss using one-hot supervision.

        Args:
            logits (Tensor): [B, T, vocab_size, state_size] model outputs.
            token_ids (Tensor): [B, T] ground truth token indices.
            state_ids (Tensor): [B] ground truth fixed state per sequence.

        Returns:
            Tensor: Scalar loss value (cross entropy).
        """
        B, T, V, S = logits.shape

        # Broadcast state_ids across time
        state_ids_exp = state_ids.unsqueeze(1).expand(-1, T)  # [B, T]
        target = state_ids_exp * V + token_ids  # [B, T]
        logits_flat = logits.view(B * T, V * S)
        return F.cross_entropy(logits_flat, target.view(-1))
