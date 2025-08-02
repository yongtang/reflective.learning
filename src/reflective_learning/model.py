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
        self.input_linear = nn.Linear(vocab_size, d_model)

        # Output logit over joint token-state space
        self.output_linear = nn.Linear(d_model, vocab_size * state_size)

        # Positional embeddings for prefix + sequence
        self.pos_embedding = nn.Embedding(max_seq_len + max_prefix_len, d_model)

        self.decoder = decoder

    def forward(
        self,
        token: torch.Tensor,  # [B, T]
        prefix: torch.Tensor,  # [B, C, D]
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass using token/state indices and prefix.

        Args:
            token: [B, T] LongTensor of token indices.
            prefix: [B, C, D] prefix embeddings to prepend.
            mask: Optional attention mask.

        Returns:
            [B, V, S] logit at the next position.
        """
        assert prefix is not None, "prefix is required"
        B, T = token.shape
        V, S = self.vocab_size, self.state_size

        # One-hot encode token
        x = F.one_hot(token, num_classes=V).float()  # [B, T, V]

        x = self.input_linear(x)  # [B, T, D]
        x = torch.cat([prefix, x], dim=1)  # [B, C+T, D]

        return self.call(mask=mask, embed=x)  # [B, V, S]

    def call(self, mask: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for precomputed embeddings.
        Args:
            mask:  [B, L, L] causal attention mask (float mask with -inf or 0)
            embed: [B, L, D]
        Returns:
            [B, V, S] logit at final position
        """
        B, L, _ = embed.shape
        assert False
        # Expand [B, L, L] → [B * head, L, L] as required by PyTorch
        head = self.decoder.layers[0].self_attn.num_heads
        mask = mask.unsqueeze(1).expand(B, head, L, L).reshape(B * head, L, L)

        # Add position embeddings
        x = torch.arange(L, device=embed.device).unsqueeze(0).expand(B, L)
        x = embed + self.pos_embedding(x)

        x = self.decoder(x, x, tgt_mask=mask)  # standard decoder call
        logit = self.output_linear(x).view(B, L, V, S)

        return logit[:, -1]  # always return final position

    def loss(
        self,
        logit: torch.Tensor,  # [B, V, S] predicted logit.
        token: torch.Tensor,  # [B] – token index per sequence.
        state: torch.Tensor,  # [B] – state index per sequence.
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss against a single (token, state) pair per sequence.

        Args:
            logit: [B, V, S] predicted logit.
            token: [B] ground truth token index per sequence.
            state: [B] ground truth state index per sequence.

        Returns:
            Scalar loss (cross entropy).
        """

        B, V, S = logit.shape

        # Select the logits for the token - shape: [B, S]
        value = logit[torch.arange(B), token]

        # compute cross-entropy loss for state class
        return F.cross_entropy(value, state)

    def collate(self, batch):
        """
        Collate function for training the ReflectiveCore model using next-token prediction.

        Each example in the input batch consists of:
            - prefix: FloatTensor [C, D], real-valued context embedding
            - token: LongTensor [T], where T > 0
            - state: LongTensor [], single state label

        This function produces a batch suitable for training a decoder-only model
        to predict the final token in the sequence (token[T-1]) using the following input:
            - prefix embedding [C, D]
            - projected input tokens token[0:T-1], each expanded with one-hot token × state,
              linearly projected to [D]

        The resulting input embedding has shape [L, D] where L = C + (T-1).
        A causal attention mask of shape [L, L] is constructed to prevent attending to future positions.

        Returns:
            A dict with the following keys:
                - "mask":  FloatTensor [B, L, L] — causal attention mask (float, with -inf for masked positions)
                - "embed": FloatTensor [B, L, D] — full input embeddings (prefix + token projections)
                - "token": LongTensor [B] — final token in each sequence (prediction target)
                - "state": LongTensor [B] — state label per example (shared across sequence)
        """
        device = next(self.parameters()).device
        D = self.d_model
        V, S = self.vocab_size, self.state_size

        mask, embed = [], []
        token_label, state_label = [], []
        # max_len = 0

        for entry in batch:
            token = entry["token"].to(device)  # [T]
            state = entry["state"].to(device)  # []
            prefix = entry["prefix"].to(device)  # [C, D]

            T = token.size(0)
            assert T > 1, "Sequence must have at least 1 token (to predict one step)"

            token_label.append(token[-1])  # []
            state_label.append(state)  # []

            # One-hot encode token and state
            x = F.one_hot(token[:-1], num_classes=V).float()  # [T-1, V]

            x = self.input_linear(x)  # [T-1, D]
            x = torch.cat([prefix, x], dim=0)  # [C+T-1, D]
            embed.append(x)

        # Length of embed
        count = torch.tensor([e.size(0) for e in embed], device=embed.device)

        # Pad to longest sequence with zeros
        embed = torch.nn.utils.rnn.pad_sequence(
            embed, batch_first=True
        )  # [B, T, D] <= max(T)

        # True for valid, False for padding
        B, T = embed.size(0), embed.size(1)
        mask = torch.arange(T, device=padded.device).expand(B, T) < count.unsqueeze(
            1
        )  # shape: [B, T]

        return {
            "mask": padded_mask,  # [B, T]
            "embed": padded_embed,  # [B, T, D]
            "token": torch.stack(token_label),  # [B]
            "state": torch.stack(state_label),  # [B]
        }
