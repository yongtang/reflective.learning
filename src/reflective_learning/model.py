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

        # Output logit over joint token-state space
        self.output_linear = nn.Linear(d_model, vocab_size * state_size)

        # Positional embeddings for prefix + sequence
        self.pos_embedding = nn.Embedding(max_seq_len + max_prefix_len, d_model)

        self.decoder = decoder

    def forward(
        self,
        token: torch.Tensor,  # [B, T]
        state: torch.Tensor,  # [B]
        prefix: torch.Tensor,  # [B, C, d_model]
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass using token/state indices and prefix.

        Args:
            token: [B, T] LongTensor of token indices.
            state: [B] LongTensor, single state index per sequence.
            prefix: [B, C, d_model] prefix embeddings to prepend.
            mask: Optional attention mask.

        Returns:
            [B, vocab_size, state_size] logit at the final position.
        """
        assert prefix is not None, "prefix is required"
        B, T = token.shape
        V, S = self.vocab_size, self.state_size

        # One-hot encode token and state
        token_onehot = F.one_hot(token, num_classes=V).float()  # [B, T, V]
        state_onehot = F.one_hot(state, num_classes=S).float().unsqueeze(1)  # [B, 1, S]

        # Outer product → [B, T, V, S], then flatten → [B, T, V*S]
        joint_input = torch.einsum("btv,bks->btvs", token_onehot, state_onehot)
        joint_flat = joint_input.view(B, T, V * S)

        # Project and prepend prefix
        x = self.input_linear(joint_flat)  # [B, T, d_model]
        x = torch.cat([prefix, x], dim=1)  # [B, C+T, d_model]

        return self.call(x, mask=mask)  # [B, V, S]

    def call(self, embed: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for precomputed embeddings.

        Args:
            embed: [B, L, d_model] real-valued embeddings (prefix + tokens).
            mask: Optional [B, L, L] attention mask.

        Returns:
            [B, vocab_size, state_size] logit at the final position.
        """
        assert embed is not None, "embed (with prefix) is required"
        B, L, _ = embed.shape

        # Add positional embeddings
        pos = torch.arange(L, device=embed.device).unsqueeze(0).expand(B, L)  # [B, L]
        x = embed + self.pos_embedding(pos)  # [B, L, d_model]

        x = self.decoder(x, x, tgt_mask=mask)  # [B, L, d_model]
        logit = self.output_linear(x).view(
            B, L, self.vocab_size, self.state_size
        )  # [B, L, V, S]

        if mask is not None:
            # Determine the last non-masked position per sequence
            is_masked = mask[:, -1, :] == float("-inf")  # [B, L]
            lengths = (~is_masked).sum(dim=1)  # [B], number of valid positions
            last_idx = lengths - 1  # [B]
            return logit[torch.arange(B), last_idx]  # [B, V, S]
        else:
            return logit[:, -1]  # [B, V, S]

    def loss(
        self,
        logit: torch.Tensor,  # [B, V, S]
        token: torch.Tensor,  # [B] – target token (final position)
        state: torch.Tensor,  # [B] – single state index per sequence
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss against a single (token, state) pair per sequence.

        Args:
            logit: [B, V, S] predicted logit at final position.
            token: [B] ground truth token index per sequence.
            state: [B] ground truth state index per sequence.

        Returns:
            Scalar loss (cross entropy).
        """
        B, V, S = logit.shape

        # Convert (token, state) pair into a flat index over V × S
        target = state * V + token  # [B]
        logit_flat = logit.view(B, V * S)  # [B, V*S]

        return F.cross_entropy(logit_flat, target)

    def collate(self, batch):
        """
        Collate function for variable-length prefix support.

        Input format (per example):
            {
                "token": LongTensor [T],       # token sequence (no prefix)
                "state": LongTensor [1],       # scalar state per sequence
                "prefix": FloatTensor [C, d_model],  # prefix embedding
            }

        Output format (batched):
            {
                "mask":  FloatTensor [B, L, L],        # causal mask
                "embed": FloatTensor [B, L, d_model],  # model input (prefix + projected tokens)
                "token": LongTensor [B, T-1],          # token prediction target
                "state": LongTensor [B],               # per-sequence state
            }
        """
        device = next(self.parameters()).device
        d_model = self.d_model
        V, S = self.vocab_size, self.state_size

        masks, embeds = [], []
        token_targets, state_targets = [], []
        max_embed_len = 0  # max length after prefix + token projection
        max_token_len = 0  # max length of token prediction target (T-1)

        for entry in batch:
            token = entry["token"].to(device)  # [T]
            state = entry["state"].to(device)  # [1]
            prefix = entry["prefix"].to(device)  # [C, d_model]

            T = token.size(0)
            x = torch.zeros((T, V, S), device=device)
            x.scatter_(1, token.view(T, 1, 1), 1.0)
            x.scatter_(2, state.view(T, 1, 1).expand(T, 1, 1), 1.0)
            x = x.view(T, V * S)
            projected = self.input_linear(x)  # [T, d_model]

            embed = torch.cat([prefix, projected], dim=0)  # [L, d_model]
            embeds.append(embed)
            L = embed.size(0)
            max_embed_len = max(max_embed_len, L)

            causal_mask = torch.triu(
                torch.ones((L, L), device=device), diagonal=1
            ).bool()
            mask = torch.zeros((L, L), device=device)
            mask.masked_fill_(causal_mask, float("-inf"))
            masks.append(mask)

            token_targets.append(token[1:])  # [T-1]
            state_targets.append(state.view(()))  # []

            max_token_len = max(max_token_len, T - 1)

        B = len(batch)
        padded_mask = torch.full(
            (B, max_embed_len, max_embed_len), float("-inf"), device=device
        )
        padded_embed = torch.zeros((B, max_embed_len, d_model), device=device)
        padded_tokens = torch.zeros((B, max_token_len), dtype=torch.long, device=device)
        padded_states = torch.stack(state_targets, dim=0)  # [B]

        for i in range(B):
            L = embeds[i].size(0)
            padded_embed[i, :L] = embeds[i]
            padded_mask[i, :L, :L] = masks[i]

            T1 = token_targets[i].size(0)
            padded_tokens[i, :T1] = token_targets[i]

        return {
            "mask": padded_mask,  # [B, L, L]
            "embed": padded_embed,  # [B, L, d_model]
            "token": padded_tokens,  # [B, T-1]
            "state": padded_states,  # [B]
        }
