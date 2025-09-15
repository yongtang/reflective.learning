import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def autocast():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                yield
        except RuntimeError:
            yield  # Fallback fp32


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

    def call(
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
        C = prefix.shape[0]
        D = self.d_model

        # One-hot encode token
        value = F.one_hot(token, num_classes=V).float()  # [T, V]

        # Input projection
        value = self.input_linear(value)  # [T, D]

        # Prepend prefix
        value = torch.cat([prefix, value], dim=0)  # [C+T, D]

        # Add batch dimension
        value = value.unsqueeze(0)  # [1, C+T, D]

        # Construct mask: 1s for all positions
        mask = torch.ones(1, C + T, dtype=torch.bool, device=value.device)  # [1, C+T]

        # Call transformer
        logit = self.forward(mask=mask, embed=value)  # [1, C+T, V]

        # Return logits for final token position
        return logit[0, -1]  # [V]

    def forward(self, mask: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for precomputed embeddings.
        Args:
            mask:  [B, T]
            embed: [B, T, D]
        Returns:
            [B, T, V] logits at each position
        """
        B, T, D = embed.shape

        # Positional embedding
        pos = (
            torch.arange(T, device=embed.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, T)
        )
        value = embed + self.pos_embedding(pos)  # [B, T, D]

        # Padding mask: [B, T], True = PAD — so invert `mask`
        src_key_padding_mask = ~mask  # [B, T]

        # Causal mask: [T, T], True = masked
        mask = torch.triu(embed.new_ones((T, T), dtype=torch.bool), diagonal=1)

        # Transformer: decoder-only via TransformerEncoder + causal mask
        value = self.decoder(
            src=value,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # shape [B, T, D]

        # Output projection to token logits
        logit = self.output_linear(value)  # [B, T, V]

        return logit

    def loss(
        self,
        logit: torch.Tensor,  # [B, L, V] predicted logits for prefix + token
        token: torch.Tensor,  # [B, T] ground truth token indices
        index: torch.Tensor,  # [B] index where token sequence begins (first token position)
        mask: torch.Tensor,  # [B, L] boolean mask for valid positions in logits
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss for next-token prediction.

        Args:
            logit: [B, L, V] predicted logits for prefix + token.
                   Each position logit[b, t] predicts token[b, t + 1].
            token: [B, T] ground truth token indices (shifted right vs logits).
            index: [B] index where token sequence starts in logits (i.e. position of token[0]).
            mask:  [B, L] boolean mask for valid positions in logits.

        Returns:
            Scalar loss (cross entropy), averaged over valid tokens.
        """
        B, L, V = logit.shape
        T = token.size(1)

        # Offsets [0, 1, 2, ...]
        I = torch.arange(T, device=logit.device).view(1, T)  # [1,T]
        start = (index - 1).view(B, 1)  # [B,1]
        position = start + I  # [B,T] positions in logits

        # Number of valid positions per example
        N = mask.sum(dim=1)  # [B]
        count = N - index  # [B]
        # count excludes the final target position — there is no "next token" to predict
        M = int(count.max().item())  # process only needed columns

        I = I[:, :M]  # [1,M]
        position = position[:, :M].clamp_(min=0, max=L - 1)  # [B,M]
        valid = I < count.view(B, 1)  # [B,M]

        # Gather logits for each token prediction step
        step = logit.gather(1, position.unsqueeze(-1).expand(B, M, V))  # [B,M,V]

        # Cross-entropy for valid positions
        loss_flat = F.cross_entropy(
            step.reshape(-1, V), token[:, :M].reshape(-1), reduction="none"
        ).view(
            B, M
        )  # [B,M]

        # Mask invalid positions
        loss_flat = loss_flat * valid.float()  # [B,M]

        # Average over valid tokens
        return loss_flat.sum() / valid.sum().clamp(min=1).float()

    def prob(self, mask, embed, token, index):
        """
        Compute sequence log-probabilities log P(y | x) in batch,
        leveraging collate() outputs.

        Args:
            mask:  [B, L]    from collate()
            embed: [B, L, D] from collate()
            token: [B, T]    from collate() (right-padded)
            index: [B]       from collate() (position of token[0] in input)

        Returns:
            logprob:  [B] sum log-probability of each sequence

        Equations (ASCII):
            For each sequence b and step t (target token y_{b,t} at position pos_{b,t}):
                logits_{b,t} = model(x_b)[pos_{b,t}, :]
                logZ_{b,t}   = logsumexp(logits_{b,t})
                logp_{b,t}   = logits_{b,t}[y_{b,t}] - logZ_{b,t]

            Let valid_{b,t} ∈ {0,1} indicate whether this step is inside the sequence.
            Sequence score (sum over valid steps):
                logP_b = sum_t valid_{b,t} * logp_{b,t}
        """
        # Run the transformer to get logits at every position
        logit = self.forward(mask=mask, embed=embed)  # [B, L, V]
        B, L, V = logit.shape
        T = token.size(1)

        # Offsets for token positions: [0, 1, 2, ...]
        I = torch.arange(T, device=mask.device).view(1, T)  # [1, T]
        start = (index - 1).view(B, 1)  # position before the first token
        position = start + I  # [B, T] positions in logits

        # Mark valid positions: inside sequence length and not padding
        valid = (position >= 0) & ((position + 1) < L) & mask.gather(1, (position + 1))
        # position = position.clamp(0, L - 1)  # ensure positions are in range

        # Gather logits for each token prediction step
        step = logit.gather(1, position.unsqueeze(-1).expand(B, T, V))  # [B, T, V]

        # Normalize with log_softmax (fp32 for stability) vs. (log-softmax via logsumexp)
        with torch.autocast(device_type=embed.device.type, enabled=False):
            logp = F.log_softmax(step, dim=-1)

        # Keep only the log-prob assigned to the reference token
        logp = logp.gather(-1, token.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # Zero-out invalid positions
        logp = logp.masked_fill(~valid, 0.0)

        # Return SUM of valid token log-probabilities per sequence
        return logp.sum(dim=1)  # [B]

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
                - "embed": FloatTensor [B, L, D] — input embeddings (prefix + tokens)
                - "token": LongTensor [B, T] — full token sequence
                - "state": LongTensor [B] — state per example
                - "index": LongTensor [B] — index where token embeddings begin in logits
        """
        device = next(self.parameters()).device
        D = self.d_model
        V = self.vocab_size

        embed_list, token_list, mask_list, state_list = [], [], [], []
        index_list = []

        for entry in batch:
            prefix = entry["prefix"].to(device, non_blocking=True)  # [C, D]
            token = entry["token"].to(device, non_blocking=True)  # [T]
            state = entry["state"].to(device, non_blocking=True)  # []

            T = token.size(0)
            assert T > 0, "Token sequence must have at least 1 token"

            value = F.one_hot(token, num_classes=V).float()  # [T, V]
            value = self.input_linear(value)  # [T, D]
            value = torch.cat([prefix, value], dim=0)  # [C + T, D]

            embed_list.append(value)
            token_list.append(
                token
            )  # target tokens: predict token[t+1] from prefix + token[:t]
            mask_list.append(
                torch.ones(value.shape[0], dtype=torch.bool, device=device)
            )
            state_list.append(state)
            index_list.append(torch.tensor(prefix.shape[0], device=device))  # scalar

        embed = torch.nn.utils.rnn.pad_sequence(
            embed_list, batch_first=True
        )  # [B, L, D]
        token = torch.nn.utils.rnn.pad_sequence(
            token_list, batch_first=True, padding_value=0
        )  # [B, T]
        mask = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True)  # [B, L]
        state = torch.stack(state_list)  # [B]
        index = torch.stack(index_list)  # [B]

        return {
            "mask": mask,  # [B, L] — attention mask for valid input embeddings
            "embed": embed,  # [B, L, D]
            "token": token,  # [B, T] — full token sequence, including token[0]
            "state": state,  # [B]
            "index": index,  # [B] — where token logits begin in output
        }
