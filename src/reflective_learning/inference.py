import torch


def sequence(
    model,
    prefix: torch.Tensor,
    state_weights: dict,
    stop_token: int,
    max_seq_len: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a single token sequence using a fixed prefix and state-weighted sampling.

    Args:
        model: Trained ReflectiveCore model.
        prefix (Tensor): [C, d_model] prefix embedding.
        state_weights (dict): Mapping from state index to probability.
        stop_token (int): Token that terminates generation.
        max_seq_len (int): Maximum number of tokens to generate.
        device (str): Device for computation.

    Returns:
        Tensor: [T] of generated token indices.
    """
    model.eval()

    with torch.no_grad():

        V = model.vocab_size
        S = model.state_size
        prefix = prefix.unsqueeze(0).to(device)  # [1, C, d_model]

        # Prepare state indices and normalized weights
        state_indices = torch.arange(S, device=device)  # [S]
        weight_list = [state_weights[s.item()] for s in state_indices]
        state_weights_tensor = torch.tensor(weight_list, device=device)
        state_weights_tensor = state_weights_tensor / state_weights_tensor.sum()

        tokens = torch.empty(0, dtype=torch.long, device=device)  # [T]

        for _ in range(max_seq_len):
            T = tokens.shape[0]

            # Expand current sequence across all states — works even when T == 0
            token_input = tokens.unsqueeze(0).expand(S, -1)  # [S, T]
            state_input = state_indices  # [S]
            prefix_input = prefix.expand(S, -1, -1)  # [S, C, d_model]

            logit = model.forward(token_input, state_input, prefix_input)  # [S, V, S]

            # Select diagonal: P(token | state=s)
            diag_logits = logit.permute(1, 0, 2).diagonal(dim1=1, dim2=2).T  # [S, V]

            # Combine across states using provided weights
            probs = (diag_logits.softmax(dim=-1) * state_weights_tensor[:, None]).sum(
                dim=0
            )  # [V]

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens = torch.cat([tokens, torch.tensor([next_token], device=device)])

            if next_token == stop_token:
                break

        return tokens
