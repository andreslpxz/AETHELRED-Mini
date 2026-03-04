import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SwiGLU
from .router import MoERouter

class MiniMoE(nn.Module):
    """
    Mini Mixture-of-Experts implementation.
    4 experts, Top-2 routing.
    """
    def __init__(self, d_model, d_ff, n_experts=4, top_k=2, capacity_factor=1.2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = MoERouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        top_probs, top_indices = self.router(x_flat)

        # Dispatching logic
        # For simplicity and T4 optimization, we use a loop or efficient gather
        # A full capacity-based implementation is more complex, here we do a direct dispatch

        out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # Mask for tokens assigned to this expert
            mask = (top_indices == i)
            if mask.any():
                # Get the indices in the flattened batch
                token_indices, expert_rank = torch.where(mask)
                # Select tokens and apply expert
                expert_out = expert(x_flat[token_indices])
                # Weight by router probability and accumulate
                # top_probs has shape [tokens, top_k]
                out[token_indices] += top_probs[token_indices, expert_rank].unsqueeze(-1) * expert_out

        return out.view(batch_size, seq_len, d_model)
