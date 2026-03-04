import torch
import torch.nn as nn
import torch.nn.functional as F

class PathRouter(nn.Module):
    """
    Decides between Standard Attention, Linear Attention, and SSM-lite paths.
    """
    def __init__(self, d_model, n_paths=3, temperature=1.0):
        super().__init__()
        self.router = nn.Linear(d_model, n_paths)
        self.temperature = temperature

    def forward(self, x):
        # x: [B, S, D]
        logits = self.router(x) / self.temperature
        # Gumbel-Softmax for differentiable path selection or just soft gating
        probs = F.softmax(logits, dim=-1)
        return probs

class MoERouter(nn.Module):
    """
    Router for Mini-MoE Experts.
    """
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        # Normalize top probs
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-6)
        return top_probs, top_indices
