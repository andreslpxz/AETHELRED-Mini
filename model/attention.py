import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask expected to be [B, 1, S, S] or [1, 1, S, S]
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = (attn_probs @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class LinearAttention(nn.Module):
    """Simplified FAVOR+ / Performer-style Linear Attention"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x):
        return F.relu(x) + 1e-6

    def forward(self, x, mask=None, kv_cache=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Linear attention for decoder: using a simple causal prefix sum for linear efficiency
        # This approximates causal linear attention
        k = k.transpose(1, 2) # [B, H, S, D]
        v = v.transpose(1, 2) # [B, H, S, D]
        q = q.transpose(1, 2) # [B, H, S, D]

        # Causal linear attention via prefix sums
        # Compute K^T @ V cumulatively
        # kv: [B, H, S, D, D]
        kv = torch.einsum('bhsd,bhsm->bhsdm', k, v)
        kv_cum = torch.cumsum(kv, dim=2)

        # Compute out: Q @ (sum K^T @ V)
        out = torch.einsum('bhsd,bhsdm->bhsm', q, kv_cum)

        # Normalization
        k_cum = torch.cumsum(k, dim=2)
        z = 1.0 / (torch.einsum('bhsd,bhsd->bhs', q, k_cum) + 1e-6)

        out = out * z.unsqueeze(-1)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)
