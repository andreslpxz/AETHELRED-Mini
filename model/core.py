import torch
import torch.nn as nn
from .layers import RMSNorm
from .attention import StandardAttention, LinearAttention
from .ssm import SSMLite
from .router import PathRouter
from .moe import MiniMoE
from config import Config

class AETHELREDBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.model.d_model

        self.norm1 = RMSNorm(self.d_model)

        # Hybrid paths
        self.standard_attn = StandardAttention(self.d_model, config.model.n_heads, config.model.dropout)
        self.linear_attn = LinearAttention(self.d_model, config.model.n_heads, config.model.dropout)
        self.ssm_path = SSMLite(self.d_model)

        self.path_router = PathRouter(self.d_model)

        self.norm2 = RMSNorm(self.d_model)
        self.moe = MiniMoE(
            self.d_model,
            config.model.d_ff,
            n_experts=config.model.n_experts,
            top_k=config.model.top_k_moe,
            capacity_factor=config.model.capacity_factor
        )

    def forward(self, x, mask=None, kv_cache=None):
        # x: [B, S, D]
        residual = x
        x = self.norm1(x)

        # Routing between hybrid paths
        path_probs = self.path_router(x) # [B, S, 3]

        out_standard = self.standard_attn(x, mask, kv_cache)
        out_linear = self.linear_attn(x, mask)
        out_ssm = self.ssm_path(x)

        # Gated fusion
        h = (
            path_probs[:, :, 0:1] * out_standard +
            path_probs[:, :, 1:2] * out_linear +
            path_probs[:, :, 2:3] * out_ssm
        )

        x = residual + h

        # MoE FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.moe(x)

        return x

class AETHELREDMini(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.model.vocab_size, config.model.d_model)

        self.layers = nn.ModuleList([
            AETHELREDBlock(config) for _ in range(config.model.layers)
        ])

        self.norm_out = RMSNorm(config.model.d_model)
        self.lm_head = nn.Linear(config.model.d_model, config.model.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embeddings.weight

    def forward(self, input_ids, mask=None, kv_caches=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if mask is None:
            # Create causal mask
            mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len)

        x = self.embeddings(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x = layer(x, mask, kv_caches[i])

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
