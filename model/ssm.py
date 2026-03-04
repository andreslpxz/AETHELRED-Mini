import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMLite(nn.Module):
    """
    A simplified SSM-like path using depthwise convolutions and state compression.
    Inspired by Mamba but simplified for T4 constraints.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        # Simplified forward pass
        batch_size, seq_len, _ = x.shape

        projected = self.in_proj(x)
        x_inner, res = projected.chunk(2, dim=-1)

        # Conv path
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :seq_len]
        x_inner = x_inner.transpose(1, 2)

        x_inner = F.silu(x_inner)

        # Simple gating/state interaction
        # In a real SSM, we'd have a recurrent step.
        # Here we simulate the effect with a data-dependent projection
        dt = torch.sigmoid(self.dt_proj(x_inner))
        out = x_inner * dt * F.silu(res)

        return self.out_proj(out)
