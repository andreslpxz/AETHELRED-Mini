import torch

class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, device="cuda"):
        self.k = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), device=device)
        self.v = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), device=device)
        self.ptr = 0

    def update(self, k_new, v_new):
        # k_new, v_new: [B, H, S_new, D]
        batch_size, n_heads, seq_len, head_dim = k_new.shape
        self.k[:batch_size, :, self.ptr:self.ptr + seq_len, :] = k_new
        self.v[:batch_size, :, self.ptr:self.ptr + seq_len, :] = v_new
        self.ptr += seq_len
        return self.k[:batch_size, :, :self.ptr, :], self.v[:batch_size, :, :self.ptr, :]

    def reset(self):
        self.ptr = 0
