import torch
import unittest
from model.attention import StandardAttention, LinearAttention
from model.router import PathRouter
from model.moe import MiniMoE
from config import Config

class TestModelComponents(unittest.TestCase):
    def setUp(self):
        self.d_model = 128
        self.n_heads = 4
        self.batch_size = 2
        self.seq_len = 16
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_standard_attention(self):
        attn = StandardAttention(self.d_model, self.n_heads)
        out = attn(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_linear_attention(self):
        attn = LinearAttention(self.d_model, self.n_heads)
        out = attn(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_path_router(self):
        router = PathRouter(self.d_model, n_paths=3)
        probs = router(self.x)
        self.assertEqual(probs.shape, (self.batch_size, self.seq_len, 3))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size, self.seq_len)))

    def test_moe(self):
        moe = MiniMoE(self.d_model, d_ff=256, n_experts=4, top_k=2)
        out = moe(self.x)
        self.assertEqual(out.shape, self.x.shape)

if __name__ == "__main__":
    unittest.main()
