import torch
import torch.nn.functional as F
import argparse
import sentencepiece as spm
from aethelred_mini.model.core import AETHELREDMini
from aethelred_mini.config import get_config
from aethelred_mini.memory.kv_cache import KVCache

class Generator:
    def __init__(self, checkpoint_path, config_path=None, device="cuda"):
        self.config = get_config(config_path)
        self.device = device
        self.model = AETHELREDMini(self.config).to(device)

        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(self.config.tokenizer_path)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9, use_cache=True):
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)

        # Initialize KV Caches for each layer
        kv_caches = None
        if use_cache:
            kv_caches = [
                KVCache(
                    max_batch_size=1,
                    max_seq_len=input_ids.shape[1] + max_new_tokens,
                    n_heads=self.config.model.n_heads,
                    head_dim=self.config.model.d_model // self.config.model.n_heads,
                    device=self.device
                ) for _ in range(self.config.model.layers)
            ]

        generated = tokens

        for i in range(max_new_tokens):
            if use_cache:
                # If using cache, we only pass the last token after the first step
                model_input = input_ids if i == 0 else input_ids[:, -1:]
                logits = self.model(model_input, kv_caches=kv_caches)[:, -1, :]
            else:
                logits = self.model(input_ids)[:, -1, :]

            logits = logits / (temperature + 1e-6)

            # Top-K / Top-P sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_id():
                break

        return self.tokenizer.decode(generated)

    def self_consistency(self, prompt, n=5, **kwargs):
        samples = []
        for _ in range(n):
            samples.append(self.generate(prompt, **kwargs))
        return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "self_consistency"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading from {args.checkpoint} on {device}...")

    try:
        gen = Generator(args.checkpoint, config_path=args.config, device=device)
        if args.mode == "self_consistency":
            results = gen.self_consistency(args.prompt, n=args.n, max_new_tokens=args.max_tokens)
            for i, res in enumerate(results):
                print(f"\nSample {i+1}:\n{res}")
        else:
            result = gen.generate(args.prompt, max_new_tokens=args.max_tokens)
            print(f"\nResult:\n{result}")
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Note: Inference requires a valid checkpoint and tokenizer model file.")
