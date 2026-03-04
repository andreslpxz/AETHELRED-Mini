# AETHELRED-Mini

AETHELRED-Mini is a hybrid LLM architecture optimized for training and inference on a single NVIDIA T4 (16GB) GPU.

## Features
- **Hybrid Architecture**: Combines Dense Attention, Linear Attention (FAVOR-style), and SSM-lite (Mamba-inspired) paths.
- **Mini-MoE**: 4 experts with Top-2 routing for efficient parameter scaling.
- **VRAM Optimized**: Support for 8-bit optimizers (`bitsandbytes`) and mixed-precision (FP16).
- **Memory**: KV-cache for fast inference and FAISS-based session memory.

## Setup

```bash
pip install -r requirements.txt
./run.sh
```

## Usage

### 1. Train Tokenizer
```bash
python aethelred_mini/tokenizer/train_tokenizer.py --input data/raw/corpus.txt --model_prefix spm --vocab_size 32000
```

### 2. Prepare Data
```bash
python aethelred_mini/data/prepare_data.py --input data/raw/train.txt --tokenizer models/spm.model --output data/processed/train.npy
```

### 3. Training
```bash
python aethelred_mini/train.py --config aethelred_mini/configs/default.yaml --device cuda
```

### 4. Fine-tuning
```bash
python aethelred_mini/finetune.py --checkpoint ckpt/latest.pt --data data/new/data.npy
```

### 5. Inference
```bash
python aethelred_mini/inference.py --checkpoint ckpt/latest.pt --prompt "Hello AETHELRED" --mode sample
```

## Architecture Details
- Layers: 12
- d_model: 768
- n_heads: 12
- d_ff: 3072
- Experts: 4 (Top-2)
- Total Params: ~180M

## Estimated VRAM Usage (T4)
- Default config (~180M params): ~8-10GB during training with `8bit Adam` and `grad_accum`.
- Inference: < 2GB.
