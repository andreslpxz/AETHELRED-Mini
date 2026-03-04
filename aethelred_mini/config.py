import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    vocab_size: int = 32000
    seq_len: int = 1024
    dropout: float = 0.1
    n_experts: int = 4
    top_k_moe: int = 2
    capacity_factor: float = 1.2

@dataclass
class TrainingConfig:
    batch_size: int = 1
    grad_accum: int = 32
    lr: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    fp16: bool = True
    use_bnb: bool = True
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    weight_decay: float = 0.1

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer_path: str = "models/spm.model"
    train_path: str = "data/processed/train.npy"
    val_path: str = "data/processed/val.npy"

def get_config(config_path: Optional[str] = None) -> Config:
    config = Config()
    if config_path:
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
            if 'model' in yaml_cfg:
                config.model = ModelConfig(**yaml_cfg['model'])
            if 'training' in yaml_cfg:
                config.training = TrainingConfig(**yaml_cfg['training'])
            if 'data' in yaml_cfg:
                config.tokenizer_path = yaml_cfg['data'].get('tokenizer_path', config.tokenizer_path)
                config.train_path = yaml_cfg['data'].get('train_path', config.train_path)
                config.val_path = yaml_cfg['data'].get('val_path', config.val_path)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="AETHELRED-Mini Configuration")
    parser.add_argument('--config', type=str, default=None, help='Path to yaml config file')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--grad_accum', type=int, default=None)
    return parser.parse_args()
