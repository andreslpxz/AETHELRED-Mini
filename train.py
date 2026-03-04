import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import argparse
import time
import os

from aethelred_mini.model.core import AETHELREDMini
from aethelred_mini.config import get_config
from aethelred_mini.utils.dataset import get_dataloader
from aethelred_mini.utils.checkpoint import save_checkpoint, load_checkpoint
from aethelred_mini.utils.logging import setup_logging

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="aethelred_mini/configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    logger = setup_logging()

    device = torch.device(args.device)
    model = AETHELREDMini(config).to(device)
    logger.info(f"Model initialized with {model.get_num_params() / 1e6:.2f}M parameters")

    # Optimizer
    if config.training.use_bnb and HAS_BNB and device.type == 'cuda':
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        logger.info("Using bitsandbytes Adam8bit optimizer")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        logger.info("Using standard AdamW optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.max_steps)
    scaler = GradScaler(enabled=config.training.fp16)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume, device=args.device)

    if os.path.exists(config.train_path):
        train_loader = get_dataloader(config.train_path, config.training.batch_size, config.model.seq_len)
    else:
        logger.warning(f"Train data not found at {config.train_path}. Training will not start.")
        return

    model.train()
    step = start_step

    logger.info("Starting training loop...")
    while step < config.training.max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.training.fp16):
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, config.model.vocab_size), y.view(-1))
                loss = loss / config.training.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % config.training.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % 100 == 0:
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                logger.info(f"Step {step} | Loss: {loss.item() * config.training.grad_accum:.4f} | VRAM: {mem:.2f}GB")

            if step > 0 and step % config.training.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, step, f"ckpt/model_step_{step}.pt")

            step += 1
            if step >= config.training.max_steps:
                break

    save_checkpoint(model, optimizer, scheduler, step, "ckpt/final_model.pt")
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
