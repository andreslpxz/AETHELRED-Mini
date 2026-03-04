import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import argparse
import os

from aethelred_mini.model.core import AETHELREDMini
from aethelred_mini.config import get_config
from aethelred_mini.utils.dataset import get_dataloader
from aethelred_mini.utils.checkpoint import load_checkpoint, save_checkpoint
from aethelred_mini.utils.logging import setup_logging

def finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="aethelred_mini/configs/default.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to new processed data (.npy)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    config = get_config(args.config)
    logger = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AETHELREDMini(config).to(device)

    # Load pretrained checkpoint
    load_checkpoint(model, None, None, args.checkpoint, device=device.type)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=config.training.fp16)

    train_loader = get_dataloader(args.data, config.training.batch_size, config.model.seq_len)

    model.train()
    logger.info(f"Starting fine-tuning on {args.data} for {args.steps} steps")

    step = 0
    while step < args.steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.training.fp16):
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, config.model.vocab_size), y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % 50 == 0:
                logger.info(f"Finetune Step {step} | Loss: {loss.item():.4f}")

            step += 1
            if step >= args.steps:
                break

    save_checkpoint(model, optimizer, None, step, "ckpt/finetuned_model.pt")
    logger.info("Fine-tuning complete.")

if __name__ == "__main__":
    finetune()
