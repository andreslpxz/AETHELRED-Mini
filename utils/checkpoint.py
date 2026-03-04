import torch
import os

def save_checkpoint(model, optimizer, scheduler, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at step {step}")

def load_checkpoint(model, optimizer, scheduler, path, device="cuda"):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
    return checkpoint['step']
