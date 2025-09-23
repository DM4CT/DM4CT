import os 
import torch

def save_checkpoint(epoch, model, optimizer, best_val_loss=None, path=None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch+1}: {path}")

def load_checkpoint(model, optimizer, path=None):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resuming training from epoch {start_epoch}: {path}")
        return start_epoch, best_val_loss
    else:
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return 0, float("inf")
    