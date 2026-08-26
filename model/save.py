import torch
import os

def save_model(model, path="saved_models/best_model.pth"):
    """
    Saves the PyTorch model's state_dict to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model successfully saved to {path}")
