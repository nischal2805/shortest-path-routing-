import os
import torch
import time

def save_model(model, model_name, save_dir="saved_models"):
    """
    Save a model to disk
    
    Args:
        model: PyTorch model to save
        model_name: Name of the model (e.g., 'ma_dqn', 'sa_gcn')
        save_dir: Directory to save models in
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pt"
    path = os.path.join(save_dir, filename)
    
    # Save the model
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
    # Also save as latest version
    latest_path = os.path.join(save_dir, f"{model_name}_latest.pt")
    torch.save(model.state_dict(), latest_path)
    
    return path

def load_model(model, model_name, path=None, save_dir="saved_models"):
    """
    Load a model from disk
    
    Args:
        model: PyTorch model instance to load weights into
        model_name: Name of the model
        path: Specific path to load from (optional)
        save_dir: Directory where models are saved
    
    Returns:
        model: The model with loaded weights
    """
    if path is None:
        # Try to load the latest version
        path = os.path.join(save_dir, f"{model_name}_latest.pt")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}, using initial weights")
    
    return model