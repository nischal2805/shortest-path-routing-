import torch

# Add debug information about GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Use CUDA if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Force CUDA device if available (even more explicit)
if device == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
