import torch
import sys

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor creation on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            print(f"Test tensor on GPU: {x}")
            print("✓ GPU test passed!")
        except Exception as e:
            print(f"GPU test failed with error: {e}")
    else:
        print("No CUDA-capable GPU detected")

if __name__ == "__main__":
    check_gpu()