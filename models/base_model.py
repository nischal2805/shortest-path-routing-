import torch
import os
import time
from abc import ABC, abstractmethod
from config import device

class BaseRoutingModel(ABC, torch.nn.Module):
    """Base class for all routing models with common functionality"""
    
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.device = device
        self.to(device)
        
    @abstractmethod
    def forward(self, state):
        """Forward pass of the model"""
        pass
        
    @abstractmethod
    def select_action(self, state):
        """Select an action given the current state"""
        pass
    
    def save(self, path=None):
        """Save model to a file"""
        if path is None:
            os.makedirs("saved_models", exist_ok=True)
            path = f"saved_models/{self.__class__.__name__}_{int(time.time())}.pt"
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")
        return path
        
    def load(self, path):
        """Load model from a file"""
        try:
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False