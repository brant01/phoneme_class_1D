
import torch
import random

class AddNoise:
    def __init__(self, 
                 std: float = 0.005, 
                 p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x
