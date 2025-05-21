
import torch
import torchaudio.transforms as T
import random

class RandomTimeMask:
    def __init__(self, max_width: int = 30, p: float = 0.5):
        self.max_width = max_width
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            masker = T.TimeMasking(time_mask_param=self.max_width)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return masker(x)
        return x

class RandomFreqMask:
    def __init__(self, max_width: int = 10, p: float = 0.5):
        self.max_width = max_width
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            masker = T.FrequencyMasking(freq_mask_param=self.max_width)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return masker(x)
        return x
