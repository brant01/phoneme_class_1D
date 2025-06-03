import torchaudio.transforms as T
import torch
import torch.nn as nn

class MFCC(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 ) -> None:
       
        super().__init__()
        self.transform = T.MFCC(sample_rate=sample_rate,
                                n_mfcc=n_mfcc)
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.transform(x)