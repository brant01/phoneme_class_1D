import torchaudio.transforms as T
import torch.nn as nn
import torch

class LogMelSpectrogram(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 64
    ) -> None:
        super().__init__()
        self.transform = T.MelSpectrogram(sample_rate=sample_rate,
                                          n_mels=n_mels)
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        mel_spec = self.transform(x)
        return self.amplitude_to_db(mel_spec)