# src/data/features.py
import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Dict, Optional

class FeatureExtractor(nn.Module):
    """Base class for feature extractors"""
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MFCCExtractor(FeatureExtractor):
    """Extract MFCC features from waveform"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        
        # Create mel spectrogram first
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate / 2
        )
        
        # Then convert to MFCC
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'f_min': f_min,
                'f_max': f_max or sample_rate / 2
            }
        )
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [1, samples] or [samples]
            
        Returns:
            mfcc: [1, n_mfcc, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Compute MFCC
        mfcc = self.mfcc(waveform)
        
        # Ensure output is [1, features, time]
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)
            
        return mfcc

class MelSpectrogramExtractor(FeatureExtractor):
    """Extract log mel spectrogram features"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate / 2
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            log_mel: [1, n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel = self.mel_spec(waveform)
        log_mel = self.amplitude_to_db(mel)
        
        if log_mel.dim() == 2:
            log_mel = log_mel.unsqueeze(0)
            
        return log_mel

class CombinedExtractor(FeatureExtractor):
    """Combine multiple feature extractors"""
    
    def __init__(self, extractors: Dict[str, FeatureExtractor]):
        super().__init__()
        self.extractors = nn.ModuleDict(extractors)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            features: [1, total_features, time_frames]
        """
        features = []
        for name, extractor in self.extractors.items():
            feat = extractor(waveform)
            features.append(feat)
            
        # Concatenate along feature dimension
        return torch.cat(features, dim=1)

# Factory function
def build_feature_extractor(config: Dict) -> FeatureExtractor:
    """Build feature extractor from config"""
    
    extractor_type = config.get("type", "mfcc")
    
    if extractor_type == "mfcc":
        return MFCCExtractor(**config.get("mfcc_params", {}))
    elif extractor_type == "mel":
        return MelSpectrogramExtractor(**config.get("mel_params", {}))
    elif extractor_type == "combined":
        extractors = {}
        if config.get("use_mfcc", True):
            extractors["mfcc"] = MFCCExtractor(**config.get("mfcc_params", {}))
        if config.get("use_mel", False):
            extractors["mel"] = MelSpectrogramExtractor(**config.get("mel_params", {}))
        return CombinedExtractor(extractors)
    else:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}")