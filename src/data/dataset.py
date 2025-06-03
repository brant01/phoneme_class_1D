

from torch.utils.data import Dataset
import torchaudio
import torch
from pathlib import Path
from typing import Callable, Optional
import random

from experiment.exp_params import ExpParams


class PhonemeDataset(Dataset):
    """
    A dataset for phoneme classification from WAV files.

    Each sample is a (waveform, label) pair, optionally transformed.
    """

    def __init__(
            self,
            file_paths: list[Path],
            labels: list[int],
            params: ExpParams,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:

        # File paths and labels must have the same length
        assert len(file_paths) == len(labels)  # src/data/dataset.py
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
import random

class PhonemeContrastiveDataset(Dataset):
    """
    Dataset that returns multiple augmented views of each audio sample.
    
    For contrastive learning, each __getitem__ call returns:
    - views: Tensor of shape [n_views, C, H, W] 
    - label: Integer phoneme label
    - metadata: Dict with file info, speaker, etc.
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        config: Dict,
        transform_pipeline: Callable,
        feature_extractor: Callable,
        mode: str = "train"
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.transform_pipeline = transform_pipeline
        self.feature_extractor = feature_extractor
        self.mode = mode
        
        self.n_views = config["views_per_sample"] if mode == "train" else 1
        self.target_sr = config["target_sr"]
        self.max_samples = int(config["max_length_ms"] * self.target_sr / 1000)
        
        # Cache mechanism for small dataset
        self.cache = {}
        self.use_cache = len(file_paths) < 500  # Cache if small dataset
        
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        # Load audio (from cache if available)
        waveform = self._load_audio(idx)
        
        # Generate multiple views
        views = []
        for view_idx in range(self.n_views):
            # Each view gets different augmentation
            aug_waveform = self._augment_waveform(waveform, seed=idx * 1000 + view_idx)
            
            # Extract features (MFCC, mel-spec, etc.)
            features = self.feature_extractor(aug_waveform)
            
            # Apply spectrogram augmentations
            if self.mode == "train":
                features = self.transform_pipeline(features, seed=idx * 2000 + view_idx)
                
            views.append(features)
        
        views = torch.stack(views) if len(views) > 1 else views[0]
        
        metadata = {
            "file_path": str(self.file_paths[idx]),
            "original_length": waveform.shape[-1],
        }
        
        return views, self.labels[idx], metadata
    
    def _load_audio(self, idx: int) -> torch.Tensor:
        if self.use_cache and idx in self.cache:
            return self.cache[idx].clone()
            
        waveform, sr = torchaudio.load(self.file_paths[idx])
        
        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Pad or trim to fixed length
        waveform = self._pad_or_trim(waveform)
        
        if self.use_cache:
            self.cache[idx] = waveform.clone()
            
        return waveform
    
    def _augment_waveform(self, waveform: torch.Tensor, seed: int) -> torch.Tensor:
        """Apply waveform-level augmentations"""
        if self.mode != "train":
            return waveform
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Time stretching (without changing pitch)
        if random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            waveform = torchaudio.functional.time_stretch(waveform, rate)
            
        # Add background noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * random.uniform(0.001, 0.005)
            waveform = waveform + noise
            
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
            
        return waveform
    
    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or trim waveform to fixed length"""
        length = waveform.shape[-1]
        
        if length > self.max_samples:
            # Random crop for training, center crop for eval
            if self.mode == "train":
                start = random.randint(0, length - self.max_samples)
            else:
                start = (length - self.max_samples) // 2
            waveform = waveform[..., start:start + self.max_samples]
        elif length < self.max_samples:
            # Random pad position for training
            pad_total = self.max_samples - length
            if self.mode == "train":
                pad_left = random.randint(0, pad_total)
            else:
                pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
            
        return waveform

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.params = params
        self.target_sr = params.target_sr
        self.max_length = params.max_length or self._estimate_max_length()
        self.n_augment = params.n_augment
        self.pad_strategy = params.pad_strategy


    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths) * self.n_augment
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        
        # Account for augmentations
        true_idx = idx % len(self.file_paths)
        
        path = self.file_paths[true_idx]
        label = self.labels[true_idx]

        waveform, sr = torchaudio.load(path)

        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Remove channel dim if it's mono
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        # Apply radom zero padding before transform
        waveform = self._pad_waveform(waveform)

        # Apply transformation if provided
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
    
    def _estimate_max_length(self) -> int:
        lengths = [torchaudio.info(str(path)).num_frames for path in self.file_paths]
        
        # Add 20% to max length to account for padding
        return int(max(lengths) * 1.2)
    
    def _pad_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length >= self.max_length:
            return waveform[..., :self.max_length]

        pad_total = self.max_length - length
        if self.pad_strategy == "random":
            pad_left = random.randint(0, pad_total)
        elif self.pad_strategy == "left":
            pad_left = 0
        elif self.pad_strategy == "right":
            pad_left = pad_total
        else:
            raise ValueError(f"Invalid pad_strategy: {self.pad_strategy}")

        pad_right = pad_total - pad_left
        return torch.nn.functional.pad(waveform, (pad_left, pad_right))
