

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
        assert len(file_paths) == len(labels)  

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
