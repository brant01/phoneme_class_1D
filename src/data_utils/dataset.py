

from torch.utils.data import Dataset
import torchaudio
import torch
from pathlib import Path
from typing import Callable, List, Optional


class PhonemeDataset(Dataset):
    """
    A dataset for phoneme classification from WAV files.

    Each sample is a (waveform, label) pair, optionally transformed.
    """

    def __init__(
            self,
            file_paths: list[Path],
            labels: list[int],
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_sr: int = 16000,
    ) -> None:
        """
        Args:
            file_paths (list[Path]): List of file paths to the audio files.
            labels (list[int]): List of integer labels corresponding to each file.
            transform (Callable, optional): Optional transform to be applied on a sample.
            target_sr (int): Target sample rate for audio files.
        """
        
        # File paths and labels must have the same length
        assert len(file_paths) == len(labels)  

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_sr = target_sr

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(path)

        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Remove channel dim if it's mono
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        # Apply transformation if provided
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label