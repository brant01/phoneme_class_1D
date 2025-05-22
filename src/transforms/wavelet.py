
import torch
import torchaudio
import pywt
import numpy as np
from typing import Callable

class WaveletTransform:
    def __init__(self, 
                 wavelet: str = "db4", 
                 level: int = 5, 
                 mode: str = "zero"):
        """
        Compute multi-scale wavelet coefficients for an audio waveform.

        Args:
            wavelet: Name of the wavelet (e.g., 'db4', 'coif1')
            level: Number of decomposition levels
            mode: Padding mode (e.g., 'zero', 'symmetric')
        """
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.squeeze(0)  # convert [1, N] to [N]

        x_np = x.cpu().numpy()
        coeffs = pywt.wavedec(x_np, self.wavelet, mode=self.mode, level=self.level)

        # Convert list of arrays to 2D array: [scale, time]
        padded_coeffs = [self._pad_to_length(c, coeffs[0].shape[0]) for c in coeffs]
        coeff_array = np.stack(padded_coeffs)

        return torch.tensor(coeff_array, dtype=torch.float32)

    def _pad_to_length(self, arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) == length:
            return arr
        elif len(arr) > length:
            return arr[:length]
        else:
            return np.pad(arr, (0, length - len(arr)), mode='constant')