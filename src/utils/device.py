from typing import Optional, Literal
import torch
import platform
import os
import logging

def get_best_device(
    device_str: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    logger: Optional[logging.Logger] = None
) -> torch.device:
    """
    Determine the best device for computation.

    Args:
        device_str: One of 'auto', 'cuda', 'mps', or 'cpu'.
        logger: Optional logger for recording device selection.

    Returns:
        torch.device
    """
    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if device_str != "auto":
        log(f"Using explicitly requested device: {device_str}")
        return torch.device(device_str)

    # Auto-selection
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        log(f"Found {device_count} CUDA device(s):")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            log(f"  GPU {i}: {device_name}")

            if "RTX" in device_name or "NVIDIA" in device_name:
                log(f"Selected NVIDIA GPU: {device_name}")
                return torch.device(f"cuda:{i}")

        log("No NVIDIA RTX GPU found, using default CUDA device")
        return torch.device("cuda:0")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log("Using Apple Silicon GPU with MPS")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        mac_version = platform.mac_ver()[0]
        if mac_version:
            major_version = int(mac_version.split('.')[0])
            if major_version < 13:
                log(f"Warning: macOS {mac_version} detected. MPS performs best on macOS 13+")

        return torch.device("mps")

    else:
        log("No GPU found, using CPU")
        cpu_count = os.cpu_count()
        if cpu_count:
            optimal_threads = max(1, cpu_count - 2)
            torch.set_num_threads(optimal_threads)
            log(f"Set PyTorch to use {optimal_threads} of {cpu_count} CPU threads")

        return torch.device("cpu")
