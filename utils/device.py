import torch 
from typing import Optional
import logging

def get_best_device(device_str: str = "auto",
                    logger: Optional[logging.Logger] = None) -> torch.device:
    if device_str == "auto":
        if torch.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)


    if logger:  
        logger.info(f"Using device: {device}")
    return device