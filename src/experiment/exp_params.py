from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from pathlib import Path
import json
from datetime import datetime


class ExpParams(BaseModel):
    # === Paths ===
    data_path: Path = Path("data/raw/New Stimuli 9-8-2024")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    run_base_dir: Path = Path("runs")

    # === Audio ===
    target_sr: int = 16000
    max_length: Optional[int] = None

    # === Dataset ===
    n_augment: int = 1
    pad_strategy: Literal["random", "left", "right"] = "random"

    # === Transforms ===
    use_mfcc: bool = True
    use_log_mel: bool = False
    use_wavelet: bool = False
    use_time_mask: bool = True
    use_freq_mask: bool = True
    use_noise: bool = True

    time_mask_p: float = 0.5
    time_mask_param: int = 30

    freq_mask_p: float = 0.5
    freq_mask_param: int = 10

    noise_std: float = 0.005
    noise_p: float = 0.3

    # === Device ===
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"

    # === Training ===
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 3e-4
    num_workers: int = 1
    pin_memory: Optional[bool] = None
    drop_last: bool = False

    # === Model ===
    embedding_dim: int = 128
    use_attention: bool = True

    # === Contrastive Loss ===
    temperature: float = 0.07

    # === Evaluation ===
    eval_classifier_every: int = 5
    use_kfold: bool = True
    n_splits: int = 5

    # === Experiment control ===
    mode: Literal["train", "evaluate", "visualize"] = "train"

    def generate_run_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def from_json(cls, path: Path) -> "ExpParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4, default=str)
