
from datetime import datetime
from pathlib import Path
import json
from pydantic import BaseModel


class ExpParams(BaseModel):
    data_path: Path
    batch_size: int
    learning_rate: float
    num_epochs: int
    imput_dim: int
    num_classes: int


    def generate_run_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def from_json(cls,
                  path: Path) -> "ExpParams":
        with path.open("r") as f:
            return cls(**json.load(f))
