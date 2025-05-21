
from pathlib import Path
from typing import Optional
from experiment.exp_params import ExpParams
import logging
import torch

from data_utils.dataset import PhonemeDataset
from data_utils.parser import parse_dataset

class Experiment:
    def __init__(self,
                 params: ExpParams,
                 run_dir: Path,
                 device: torch.device,
                 logger: logging.Logger,
    ) -> None:
        self.params = params
        self.run_dir = run_dir
        self.device = device
        self.logger = logger

    def train(self) -> None:
        self.logger.info("Starting training...")
        # Training logic goes here
        # For example:
        # model = MyModel(self.params)
        # model.train()
        self.logger.info("Training completed.")

    def evaluate(self) -> None:
        self.logger.info("Starting evaluation...")

    def visualize(self) -> None:
        self.logger.info("Starting visualization...")
