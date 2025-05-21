from experiment.exp_params import ExpParams

from data_utils.dataset import PhonemeDataset
from data_utils.parser import parse_dataset
from utils.logger import create_logger
from utils.device import get_best_device

class Experiment:
    def __init__(self, params: ExpParams) -> None:
        self.params = params
        self.run_dir = self.params.run_base_dir / self.params.generate_run_id()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(self.run_dir)
        self.device = get_best_device(device_str=self.params.device, logger=self.logger)

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Run directory: {self.run_dir}")

        # Save config to run directory
        self.params.to_json(self.run_dir / "config.json")

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