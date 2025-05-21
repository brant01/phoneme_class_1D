from experiment.exp_params import ExpParams

from data_utils.dataset import PhonemeDataset
from data_utils.parser import parse_dataset
from utils.logging import create_logger
from utils.device import get_best_device
from utils.system_resources import adjust_exp_params_for_system
from transforms.build_transforms import build_transforms
from torch.utils.data import DataLoader

class Experiment:
    def __init__(self, params: ExpParams) -> None:
        self.params = params
        self.run_dir = self.params.run_base_dir / self.params.generate_run_id()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(self.run_dir)
        self.device = get_best_device(device_str=self.params.device, logger=self.logger)

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Run directory: {self.run_dir}")

        self.params = adjust_exp_params_for_system(self.params, self.device)

        # Save config to run directory
        self.params.to_json(self.run_dir / "config.json")

    def train(self) -> None:
        self.logger.info("Starting training...")

        # Load data
        file_paths, int_labels, label_map, _ = parse_dataset(self.params.data_path, logger=self.logger)
        self.logger.info(f"Loaded {len(file_paths)} files with {len(label_map)} labels")

        # Build transform pipeline
        transform = build_transforms(self.params)

        # Create dataset and dataloader
        dataset = PhonemeDataset(file_paths, int_labels, params=self.params, transform=transform)
        
        dataloader = DataLoader(
                    dataset,
                    batch_size=self.params.batch_size,
                    num_workers=self.params.num_workers,
                    pin_memory=bool(self.params.pin_memory),
                    drop_last=self.params.drop_last,
                    shuffle=True
                )



        # Placeholder for training loop
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            # TODO: forward pass, loss, backprop, optimize, etc.
            if batch_idx == 0:
                self.logger.info(f"First batch shapes: x={x.shape}, y={y.shape}")

        self.logger.info("Training completed.")

    def evaluate(self) -> None:
        self.logger.info("Starting evaluation...")

    def visualize(self) -> None:
        self.logger.info("Starting visualization...")
