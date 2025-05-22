from experiment.exp_params import ExpParams
from data_utils.parser import parse_dataset
from data_utils.dataset import PhonemeDataset
from models.phoneme_net import PhonemeNet
from models.losses import nt_xent_loss
from transforms.build_transforms import build_transforms
from utils.logging import create_logger
from utils.device import get_best_device
from utils.system_resources import adjust_exp_params_for_system

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  



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
        self.params.to_json(self.run_dir / "config.json")

    def train(self) -> None:
        self.logger.info("Starting training...")

        if self.params.n_augment != 2:
            raise ValueError("NT-Xent loss requires n_augment=2. Please update your config.")

        # Load dataset
        file_paths, int_labels, label_map, _ = parse_dataset(self.params.data_path, logger=self.logger)
        transform = build_transforms(self.params)
        dataset = PhonemeDataset(file_paths, int_labels, params=self.params, transform=transform)

        dataloader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=bool(self.params.pin_memory),
            drop_last=self.params.drop_last,
            shuffle=True,
        )

        model = PhonemeNet(
            in_channels=1,
            embedding_dim=self.params.embedding_dim,
            use_attention=self.params.use_attention,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.learning_rate)
        model.train()

        for epoch in range(self.params.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.params.epochs}")
            model.train()

            epoch_loss = 0.0
            for batch_idx, (x, _) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
                x = x.to(self.device)

                embeddings = model(x)
                loss = nt_xent_loss(embeddings, temperature=self.params.temperature)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")


        self.logger.info("Training completed.")

    def evaluate(self) -> None:
        self.logger.info("Starting evaluation...")

        file_paths, int_labels, label_map, _ = parse_dataset(self.params.data_path, logger=self.logger)
        transform = build_transforms(self.params)
        dataset = PhonemeDataset(file_paths, int_labels, params=self.params, transform=transform)

        dataloader = DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=bool(self.params.pin_memory),
            drop_last=False,
            shuffle=False,
        )

        model = PhonemeNet(
            in_channels=1,
            embedding_dim=self.params.embedding_dim,
            use_attention=self.params.use_attention,
        ).to(self.device)

        model.eval()
        all_embeddings = []
        all_labels = []
        all_filenames = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                emb = model(x)  # shape [B, D]
                all_embeddings.append(emb.cpu())
                all_labels.append(y.cpu())
                all_filenames.extend([""] * x.size(0))  # placeholder

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        torch.save(embeddings, self.run_dir / "embeddings.pt")
        torch.save(labels, self.run_dir / "labels.pt")
        with open(self.run_dir / "filenames.txt", "w") as f:
            for name in all_filenames:
                f.write(f"{name}\n")

        self.logger.info(f"Saved embeddings, labels, and filenames to {self.run_dir}")

    def visualize(self) -> None:
        self.logger.info("Starting visualization...")
        # Placeholder — will be implemented after evaluate outputs
