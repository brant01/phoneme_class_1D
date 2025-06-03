# experiment.py

import logging
from pathlib import Path
from experiment.exp_params import ExpParams
from data_utils.parser import parse_dataset
from data_utils.dataset import PhonemeDataset
from utils.device import get_best_device
from utils.logging import create_logger
from utils.system_resources import adjust_exp_params_for_system
from transforms.build_transforms import build_transforms
from models.phoneme_net import PhonemeNet
from models.losses import SupervisedContrastiveLoss
from utils.evaluate_latent_classification import evaluate_latent_classification
from utils.samplers import MultiViewBatchSampler

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import numpy as np
import csv
from tqdm import tqdm


class Experiment:
    def __init__(self, params: ExpParams) -> None:
        self.device = get_best_device(device_str=params.device)

        run_id = params.generate_run_id()
        self.run_dir = params.run_base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(
            self.run_dir / "logs", console_log_level=params.console_log_level
        )
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Run directory: {self.run_dir}")

        self.params = adjust_exp_params_for_system(params, self.device, logger=self.logger)

        self.logger.debug("Adjusted experiment parameters:")
        self.logger.debug(self.params.model_dump_json(indent=2))

        self.params.to_json(self.run_dir / "config.json")

    def train(self) -> None:
        self.logger.info("Starting training...")

        file_paths, int_labels, label_map, _ = parse_dataset(
            self.params.data_path, logger=self.logger
        )

        self.logger.info(f"Found {len(file_paths)} usable audio files")
        self.logger.info(f"Detected {len(label_map)} unique phoneme labels")
        self.logger.debug(f"Phoneme labels: {sorted(label_map.keys())}")

        transform = build_transforms(self.params)
        self.logger.debug(f"Transform pipeline: {transform}")

        dataset = PhonemeDataset(file_paths, int_labels, params=self.params, transform=transform)

        if self.params.use_kfold:
            self._run_kfold_training(dataset, int_labels)
        else:
            self._run_single_fold(dataset, list(range(len(dataset))), fold_id=None)

    def _run_kfold_training(self, dataset, labels):
        kf = KFold(n_splits=self.params.n_splits, shuffle=True, random_state=42)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(labels)))):
            self.logger.info(f"--- Fold {fold_idx + 1}/{self.params.n_splits} ---")
            self._run_single_fold(dataset, train_idx, val_idx, fold_id=fold_idx)

    def _run_single_fold(self, dataset, train_idx, val_idx=None, fold_id=None):
        fold_dir = self.run_dir / f"fold_{fold_id}" if fold_id is not None else self.run_dir
        (fold_dir / "models").mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics").mkdir(parents=True, exist_ok=True)

        if val_idx is None:
            val_split = int(0.8 * len(train_idx))
            val_idx = train_idx[val_split:]
            train_idx = train_idx[:val_split]

        train_labels = [dataset[i][1] for i in train_idx]
        sampler = MultiViewBatchSampler(
            labels=train_labels,
            n_views=2,
            n_classes_per_batch=self.params.batch_size // 2
        )

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_sampler=sampler,
            num_workers=self.params.num_workers,
            pin_memory=bool(self.params.pin_memory),
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=bool(self.params.pin_memory),
            drop_last=False,
        )

        self.logger.debug(
            "Dataloader settings:\n"
            f"  Batch size:   {self.params.batch_size}\n"
            f"  Num workers:  {self.params.num_workers}\n"
            f"  Pin memory:   {self.params.pin_memory}\n"
            f"  Drop last:    {self.params.drop_last}"
        )

        model = PhonemeNet(
            in_channels=1,
            embedding_dim=self.params.embedding_dim,
            use_attention=self.params.use_attention,
        ).to(self.device)

        self.logger.info(
            f"Model initialized | embedding_dim={self.params.embedding_dim}, "
            f"use_attention={self.params.use_attention}"
        )
        self.logger.debug(str(model))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.learning_rate)
        
        criterion = SupervisedContrastiveLoss(
            temperature=self.params.temperature,
            logger=self.logger
        )


        self.logger.debug(f"Optimizer: Adam (lr={self.params.learning_rate})")
        self.logger.info(f"Loss function: SupervisedContrastiveLoss (temperature={self.params.temperature})")

        best_acc = 0.0
        acc_file = fold_dir / "metrics" / "accuracy.csv"
        with acc_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "accuracy"])

        for epoch in range(self.params.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.params.epochs}")
            model.train()
            total_loss = 0.0

            for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                embeddings = model(x)
                self.logger.debug(f"Embeddings mean: {embeddings.mean().item():.4f}, std: {embeddings.std().item():.4f}")
                loss = criterion(embeddings, y)
                loss.backward()
                self.logger.debug(f"Contrastive loss (batch): {loss.item():.6f}")
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.params.eval_classifier_every == 0:
                acc = evaluate_latent_classification(model, val_loader, device=self.device)
                self.logger.info(f"Diagnostic classifier accuracy: {acc:.4f}")
                with acc_file.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, acc])

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), fold_dir / "models" / "best.pt")
                    self.logger.info(f"Saved new best model with accuracy {acc:.4f}")

        torch.save(model.state_dict(), fold_dir / "models" / "last.pt")
        self.logger.info("Final model saved.")
