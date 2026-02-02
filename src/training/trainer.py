"""
Training engine for hierarchical classifier.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """
    Handles training and validation loops.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5
        )

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):
            images = batch["image"].to(self.device)
            major_labels = batch["major_label"].to(self.device)
            sub_labels = batch["sub_label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss_major = self.criterion(
                    outputs["major_logits"], major_labels
                )
                loss_sub = self.criterion(
                    outputs["sub_logits"], sub_labels
                )
                loss = loss_major + loss_sub

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", leave=False):
                images = batch["image"].to(self.device)
                major_labels = batch["major_label"].to(self.device)
                sub_labels = batch["sub_label"].to(self.device)

                outputs = self.model(images)

                loss_major = self.criterion(
                    outputs["major_logits"], major_labels
                )
                loss_sub = self.criterion(
                    outputs["sub_logits"], sub_labels
                )
                loss = loss_major + loss_sub

                total_loss += loss.item()

        return total_loss / len(loader)
