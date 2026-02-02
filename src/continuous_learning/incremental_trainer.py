"""
Incremental training logic.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.cnn_classifier import HierarchicalCNN
from src.training.trainer import Trainer
from src.utils.logger import setup_logger

logger = setup_logger("incremental_trainer")


class IncrementalTrainer:
    """
    Fine-tunes existing model using new feedback data.
    """

    def retrain(
        self,
        base_model_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_major: int,
        num_sub: int,
        device: torch.device,
        epochs: int = 10,
    ) -> float:
        """
        Incrementally retrain model.

        Returns:
            Validation loss
        """
        model = HierarchicalCNN(num_major, num_sub)
        model.load_state_dict(torch.load(base_model_path, map_location=device))

        # Freeze backbone
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        trainer = Trainer(model, device, lr=1e-4)

        best_val = float("inf")

        for epoch in range(epochs):
            logger.info("Incremental Epoch %d/%d", epoch + 1, epochs)
            trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)

            if val_loss < best_val:
                best_val = val_loss

        return best_val
