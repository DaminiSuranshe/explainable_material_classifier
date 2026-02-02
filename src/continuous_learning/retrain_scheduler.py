"""
Retraining scheduler (manual trigger or cron).
"""

import torch
from torch.utils.data import DataLoader

from src.continuous_learning.incremental_trainer import IncrementalTrainer
from src.continuous_learning.model_registry import ModelRegistry
from src.data_management.category_manager import CategoryManager
from src.utils.logger import setup_logger

logger = setup_logger("retrain_scheduler")


def run_retraining(
    base_model: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    category_file: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cat_mgr = CategoryManager(category_file)
    num_major, num_sub = cat_mgr.num_classes()

    trainer = IncrementalTrainer()
    val_loss = trainer.retrain(
        base_model,
        train_loader,
        val_loader,
        num_major,
        num_sub,
        device,
    )

    registry = ModelRegistry()
    registry.register(
        {
            "version": f"incremental_{len(registry.latest())}",
            "base_model": base_model,
            "val_loss": val_loss,
        }
    )

    logger.info("Retraining completed (val loss %.4f)", val_loss)
