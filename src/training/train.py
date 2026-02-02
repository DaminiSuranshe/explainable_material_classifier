"""
CLI training entrypoint.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from src.data_management.category_manager import CategoryManager
from src.data_management.dataset_loader import HierarchicalImageDataset
from src.preprocessing.image_processor import ImageProcessor
from src.models.cnn_classifier import HierarchicalCNN
from src.training.trainer import Trainer
from src.utils.logger import setup_logger

logger = setup_logger("train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/splits/train")
    parser.add_argument("--val-dir", default="dataset/splits/validation")
    parser.add_argument("--categories", default="config/categories.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    category_manager = CategoryManager(args.categories)
    num_major, num_sub = category_manager.num_classes()

    processor = ImageProcessor()

    train_ds = HierarchicalImageDataset(
        args.train_dir, category_manager, processor
    )
    val_ds = HierarchicalImageDataset(
        args.val_dir, category_manager, processor
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = HierarchicalCNN(num_major, num_sub)
    trainer = Trainer(model, device, args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)

        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        trainer.scheduler.step(val_loss)

        logger.info(
            "Train Loss: %.4f | Val Loss: %.4f", train_loss, val_loss
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                "models/saved_models/hierarchical_cnn_best.pth",
            )
            logger.info("✅ Saved new best model")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
