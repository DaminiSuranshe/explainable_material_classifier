"""
Small-data training script using transfer learning.

Optimized for:
- 10–50 images per category
- Heavy augmentation
- Frozen backbone training
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_management.category_manager import CategoryManager
from src.data_management.small_data_loader import SmallDataImageDataset
from src.models.transfer_learning_model import TransferLearningModel


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance and small datasets.
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs["logits"], labels)
            total_loss += loss.item()

            preds = outputs["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/training_config_small_data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone", default="mobilenetv2")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    category_manager = CategoryManager("config/categories.json")
    num_classes = category_manager.num_classes()[1]

    train_ds = SmallDataImageDataset(
        "dataset/splits/train",
        category_manager,
        is_training=True,
        augmentation_multiplier=cfg["dataset"]["augmentation_multiplier"],
    )

    val_ds = SmallDataImageDataset(
        "dataset/splits/validation",
        category_manager,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size
        or cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size
        or cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    model = TransferLearningModel(
        num_classes=num_classes,
        backbone=args.backbone,
    ).to(device)

    model.freeze_backbone()

    class_weights = train_ds.get_class_weights().to(device)

    criterion = FocalLoss(
        gamma=cfg["loss"]["focal_loss"]["gamma"],
        alpha=class_weights,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )

    best_acc = 0.0

    for epoch in range(args.epochs or cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                "models/saved_models/small_data_v1.0.pth",
            )
            print("✅ Saved best model")

    print("Training complete")


if __name__ == "__main__":
    main()
