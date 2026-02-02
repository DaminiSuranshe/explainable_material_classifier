"""
Progressive training controller for small-data transfer learning.

Automatically advances training stages as dataset grows:
Stage 1: Frozen backbone (10–25 images/class)
Stage 2: Unfreeze top layers (30–50 images/class)
Stage 3: Full fine-tuning (50+ images/class)
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict

from src.models.transfer_learning_model import TransferLearningModel
from src.data_management.small_data_loader import SmallDataImageDataset
from src.data_management.category_manager import CategoryManager
from src.training.train_small_data import FocalLoss, train_epoch, validate


class ProgressiveTrainer:
    """
    Controls progressive unfreezing and fine-tuning.
    """

    def __init__(
        self,
        config: Dict,
        backbone: str = "mobilenetv2",
    ):
        self.cfg = config
        self.backbone = backbone

    def _count_images_per_class(self, dataset_root: str) -> int:
        """
        Estimate images per class.
        """
        counts = {}
        for major in dataset_root.iterdir():
            for sub in major.iterdir():
                counts[sub.name] = len(list(sub.glob("*.*")))

        return min(counts.values())

    def run(
        self,
        base_model_path: str,
        dataset_root: str = "dataset/splits/train",
        stage: int = None,
    ) -> None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cat_mgr = CategoryManager("config/categories.json")
        num_classes = cat_mgr.num_classes()[1]

        dataset_root = torch.Path(dataset_root)
        min_images = self._count_images_per_class(dataset_root)

        if stage is None:
            if min_images >= 50:
                stage = 3
            elif min_images >= 30:
                stage = 2
            else:
                stage = 1

        print(f"🔁 Progressive Training Stage {stage}")

        train_ds = SmallDataImageDataset(
            "dataset/splits/train",
            cat_mgr,
            is_training=True,
            augmentation_multiplier=self.cfg["dataset"]["augmentation_multiplier"],
        )

        val_ds = SmallDataImageDataset(
            "dataset/splits/validation",
            cat_mgr,
            is_training=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
        )

        model = TransferLearningModel(
            num_classes=num_classes,
            backbone=self.backbone,
        ).to(device)

        model.load_state_dict(torch.load(base_model_path))

        if stage == 1:
            model.freeze_backbone()
        elif stage == 2:
            model.freeze_backbone()
            model.unfreeze_last_layers(
                self.cfg["model"]["unfreeze_strategy"]["stage_2"][
                    "unfreeze_last_n_layers"
                ]
            )
        elif stage == 3:
            model.unfreeze_all()

        class_weights = train_ds.get_class_weights().to(device)

        criterion = FocalLoss(
            gamma=self.cfg["loss"]["focal_loss"]["gamma"],
            alpha=class_weights,
        )

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg["training"]["learning_rate"] / stage,
        )

        best_acc = 0.0

        for epoch in range(10):
            print(f"Epoch {epoch+1}")

            train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            _, val_acc = validate(
                model, val_loader, criterion, device
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"models/saved_models/small_data_stage{stage}.pth",
                )
                print("✅ Updated progressive model")

        print("Progressive training complete")
