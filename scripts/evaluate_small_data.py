"""
Evaluation script: confusion matrix + metrics.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.transfer_learning_model import TransferLearningModel
from src.data_management.small_data_loader import SmallDataImageDataset
from src.data_management.category_manager import CategoryManager


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cat_mgr = CategoryManager("config/categories.json")
    num_classes = cat_mgr.num_classes()[1]
    class_names = list(cat_mgr.id_to_sub.values())

    dataset = SmallDataImageDataset(
        "dataset/splits/test",
        cat_mgr,
        is_training=False,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = TransferLearningModel(num_classes=num_classes)
    model.load_state_dict(
        torch.load("models/saved_models/small_data_v1.0.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)["logits"]
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("✅ Confusion matrix saved as confusion_matrix.png")


if __name__ == "__main__":
    main()
