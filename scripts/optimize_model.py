"""
TorchScript optimization.
"""

import torch
from src.models.cnn_classifier import HierarchicalCNN
from src.data_management.category_manager import CategoryManager


def main():
    cat = CategoryManager("config/categories.json")
    num_major, num_sub = cat.num_classes()

    model = HierarchicalCNN(num_major, num_sub)
    model.load_state_dict(
        torch.load("models/saved_models/hierarchical_cnn_best.pth")
    )
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save("models/saved_models/model_optimized.pt")

    print("✅ Optimized model saved")


if __name__ == "__main__":
    main()
