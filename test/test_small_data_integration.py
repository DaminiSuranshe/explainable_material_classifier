"""
Integration tests for small-data training.
"""

import torch
from src.models.transfer_learning_model import TransferLearningModel


def test_model_forward():
    model = TransferLearningModel(num_classes=3)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert "logits" in out
    assert "features" in out
    assert out["logits"].shape[1] == 3


if __name__ == "__main__":
    test_model_forward()
    print("✅ Integration test passed")
