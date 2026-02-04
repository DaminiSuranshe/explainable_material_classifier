import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.class_weights,
        )

        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        return loss.mean()
