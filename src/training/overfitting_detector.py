"""
Overfitting detection for small-data training.
"""

class OverfittingDetector:
    """
    Monitors train/validation gap.
    """

    def __init__(self, gap_threshold: float = 0.15):
        self.gap_threshold = gap_threshold

    def check(self, train_acc: float, val_acc: float) -> str:
        gap = train_acc - val_acc

        if gap > self.gap_threshold:
            return (
                "⚠️ Overfitting detected.\n"
                "Suggested actions:\n"
                "- Increase augmentation\n"
                "- Increase dropout\n"
                "- Reduce learning rate\n"
                "- Collect 5–10 more images per class"
            )

        return "✅ No overfitting detected"
