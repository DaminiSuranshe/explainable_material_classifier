"""
Estimate dataset sufficiency and expected accuracy.
"""

from pathlib import Path


def estimate_accuracy(images_per_class: int) -> str:
    if images_per_class < 10:
        return "❌ Too few images"
    if images_per_class < 20:
        return "60–75%"
    if images_per_class < 50:
        return "75–85%"
    if images_per_class < 100:
        return "85–92%"
    return "92–97%"


def main():
    root = Path("dataset/raw")

    for major in root.iterdir():
        for sub in major.iterdir():
            count = len(list(sub.glob("*.*")))
            print(
                f"{sub.name}: {count} images → "
                f"Expected accuracy: {estimate_accuracy(count)}"
            )


if __name__ == "__main__":
    main()
