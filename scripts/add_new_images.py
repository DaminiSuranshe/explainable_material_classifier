"""
Add new images to dataset and trigger retraining.
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--image-folder", required=True)

    args = parser.parse_args()

    dest = Path("dataset/raw") / args.category
    dest.mkdir(parents=True, exist_ok=True)

    added = 0
    for img_path in Path(args.image_folder).glob("*.*"):
        try:
            img = Image.open(img_path)
            img.verify()
            shutil.copy(img_path, dest / img_path.name)
            added += 1
        except Exception:
            continue

    print(f"✅ Added {added} images to {args.category}")
    print("🔁 Re-run dataset preparation and training")


if __name__ == "__main__":
    main()
