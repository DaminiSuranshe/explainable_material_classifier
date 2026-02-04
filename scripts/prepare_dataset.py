"""
Dataset preparation script.
"""

import argparse
from tkinter.filedialog import test

from src.data_management.data_splitter import split_dataset
from src.utils.logger import get_logger


logger = get_logger(
    "prepare_dataset",
    log_dir="logs",
    json_logs=False,
)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--raw", default="dataset/raw")
    parser.add_argument("--output", default="dataset/splits")
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=(0.7, 0.15, 0.15),
        help="Train/Val/Test split ratios",
    )

    args = parser.parse_args()

    split_dataset(
        raw_dir=args.raw,
        output_dir=args.output,
        split_ratio=tuple(args.split),
    )

logger.info("Starting dataset preparation")

if __name__ == "__main__":
    main()
