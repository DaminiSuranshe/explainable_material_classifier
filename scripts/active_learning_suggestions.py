"""
Suggests which categories need more data next.
"""

from pathlib import Path
import json


def main():
    report_file = Path("logs/last_classification_report.json")

    if not report_file.exists():
        print("❌ No report found. Run evaluation first.")
        return

    report = json.loads(report_file.read_text())

    print("\n🧠 Active Learning Suggestions:\n")

    for cls, metrics in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue

        recall = metrics["recall"]
        precision = metrics["precision"]

        if recall < 0.7:
            print(
                f"⚠️ {cls}: LOW recall ({recall:.2f}) → "
                "Collect more diverse images"
            )
        elif precision < 0.7:
            print(
                f"⚠️ {cls}: LOW precision ({precision:.2f}) → "
                "Collect confusing negatives"
            )
        else:
            print(f"✅ {cls}: OK")


if __name__ == "__main__":
    main()
