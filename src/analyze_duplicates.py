
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION - DUPLICATE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("Total rows:", len(df))

    # Count duplicate rows, excluding the first occurrence
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    print("Exact duplicate rows:", duplicate_count)
    print("Unique rows:", len(df) - duplicate_count)

    # Examine duplicate records, including all copies
    duplicate_records = df[df.duplicated(keep=False)].copy()

    print(
        "Rows belonging to duplicate groups:",
        len(duplicate_records)
    )

    # Check class distribution among duplicate rows
    if not duplicate_records.empty:
        print("\nClass distribution in duplicate groups:")
        print(duplicate_records["Class"].value_counts())

        print("\nFraud percentage in duplicate groups:")
        print(
            (duplicate_records["Class"].value_counts(normalize=True) * 100)
            .round(4)
            .astype(str) + "%"
        )

    # Compare full dataset and unique-only dataset
    unique_df = df.drop_duplicates(keep="first")

    print("\nOriginal class counts:")
    print(df["Class"].value_counts().sort_index())

    print("\nUnique-only class counts:")
    print(unique_df["Class"].value_counts().sort_index())

    # Save a summary report
    summary = pd.DataFrame({
        "Metric": [
            "Total rows",
            "Exact duplicate rows (extra copies)",
            "Rows in duplicate groups",
            "Unique rows after removing exact duplicates",
            "Original fraud cases",
            "Fraud cases after removing exact duplicates",
            "Original legitimate cases",
            "Legitimate cases after removing exact duplicates",
        ],
        "Value": [
            len(df),
            duplicate_count,
            len(duplicate_records),
            len(unique_df),
            int((df["Class"] == 1).sum()),
            int((unique_df["Class"] == 1).sum()),
            int((df["Class"] == 0).sum()),
            int((unique_df["Class"] == 0).sum()),
        ],
    })

    report_path = REPORT_DIR / "duplicate_analysis.csv"
    summary.to_csv(report_path, index=False)

    print("\nReport saved to:", report_path)
    print("\nNo original dataset rows were deleted.")
    print("Step 8.1 completed!")


if __name__ == "__main__":
    main()
