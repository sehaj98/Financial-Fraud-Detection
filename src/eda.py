
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
REPORTS_DIR = PROJECT_DIR / "reports"


def main():
    print("=" * 50)
    print("FINANCIAL FRAUD DETECTION - EDA")
    print("=" * 50)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("Total transactions:", len(df))
    print("Total columns:", len(df.columns))

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts().sort_index())

    print("\nClass percentages:")
    print(
        (df["Class"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(4)
    )

    print("\nAmount statistics:")
    print(df["Amount"].describe())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save class distribution
    class_counts = df["Class"].value_counts().sort_index()
    class_counts.to_csv(
        REPORTS_DIR / "class_distribution.csv",
        header=["Transaction_Count"]
    )

    # Plot transaction classes
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Class")
    plt.title("Legitimate vs Fraudulent Transactions")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "class_distribution.png",
        dpi=150
    )
    plt.close()

    # Plot transaction amounts
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Class", y="Amount")
    plt.title("Transaction Amount by Class")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Amount")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "transaction_amounts.png",
        dpi=150
    )
    plt.close()

    print("\nEDA completed successfully!")
    print("Reports saved in:", REPORTS_DIR)


if __name__ == "__main__":
    main()
