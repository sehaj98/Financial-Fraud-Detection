
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports"

FILES = {
    "Logistic Regression": REPORT_DIR / "model_metrics.csv",
    "Random Forest": REPORT_DIR / "random_forest_metrics.csv",
}

def main():
    results = []

    for model_name, file_path in FILES.items():
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            return

        df = pd.read_csv(file_path)
        df.columns = [str(col).strip().lower() for col in df.columns]

        if not {"metric", "value"}.issubset(df.columns):
            print(f"Unexpected CSV format: {file_path}")
            print("Expected columns: Metric, Value")
            return

        metrics = {
            str(row["metric"]).strip().lower(): row["value"]
            for _, row in df.iterrows()
        }

        results.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy"),
            "Precision": metrics.get("precision"),
            "Recall": metrics.get("recall"),
            "F1_Score": metrics.get("f1_score"),
            "PR_AUC": metrics.get("pr_auc_average_precision"),
        })

    comparison = pd.DataFrame(results)
    output_path = REPORT_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\nMODEL COMPARISON")
    print("=" * 75)
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to: {output_path}")

    metrics_to_plot = [
        "Accuracy", "Precision", "Recall", "F1_Score", "PR_AUC"
    ]

    comparison.set_index("Model")[metrics_to_plot].plot(
        kind="bar", figsize=(10, 6), ylim=(0, 1)
    )

    plt.title("Fraud Detection Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.legend(loc="best")
    plt.tight_layout()

    plot_path = REPORT_DIR / "model_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Comparison chart saved to: {plot_path}")
    print("\nStep 5 completed!")

if __name__ == "__main__":
    main()
