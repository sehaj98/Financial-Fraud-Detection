
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_DIR / "models" / "fraud_random_forest.joblib"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("FINANCIAL FRAUD DETECTION - THRESHOLD TUNING")
    print("=" * 65)

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2. Recreate the same test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Load the trained Random Forest model
    model = joblib.load(MODEL_PATH)

    # 4. Predict fraud probabilities
    fraud_probabilities = model.predict_proba(X_test)[:, 1]

    print("\nTest transactions:", len(y_test))
    print("Actual fraud cases:", int(y_test.sum()))
    print("\nEvaluating different thresholds...\n")

    # 5. Thresholds to evaluate
    thresholds = [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90,
    ]

    results = []

    # 6. Evaluate every threshold
    for threshold in thresholds:

        predictions = (
            fraud_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
        })

    results_df = pd.DataFrame(results)

    # 7. Save all threshold results
    results_path = REPORT_DIR / "threshold_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # 8. Show results
    print(results_df.round(4).to_string(index=False))

    # 9. Identify threshold with the highest F1-score
    best_index = results_df["F1_Score"].idxmax()
    best_result = results_df.loc[best_index]

    print("\n" + "=" * 65)
    print("BEST THRESHOLD BY F1-SCORE")
    print("=" * 65)

    print(f"Threshold: {best_result['Threshold']:.2f}")
    print(f"Precision: {best_result['Precision']:.4f}")
    print(f"Recall:    {best_result['Recall']:.4f}")
    print(f"F1-score:  {best_result['F1_Score']:.4f}")

    print("\nConfusion Matrix Counts:")
    print("True negatives:", int(best_result["True_Negatives"]))
    print("False positives:", int(best_result["False_Positives"]))
    print("False negatives:", int(best_result["False_Negatives"]))
    print("True positives:", int(best_result["True_Positives"]))

    # 10. Save best threshold summary
    best_path = REPORT_DIR / "best_threshold_summary.csv"
    pd.DataFrame([best_result]).to_csv(
        best_path, index=False
    )

    # 11. Calculate PR-AUC / Average Precision
    pr_auc = average_precision_score(
        y_test, fraud_probabilities
    )

    print(f"\nPR-AUC (Average Precision): {pr_auc:.4f}")
    print("\nSaved:", results_path)
    print("Saved:", best_path)
    print("\nStep 6 completed successfully!")


if __name__ == "__main__":
    main()
