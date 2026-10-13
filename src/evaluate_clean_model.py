
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("CLEAN FRAUD MODEL - VALIDATION AND FINAL TEST")
    print("=" * 65)

    # 1. Load data and remove exact duplicate rows for this experiment.
    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    print("Original rows:", original_rows)
    print("Rows after exact deduplication:", len(df))
    print("Rows removed:", original_rows - len(df))

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2. Create a held-out test set first.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Split the development data into training and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        random_state=42,
        stratify=y_development,
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Final test rows:", len(X_test))
    print("Training fraud cases:", int(y_train.sum()))
    print("Validation fraud cases:", int(y_val.sum()))
    print("Test fraud cases:", int(y_test.sum()))

    # 4. Train a new model only on training data.
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining clean Random Forest...")
    model.fit(X_train, y_train)

    # 5. Choose a candidate threshold using validation data only.
    val_probabilities = model.predict_proba(X_val)[:, 1]

    threshold_rows = []

    for threshold in [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85, 0.90,
    ]:
        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        threshold_rows.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, val_predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val, val_predictions, zero_division=0
            ),
            "F1_Score": f1_score(
                y_val, val_predictions, zero_division=0
            ),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        REPORT_DIR / "clean_validation_thresholds.csv",
        index=False,
    )

    best_idx = threshold_df["F1_Score"].idxmax()
    best_threshold = float(
        threshold_df.loc[best_idx, "Threshold"]
    )

    print("\nBest validation threshold:", best_threshold)
    print(
        threshold_df.loc[best_idx].to_string()
    )

    # 6. Evaluate once on the held-out test set.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Rows_After_Deduplication": len(df),
        "Threshold_Selected_On_Validation": best_threshold,
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, test_predictions, zero_division=0
        ),
        "PR_AUC_Average_Precision": average_precision_score(
            y_test, test_probabilities
        ),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Test_Transactions": len(y_test),
        "Test_Fraud_Cases": int(y_test.sum()),
    }

    print("\n" + "=" * 65)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 65)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"],
    ).to_csv(
        REPORT_DIR / "clean_final_test_metrics.csv",
        index=False,
    )

    # Save this model separately so the earlier model is preserved.
    joblib.dump(
        model,
        MODEL_DIR / "fraud_random_forest_clean.joblib",
    )

    print("\nMetrics saved to reports/clean_final_test_metrics.csv")
    print("Threshold results saved to reports/clean_validation_thresholds.csv")
    print("Model saved to models/fraud_random_forest_clean.joblib")
    print("\nClean evaluation completed!")


if __name__ == "__main__":
    main()
