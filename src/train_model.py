
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def main():
    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' is missing.")

    if df.isnull().any().any():
        raise ValueError(
            "Missing values found. Investigate before training."
        )

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    print("\nDataset loaded:", df.shape)

    # We are deliberately retaining duplicate rows for now.
    # Their cause should be investigated before removal.


    # --------------------------------------------------
    # 3. FEATURES AND TARGET
    # --------------------------------------------------

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("\nFeatures:", X.shape[1])
    print("Legitimate transactions:", int((y == 0).sum()))
    print("Fraudulent transactions:", int((y == 1).sum()))


    # --------------------------------------------------
    # 4. TRAIN-TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining transactions:", len(X_train))
    print("Testing transactions:", len(X_test))
    print("Training fraud cases:", int(y_train.sum()))
    print("Testing fraud cases:", int(y_test.sum()))


    # --------------------------------------------------
    # 5. PREPROCESSING
    # --------------------------------------------------

    # Time and Amount have different numerical scales.
    # The V1-V28 columns are already PCA-transformed.

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scale_time_amount",
                StandardScaler(),
                ["Time", "Amount"]
            )
        ],
        remainder="passthrough"
    )


    # --------------------------------------------------
    # 6. CREATE MODEL PIPELINE
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )


    # --------------------------------------------------
    # 7. TRAIN MODEL
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    print("Model training completed!")


    # --------------------------------------------------
    # 8. PREDICTIONS
    # --------------------------------------------------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------
    # 9. EVALUATION METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    pr_auc = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Breakdown:")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)


    # --------------------------------------------------
    # 10. SIMPLE BASELINE COMPARISON
    # --------------------------------------------------

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_pred
    )
    baseline_recall = recall_score(
        y_test, baseline_pred, zero_division=0
    )

    print("\nMajority-Class Baseline:")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Fraud Recall: {baseline_recall:.4f}")


    # --------------------------------------------------
    # 11. SAVE CONFUSION MATRIX PLOT
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot(values_format="d")
    plt.title("Fraud Detection - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=150
    )
    plt.close()


    # --------------------------------------------------
    # 12. SAVE METRICS
    # --------------------------------------------------

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "PR_AUC_Average_Precision": pr_auc,
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Baseline_Accuracy": baseline_accuracy,
        "Baseline_Recall": baseline_recall
    }

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )

    metrics_df.to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False
    )

    with open(
        REPORT_DIR / "model_report.txt",
        "w",
        encoding="utf-8"
    ) as report:
        report.write("FINANCIAL FRAUD DETECTION\n")
        report.write("Logistic Regression Baseline\n\n")

        for name, value in metrics.items():
            report.write(f"{name}: {value}\n")

        report.write("\nClassification Report:\n")
        report.write(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                zero_division=0
            )
        )


    # --------------------------------------------------
    # 13. SAVE TRAINED MODEL
    # --------------------------------------------------

    model_path = MODEL_DIR / "fraud_detection_model.joblib"

    joblib.dump(model, model_path)

    print("\nModel saved to:", model_path)
    print("Metrics saved to:", REPORT_DIR / "model_metrics.csv")
    print("Report saved to:", REPORT_DIR / "model_report.txt")
    print("Confusion matrix saved to:", REPORT_DIR / "confusion_matrix.png")

    print("\nModel training and evaluation completed!")


if __name__ == "__main__":
    main()
