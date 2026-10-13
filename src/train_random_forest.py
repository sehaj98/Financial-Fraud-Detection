
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("RANDOM FOREST - FRAUD DETECTION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Use the same split as Logistic Regression
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training completed!")

    # Default prediction threshold is 0.5
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "PR_AUC_Average_Precision": average_precision_score(
            y_test, y_prob
        ),
    }

    print("\nRANDOM FOREST RESULTS")
    print("-" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\nTrue negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    metrics["True_Negatives"] = int(tn)
    metrics["False_Positives"] = int(fp)
    metrics["False_Negatives"] = int(fn)
    metrics["True_Positives"] = int(tp)

    pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"],
    ).to_csv(
        REPORT_DIR / "random_forest_metrics.csv",
        index=False,
    )

    model_path = MODEL_DIR / "fraud_random_forest.joblib"
    joblib.dump(model, model_path)

    print("\nModel saved to:", model_path)
    print(
        "Metrics saved to:",
        REPORT_DIR / "random_forest_metrics.csv",
    )
    print("\nRandom Forest evaluation completed!")


if __name__ == "__main__":
    main()
