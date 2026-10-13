
from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "fraud_random_forest_clean.joblib"
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"

# Selected using the validation dataset in Step 8
FRAUD_THRESHOLD = 0.50


def load_model_and_feature_names():
    """Load the clean model and original feature names."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Clean model not found: {MODEL_PATH}\n"
            "Run python src/evaluate_clean_model.py first."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH, nrows=1)
    feature_names = [
        column for column in df.columns if column != "Class"
    ]

    return model, feature_names


def predict_transaction(model, feature_names, values):
    """Return prediction and estimated fraud probability."""

    transaction = pd.DataFrame(
        [values],
        columns=feature_names
    )

    probability = model.predict_proba(transaction)[0, 1]
    prediction = int(probability >= FRAUD_THRESHOLD)

    return prediction, probability


def main():
    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION - CLEAN MODEL")
    print("=" * 60)

    model, feature_names = load_model_and_feature_names()

    print("\nClean model loaded successfully!")
    print("Required features:", len(feature_names))
    print("Decision threshold:", FRAUD_THRESHOLD)

    choice = input(
        "\nChoose mode:\n"
        "1. Test a transaction from the dataset\n"
        "2. Enter feature values manually\n"
        "Enter 1 or 2: "
    ).strip()

    actual_class = None

    if choice == "1":
        df = pd.read_csv(DATA_PATH)

        try:
            row_number = int(
                input(
                    f"Enter row number (0 to {len(df) - 1}): "
                )
            )
        except ValueError:
            print("Please enter a valid integer row number.")
            return

        if not 0 <= row_number < len(df):
            print("Invalid row number.")
            return

        row = df.iloc[row_number]
        values = row[feature_names].tolist()
        actual_class = int(row["Class"])

    elif choice == "2":
        print(
            "\nEnter all 30 numeric features in the exact order shown."
        )
        print(
            "Use values from a compatible dataset; "
            "random values are not meaningful."
        )

        values = []

        for feature in feature_names:
            while True:
                try:
                    value = float(input(f"{feature}: ").strip())
                    values.append(value)
                    break
                except ValueError:
                    print("Invalid input. Enter a numeric value.")

    else:
        print("Invalid choice. Run the program again.")
        return

    prediction, probability = predict_transaction(
        model,
        feature_names,
        values
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Estimated fraud probability: {probability:.2%}")
    print(f"Decision threshold: {FRAUD_THRESHOLD:.2f}")

    if prediction == 1:
        print("Model decision: FLAG FOR FRAUD REVIEW")
    else:
        print("Model decision: NOT FLAGGED BY THE MODEL")

    if actual_class is not None:
        print(
            "Actual dataset label:",
            "Fraud" if actual_class == 1 else "Legitimate"
        )

    print("\nA prediction is not proof of fraud.")
    print("This model is an educational prototype.")


if __name__ == "__main__":
    main()
