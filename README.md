# Financial Fraud Detection System Using Machine Learning

## 1. Project Overview

The Financial Fraud Detection System is a machine learning project that identifies potentially fraudulent credit card transactions. It uses a Random Forest classifier to predict whether a transaction should be flagged for fraud review.

The project also compares multiple classification models, investigates duplicate records, tunes classification thresholds, and evaluates model performance on a separate test dataset.

## 2. Objectives

- Analyze credit card transaction data.
- Identify class imbalance and duplicate records.
- Train and compare Logistic Regression and Random Forest models.
- Evaluate precision, recall, F1-score, and PR-AUC.
- Select a classification threshold using validation data.
- Generate predictions for individual transactions.
- Save trained models and evaluation reports.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- CSV datasets
- Visual Studio Code

## 4. Dataset

The project uses the Credit Card Fraud Detection dataset containing transaction features and a target column named `Class`.

- Original transactions: 284,807
- Original features: 30
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Exact duplicate copies identified: 1,081
- Rows after exact deduplication: 283,726

Target labels:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

## 5. Machine Learning Models

### Logistic Regression

The baseline model uses class weighting to help address the imbalanced target classes.

### Random Forest

The Random Forest classifier uses multiple decision trees to identify potentially fraudulent transactions.

### Threshold Tuning

Different classification thresholds were tested to understand the trade-off between precision, recall, and false positives.

## 6. Final Clean Evaluation

The clean Random Forest model was trained on deduplicated data. A validation split was used to select the threshold, and a separate held-out test set was used for final evaluation.

| Metric | Test Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 94.44% |
| Recall | 71.58% |
| F1-score | 0.8144 |
| PR-AUC (Average Precision) | 0.8088 |
| True Positives | 68 |
| False Positives | 4 |
| False Negatives | 27 |
| True Negatives | 56,647 |

These results apply to the held-out test set from this experiment. They do not guarantee the same performance on future transactions.

## 7. Project Structure

```text
Financial-Fraud-Detection/
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_detection_model.joblib
│   ├── fraud_random_forest.joblib
│   └── fraud_random_forest_clean.joblib
├── reports/
│   ├── duplicate_analysis.csv
│   ├── clean_final_test_metrics.csv
│   ├── clean_validation_thresholds.csv
│   ├── threshold_comparison.csv
│   ├── model_comparison.csv
│   └── other EDA and evaluation reports
├── src/
│   ├── eda.py
│   ├── train_model.py
│   ├── train_random_forest.py
│   ├── tune_threshold.py
│   ├── predict_fraud.py
│   ├── analyze_duplicates.py
│   └── evaluate_clean_model.py
├── requirement.txt
└── README.md
```

## 8. How to Run the Project

Install the required packages:

```bash
pip install pandas scikit-learn matplotlib joblib
```

Run exploratory data analysis:

```bash
python src/eda.py
```

Train the Random Forest model:

```bash
python src/train_random_forest.py
```

Analyze duplicates:

```bash
python src/analyze_duplicates.py
```

Run the clean evaluation:

```bash
python src/evaluate_clean_model.py
```

Run the prediction program:

```bash
python src/predict_fraud.py
```

Run these commands from the project root directory.

## 9. Limitations

- Fraudulent transactions are highly underrepresented in the dataset.
- Some fraudulent transactions may be missed.
- Some legitimate transactions may be incorrectly flagged.
- Dataset performance does not guarantee real-world performance.
- Exact duplicate rows may require further investigation based on the dataset's context.
- The current model is an educational prototype, not a production banking system.

## 10. Future Improvements

- Test additional classification algorithms.
- Improve recall while controlling false positives.
- Perform cross-validation and additional robustness checks.
- Develop an interactive dashboard.
- Build an API for controlled model inference.
- Monitor performance on new, appropriately labelled transaction data.

## 11. Conclusion

This project demonstrates an end-to-end machine learning workflow for credit card fraud detection, including exploratory analysis, model training, comparison, threshold selection, duplicate analysis, and held-out evaluation. It provides a foundation for further research into fraud detection and imbalanced classification.
