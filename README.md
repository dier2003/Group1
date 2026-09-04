# Customer Risk Prediction

An end-to-end machine learning classification project for identifying high-risk customers from demographic and behavioral data.

## Project Overview

A financial company needs a reliable way to identify customers who are likely to be high-risk. This project builds and evaluates a complete machine learning workflow using a synthetic customer dataset, with special attention to data leakage, model validation, and generalization.

**Task:** Binary Classification  
**Target:** `target` (`0 = Low Risk`, `1 = High Risk`)  
**Dataset size:** 5,000 samples  
**Original features:** age, income, transactions, risk_score

## Key Challenge: Data Leakage

During model investigation, the `risk_score` feature was discovered to be directly related to how the target variable was constructed. Keeping it produced unrealistically perfect model performance.

To build a more trustworthy model, `risk_score` was removed and the models were retrained using genuine predictive features.

## Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 94.9% | 94.3% | 95.6% | 94.9% | 0.990 |
| Decision Tree | 98.5% | 98.0% | 99.0% | 98.5% | 0.985 |
| Random Forest | **99.0%** | **98.8%** | **99.2%** | **99.0%** | **0.999** |
| SVM | 98.8% | 98.4% | 99.2% | 98.8% | 0.9996 |

**Selected model:** Random Forest

## Generalization Test

The final model was also tested on a completely new synthetic dataset generated with a different random seed.

- Original test accuracy: **99.0%**
- New unseen data accuracy: **97.95%**
- Generalization gap: **1.05%**

This additional test helps demonstrate that the final model is not simply memorizing the original split.

## Key Findings

- `transactions` was the strongest genuine predictor, with a reported correlation of approximately `0.807` with the target.
- Extremely high performance should always trigger investigation for leakage.
- Testing on unseen data provides stronger evidence of generalization than relying on a single train/test split.
- Feature investigation is as important as model selection.

## Repository Structure

```text
Group1/
├── app/          # Application / deployment files
├── data/         # Project datasets
├── models/       # Saved model artifacts
├── notebooks/    # Exploration and model-development notebooks
├── src/          # Supporting source code
├── requirement.txt
└── README.md
```

## Tools & Technologies

Python · Pandas · NumPy · Scikit-learn · Machine Learning · Classification · Model Evaluation

## What This Project Demonstrates

- Exploratory data analysis
- Feature investigation
- Data leakage detection
- Model comparison
- Classification metrics
- Generalization testing
- Reproducible ML project organization

## Author

**Dier Focus**  
GitHub: [@dier2003](https://github.com/dier2003)
