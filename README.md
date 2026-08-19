Group 1 : **Customer Risk Prediction**

Machine Learning Weekend Challenge Group 1

&#x20;Business Scenario

A financial company wants to identify customers who are likely to be high-risk. This project develops a complete machine learning pipeline from data investigation to model deployment using a synthetic customer dataset.



&#x20; **Problem Definition**

Task Type: **Binary Classification**

Target Variable: target (0 = Low Risk, 1 = High Risk)

Goal: Predict whether a customer belongs to the high-risk category based on their demographic and behavioral data



Datasets

properties: value

Samples: 5000

Features: age, income, transactions, risk\_score

Targets: target(balanced: 2500 by 2500)

Source: Synthetically generated

Key Investigation Findings

1\. Data Leakage Discovered

The risk\_score feature was used directly to construct the target variable:

Code

This caused Decision Tree and Random Forest to achieve an unrealistic 100% accuracy.



2\. Correction Applied

risk\_score was removed from the feature set. Models were retrained using only age, income\_log, and transactions.

3\. Realistic Performance After Correction

Model for this parameters:

Accuracy

Precision

Recall

F1

ROC-AUC

Logistic Regression

94.9%

94.3%

95.6%

94.9%

0.990

Decision Tree

98.5%

98.0%

99.0%

98.5%

0.985

Random Forest 

99.0%

98.8%

99.2%

99.0%

0.999

SVM

98.8%

98.4%

99.2%

98.8%

0.9996

4\. Strongest True Predictor

transactions showed the highest correlation with the target (r = 0.807), independently confirmed through EDA — not from the leaked feature.

5\. Generalization Test

The final model was evaluated on a completely new, unseen synthetic dataset (different random seed):

Original test accuracy: 99.0%

New unseen data accuracy: 97.95%

Generalization gap: only 1.05% → model generalizes well

&#x20;Models Trained

Logistic Regression

Decision Tree

Random Forest  (Final selected model)

Support Vector Machine (SVM)



Lessons Learned

Always investigate unusually high model performance before trusting it

Data leakage can silently inflate accuracy without any coding error

Cross-validation and testing on completely unseen data are essential for verifying true generalization

The strongest correlated feature (transactions) was not the same as the leakage source (risk\_score) — both had to be investigated separately

