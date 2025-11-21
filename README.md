# Credit-Default-Early-Warning-System--ML-Based-Risk-Scoring
Built using GiveMeSomeCredit Dataset 

# Project Overview

This project builds a Machine Learning–based Credit Default Early Warning System designed to help lenders, NBFCs, and financial institutions identify high-risk borrowers before loan issuance. Using the GiveMeSomeCredit consumer finance dataset, the model predicts whether a borrower is likely to become seriously delinquent within the next 2 years.

The system utilizes behavioral credit indicators, such as past delinquencies, credit utilization, age, and debt ratios, to generate a probability score for default.

## Key Features

1. End-to-end ML pipeline: preprocessing, EDA, feature engineering, modeling
2. Random Forest–based credit risk scoring model
3. Handles missing values, outliers, and class imbalance
4. Produces a submission-ready prediction file for Kaggle
5. Includes detailed EDA insights
6. Identifies top drivers of borrower default
7. Generates default probability for each applicant

# Dataset

Source: Kaggle, Give Me Some Credit
Files used:
- cs-training.csv
- cs-test.csv
- Data Dictionary.xls

Dataset includes:
Revolving credit utilization, 
Debt ratio, 
Delinquency history (30–59, 60–89, 90+ days late),
Monthly income,
Age,
Number of dependents
Target variable: SeriousDlqin2yrs (1 = default, 0 = non-default)

# Exploratory Data Analysis
1️. Target Imbalance: 
Only 6.7% of borrowers defaulted, highly imbalanced classification problem.

2️.  Missing Values: 
MonthlyIncome and NumberOfDependents had missing values → handled via median imputation.

3️. Outliers:
Significant outliers in:
- RevolvingUtilizationOfUnsecuredLines
- DebtRatio
- Delinquency counts
These represent real credit behavior, so they were retained.

4️.  Feature Importance (Top Predictors):
The Random Forest model highlighted the following as strongest signals:

# Feature	Why it matters
- RevolvingUtilizationOfUnsecuredLines	High utilization → high credit stress
- NumberOfTimes90DaysLate	Severe past delinquency predicts future default
- NumberOfTime30-59DaysPastDueNotWorse	Repeated mild delays show poor financial discipline
- NumberOfTime60-89DaysPastDueNotWorse	Medium-risk behavioral pattern
- age	Younger borrowers show higher default tendencies

These insights match real-world NBFC underwriting patterns.

# Modeling Approach
# 1. Algorithms Tested

- Logistic Regression (baseline)

- Random Forest Classifier (final model)

# 2. Why Random Forest?
- Performs well on imbalanced datasets

- Captures non-linear relationships

- Robust to outliers

- Provides interpretable feature importance

# 3. Final Model Performance

- ROC-AUC: ~0.86–0.89

- Improved recall on high-risk borrowers via class balancing

- Strong early warning capability

