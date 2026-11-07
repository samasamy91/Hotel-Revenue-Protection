
# Hotel Revenue Protection: Cancellation Predictor

An AI-powered hotel booking cancellation prediction system that helps hotels identify cancellation risks and recommend suitable deposit policies.

## Problem Statement

Hotel cancellations can cause revenue losses and make room and occupancy planning difficult.

This project predicts the probability of booking cancellation and recommends a deposit policy based on the estimated risk.

## Dataset

Hotel Booking Demand Dataset from Kaggle.

- Original records: Approximately 119,390
- Original columns: 32
- Target: `is_canceled`

## Machine Learning Models

- Dummy Classifier (Baseline)
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest was selected and tuned using RandomizedSearchCV.

## Test Results

- Accuracy: 81.73%
- Precision: 64.30%
- Recall: 75.57%
- F1-Score: 69.48%
- ROC-AUC: 0.8897

## Deposit Recommendations

| Cancellation Risk | Recommendation |
|---                |---             |
| Below 30%         | No Deposit     |
| 30% to below 60%  | Partial Deposit|
| 60% or above      | Full Deposit   |

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Web Application

The application allows hotel employees to enter booking details and receive a cancellation probability, predicted status, and deposit recommendation.

## Limitations

The model learns from historical booking patterns. Its recommendations do not guarantee that deposits will prevent cancellations or protect a specific amount of revenue.
