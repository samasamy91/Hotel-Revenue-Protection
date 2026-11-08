# 🏨 Hotel Revenue Protection
### AI-Powered Booking Cancellation Predictor

An end-to-end Machine Learning project that predicts hotel booking cancellation risk and provides intelligent deposit recommendations to support better hotel revenue management.

[![Live App]](https://hotel-revenue-protection.streamlit.app/)

---

## Project Overview

Hotel booking cancellations create significant challenges for the hospitality industry, including lost revenue, unused room capacity, and difficulties in planning hotel operations.

**Hotel Revenue Protection** uses machine learning to estimate the probability that a hotel reservation will be canceled.

Based on the predicted cancellation probability, the application recommends one of three deposit policies:

- **No Deposit:** For low-risk bookings.
- **Partial Deposit:** For medium-risk bookings.
- **Full Deposit:** For high-risk bookings.

The project covers the complete machine learning workflow, from data cleaning and exploratory data analysis to model development, evaluation, and deployment through an interactive Streamlit application.

## Project Objectives

- Analyze historical hotel booking data to identify cancellation patterns.
- Build a machine learning model to predict booking cancellation risk.
- Compare multiple classification algorithms.
- Optimize model performance using hyperparameter tuning.
- Translate predictions into actionable deposit recommendations.
- Deploy the final model as an interactive web application.

## Dataset

**Dataset:** [Hotel Booking Demand – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

The dataset contains historical booking records from two hotel types:

- City Hotel
- Resort Hotel

| Description | Value |
|---|---:|
| Original records | 119,390 |
| Original features | 32 |
| Records after cleaning | 87,229 |
| Model input features | 23 |
| Target variable | `is_canceled` |

### Target Variable

The target is `is_canceled`:

- `0` – Booking was not canceled.
- `1` – Booking was canceled.

### Class Distribution

- Not canceled: approximately 72.5%.
- Canceled: approximately 27.5%.

  <img width="635" height="470" alt="image" src="https://github.com/user-attachments/assets/bae1760d-ed92-4bd5-8102-4434f6897937" />


## Data Cleaning and Preprocessing

The dataset was prepared through several steps:

1. Removed duplicate records.
2. Removed bookings with zero guests.
3. Removed invalid records containing negative ADR values.
4. Handled missing values.
5. Excluded columns with excessive missing values or unsuitable information.
6. Removed data leakage features.
7. Applied feature engineering.

### Preventing Data Leakage

The following features were excluded:

- `reservation_status`
- `reservation_status_date`

These fields reveal information about the final reservation outcome and would cause unrealistic model performance.

Other fields were excluded because they were unavailable at the time of the initial booking or were unsuitable for the intended recommendation workflow.

### Feature Engineering

Additional features were created:

- `total_nights`
- `total_guests`
- `arrival_day_of_week`
- `arrival_month_num`

### Preprocessing Pipeline

A Scikit-learn preprocessing pipeline was implemented using:

**Numerical features**
- Missing values: Median imputation.
- Scaling: StandardScaler.

**Categorical features**
- Missing values: Constant imputation (`Unknown`).
- Encoding: OneHotEncoder with `handle_unknown="ignore"`.

A ColumnTransformer combines both preprocessing branches.

All preprocessing is fitted using training data through the machine learning pipeline to prevent data leakage.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand booking behavior and identify patterns associated with cancellations.

The analysis included:

- Univariate analysis.
- Bivariate analysis.
- Multivariate analysis.
- Numerical feature distributions.
- Categorical feature comparisons.
- Cancellation patterns across booking characteristics.

<img width="1208" height="997" alt="image" src="https://github.com/user-attachments/assets/4d27a4d1-859d-4513-b822-4634d5991965" />

The findings helped guide feature preparation and model development.

## Machine Learning Models

Four classification approaches were evaluated:

1. Dummy Classifier – Baseline model.
2. Logistic Regression.
3. Decision Tree Classifier.
4. Random Forest Classifier.

### Model Comparison

Stratified 5-fold cross-validation was used to compare the models.

| Model | ROC-AUC | Recall | F1-score |
|---|---:|---:|---:|
| Random Forest | 0.8691 | 0.8269 | 0.6536 |
| Decision Tree | 0.8526 | 0.7784 | 0.6460 |
| Logistic Regression | 0.8272 | 0.7660 | 0.6160 |
| Dummy Classifier | 0.5000 | 0.0000 | 0.0000 |

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/826d6ade-fcee-47d0-b8e8-a6c172098b29" />

**Selected Model: Random Forest Classifier**

Random Forest achieved the strongest cross-validation ROC-AUC, recall, and F1-score among the evaluated models.

## Hyperparameter Tuning

RandomizedSearchCV was used to optimize the Random Forest model.

| Setting | Value |
|---|---|
| Search method | RandomizedSearchCV |
| Number of sampled combinations | 10 |
| Cross-validation folds | 3 |
| Optimization metric | ROC-AUC |
| Best CV ROC-AUC | 0.8798 |

### Best Parameters

```python
{
    "n_estimators": 150,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "max_depth": None
}
```

Hyperparameter tuning was performed using training data only. The test set remained untouched until final evaluation.

## Final Model Evaluation

The tuned model was evaluated on the unseen test set.

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 81.73% |
| Precision | 64.30% |
| Recall | 75.57% |
| F1-score | 69.48% |
| ROC-AUC | 0.8897 |

### Confusion Matrix

| | Predicted: Not Canceled | Predicted: Canceled |
|---|---:|---:|
| Actual: Not Canceled | 10,629 | 2,015 |
| Actual: Canceled | 1,173 | 3,629 |

<img width="625" height="470" alt="image" src="https://github.com/user-attachments/assets/01234e51-10d4-41a2-9e10-ef3d73c5c78a" />


### Interpretation

- The model correctly identifies approximately 75.57% of actual cancellations.
- When the model predicts cancellation, approximately 64.30% of those predictions are correct.
- The ROC-AUC score of 0.8897 indicates strong ability to distinguish between canceled and non-canceled bookings.

## Intelligent Deposit Recommendation

The system converts cancellation probabilities into deposit recommendations using business-defined risk thresholds.

| Predicted Risk | Probability | Recommendation |
|---|---|---|
| Low Risk | Below 30% | No Deposit |
| Medium Risk | 30% to below 60% | Partial Deposit |
| High Risk | 60% or higher | Full Deposit |

### Python Implementation

```python
def recommend_deposit(probability):

    if probability < 0.30:
        return "No Deposit"

    elif probability < 0.60:
        return "Partial Deposit"

    else:
        return "Full Deposit"
```

### Test Set Risk Analysis

| Recommendation | Bookings | Actual Cancellation Rate |
|---|---:|---:|
| No Deposit | 9,049 | 4.85% |
| Partial Deposit | 3,871 | 31.23% |
| Full Deposit | 4,526 | 69.69% |

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/98098b63-5128-4fc3-8fce-8d6fd42441eb" />

These results demonstrate that the model separates bookings into groups with different historical cancellation risks.

**Important:** These recommendations are decision-support suggestions. The analysis does not establish that collecting deposits directly prevents cancellations or guarantees revenue protection.

## Live Web Application

The trained model is deployed using Streamlit.

**Try the application:**

 [Hotel Revenue Protection – Live Demo](https://hotel-revenue-protection.streamlit.app/)

### Application Features

- Interactive hotel booking input form.
- Cancellation probability prediction.
- Predicted booking status.
- Cancellation risk classification.
- Intelligent deposit recommendation.
- User-friendly interface.

### Example Output

The application displays:

- Cancellation Probability
- Predicted Status
- Risk Level
- Recommended Deposit Policy

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Model serialization |
| Streamlit | Web application |
| Jupyter Notebook | Experimentation and development |
| Git & GitHub | Version control and project hosting |

## Project Structure

```text
Hotel-Revenue-Protection/
│
├── app.py
├── hotel_cancellation_model.pkl
├── deposit_thresholds.pkl
├── requirements.txt
├── README.md
└── hotel_cancellation_prediction.ipynb
```

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/samasamy91/Hotel-Revenue-Protection.git
```

### 2. Navigate to the Project Directory

```bash
cd Hotel-Revenue-Protection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Limitations and Future Improvements

### Current Limitations

- The model is trained on historical booking data, which may not fully represent current hotel booking behavior.
- Some potentially useful features were excluded due to missingness or prediction-time availability.
- Predictions depend on the accuracy and availability of booking information.
- Deposit recommendations are based on selected business thresholds and are not guaranteed to reduce cancellations.

### Future Improvements

- Train and validate the model using newer booking data.
- Add explainable AI techniques to identify the main factors influencing predictions.
- Optimize deposit thresholds using financial costs and business objectives.
- Monitor model performance and data drift.
- Integrate real-time hotel reservation systems.
- Evaluate deposit strategies through controlled experiments.

## Disclaimer

Predictions are based on historical booking patterns and should not be treated as guarantees of future customer behavior or financial outcomes.
