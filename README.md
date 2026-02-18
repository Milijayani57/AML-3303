# Customer Churn Prediction — ConnectWave Telecom

## Overview

This project builds a machine learning pipeline to predict customer churn for **ConnectWave Telecom**. By identifying customers at risk of leaving, the company can proactively deploy retention strategies such as personalized offers, loyalty programs, and service improvements.

## Problem Statement

ConnectWave Telecom has observed increasing churn, particularly among prepaid and short-term contract users. Without a data-driven approach, retention campaigns are inefficient and costly. This project addresses that gap by developing a predictive model that classifies customers as likely to churn or not, based on demographic, service usage, and billing data.


## Objectives

- Analyze customer data to identify factors that drive churn.
- Perform Exploratory Data Analysis (EDA) to uncover trends and relationships.
- Preprocess data by handling missing values, encoding categorical variables, and scaling features.
- Train and evaluate machine learning models for churn classification.
- Generate actionable business insights to help reduce churn.

## Dataset

**File:** `connectwave_customer_churn_dataset.csv`

The dataset contains **7,043 customer records** with **21 features**, including:

| Feature | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen (0/1) |
| `Partner` / `Dependents` | Household status |
| `tenure` | Months the customer has been with the company |
| `PhoneService`, `MultipleLines` | Phone-related services |
| `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV` | Internet and add-on services |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `PaperlessBilling` | Paperless billing flag |
| `PaymentMethod` | Payment method used |
| `MonthlyCharges`, `TotalCharges` | Billing amounts |
| `Churn` | Target variable — whether the customer churned (Yes/No) |

## Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualization
  - `scikit-learn` — preprocessing, modeling, evaluation
  - `requests` — Slack alert integration


## Project Workflow

1. **Import Libraries** — Load all required dependencies.
2. **Data Understanding** — Inspect shape, data types, and summary statistics.
3. **Exploratory Data Analysis (EDA)** — Visualize distributions, correlations, and churn patterns using bar charts, heatmaps, and boxplots.
4. **Data Preprocessing** — Handle missing values, encode categorical features using `LabelEncoder`, and scale numerical features with `StandardScaler`.
5. **Model Training** — Train two classifiers:
   - Logistic Regression
   - Random Forest Classifier
6. **Model Evaluation** — Assess performance using classification report, confusion matrix, ROC-AUC score, and ROC curve.
7. **Model Explainability** — Analyze feature importances to understand which factors most influence churn predictions.
8. **Slack Alerting** — Automatically send Slack notifications for high-risk customers (churn probability > 70%).


## Key Findings

- **Tenure** is one of the strongest predictors — shorter tenure correlates strongly with higher churn risk.
- **Contract type** matters significantly — month-to-month customers churn at much higher rates.
- **Monthly charges** positively correlate with churn, suggesting cost sensitivity among departing customers.
- **Service features** (e.g., online security, tech support) also influence churn behavior, highlighting the role of customer experience.


## Recommendations

- Launch **loyalty programs** targeting new customers with short tenure.
- Offer **discounted long-term contracts** to month-to-month customers at risk.
- Provide **personalized offers** to high-risk customers flagged by the model.
- **Bundle services** to increase customer engagement and switching costs.
- **Retrain the model** regularly with fresh data to maintain prediction accuracy.


## Slack Alert Integration

The notebook includes an automated alerting system that sends Slack notifications for high-risk customers:

```python
# Customers with churn probability > 70% trigger an alert
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."

def send_slack_alert(customer_id, churn_probability):
    msg = f":warning: High Churn Risk! Customer ID: {customer_id} | Probability: {churn_probability:.2%}"
    requests.post(SLACK_WEBHOOK_URL, json={"text": msg})
```

> **Note:** Replace the Slack webhook URL with your own before deploying. Do not commit live webhook URLs to version control.


## Notes

- Class imbalance (fewer churn vs. non-churn cases) was accounted for during modeling.
- Feature importance analysis was performed to support interpretability.
- The dataset simulates realistic telecom customer behavior and is suitable for business-oriented use cases.

## How to Run

1. Clone or download this repository.
2. Place `connectwave_customer_churn_dataset.csv` in the same directory as the notebook.
3. Open the notebook in Google Colab or Jupyter.
4. Run all cells in order.
5. (Optional) Update the Slack webhook URL to receive real-time churn alerts.
