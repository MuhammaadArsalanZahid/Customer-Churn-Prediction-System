# Customer Churn Prediction System

**Demo video:** https://drive.google.com/file/d/17e81ELdX2EbOKE8Zh7HFpv7S3YclLbWx/view?usp=drive_link

An end-to-end machine-learning app that estimates telecom customer churn
probability, explains model factors and supports batch CSV predictions.

## Features

- Eight exploratory charts with written, data-derived insights.
- Data cleaning, duplicate removal and missing `TotalCharges` handling.
- Feature engineering: tenure group, services used, average charge per month,
  and monthly charge per service.
- Preprocessing inside a scikit-learn pipeline.
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost.
- Class weights and XGBoost positive-class weighting for class imbalance.
- Stratified cross-validation and tuning of the two leading baseline models.
- Validation-selected decision threshold and held-out test evaluation.
- Global and individual-customer SHAP explanations.
- Streamlit single-customer form, CSV upload/download and dashboard.
- Automated tests for feature engineering, pipeline predictions and thresholding.

## Workflow

CSV → cleaning → stratified fit/validation/test split → feature engineering
and preprocessing → model comparison and tuning → validation threshold
selection → held-out test evaluation → explanations and Streamlit app.

## Tools

Python 3.10 or 3.11, pandas, scikit-learn, XGBoost, SHAP, Streamlit,
matplotlib and seaborn.

## Dataset

Source: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download the CSV to `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. The dataset
is not included here. Verify the current Kaggle license and terms before use,
and document the verified license before publishing.

## Setup

Create and activate a virtual environment:

    python -m venv .venv

Windows:

    .venv\Scripts\activate

macOS/Linux:

    source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Generate EDA charts and insights:

    python scripts/eda.py

Train and evaluate the models:

    python scripts/train.py

Run automated tests and start the app:

    pytest -q
    streamlit run app.py

Training creates `models/churn_model.joblib` and reports under `reports/`.

## Results

No scores are hard-coded. Run the training script on the downloaded dataset
to generate actual results. Review:

- `reports/final_metrics.json`
- `reports/model_comparison.csv`
- `reports/classification_report.txt`
- `reports/figures/`
- `reports/shap_global_importance.csv`

Add the actual generated metrics here before publishing. The final test metrics
use a threshold selected on validation data. Recall matters when the business
wants to identify more potential churners; precision matters because unnecessary
retention outreach also has a cost.

## Limitations

- This is one telecom sample dataset and may not generalize to another company.
- Churn probability is an estimate, not certainty.
- SHAP describes how the fitted model uses features; it does not prove causation.
- Retention suggestions are examples and should be tested before business use.
- Review dataset licensing, privacy and fairness before deployment.

## Internship attribution

**Built during the AI/ML Internship at Taviq AI Solutions**

Name: Muhammad Arsalan Zahid  
LinkedIn: https://www.linkedin.com/in/muhammad-arsalan-zahid-aa255b330/  
GitHub: https://github.com/MuhammaadArsalanZahid/Customer-Churn-Prediction-System/
 Demo video: https://drive.google.com/file/d/17e81ELdX2EbOKE8Zh7HFpv7S3YclLbWx/view?usp=drive_link
