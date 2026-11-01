import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from source.config import FIGURES_DIR, MODEL_PATH, REPORTS_DIR
from source.inference import (
    customer_shap_reasons,
    load_model_bundle,
    predict_batch,
    retention_suggestions,
    risk_level,
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_model_bundle(MODEL_PATH)


try:
    bundle = get_bundle()
except FileNotFoundError:
    bundle = None

st.title("Customer Churn Prediction System")
st.caption(
    "Estimate churn probability, review model factors and prepare "
    "a customer-retention follow-up."
)

if bundle is None:
    st.warning(
        "Trained model not found. Add the dataset and run "
        "`python scripts/train.py` first."
    )
else:
    st.sidebar.success(
        f"Model ready: {bundle.get('model_name', 'trained model')}"
    )
    st.sidebar.metric(
        "Decision threshold",
        f"{float(bundle.get('threshold', 0.5)):.2f}",
    )

single_tab, batch_tab, dashboard_tab, about_tab = st.tabs([
    "Single customer",
    "CSV predictions",
    "Dashboard",
    "About",
])

with single_tab:
    st.subheader("Assess one customer")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"],
            )

        with col2:
            internet = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
            )
            security = st.selectbox(
                "Online security",
                ["No", "Yes", "No internet service"],
            )
            backup = st.selectbox(
                "Online backup",
                ["No", "Yes", "No internet service"],
            )
            device = st.selectbox(
                "Device protection",
                ["No", "Yes", "No internet service"],
            )
            support = st.selectbox(
                "Tech support",
                ["No", "Yes", "No internet service"],
            )
            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            movies = st.selectbox(
                "Streaming movies",
                ["No", "Yes", "No internet service"],
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless = st.selectbox("Paperless billing", ["No", "Yes"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )
            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0,
                step=10.0,
            )

        submitted = st.form_submit_button(
            "Assess churn risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if internet == "No":
            security = backup = device = support = tv = movies = (
                "No internet service"
            )
        if phone == "No":
            multiple = "No phone service"

        customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        customer_frame = pd.DataFrame([customer])

        if bundle is None:
            st.error("Train the model before making predictions.")
        else:
            try:
                with st.spinner("Scoring customer and preparing explanation..."):
                    probability = float(
                        bundle["pipeline"].predict_proba(customer_frame)[0, 1]
                    )
                    threshold = float(bundle.get("threshold", 0.5))
                    level = risk_level(probability)

                    try:
                        reasons = customer_shap_reasons(
                            bundle, customer_frame
                        )
                        explanation_error = None
                    except Exception as exc:
                        reasons = None
                        explanation_error = str(exc)

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Churn probability", f"{probability:.1%}")
                metric2.metric("Risk level", level)
                metric3.metric(
                    "Model decision",
                    "Flag for follow-up"
                    if probability >= threshold
                    else "Not flagged",
                )

                st.progress(max(0.0, min(1.0, probability)))
                st.caption(
                    f"Decision threshold: {threshold:.3f}. "
                    "Risk bands: Low < 30%, Medium 30–60%, High ≥ 60%."
                )

                if level == "High":
                    st.error(
                        "High estimated risk. Consider a respectful customer "
                        "check-in; this prediction is not a certainty."
                    )
                elif level == "Medium":
                    st.warning(
                        "Medium estimated risk. Consider monitoring the account."
                    )
                else:
                    st.success(
                        "Low estimated risk under the configured risk bands."
                    )

                st.subheader("Individual customer explanation")
                st.caption(
                    "SHAP values show model contribution, not causal effects "
                    "or percentage-point changes."
                )

                if reasons is None:
                    st.info(
                        "Prediction succeeded, but SHAP was unavailable: "
                        f"{explanation_error}"
                    )
                else:
                    reason1, reason2 = st.columns(2)

                    with reason1:
                        st.markdown("**Factors pushing toward churn**")
                        for item in reasons["risk_factors"]:
                            st.write(
                                f"• {item['feature']}: "
                                f"SHAP {item['impact']:+.3f}"
                            )
                        if not reasons["risk_factors"]:
                            st.write("No positive SHAP factors returned.")

                    with reason2:
                        st.markdown("**Factors pushing away from churn**")
                        for item in reasons["protective_factors"]:
                            st.write(
                                f"• {item['feature']}: "
                                f"SHAP {item['impact']:+.3f}"
                            )
                        if not reasons["protective_factors"]:
                            st.write("No negative SHAP factors returned.")

                if level == "High":
                    st.subheader("Possible retention actions")
                    for suggestion in retention_suggestions(customer):
                        st.write(f"• {suggestion}")

            except Exception as exc:
                st.error(f"Could not score this customer: {exc}")

with batch_tab:
    st.subheader("Batch predictions from CSV")
    st.write(
        "Upload a CSV with the Telco input columns. If a `Churn` label is "
        "present, it is not used for prediction."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)
            st.caption(f"Uploaded rows: {len(source_df):,}")

            if st.button("Generate predictions", type="primary"):
                if bundle is None:
                    st.error("Train the model before making predictions.")
                else:
                    with st.spinner("Predicting uploaded customers..."):
                        st.session_state["batch_result"] = predict_batch(
                            bundle,
                            source_df,
                        )

            if "batch_result" in st.session_state:
                result = st.session_state["batch_result"]
                st.dataframe(result.head(100), use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")

with dashboard_tab:
    st.subheader("Model performance and EDA dashboard")

    if bundle is not None:
        metrics = bundle.get("test_metrics", {})
        columns = st.columns(5)
        for column, label, key in zip(
            columns,
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
        ):
            value = metrics.get(key)
            column.metric(
                label,
                f"{value:.3f}"
                if isinstance(value, (int, float))
                else "N/A",
            )

        st.caption(
            "Held-out test metrics using a threshold selected on validation data."
        )

        for filename, caption in [
            ("model_comparison.png", "Cross-validation model comparison"),
            ("confusion_matrix.png", "Held-out test confusion matrix"),
            ("shap_global_importance.png", "Global SHAP importance"),
        ]:
            image_path = FIGURES_DIR / filename
            if image_path.exists():
                st.image(
                    str(image_path),
                    caption=caption,
                    use_container_width=True,
                )

    manifest = REPORTS_DIR / "eda_summary.json"
    if manifest.exists():
        try:
            items = json.loads(manifest.read_text(encoding="utf-8"))
            st.markdown("### Exploratory charts and insights")
            for item in items:
                st.markdown(f"**{item['title']}**")
                image_path = FIGURES_DIR / item["image"]
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                st.write(f"Insight: {item['insight']}")
        except Exception as exc:
            st.info(f"Could not load EDA report: {exc}")
    else:
        st.info("Generate EDA charts with `python scripts/eda.py`.")

with about_tab:
    st.subheader("Workflow")
    st.markdown(
        """
        1. Clean the Telco CSV and convert blank charges to missing values.
        2. Engineer tenure, service-count and charge-related features.
        3. Fit imputation, scaling and encoding inside an sklearn pipeline.
        4. Compare five models and tune the two leading candidates.
        5. Select the operating threshold using validation data.
        6. Report final metrics on a separate held-out test set.
        7. Explain global and individual predictions with SHAP.
        """
    )
    st.markdown("### Limitations")
    st.write(
        "This educational model uses one public telecom sample dataset. Its "
        "probabilities are estimates, not certainty. SHAP values describe model "
        "associations, not causes. Retention suggestions are examples and "
        "should be tested and reviewed for fairness before business use."
    )
