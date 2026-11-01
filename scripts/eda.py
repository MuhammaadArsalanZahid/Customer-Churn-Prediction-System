import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from source.config import DATA_PATH, FIGURES_DIR, REPORTS_DIR
from source.data import load_telco_data

sns.set_theme(style="whitegrid")


def save_group_chart(df, column, title, filename):
    work = df[[column, "Churn"]].copy()
    work["_group"] = work[column].fillna("Unknown").astype(str)

    if column == "SeniorCitizen":
        work["_group"] = work["_group"].replace({"0": "No", "1": "Yes"})

    rates = work.groupby("_group", dropna=False)["Churn"].mean()
    rates = rates.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(rates.index.astype(str), rates.values, color="#536dfe")
    ax.set_title(title)
    ax.set_ylabel("Observed churn rate")
    ax.set_xlabel("")
    ax.set_ylim(0, min(1.0, max(0.5, float(rates.max()) + 0.1)))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

    if rates.empty:
        insight = "No valid category values were available for this chart."
    else:
        insight = (
            f"Observed churn is highest for **{rates.index[0]}** "
            f"({rates.iloc[0]:.1%}) and lowest for **{rates.index[-1]}** "
            f"({rates.iloc[-1]:.1%}). This is an association, not proof of cause."
        )

    return {"title": title, "image": filename, "insight": insight}


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df, cleaning = load_telco_data(DATA_PATH)
    items = []

    counts = df["Churn"].value_counts().reindex([0, 1], fill_value=0)
    churn_rate = float(df["Churn"].mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["No churn", "Churn"],
        counts.values,
        color=["#26a69a", "#ef5350"],
    )
    ax.set_title("Customer churn distribution")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    filename = "eda_01_churn_distribution.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

    items.append({
        "title": "Customer churn distribution",
        "image": filename,
        "insight": (
            f"{churn_rate:.1%} of cleaned rows are labelled as churn. "
            "This is why recall, precision and ROC-AUC are reported alongside "
            "accuracy."
        ),
    })

    items.append(save_group_chart(
        df, "Contract", "Churn rate by contract type", "eda_02_contract.png"
    ))

    tenure_df = df.copy()
    tenure_df["TenureGroup"] = pd.cut(
        tenure_df["tenure"],
        bins=[-np.inf, 12, 24, 48, 60, np.inf],
        labels=[
            "0-12 months", "13-24 months", "25-48 months",
            "49-60 months", "61+ months",
        ],
    ).astype(object)
    items.append(save_group_chart(
        tenure_df, "TenureGroup", "Churn rate by tenure",
        "eda_03_tenure.png",
    ))

    charges = df[["MonthlyCharges", "Churn"]].dropna().copy()
    charges["Outcome"] = charges["Churn"].map({
        0: "No churn", 1: "Churn"
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=charges, x="Outcome", y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly charges by churn outcome")
    fig.tight_layout()
    filename = "eda_04_monthly_charges.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

    medians = charges.groupby("Outcome")["MonthlyCharges"].median()
    items.append({
        "title": "Monthly charges by churn outcome",
        "image": filename,
        "insight": (
            f"Median monthly charges are "
            f"{medians.get('Churn', float('nan')):.2f} for churned customers "
            f"and {medians.get('No churn', float('nan')):.2f} for retained "
            "customers. This is descriptive, not causal."
        ),
    })

    items.append(save_group_chart(
        df, "InternetService", "Churn rate by internet service",
        "eda_05_internet_service.png",
    ))
    items.append(save_group_chart(
        df, "PaymentMethod", "Churn rate by payment method",
        "eda_06_payment_method.png",
    ))
    items.append(save_group_chart(
        df, "TechSupport", "Churn rate by technical support",
        "eda_07_tech_support.png",
    ))
    items.append(save_group_chart(
        df, "SeniorCitizen", "Churn rate by senior-citizen indicator",
        "eda_08_senior_citizen.png",
    ))

    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(items, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Exploratory Data Analysis",
        "",
        f"- Original rows: {cleaning['original_rows']}",
        f"- Duplicate rows removed: {cleaning['duplicate_rows_removed']}",
        f"- Rows without a valid target removed: "
        f"{cleaning['rows_without_valid_target_removed']}",
        f"- Clean rows: {cleaning['clean_rows']}",
        f"- Missing TotalCharges kept for pipeline imputation: "
        f"{cleaning['missing_total_charges']}",
        "",
    ]
    for item in items:
        lines.extend([
            f"## {item['title']}",
            "",
            f"![{item['title']}](figures/{item['image']})",
            "",
            f"**Insight:** {item['insight']}",
            "",
        ])

    (REPORTS_DIR / "eda_insights.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Created {len(items)} charts in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
