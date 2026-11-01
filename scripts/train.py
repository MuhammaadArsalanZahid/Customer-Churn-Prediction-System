import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from source.config import (
    DATA_PATH,
    FIGURES_DIR,
    MODEL_PATH,
    MODELS_DIR,
    RAW_FEATURE_COLUMNS,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from source.data import load_telco_data
from source.explain import calculate_shap_values, transformed_features
from source.metrics import choose_business_threshold, evaluate_probabilities
from source.modeling import make_pipeline, model_catalog, parameter_grids


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument(
        "--min-validation-precision",
        type=float,
        default=0.35,
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, cleaning = load_telco_data(args.data)
    X = df.loc[:, RAW_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    if y.nunique() != 2:
        raise ValueError("Training data must contain both churn classes.")

    # Stratified 60/20/20 fit, validation and held-out test split.
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        stratify=y_train_valid,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    catalog = model_catalog(y_train)
    rows = []

    for name, classifier in catalog.items():
        print(f"Cross-validating {name}...")
        scores = cross_validate(
            make_pipeline(classifier),
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values))
        row["tuned_cv_f1"] = np.nan
        row["tuned_parameters"] = ""
        rows.append(row)

    top_names = [
        row["model"]
        for row in sorted(
            rows,
            key=lambda item: (
                item["mean_f1"],
                item["mean_recall"],
            ),
            reverse=True,
        )[:2]
    ]

    tuned = []
    grids = parameter_grids()

    for name in top_names:
        print(f"Tuning {name}...")
        search = GridSearchCV(
            make_pipeline(catalog[name]),
            grids[name],
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        tuned.append({
            "name": name,
            "pipeline": search.best_estimator_,
            "cv_f1": float(search.best_score_),
            "parameters": search.best_params_,
        })

        for row in rows:
            if row["model"] == name:
                row["tuned_cv_f1"] = float(search.best_score_)
                row["tuned_parameters"] = json.dumps(
                    search.best_params_,
                    sort_keys=True,
                )

    selected = max(tuned, key=lambda item: item["cv_f1"])
    pipeline = selected["pipeline"]

    # Choose the decision threshold using validation data only.
    validation_probabilities = pipeline.predict_proba(X_valid)[:, 1]
    threshold, threshold_policy = choose_business_threshold(
        y_valid,
        validation_probabilities,
        min_precision=args.min_validation_precision,
    )
    validation_metrics = evaluate_probabilities(
        y_valid,
        validation_probabilities,
        threshold,
    )

    # Evaluate once on the held-out test split.
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(
        y_test,
        test_probabilities,
        threshold,
    )
    test_predictions = (test_probabilities >= threshold).astype(int)

    comparison = pd.DataFrame(rows).sort_values(
        "mean_f1",
        ascending=False,
    )
    comparison.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )

    ordered = comparison.sort_values("mean_f1")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        positions - 0.18,
        ordered["mean_f1"],
        0.36,
        label="CV F1",
        color="#536dfe",
    )
    ax.barh(
        positions + 0.18,
        ordered["mean_recall"],
        0.36,
        label="CV recall",
        color="#26a69a",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean 5-fold training cross-validation score")
    ax.set_title("Candidate model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(test_metrics["confusion_matrix"]),
        display_labels=["No churn", "Churn"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Held-out test confusion matrix — {selected['name']}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
        zero_division=0,
    )
    (REPORTS_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    # Explain a validation sample with SHAP.
    background_frame = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42,
    )
    shap_background, _ = transformed_features(pipeline, background_frame)

    explain_frame = X_valid.sample(
        n=min(250, len(X_valid)),
        random_state=42,
    )
    shap_values, feature_names = calculate_shap_values(
        pipeline,
        explain_frame,
        background=shap_background,
    )

    importance = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(
        REPORTS_DIR / "shap_global_importance.csv",
        index=False,
    )

    top = importance_df.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#7e57c2")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global SHAP importance — validation sample")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=150)
    plt.close(fig)

    insight_lines = [
        "# Manager-facing model insights",
        "",
        "The features below have the largest mean absolute SHAP values in a "
        "validation sample. They describe model associations, not causes.",
        "",
    ]
    for number, feature in enumerate(
        importance_df.head(5)["feature"],
        start=1,
    ):
        insight_lines.append(
            f"{number}. Review customer journeys associated with **{feature}**. "
            "Validate any retention action with a controlled pilot before "
            "changing business policy."
        )
    insight_lines.extend([
        "",
        "SHAP values do not prove that a feature causes churn. Review fairness, "
        "customer impact and retention outcomes before deployment.",
    ])
    (REPORTS_DIR / "manager_insights.md").write_text(
        "\n".join(insight_lines),
        encoding="utf-8",
    )

    results = {
        "selected_model": selected["name"],
        "threshold": float(threshold),
        "threshold_policy": threshold_policy,
        "cleaning_summary": cleaning,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "tuned_cv_f1": selected["cv_f1"],
        "best_parameters": selected["parameters"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (REPORTS_DIR / "final_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": float(threshold),
            "model_name": selected["name"],
            "threshold_policy": threshold_policy,
            "test_metrics": test_metrics,
            "validation_metrics": validation_metrics,
            "shap_background": shap_background,
            "feature_names": feature_names,
        },
        MODEL_PATH,
        compress=3,
    )

    print(f"Selected model: {selected['name']}")
    print(f"Validation-selected threshold: {threshold:.4f}")
    print("Held-out test metrics:")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {test_metrics[metric]:.4f}")
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
