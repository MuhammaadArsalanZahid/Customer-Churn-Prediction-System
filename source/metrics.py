import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_business_threshold(y_true, probabilities, min_precision=0.35):
    """Choose the threshold on validation data, prioritizing recall."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5, {"method": "fallback: no thresholds available"}

    valid = np.flatnonzero(precision[:-1] >= min_precision)
    if len(valid):
        index = valid[np.argmax(recall[valid])]
        method = (
            f"maximum recall with validation precision >= {min_precision:.2f}"
        )
    else:
        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        index = int(np.argmax(f1))
        method = "fallback: maximum validation F1"

    return float(thresholds[index]), {
        "method": method,
        "validation_precision_floor": float(min_precision),
    }


def evaluate_probabilities(y_true, probabilities, threshold):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": auc,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }
