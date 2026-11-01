import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression


def transformed_features(pipeline, raw_frame):
    engineered = pipeline.named_steps["features"].transform(raw_frame)
    matrix = pipeline.named_steps["preprocessor"].transform(engineered)

    if sparse.issparse(matrix):
        matrix = matrix.toarray()

    matrix = np.asarray(matrix, dtype=float)
    names = list(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    return matrix, names


def positive_class_shap_values(values, rows, columns):
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
        elif values.shape[1] == 2:
            values = values[:, 1, :]

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape == (columns, rows):
        values = values.T
    if values.shape != (rows, columns):
        raise ValueError(f"Unexpected SHAP result shape: {values.shape}")

    return values.astype(float)


def calculate_shap_values(pipeline, raw_frame, background=None):
    import shap

    matrix, names = transformed_features(pipeline, raw_frame)
    model = pipeline.named_steps["model"]

    if isinstance(model, LogisticRegression):
        reference = matrix if background is None else np.asarray(background)
        explainer = shap.LinearExplainer(model, reference)
        values = explainer.shap_values(matrix)
    else:
        explainer = shap.TreeExplainer(model)
        try:
            values = explainer.shap_values(
                matrix, check_additivity=False
            )
        except TypeError:
            values = explainer.shap_values(matrix)

    values = positive_class_shap_values(
        values, matrix.shape[0], matrix.shape[1]
    )
    return values, names
