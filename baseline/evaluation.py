from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    out = {"Accuracy": acc, "F1_macro": f1}
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            out["AUC"] = auc
        except Exception:
            pass
    return out


def calibration_plot(y_true, y_prob, out_path: str, n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=1)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = float("nan")
    return brier