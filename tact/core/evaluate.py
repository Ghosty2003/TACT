"""Evaluation metrics: MSE, Spearman, pairwise accuracy."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tact.core.model import load_csv, load_eval_csv  # noqa: F401

PAIR_MARGIN = 0.05


def pairwise_accuracy(pred, y, margin=PAIR_MARGIN):
    """Fraction of sample pairs with |y_i - y_j| > margin whose predicted order matches the label order.
    A tied prediction counts as wrong."""
    pred, y = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dy = y[:, None] - y[None, :]
    dp = pred[:, None] - pred[None, :]
    i, j = np.triu_indices(len(y), k=1)
    keep = np.abs(dy[i, j]) > margin
    if not keep.any():
        return float("nan")
    return float((np.sign(dy[i, j][keep]) == np.sign(dp[i, j][keep])).mean())


def metrics_from_pred(pred, y):
    pred, y = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return {
        "mse": float(np.mean((pred - y) ** 2)),
        "spearman": float(spearmanr(pred, y).statistic),
        "pairwise_acc": pairwise_accuracy(pred, y),
    }


def evaluate(model, csv):
    """csv: path (absolute, relative, or a name inside the data dir) or a DataFrame."""
    df = csv if isinstance(csv, pd.DataFrame) else load_csv(csv)
    return metrics_from_pred(model.predict(df), df["score"].values)


def habit_mask(df):
    """Rows whose label was produced by a habit rule (habit_hit non-empty). Evaluation only."""
    return df["habit_hit"].notna().values


def subset_metrics(pred, df, mask):
    """MSE / Spearman / pairwise acc on df[mask]; NaN when the subset is too small."""
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "mse": float("nan"), "spearman": float("nan"), "pairwise_acc": float("nan")}
    return {"n": n, **metrics_from_pred(np.asarray(pred)[mask], df["score"].values[mask])}


def diagnostics(model, test_full, train_full, pred_test=None):
    """Noise + habit-subset diagnostics (evaluation only; uses score_clean and habit_hit).

    test_full / train_full come from load_eval_csv; train_full is the slice the model was trained on."""
    pt = model.predict(test_full) if pred_test is None else pred_test
    ptr = model.predict(train_full)
    m = habit_mask(test_full)
    h, nh = subset_metrics(pt, test_full, m), subset_metrics(pt, test_full, ~m)
    return {
        "mse_score": float(np.mean((pt - test_full["score"].values) ** 2)),
        "mse_clean": float(np.mean((pt - test_full["score_clean"].values) ** 2)),
        "train_mse_score": float(np.mean((ptr - train_full["score"].values) ** 2)),
        "train_mse_clean": float(np.mean((ptr - train_full["score_clean"].values) ** 2)),
        "habit_n": h["n"], "habit_mse": h["mse"], "habit_spearman": h["spearman"],
        "habit_pairwise_acc": h["pairwise_acc"], "nonhabit_mse": nh["mse"],
    }
