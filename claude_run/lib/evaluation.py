"""One evaluation function used by every notebook, plus the noise/uncertainty tools from c2."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .data import CLASSES, NO_TUMOR


def tumor_score(prob: np.ndarray) -> np.ndarray:
    """P(any tumour) = 1 - P(no_tumor)."""
    return 1.0 - prob[:, NO_TUMOR]


def evaluate(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """All metrics decided in c2. prob: (N, 4) softmax outputs."""
    y_true = np.asarray(y_true)
    y_pred = prob.argmax(1)
    is_tumor = y_true != NO_TUMOR
    pred_tumor = y_pred != NO_TUMOR
    rec = recall_score(y_true, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    return {
        "bal_acc": balanced_accuracy_score(y_true, y_pred),          # primary
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": dict(zip(CLASSES, rec.round(4).tolist())),
        "precision": dict(zip(CLASSES, prec.round(4).tolist())),
        "tumor_auc": roc_auc_score(is_tumor, tumor_score(prob)),
        "tumor_sens": float((pred_tumor & is_tumor).sum() / is_tumor.sum()),     # at argmax
        "tumor_spec": float((~pred_tumor & ~is_tumor).sum() / (~is_tumor).sum()),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
    }


def wilson(k: int, n: int, conf: float = 0.95):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=conf, method="wilson")
    return ci.low, ci.high


def bootstrap_ci(y_true, y_pred, metric=balanced_accuracy_score, n_boot=2000, seed=0, conf=0.95):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = np.array([metric(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    a = (1 - conf) / 2
    return np.quantile(stats, a), np.quantile(stats, 1 - a)


def paired_bootstrap(y_true, pred_a, pred_b, metric=balanced_accuracy_score, n_boot=2000, seed=0):
    """CI of metric(b) - metric(a) on the same resampled images, and P(b <= a)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(metric(y_true[i], pred_b[i]) - metric(y_true[i], pred_a[i]))
    d = np.array(d)
    return {"diff": metric(y_true, pred_b) - metric(y_true, pred_a),
            "ci": (np.quantile(d, 0.025), np.quantile(d, 0.975)), "p_b_not_better": float((d <= 0).mean())}
