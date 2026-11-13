"""Comparing options the same way in every notebook (rule from c2, sigma_seed from c4)."""
from __future__ import annotations

import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ARTIFACTS, SEED
from .train import Config, run

SEEDS3 = [SEED, SEED + 1, SEED + 2]

# c4 fixed a 60-epoch cap (artifacts/protocol.json). Round r3 raised it to 200 from c5 on (early stopping, patience
# 20 on val balanced accuracy, best weights restored); its final model scored lower (0.952 vs 0.968, see
# history/README.md), so the reported pipeline (round r2) uses 60. Set 200 to redo r3.
EPOCH_CAP = 60


def protocol() -> dict:
    """c4's protocol (monitor, patience) with the epoch cap EPOCH_CAP."""
    p = json.loads((ARTIFACTS / "protocol.json").read_text())
    return {**p, "epochs": max(p["epochs"], EPOCH_CAP)}


def sigma_seed() -> float:
    return json.loads((ARTIFACTS / "sigma_seed.json").read_text())["sigma"]


def run_seeds(cfg: Config, seeds=SEEDS3) -> list[dict]:
    return [run(replace(cfg, name=f"{cfg.name}_s{s}", seed=s)) for s in seeds]


def compare(options: dict[str, list[dict]], baseline: str, title: str = "", plot: bool = True) -> pd.DataFrame:
    """options: name -> list of run results (one per seed). Table of every option with its delta to the baseline.

    'noise_2se' is 2 * sigma_seed * sqrt(1/n + 1/n_base), two standard errors of a difference of means, shown only
    as a noise reference. It does not decide anything: `pick` takes the highest mean.
    """
    sig = sigma_seed()
    rows = {}
    for name, rs in options.items():
        b = np.array([r["val"]["bal_acc"] for r in rs])
        rows[name] = {"bal_acc": b.mean(), "sd": b.std(ddof=1) if len(b) > 1 else np.nan, "n": len(b),
                      "macro_f1": np.mean([r["val"]["macro_f1"] for r in rs]),
                      "tumor_auc": np.mean([r["val"]["tumor_auc"] for r in rs]),
                      "no_tumor_rec": np.mean([r["val"]["recall"]["no_tumor"] for r in rs]),
                      "epochs": np.mean([r["epochs_run"] for r in rs]),
                      "params": rs[0]["n_params"], "s/ep": np.mean([r["sec_per_epoch"] for r in rs])}
    df = pd.DataFrame(rows).T
    base = df.loc[baseline]
    df["delta"] = df.bal_acc - base.bal_acc
    thr = 2 * sig * np.sqrt(1 / df.n + 1 / base.n)
    df["noise_2se"] = thr
    if plot:
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(df)), 3.6))
        for i, (name, rs) in enumerate(options.items()):
            b = [r["val"]["bal_acc"] for r in rs]
            ax.scatter([i] * len(b), b, color="C0", alpha=0.6, s=25)
            ax.hlines(np.mean(b), i - 0.25, i + 0.25, color="k")
        ax.axhline(base.bal_acc, color="grey", ls="--", lw=1, label="baseline mean")
        ax.set_xticks(range(len(df)), df.index, rotation=20, ha="right")
        ax.set(ylabel="val balanced accuracy", title=title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    cols = ["bal_acc", "sd", "n", "delta", "noise_2se", "macro_f1", "tumor_auc", "no_tumor_rec", "epochs",
            "params", "s/ep"]
    return df[cols].round(4)


def pick(table: pd.DataFrame, baseline: str) -> str:
    """The option with the highest mean val balanced accuracy (as shown, 4 decimals), regardless of size or cost.
    On an exact tie the first option listed wins. `baseline` gets no preference."""
    return table.bal_acc.astype(float).idxmax()


def log_progress(stage: str, runs: list[dict]) -> float:
    """Record the mean val balanced accuracy of the configuration chosen at a step (for the c9 progress plot)."""
    f = ARTIFACTS / "progress.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[stage] = float(np.mean([r["val"]["bal_acc"] for r in runs]))
    f.write_text(json.dumps(d, indent=1))
    return d[stage]


def curve_test(options: dict[float, list[dict]]) -> dict:
    """Ordered numeric setting (e.g. learning rate): test whether the whole curve has a shape.

    Fits val balanced accuracy of every single run against log10(value), once with a line and once with a
    parabola. The pairwise rule compares each option with the baseline alone and ignores that the options
    form a curve; a trend across all runs uses every run at once and is much more sensitive.
    Returns slope per decade with its 95 % CI, p-values, and the option with the highest mean.
    """
    from scipy import stats
    x = np.concatenate([[np.log10(v)] * len(rs) for v, rs in options.items()])
    y = np.concatenate([[r["val"]["bal_acc"] for r in rs] for rs in options.values()])
    lin = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    # quadratic term: is there a peak (or valley) inside the range?
    X = np.column_stack([np.ones_like(x), x - x.mean(), (x - x.mean()) ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(x) - 3
    s2 = float(((y - X @ beta) ** 2).sum() / dof)
    se_q = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
    p_quad = 2 * stats.t.sf(abs(beta[2] / se_q), dof)
    means = {v: float(np.mean([r["val"]["bal_acc"] for r in rs])) for v, rs in options.items()}
    best = max(means, key=means.get)
    keys = sorted(options)
    return {"slope_per_decade": float(lin.slope),
            "slope_ci": (float(lin.slope - tcrit * lin.stderr), float(lin.slope + tcrit * lin.stderr)),
            "p_linear": float(lin.pvalue), "p_quadratic": float(p_quad), "has_shape": bool(lin.pvalue < 0.05 or p_quad < 0.05),
            "best": best, "best_at_edge": bool(best in (keys[0], keys[-1])), "means": means}
