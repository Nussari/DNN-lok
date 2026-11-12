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


def sigma_seed() -> float:
    return json.loads((ARTIFACTS / "sigma_seed.json").read_text())["sigma"]


def run_seeds(cfg: Config, seeds=SEEDS3) -> list[dict]:
    return [run(replace(cfg, name=f"{cfg.name}_s{s}", seed=s)) for s in seeds]


def compare(options: dict[str, list[dict]], baseline: str, title: str = "", plot: bool = True) -> pd.DataFrame:
    """options: name -> list of run results (one per seed). Verdict per option vs the baseline.

    Rule (c2 + c4): with n seeds per option, a difference of means is 'real' if it exceeds
    2 * sigma_seed * sqrt(2/n) (two standard errors of a difference of two n-seed means).
    Ties go to the simpler/cheaper option, decided in the notebook text.
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
    df["needed"] = thr
    df["verdict"] = np.where(df.index == baseline, "baseline",
                             np.where(df.delta > thr, "better", np.where(df.delta < -thr, "worse", "tie")))
    if plot:
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(df)), 3.6))
        for i, (name, rs) in enumerate(options.items()):
            b = [r["val"]["bal_acc"] for r in rs]
            ax.scatter([i] * len(b), b, color="C0", alpha=0.6, s=25)
            ax.hlines(np.mean(b), i - 0.25, i + 0.25, color="k")
        t = thr.mean()
        ax.axhspan(base.bal_acc - t, base.bal_acc + t, color="grey", alpha=0.15,
                   label="baseline ± needed difference")
        ax.set_xticks(range(len(df)), df.index, rotation=20, ha="right")
        ax.set(ylabel="val balanced accuracy", title=title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    cols = ["bal_acc", "sd", "n", "delta", "needed", "verdict", "macro_f1", "tumor_auc", "no_tumor_rec", "epochs",
            "params", "s/ep"]
    return df[cols].round(4)


def pick(table: pd.DataFrame, baseline: str) -> str:
    """Apply the rule: the best option that is clearly 'better' than the baseline, otherwise the baseline."""
    better = table[table.verdict == "better"]
    return better.bal_acc.astype(float).idxmax() if len(better) else baseline


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
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(x) - 3
    s2 = float(((y - X @ beta) ** 2).sum() / dof)
    se_q = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
    p_quad = 2 * stats.t.sf(abs(beta[2] / se_q), dof)
    means = {v: float(np.mean([r["val"]["bal_acc"] for r in rs])) for v, rs in options.items()}
    best = max(means, key=means.get)
    keys = sorted(options)
    return {"slope_per_decade": lin.slope, "slope_ci": (lin.slope - tcrit * lin.stderr, lin.slope + tcrit * lin.stderr),
            "p_linear": lin.pvalue, "p_quadratic": p_quad, "has_shape": bool(lin.pvalue < 0.05 or p_quad < 0.05),
            "best": best, "best_at_edge": best in (keys[0], keys[-1]), "means": means}
