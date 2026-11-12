import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c2_text import T

nb = NB()
nb.md("""
# c2 · Step 2: Evaluation setup

Decides, before any model exists: the **primary metric** (picks models/checkpoints), the **secondary metrics**
(always reported), the **noise level** (what counts as a real difference), the **val protocol** and the
**severity rule** for the step-10 threshold. All metrics are computed by `lib/evaluation.py::evaluate`, which
every later notebook uses.

Evaluation set: **clean val** (832 images; c1 removed 60 copies of train images).
""")
nb.code("""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, ".")
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             cohen_kappa_score)
from lib.data import CLASSES, NO_TUMOR, SEED, list_split, excluded_paths, DATA_ROOT
from lib.evaluation import wilson, bootstrap_ci

def labels(split, clean=True):
    ex = excluded_paths() if clean else set()
    return np.array([c for p, c in list_split(split) if str(p.relative_to(DATA_ROOT)) not in ex])
y_train, y_val = labels("train"), labels("val")
print("clean val class counts:", dict(zip(CLASSES, np.bincount(y_val))))
""")

nb.md("""
## 1. Primary metric: options

| Metric | What it measures |
|---|---|
| accuracy | share of correct predictions |
| balanced accuracy | mean of per-class recall |
| macro-F1 | mean of per-class F1 (recall and precision) |
| weighted-F1 | per-class F1 weighted by class size |
| MCC / Cohen's κ | correlation between prediction and truth, corrected for chance |

Two tests: what do **dumb classifiers** score, and how does each metric react to the **same number of errors
placed differently**?
""")
nb.code("""
METRICS = {
    "accuracy": accuracy_score,
    "balanced acc": balanced_accuracy_score,
    "macro F1": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
    "weighted F1": lambda t, p: f1_score(t, p, average="weighted", zero_division=0),
    "MCC": matthews_corrcoef,
    "kappa": cohen_kappa_score,
}
rows = {}
for name, strat in [("predict largest", "most_frequent"), ("proportional random", "stratified"),
                    ("uniform random", "uniform")]:
    scores = []
    for s in range(200 if strat != "most_frequent" else 1):  # average random guessers over 200 seeds
        clf = DummyClassifier(strategy=strat, random_state=s).fit(np.zeros((len(y_train), 1)), y_train)
        p = clf.predict(np.zeros((len(y_val), 1)))
        scores.append({m: f(y_val, p) for m, f in METRICS.items()})
    rows[name] = pd.DataFrame(scores).mean()
display(pd.DataFrame(rows).T.round(3))
""")
nb.md(T["dumb"])
nb.code("""
def corrupt(errors):
    y_pred = y_val.copy()
    for t, w, n in errors:
        idx = np.where((y_val == t) & (y_pred == t))[0][:n]
        y_pred[idx] = w
    return y_pred

G, M, NT, P = range(4)
def scenarios(n):
    return {"no_tumor -> glioma (small class missed)": [(NT, G, n)],
            "glioma -> no_tumor (large class, into small)": [(G, NT, n)],
            "glioma -> meningioma (large class, into mid)": [(G, M, n)],
            "spread evenly": [(c, (c + 1) % 4, n // 4) for c in range(4)]}
ns = np.arange(0, 93, 4)
show = ["accuracy", "balanced acc", "macro F1", "MCC"]
res = {m: {s: [] for s in scenarios(0)} for m in show}
for n in ns:
    for s, e in scenarios(n).items():
        p = corrupt(e)
        for m in show:
            res[m][s].append(METRICS[m](y_val, p))
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
for ax, m in zip(axes, show):
    for (s, v), ls in zip(res[m].items(), ["-", "--", ":", "-."]):
        ax.plot(ns, v, ls=ls, lw=2, label=s)
    ax.set(title=m, xlabel="errors on clean val"); ax.grid(alpha=0.3)
axes[0].set_ylabel("score"); axes[0].legend(fontsize=8, loc="lower left")
plt.suptitle("Same number of errors, different placement"); plt.tight_layout(); plt.show()
""")
nb.md(T["placement"])

nb.md("""
## 2. Noise: how big must a difference be?
Three sources of noise. (a) and (b) are computed here, (c) needs a trained model and is measured in c4.

* (a) **per-class recall** on a fixed val set: Wilson 95 % interval
* (b) **balanced accuracy** on a fixed val set: bootstrap of a simulated classifier
* (c) **retraining** the same config with another seed
""")
nb.code("""
n_cls = np.bincount(y_val)
rows = {}
for r in [0.80, 0.90, 0.95]:
    rows[f"recall {r:.2f}"] = {c: "±{:.1f}".format(50 * np.subtract(*wilson(round(r * n), n)[::-1]) * 2 / 2)
                               for c, n in zip(CLASSES, n_cls)}
print("Wilson 95 % interval half-width, in percentage points:")
display(pd.DataFrame(rows).T)
""")
nb.code("""
rng = np.random.default_rng(SEED)
rows = {}
for acc in [0.80, 0.90, 0.95]:
    p = y_val.copy()
    wrong = rng.random(len(p)) > acc                      # errors spread at random over all classes
    p[wrong] = (p[wrong] + rng.integers(1, 4, wrong.sum())) % 4
    lo, hi = bootstrap_ci(y_val, p, n_boot=2000, seed=SEED)
    rows[f"~{acc:.2f} correct"] = {"bal_acc": balanced_accuracy_score(y_val, p), "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                                   "half-width (pp)": round(50 * (hi - lo), 1)}
display(pd.DataFrame(rows).T)
""")
nb.md(T["noise"])

nb.md("""
## 3. Binary view and prevalence
For the tumour vs no-tumour decision: does precision (PPV) transfer from this dataset to practice?
Example classifier with sensitivity 0.98 and specificity 0.90, evaluated at the dataset's tumour prevalence and at
lower prevalences (Bayes' rule).
""")
nb.code("""
sens, spec = 0.98, 0.90
prev_data = (y_val != NO_TUMOR).mean()
rows = {}
for name, prev in [("this dataset", prev_data), ("specialist referral (assumed)", 0.30),
                   ("routine headache MRI (assumed)", 0.01)]:
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev)
    rows[name] = {"tumour prevalence": prev, "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv}
display(pd.DataFrame(rows).T.round(3))
""")
nb.md(T["prev"])

nb.md("## 4. Val protocol")
nb.md(T["protocol"])
nb.md("## 5. Severity rule for the threshold (step 10)")
nb.md(T["severity"])
nb.md("## Decisions")
nb.md(T["summary"])
nb.write("c2_eval_setup.ipynb")
