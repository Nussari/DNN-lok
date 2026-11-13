import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c10_text import T

nb = NB()
nb.md("""
# c10 · Step 10: Decision threshold (ROC), on val

Rule fixed in c2, before any model existed: **predict "tumour" when P(tumour) = 1 - P(no_tumor) > t, with t the
highest cutoff that gives tumour sensitivity ≥ 0.98 on clean val.** If it says tumour, the tumour type is the most
likely of the three tumour classes. For comparison: argmax (no threshold), Youden's J, and a 10:1 cost ratio
(a missed tumour counts 10x a false alarm).

The test set is not used here. The rule decided below is applied to the test set in `test_evaluation.ipynb`.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
sys.path.insert(0, "..")
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, confusion_matrix
from lib.data import ARTIFACTS, CLASSES, NO_TUMOR, SEED
from lib.train import RESULTS
from lib.evaluation import wilson, tumor_score
FINAL = json.loads((ARTIFACTS / "final.json").read_text())
d = np.load(RESULTS / "c9" / f"final_s{SEED}_val.npz"); y, prob = d["y"], d["prob"]
is_tumor = y != NO_TUMOR; score = tumor_score(prob)
print(f"clean val: {is_tumor.sum()} tumour, {(~is_tumor).sum()} no-tumour images")
""")

nb.md("## 1. ROC curve and the candidate cutoffs")
nb.code("""
fpr, tpr, thr = roc_curve(is_tumor, score)
auc = roc_auc_score(is_tumor, score)
rng = np.random.default_rng(SEED)
boot = [roc_auc_score(is_tumor[i], score[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if is_tumor[i].any() and (~is_tumor[i]).any()]
lo, hi = np.quantile(boot, [0.025, 0.975])
TARGET = 0.98
i_target = np.where(tpr >= TARGET)[0][0]                      # highest threshold reaching the target
i_youden = np.argmax(tpr - fpr)
P_t, P_n = is_tumor.mean(), 1 - is_tumor.mean()
cost = 10 * (1 - tpr) * P_t + 1 * fpr * P_n
i_cost = np.argmin(cost)
pred_arg_tumor = prob.argmax(1) != NO_TUMOR
arg_tpr, arg_fpr = pred_arg_tumor[is_tumor].mean(), pred_arg_tumor[~is_tumor].mean()
cuts = {"argmax": (None, arg_tpr, arg_fpr), f"sensitivity >= {TARGET}": (thr[i_target], tpr[i_target], fpr[i_target]),
        "Youden J": (thr[i_youden], tpr[i_youden], fpr[i_youden]), "cost 10:1": (thr[i_cost], tpr[i_cost], fpr[i_cost])}

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, lim in zip(ax, [(0, 1), (0, 0.4)]):
    a.plot(fpr, tpr, lw=2, label=f"ROC, AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")
    a.plot([0, 1], [0, 1], "k:", lw=1)
    for (k, (t, tp, fp)), mk in zip(cuts.items(), ["o", "*", "s", "D"]):
        a.scatter(fp, tp, s=90 if mk == "*" else 50, marker=mk, zorder=3, label=f"{k}" + (f" (t={t:.3f})" if t is not None else ""))
    a.set(xlim=lim, ylim=(0.85 if lim[1] < 1 else 0, 1.005), xlabel="false positive rate (1 - specificity)",
          ylabel="sensitivity (tumour recall)"); a.grid(alpha=0.3)
ax[0].set_title("tumour vs no tumour, clean val"); ax[1].set_title("zoom"); ax[0].legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")
nb.code("""
rows = {}
for k, (t, tp, fp) in cuts.items():
    pt = pred_arg_tumor if t is None else score >= t
    kt, nt = int((pt & is_tumor).sum()), int(is_tumor.sum()); kn, nn = int((~pt & ~is_tumor).sum()), int((~is_tumor).sum())
    s_lo, s_hi = wilson(kt, nt); p_lo, p_hi = wilson(kn, nn)
    rows[k] = {"threshold": t, "sensitivity": kt / nt, "sens 95% CI": f"[{s_lo:.3f}, {s_hi:.3f}]", "missed tumours": nt - kt,
               "specificity": kn / nn, "spec 95% CI": f"[{p_lo:.3f}, {p_hi:.3f}]", "false alarms": nn - kn}
display(pd.DataFrame(rows).T)
""")
nb.md(T["roc"])

nb.md("## 2. What the chosen rule does to the 4-class predictions")
nb.code("""
T_STAR = float(thr[i_target])
def predict_with_threshold(prob, t):
    p = prob.copy(); p[:, NO_TUMOR] = -1                   # best tumour class
    tumour_type = p.argmax(1)
    return np.where(tumor_score(prob) >= t, tumour_type, NO_TUMOR)   # >= as in roc_curve
pred_t = predict_with_threshold(prob, T_STAR)
for name, pr in [("argmax", prob.argmax(1)), (f"threshold {T_STAR:.3f}", pred_t)]:
    cm = confusion_matrix(y, pr, labels=range(4))
    print(f"{name}: balanced accuracy {balanced_accuracy_score(y, pr):.4f}")
    print(pd.DataFrame(cm, index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES]), "\\n")
""")
nb.md(T["effect"])

nb.md("""
## 3. Decision: the 0.98 sensitivity is a floor, not a target
**Restated rule** (decided on val, before any test evaluation): use **argmax**. Only if argmax misses the floor,
i.e. tumour sensitivity on clean val is below 0.98, move the cutoff down to the highest value that reaches 0.98.
Applied to the reported model and to the three compressed models (d6), seed 1738 of each.
""")
nb.code("""
PICKS = json.loads((ARTIFACTS / "compress_final.json").read_text())        # the compressed models chosen in d6
MODELS = {"large": RESULTS / "c9" / f"final_s{SEED}_val.npz",
          **{f"{p['stored kB']:.1f} kB": RESULTS / f"{p['record']}_s{SEED}_val.npz" for p in PICKS.values()}}
def floor_rule(prob, y, target=TARGET):
    tum = y != NO_TUMOR
    if (prob.argmax(1) != NO_TUMOR)[tum].mean() >= target:
        return None
    f, t, th = roc_curve(tum, tumor_score(prob))
    return float(th[np.where(t >= target)[0][0]])
rows, DECISION = {}, {}
for k, f in MODELS.items():
    d = np.load(f); yy, pp = d["y"], d["prob"]; t = floor_rule(pp, yy); DECISION[k] = t
    pr = pp.argmax(1) if t is None else predict_with_threshold(pp, t); tum = yy != NO_TUMOR
    pa = pp.argmax(1)
    rows[k] = {"decision": "argmax" if t is None else f"cutoff {t:.3f}",
               "argmax: missed tumours": int((pa[tum] == NO_TUMOR).sum()), "argmax: false alarms": int((pa[~tum] != NO_TUMOR).sum()),
               "argmax: bal. acc": round(balanced_accuracy_score(yy, pa), 4),
               "sensitivity": round((pr[tum] != NO_TUMOR).mean(), 4), "missed tumours": int((pr[tum] == NO_TUMOR).sum()),
               "specificity": round((pr[~tum] == NO_TUMOR).mean(), 4), "false alarms": int((pr[~tum] != NO_TUMOR).sum()),
               "bal. acc": round(balanced_accuracy_score(yy, pr), 4)}
display(pd.DataFrame(rows).T)
_ = (ARTIFACTS / "threshold.json").write_text(json.dumps({
    "rule": f"argmax; if tumour sensitivity on clean val < {TARGET}, the highest cutoff on P(tumour) reaching {TARGET}",
    "decisions": {k: (None if t is None else t) for k, t in DECISION.items()}, "files": {k: str(v.name) for k, v in MODELS.items()},
    "threshold": DECISION["large"], "val_auc": auc,
    "superseded": {"rule": f"highest cutoff with tumour sensitivity >= {TARGET} (c2 wording)", "threshold": T_STAR}}, indent=1))
""")
nb.md(T["decision"])

nb.write("c10_threshold.ipynb")
