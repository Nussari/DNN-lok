import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c9_text import T

nb = NB()
nb.md("""
# c9 · Step 9: Learning rate again, and the final model

The learning rate was tuned in c6 on the small, unregularised model. Depth, width, augmentation and weight decay
all shift the best value, so it is re-checked on the final configuration: ×⅓, ×1 and ×3 of the current value.
Then the **final model** is trained.

**Which run is "the" final model?** Picking the best of several seeds on val would be cherry-picking. So the
rule is fixed in advance: the run with **seed 1738** is the final model, and the other seeds only show how much
retraining varies.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, ".")
from dataclasses import replace
from lib.train import Config, run, build_model, RESULTS, CKPT
from lib.report import run_seeds, compare, pick, SEEDS3, curve_test
from lib.evaluation import bootstrap_ci, wilson
from lib.models import n_params, n_conv_layers
from lib.data import ARTIFACTS, CLASSES, SEED
proto = json.loads((ARTIFACTS / "protocol.json").read_text())
INPUT = json.loads((ARTIFACTS / "input.json").read_text())
OPTIM = json.loads((ARTIFACTS / "optim.json").read_text())
REG = json.loads((ARTIFACTS / "reg.json").read_text())
ARCH = {k: tuple(v) if isinstance(v, list) else v for k, v in REG["arch"].items()}
SEEDS = SEEDS3 if REG["winner"] == "small" else SEEDS3[:2]
CUR = replace(Config(exp="c9", name="x", monitor=proto["monitor"], patience=proto["patience"], epochs=proto["epochs"]), **INPUT, **OPTIM, **ARCH,
              aug=REG["aug"], optimizer=REG["optimizer"], weight_decay=REG["weight_decay"], dropout=REG["dropout"])
print(CUR)
""")

nb.md("## 1. Learning rate re-check")
nb.code("""
lr0 = CUR.lr
# the x1 runs are named "final" so that, if the LR stays, they are reused as the final model's runs
lr_runs = {lr0 * f: run_seeds(replace(CUR, name="final" if f == 1 else f"lr{lr0 * f:.2g}", lr=lr0 * f), SEEDS) for f in [1 / 3, 1, 3]}
t = compare({f"{k:.2g}": v for k, v in lr_runs.items()}, f"{lr0:.2g}", "learning rate on the final configuration"); display(t)
ct = curve_test(lr_runs)   # same rule as c6 for ordered settings
print({k: (round(float(v), 4) if not isinstance(v, (dict, tuple, bool)) else v) for k, v in ct.items() if k != "means"})
LR = float(ct["best"]) if ct["has_shape"] else lr0
CUR = replace(CUR, lr=LR); print("final learning rate:", LR)
""")
nb.md(T["lr"])

nb.md("## 2. The final model")
nb.code("""
final_runs = run_seeds(replace(CUR, name="final" if LR == lr0 else f"final_lr{LR:.2g}"), SEEDS3)
FINAL = final_runs[0]; assert FINAL["config"]["seed"] == SEED
net = build_model(CUR, 3 if CUR.mode == "rgb" else 1)
print(f"final model: {n_conv_layers(net)} conv layers, {n_params(net):,} parameters, {FINAL['sec_per_epoch']:.1f} s/epoch on {FINAL['device']}")
rows = {f"seed {r['config']['seed']}": {"bal_acc": r["val"]["bal_acc"], "macro_f1": r["val"]["macro_f1"],
        "tumor_auc": r["val"]["tumor_auc"], **{f"rec {c}": r["val"]["recall"][c] for c in CLASSES},
        "best_epoch": r["best_epoch"]} for r in final_runs}
S = pd.DataFrame(rows).T; S.loc["mean ± sd"] = [f"{S[c].mean():.3f} ± {S[c].std():.3f}" for c in S.columns]
display(S)
""")
nb.code("""
d = np.load(RESULTS / "c9" / f"final_s{SEED}_val.npz"); y, prob = d["y"], d["prob"]; pred = prob.argmax(1)
lo, hi = bootstrap_ci(y, pred)
print(f"FINAL (seed {SEED}) on clean val: balanced accuracy {FINAL['val']['bal_acc']:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
for i, c in enumerate(CLASSES):
    k, n = int(((pred == i) & (y == i)).sum()), int((y == i).sum()); l, h = wilson(k, n)
    print(f"  recall {c:11s} {k / n:.3f}  [{l:.3f}, {h:.3f}]  ({k}/{n})")
cm = np.array(FINAL["val"]["confusion"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
im = ax[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
ax[0].set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="predicted", ylabel="true",
          title="confusion matrix (clean val, argmax), colour = row %")
h = pd.DataFrame(FINAL["history"])
ax[1].plot(h.epoch, h.train_bal_acc, label="train"); ax[1].plot(h.epoch, h.val_bal_acc, label="val")
ax[1].axvline(FINAL["best_epoch"], color="k", ls=":", label="selected epoch")
ax[1].set(xlabel="epoch", ylabel="balanced accuracy", title="final model learning curves", ylim=(0.5, 1.01)); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")
nb.md("""
## 3. How far did each step get us?
Mean val balanced accuracy of the chosen configuration after each step (3 seeds; 2 for deep runs).
""")
nb.code("""
from lib.report import sigma_seed
steps = json.loads((ARTIFACTS / "progress.json").read_text()) if (ARTIFACTS / "progress.json").exists() else {}
steps["c9 final (LR re-checked)"] = float(np.mean([r["val"]["bal_acc"] for r in final_runs]))
P = pd.Series(steps)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(range(len(P)), P.values, marker="o"); ax.set_xticks(range(len(P)), P.index, rotation=20, ha="right")
ax.axhline(0.25, color="grey", ls="--", label="dumb baselines (0.25)")
ax.set(ylabel="val balanced accuracy", title="progress through the steps"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show(); display(P.round(4))
""")
nb.md(T["final"])
nb.code("""
_ = (ARTIFACTS / "final.json").write_text(json.dumps({"config": FINAL["config"],
     "checkpoint": str((CKPT / "c9" / f"final_s{SEED}.pt").relative_to(CKPT.parent))}, indent=1))
""")
nb.write("c9_final_model.ipynb")
