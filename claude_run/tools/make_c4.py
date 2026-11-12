import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c4_text import T

nb = NB()
nb.md("""
# c4 · Step 4: Initial model and sanity checks

Goals:
1. A small, standard CNN that **provably learns** (it can overfit a tiny set, and it beats the 0.25 baselines).
2. Decide the **early-stopping monitor** (left open in c2) with a fair test.
3. Measure **seed noise** σ_seed, which completes the "meaningful difference" rule from c2.

All later notebooks change *one thing at a time* relative to this model.
""")
nb.code("""
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
sys.path.insert(0, ".")
from dataclasses import replace
from sklearn.metrics import balanced_accuracy_score
from lib.train import Config, run, build_model, RESULTS, summary
from lib.models import n_params, n_conv_layers
from lib.evaluation import bootstrap_ci
from lib.data import SEED, CLASSES
""")

nb.md("""
## 1. The initial model ("semi-guessed" standard defaults)

| Part | Choice | Reason |
|---|---|---|
| input | 1-channel grayscale, head crop, 128×128 | c1/c3 provisional choice (tested in c5) |
| body | stem conv + 4 stages × 1 conv (3×3, BatchNorm, ReLU), max-pool between stages, widths 16-32-64-128 | the most common small-CNN recipe |
| head | global average pool → linear (4 classes), softmax inside the loss | far fewer parameters than flatten + dense |
| training | Adam, lr 1e-3, batch 64, cross-entropy, no schedule, no augmentation, no weight decay | standard defaults; tuned in c6 |
""")
nb.code("""
BASE = Config(exp="c4", name="initial")
m = build_model(BASE, 1)
print(m)
print(f"parameters: {n_params(m):,} | conv layers: {n_conv_layers(m)}")
""")

nb.md("""
## 2. Sanity check: can it overfit 64 images?
If the pipeline (labels, loss, optimiser, data flow) is correct, the model must reach about 100 % **training**
accuracy on a tiny set. Val accuracy is irrelevant here.
""")
nb.code("""
r = run(replace(BASE, name="overfit64", train_n=64, epochs=150, patience=1000))
h = pd.DataFrame(r["history"])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(h.epoch, h.train_loss); ax[0].set(yscale="log", title="train loss (64 images)", xlabel="epoch")
ax[1].plot(h.epoch, h.train_bal_acc); ax[1].set(title="train balanced accuracy (64 images)", xlabel="epoch", ylim=(0, 1.02))
plt.tight_layout(); plt.show()
print(f"final train loss {h.train_loss.iloc[-1]:.4f}, train bal. acc {h.train_bal_acc.iloc[-1]:.3f}, "
      f"first epoch at 100 %: {int(h.epoch[h.train_bal_acc >= 0.999].min()) if (h.train_bal_acc >= 0.999).any() else None}")
""")
nb.md(T["overfit"])

nb.md("""
## 3. Full training, long runs, 5 seeds
Each run trains for 60 epochs **without** early stopping and stores the val probabilities of every epoch. That
gives the full curves, and lets section 4 replay any early-stopping rule afterwards.
""")
nb.code("""
SEEDS = [SEED + i for i in range(5)]
long = {s: run(replace(BASE, name=f"long_s{s}", epochs=60, patience=1000, seed=s, extra={"save_epoch_probs": True}))
        for s in SEEDS}
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for s, r in long.items():
    h = pd.DataFrame(r["history"])
    axes[0].plot(h.epoch, h.train_loss, "C0", alpha=0.5); axes[0].plot(h.epoch, h.val_loss, "C1", alpha=0.7)
    axes[1].plot(h.epoch, h.train_bal_acc, "C0", alpha=0.5); axes[1].plot(h.epoch, h.val_bal_acc, "C1", alpha=0.7)
    axes[2].plot(h.epoch, h.val_tumor_auc, "C2", alpha=0.7)
axes[0].set(title="loss (blue train, orange val)", xlabel="epoch", yscale="log")
axes[1].set(title="balanced accuracy (blue train, orange val)", xlabel="epoch", ylim=(0.5, 1.01))
axes[2].set(title="val tumour-vs-no-tumour AUC", xlabel="epoch")
for a in axes: a.grid(alpha=0.3)
plt.suptitle("Initial model, 5 seeds, 60 epochs, no early stopping"); plt.tight_layout(); plt.show()
H = pd.concat([pd.DataFrame(r["history"]).assign(seed=s) for s, r in long.items()])
print("epoch of min val loss per seed:", H.loc[H.groupby("seed").val_loss.idxmin()].epoch.tolist() if False else
      [int(pd.DataFrame(r["history"]).val_loss.idxmin()) + 1 for r in long.values()])
print("epoch of max val bal. acc per seed:", [int(pd.DataFrame(r["history"]).val_bal_acc.idxmax()) + 1 for r in long.values()])
print(f"epoch-to-epoch jitter of val bal. acc (std of differences, epochs 15-60): "
      f"{H[H.epoch >= 15].groupby('seed').val_bal_acc.apply(lambda s: s.diff().std()).mean():.4f}")
print(f"s/epoch: {np.mean([r['sec_per_epoch'] for r in long.values()]):.1f}")
""")
nb.md(T["curves"])

nb.md("""
## 4. Fair test of the early-stopping monitor
Picking the epoch with the best val balanced accuracy and then *reporting* that same number is optimistically
biased. A fair comparison needs data the choice did not see. So:

* split clean val randomly into halves A and B (stratified), 200 times
* on **A**, choose the epoch with each rule
* on **B**, measure the balanced accuracy of that epoch

Rules compared: min val loss, max val balanced accuracy, and max of a 3-epoch moving average of balanced accuracy
(a smoother variant). The "patience" question is replayed the same way: with patience P, training would have
stopped P epochs after the last improvement on A.
""")
nb.code("""
from sklearn.model_selection import StratifiedShuffleSplit

def replay(rule, P, prob, y, A):
    loss = np.array([-np.log(p[A][np.arange(len(A)), y[A]] + 1e-7).mean() for p in prob.astype(np.float32)])
    acc = np.array([balanced_accuracy_score(y[A], p[A].argmax(1)) for p in prob])
    score = {"val_loss": -loss, "val_bal_acc": acc,
             "bal_acc_ma3": pd.Series(acc).rolling(3, min_periods=1).mean().to_numpy()}[rule]
    best, best_ep, bad = -np.inf, 0, 0
    for e, sc in enumerate(score):
        if sc > best: best, best_ep, bad = sc, e, 0
        else: bad += 1
        if bad >= P: break
    return best_ep

rows = []
for s in SEEDS:
    d = np.load(RESULTS / "c4" / f"long_s{s}_epochs.npz"); prob, y = d["prob"], d["y"]
    splits = StratifiedShuffleSplit(n_splits=40, test_size=0.5, random_state=s).split(y, y)
    for k, (A, B) in enumerate(splits):
        for rule in ["val_loss", "val_bal_acc", "bal_acc_ma3"]:
            for P in [5, 10, 20, 60]:
                e = replay(rule, P, prob, y, A)
                rows.append(dict(seed=s, split=k, rule=rule, patience=P, epoch=e + 1,
                                 bal_acc_B=balanced_accuracy_score(y[B], prob[e][B].argmax(1))))
R = pd.DataFrame(rows)
tab = R.groupby(["rule", "patience"]).agg(bal_acc_on_B=("bal_acc_B", "mean"), sd=("bal_acc_B", "std"),
                                          mean_stop_epoch=("epoch", "mean")).round(4)
display(tab.unstack("patience"))
""")
nb.md(T["monitor"])

nb.md("""
## 5. Seed noise σ_seed
Same config, 5 seeds, using the chosen rule (applied to the whole clean val, as in every later run).
""")
nb.code("""
MONITOR, PATIENCE, EPOCHS = "val_bal_acc", 20, 60   # decided in section 4
seed_runs = {s: run(replace(BASE, name=f"seed_s{s}", seed=s, monitor=MONITOR, patience=PATIENCE, epochs=EPOCHS)) for s in SEEDS}
S = summary(seed_runs)
display(S.astype(float).round(3))
sigma = S.bal_acc.astype(float).std()
print(f"mean bal. acc {S.bal_acc.mean():.4f}, sigma_seed {sigma:.4f} -> adoption threshold 2*sigma = {2 * sigma:.4f}")
r0 = seed_runs[SEED]
y = np.load(RESULTS / "c4" / f"seed_s{SEED}_val.npz")["y"]; p = np.load(RESULTS / "c4" / f"seed_s{SEED}_val.npz")["prob"]
lo, hi = bootstrap_ci(y, p.argmax(1))
print(f"seed {SEED}: bal. acc {r0['val']['bal_acc']:.3f} [95% CI {lo:.3f}, {hi:.3f}] vs dumb baselines 0.25")
print("per-class recall:", r0["val"]["recall"])
print("confusion (rows true, cols pred):", CLASSES); print(np.array(r0["val"]["confusion"]))
import json
from lib.data import ARTIFACTS
_ = (ARTIFACTS / "sigma_seed.json").write_text(json.dumps({"sigma": float(sigma), "seeds": SEEDS, "mean": float(S.bal_acc.mean())}))
_ = (ARTIFACTS / "protocol.json").write_text(json.dumps({"monitor": MONITOR, "patience": PATIENCE, "epochs": EPOCHS}))
from lib.report import log_progress
_ = log_progress("c4 initial", list(seed_runs.values())[:3])
""")
nb.md(T["seed"])
nb.md("## Decisions")
nb.md(T["summary"])
nb.write("c4_initial_model.ipynb")
