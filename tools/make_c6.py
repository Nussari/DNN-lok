import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c6_text import T

nb = NB()
nb.md("""
# c6 · Step 6: Optimisation tuning

Input from c5, model still the c4 initial model. Settings are tuned in order of how much they usually matter,
**one at a time**; each section starts from the winners of the previous ones. 3 seeds per option. Rule
(`compare`/`pick`): the option with the highest mean val balanced accuracy wins, whatever its size or cost.

| # | Setting | Options |
|---|---|---|
| 1 | learning rate (Adam) | 1e-4, 3e-4, **1e-3**, 3e-3, 1e-2 (extended if the best is at the edge) |
| 2 | optimiser | **Adam** vs SGD + Nesterov momentum (own LR sweep: 3e-3, 1e-2, 3e-2, 1e-1, extended to 0.3 because 0.1 was at the edge) |
| 3 | batch size | 32, **64**, 128 |
| 4 | LR schedule | **none**, cosine annealing, step decay (÷10 at 50 % and 75 %), reduce-on-plateau |
| 5 | loss / imbalance | **cross-entropy**, class-weighted CE, CE + weighted sampler, label smoothing 0.1, focal (γ=2) |

Bold = the current default, i.e. the baseline of each comparison.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from dataclasses import replace
from lib.train import Config, run
from lib.report import run_seeds, compare, pick, sigma_seed, log_progress, curve_test
from lib.data import ARTIFACTS
from lib.report import protocol; proto = protocol()   # c4 protocol, epoch cap from lib.report.EPOCH_CAP
INPUT = json.loads((ARTIFACTS / "input.json").read_text())
CUR = replace(Config(exp="c6", name="x", monitor=proto["monitor"], patience=proto["patience"], epochs=proto["epochs"]), **INPUT)
BASE_RUNS = run_seeds(replace(CUR, exp="c4" if CUR.epochs == 60 else "c5", name="seed" if CUR.epochs == 60 else "crop")) if INPUT == dict(mode="gray", resize="crop", size=128, normalize="global") \\
    else run_seeds(replace(CUR, exp="c5", name="input_choice"))
print(INPUT, proto)
""")

nb.md("""
## 0. Device check
c1 to c5 were trained on Apple MPS; from here on the runs are trained on CUDA. The same seed and configuration do
not give identical numbers on different devices, so the c4 baseline (3 seeds) was retrained on CUDA. If the
difference is well inside the seed noise, runs from both devices can be compared.
""")
nb.code("""
C4 = Config(exp="c4", name="seed", monitor=proto["monitor"], patience=proto["patience"], epochs=60)
mps_runs, cuda_runs = run_seeds(C4), run_seeds(replace(C4, exp="c6", name="devcheck_cuda"))
print("devices:", {r["device"] for r in mps_runs}, "vs", {r["device"] for r in cuda_runs})
display(compare({"c4 baseline, MPS": mps_runs, "c4 baseline, CUDA": cuda_runs}, "c4 baseline, MPS",
                "same configuration, different device"))
""")

nb.md("""
## 1. Learning rate

**Decision rule for ordered numeric settings.** Take the value with the highest mean. The trend and peak fits
(val balanced accuracy of every run regressed on log10(LR), line and parabola) are reported to show how strong
the effect is, but they do not gate the choice. If the best value is at the edge of the sweep, extend the sweep
one step (×3 or ×1/3) and repeat, at most twice.
""")
nb.code("""
def lr_sweep(cfg, lrs, prefix, base_runs=None, base_lr=None, max_ext=2):
    runs = {lr: (base_runs if lr == base_lr else run_seeds(replace(cfg, name=f"{prefix}_lr{lr:g}", lr=lr))) for lr in lrs}
    for _ in range(max_ext):
        ct = curve_test(runs)
        if not ct["best_at_edge"]:
            break
        lo, hi = min(runs), max(runs)
        step = lambda v, up: float(f"{v * (10 / 3 if (f'{v:e}'[0] == '3') == up else 3) ** (1 if up else -1):.1g}")  # 1-3-10 grid
        new = step(hi, True) if ct["best"] == hi else step(lo, False)
        runs[new] = run_seeds(replace(cfg, name=f"{prefix}_lr{new:g}", lr=new))
        runs = dict(sorted(runs.items()))
    return runs, curve_test(runs)

def show_sweep(runs, ct, title, ax):
    xs = sorted(runs); m = [ct["means"][x] for x in xs]
    for x in xs:
        ax.scatter([x] * len(runs[x]), [r["val"]["bal_acc"] for r in runs[x]], color="C0", alpha=0.4, s=18)
    ax.plot(xs, m, "o-", color="C0", label="mean of 3 seeds")
    xx = np.logspace(np.log10(xs[0]), np.log10(xs[-1]), 50)
    ax.plot(xx, np.mean(m) + ct["slope_per_decade"] * (np.log10(xx) - np.mean(np.log10(xs))), "k--", lw=1,
            label=f"trend {ct['slope_per_decade'] * 100:+.1f} pp/decade (p={ct['p_linear']:.1g})")
    ax.axvline(ct["best"], color="C3", ls=":", label=f"chosen {ct['best']:g}")
    ax.set(xscale="log", xlabel="learning rate", ylabel="val balanced accuracy", title=title); ax.grid(alpha=0.3); ax.legend(fontsize=8)

adam_runs, ct_adam = lr_sweep(CUR, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "adam", BASE_RUNS, 1e-3)
t = compare({f"{k:g}": v for k, v in adam_runs.items()}, "0.001", "Adam learning rate (pairwise view)"); display(t)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ct_adam.items() if k != "means"})
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
show_sweep(adam_runs, ct_adam, "Adam: LR sweep", ax[0])
for (k, rs), c in zip(adam_runs.items(), plt.cm.viridis(np.linspace(0, 1, len(adam_runs)))):
    h = pd.DataFrame(rs[0]["history"]); ax[1].plot(h.epoch, h.val_bal_acc, color=c, label=f"lr {k:g}", lw=1)
ax[1].set(xlabel="epoch", ylabel="val balanced accuracy", title="val curves (seed 1738)", ylim=(0.5, 1)); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
LR = float(ct_adam["best"])
CUR = replace(CUR, lr=LR); lr_runs = adam_runs; print("chosen lr:", LR)
""")
nb.code("""
# Is the trend an artefact of picking the best epoch? A higher LR could give a jumpier val curve, and the max of a
# jumpier curve is higher by luck. Check: jitter of the val curve after epoch 15, and the mean of the last 10 epochs
# (no selection at all).
rows = {}
for lr, rs in adam_runs.items():
    H = [pd.DataFrame(r["history"]).val_bal_acc.values for r in rs]
    rows[f"{lr:g}"] = {"best epoch (selected)": np.mean([r["val"]["bal_acc"] for r in rs]),
                       "mean of last 10 epochs": np.mean([h[-10:].mean() for h in H]),
                       "jitter (sd of epoch-to-epoch change)": np.mean([np.diff(h[15:]).std() for h in H])}
display(pd.DataFrame(rows).T.round(4))
""")
nb.md(T["lr"])

nb.md("## 2. Optimiser")
nb.code("""
sgd_runs, ct_sgd = lr_sweep(replace(CUR, optimizer="sgd"), [3e-3, 1e-2, 3e-2, 1e-1], "sgd")
SGD_LR = float(ct_sgd["best"])
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ct_sgd.items() if k != "means"})
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
show_sweep(adam_runs, ct_adam, "Adam", ax[0]); show_sweep(sgd_runs, ct_sgd, "SGD + Nesterov momentum 0.9", ax[1])
plt.tight_layout(); plt.show()
cur_runs = lr_runs[LR]
t_sgd = compare({f"adam {LR:g}": cur_runs, f"sgd {SGD_LR:g}": sgd_runs[SGD_LR]}, f"adam {LR:g}", "best Adam vs best SGD"); display(t_sgd)
OPT = "sgd" if pick(t_sgd, f"adam {LR:g}") == f"sgd {SGD_LR:g}" else "adam"
if OPT == "sgd":
    CUR = replace(CUR, optimizer="sgd", lr=SGD_LR); cur_runs = sgd_runs[SGD_LR]
print("chosen optimiser:", OPT, CUR.lr)
""")
nb.md(T["opt"])

nb.md("## 3. Batch size")
nb.code("""
bs_runs = {str(bs): (cur_runs if bs == 64 else run_seeds(replace(CUR, name=f"{OPT}_lr{CUR.lr:g}_bs{bs}", batch_size=bs)))
           for bs in [32, 64, 128]}
t = compare(bs_runs, "64", "batch size (learning rate kept fixed)"); display(t)
ct_bs = curve_test({int(k): v for k, v in bs_runs.items()})
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ct_bs.items() if k != "means"})
BS = int(ct_bs["best"]); CUR = replace(CUR, batch_size=BS); cur_runs = bs_runs[str(BS)]; print("chosen batch size:", BS)
""")
nb.md(T["bs"])

nb.md("## 4. Learning-rate schedule")
nb.code("""
tag = f"{OPT}_lr{CUR.lr:g}_bs{BS}"
sch = {"none": cur_runs, **{s: run_seeds(replace(CUR, name=f"{tag}_{s}", schedule=s)) for s in ["cosine", "step", "plateau"]}}
t = compare(sch, "none", "LR schedule (60 epochs max)"); display(t)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
for k, rs in sch.items():
    h = pd.DataFrame(rs[0]["history"])
    ax[0].plot(h.epoch, h.lr, label=k); ax[1].plot(h.epoch, h.val_loss, label=k)
ax[0].set(yscale="log", title="learning rate (seed 1)", xlabel="epoch"); ax[1].set(yscale="log", title="val loss (seed 1)", xlabel="epoch")
for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
SCHED = pick(t, "none"); CUR = replace(CUR, schedule=SCHED); cur_runs = sch[SCHED]; print("chosen schedule:", SCHED)
""")
nb.md(T["sched"])

nb.md("## 5. Loss function and class imbalance")
nb.code("""
tag = f"{OPT}_lr{CUR.lr:g}_bs{BS}_{SCHED}"
loss = {"ce": cur_runs,
        "weighted ce": run_seeds(replace(CUR, name=f"{tag}_wce", loss="weighted_ce")),
        "ce + weighted sampler": run_seeds(replace(CUR, name=f"{tag}_sampler", sampler="weighted")),
        "label smoothing": run_seeds(replace(CUR, name=f"{tag}_ls", loss="label_smoothing")),
        "focal": run_seeds(replace(CUR, name=f"{tag}_focal", loss="focal"))}
t = compare(loss, "ce", "loss / imbalance handling"); display(t)
rec = pd.DataFrame({k: pd.DataFrame([r["val"]["recall"] for r in rs]).mean() for k, rs in loss.items()}).T
print("mean per-class recall:"); display(rec.round(3))
LOSS = pick(t, "ce")
key = {"ce": {}, "weighted ce": {"loss": "weighted_ce"}, "ce + weighted sampler": {"sampler": "weighted"},
       "label smoothing": {"loss": "label_smoothing"}, "focal": {"loss": "focal"}}[LOSS]
CUR = replace(CUR, **key); print("chosen:", LOSS)
""")
nb.md(T["loss"])
nb.md("## Decisions")
nb.md(T["summary"])
nb.code("""
OPTIM = dict(optimizer=CUR.optimizer, lr=CUR.lr, batch_size=CUR.batch_size, schedule=CUR.schedule,
             loss=CUR.loss, sampler=CUR.sampler)
_ = (ARTIFACTS / "optim.json").write_text(json.dumps(OPTIM))
print(OPTIM, "-> mean val bal. acc", round(log_progress("c6 optimisation", loss[LOSS]), 4))
""")
nb.write("c6_optimisation.ipynb")
