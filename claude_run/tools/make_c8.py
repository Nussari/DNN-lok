import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c8_text import T

nb = NB()
nb.md("""
# c8 · Step 8: Regularisation

Both architecture candidates from c7 (the best **small** and the best **deep** model) are regularised, because
the benefit of regularisation depends on how much a model overfits. Order: augmentation → weight decay →
dropout, each starting from the previous winner.

| # | Technique | Options |
|---|---|---|
| 1 | augmentation (train only, on the GPU) | **none**, geometric, mild (geometric + intensity), strong |
| 2 | weight decay (decoupled, i.e. AdamW for Adam) | **0**, two strengths |
| 3 | dropout before the classifier | **0**, 0.3 |
| 4 | small vs deep | best regularised version of each |

All runs use the protocol from c4 (60-epoch cap, patience 20 on val balanced accuracy), which leaves augmented
runs, which learn more slowly, enough time. Seeds: 3 for the small model; 2 for the
deep model, because it is ~5-10x slower. The `compare` rule accounts for the smaller n with a wider threshold.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, ".")
from dataclasses import replace
from lib.train import Config, run
from lib.report import run_seeds, compare, pick, log_progress, SEEDS3
from lib.data import ARTIFACTS
from lib.report import protocol; proto = protocol()   # c4 protocol with the raised epoch cap
INPUT = json.loads((ARTIFACTS / "input.json").read_text())
OPTIM = json.loads((ARTIFACTS / "optim.json").read_text())
ARCH = {k: {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
        for k, v in json.loads((ARTIFACTS / "arch.json").read_text()).items()}
BASE = replace(Config(exp="c8", name="x", monitor=proto["monitor"], patience=proto["patience"], epochs=proto["epochs"]), **INPUT, **OPTIM)
CAND = {"small": (replace(BASE, **ARCH["small"]), SEEDS3), "deep": (replace(BASE, **ARCH["deep"]), SEEDS3[:2])}
print(OPTIM); print(ARCH)
""")

nb.md("## 1. Augmentation")
nb.code("""
def gap(rs):  # train minus val balanced accuracy at the selected epoch (overfitting indicator)
    return np.mean([r["history"][r["best_epoch"] - 1]["train_bal_acc"] - r["val"]["bal_acc"] for r in rs])

aug_runs, chosen, tables = {}, {}, {}
for m, (cfg, seeds) in CAND.items():
    aug_runs[m] = {a: run_seeds(replace(cfg, name=f"{m}_aug_{a}", aug=a), seeds) for a in ["none", "geometric", "mild", "strong"]}
    tables[m] = compare(aug_runs[m], "none", f"{m}: augmentation")
    tables[m]["train-val gap"] = [round(gap(rs), 4) for rs in aug_runs[m].values()]
    display(tables[m])
    chosen[m] = pick(tables[m], "none")
    print(m, "-> chosen augmentation:", chosen[m])
""")
nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
for ax, m in zip(axes, CAND):
    for a, rs in aug_runs[m].items():
        h = pd.DataFrame(rs[0]["history"])
        l, = ax.plot(h.epoch, h.val_bal_acc, label=f"{a} (val)")
        ax.plot(h.epoch, h.train_bal_acc, ls=":", color=l.get_color(), alpha=0.7)
    ax.set(title=f"{m}: balanced accuracy, dotted = train (seed 1)", xlabel="epoch", ylim=(0.6, 1.01)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
nb.md(T["aug"])

nb.md("## 2. Weight decay")
nb.code("""
WD = [0.01, 0.1] if OPTIM["optimizer"] in ("adam", "adamw") else [5e-4, 5e-3]
wd_runs = {}
for m, (cfg, seeds) in CAND.items():
    cfg = replace(cfg, aug=chosen[m]); CAND[m] = (cfg, seeds)
    opt = "adamw" if OPTIM["optimizer"] in ("adam", "adamw") else OPTIM["optimizer"]
    wd_runs[m] = {"0": aug_runs[m][chosen[m]],
                  **{f"{w:g}": run_seeds(replace(cfg, name=f"{m}_aug_{chosen[m]}_wd{w:g}", optimizer=opt, weight_decay=w), seeds) for w in WD}}
    t = compare(wd_runs[m], "0", f"{m}: weight decay ({opt})"); display(t)
    w = pick(t, "0")
    if w != "0":
        CAND[m] = (replace(cfg, optimizer=opt, weight_decay=float(w)), seeds)
    print(m, "-> chosen weight decay:", w)
""")
nb.md(T["wd"])

nb.md("## 3. Dropout")
nb.code("""
do_runs = {}
for m, (cfg, seeds) in CAND.items():
    cur = wd_runs[m]["0" if cfg.weight_decay == 0 else f"{cfg.weight_decay:g}"]
    do_runs[m] = {"0": cur, "0.3": run_seeds(replace(cfg, name=f"{m}_aug_{cfg.aug}_wd{cfg.weight_decay:g}_do0.3", dropout=0.3), seeds)}
    t = compare(do_runs[m], "0", f"{m}: dropout"); display(t)
    d = pick(t, "0")
    if d != "0":
        CAND[m] = (replace(cfg, dropout=0.3), seeds)
    print(m, "-> chosen dropout:", d)
""")
nb.md(T["dropout"])

nb.md("## 4. Small vs deep, both regularised")
nb.code("""
final = {m: do_runs[m]["0.3" if cfg.dropout else "0"] for m, (cfg, _) in CAND.items()}
t = compare(final, "small", "regularised small vs regularised deep"); display(t)
for m, rs in final.items():
    print(m, "per-class recall:", pd.DataFrame([r["val"]["recall"] for r in rs]).mean().round(3).to_dict())
""")
nb.md(T["final"])
nb.md("## Decisions")
nb.md(T["summary"])
nb.code("""
WINNER = pick(t, "small")   # highest mean val balanced accuracy wins, whatever its size
cfg = CAND[WINNER][0]
REG = dict(aug=cfg.aug, optimizer=cfg.optimizer, weight_decay=cfg.weight_decay, dropout=cfg.dropout)
_ = (ARTIFACTS / "reg.json").write_text(json.dumps({"winner": WINNER, "arch": ARCH[WINNER], **REG}, default=list))
for m in CAND: _ = log_progress(f"c8 regularised {m}", final[m])
WINNER, REG
""")
nb.write("c8_regularisation.ipynb")
