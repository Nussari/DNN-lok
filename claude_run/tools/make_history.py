"""Build history_all_rounds.ipynb: every round, every configuration tried, its outcome and the decision it led to.

    python tools/make_history.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB

nb = NB()
nb.md("""
# History of all rounds

The c-notebooks show the **latest** round only. This notebook keeps the record of every round: what was tried,
what it scored, and which decision it led to. Nothing here is recomputed; it reads the saved run records
(config, training history and val scores of every run) from `history/` and `results/`.

| Round | Where | Selection rule | Epoch cap | Notebooks |
|---|---|---|---|---|
| r0 | Mac (M1, MPS) | tie rule: change only if better by > 2·σ_seed·√(2/n) ≈ 5 pp, else keep simpler/default | 60 | c1 to c5 complete, c6 half, c7 started |
| r1 | desktop (RTX 2080 Ti) | tie rule (as r0) | 60 | c6 to c10 |
| r2 | desktop | highest mean val balanced accuracy wins | 60 | c5 to c10 |
| r3 | desktop | highest mean val balanced accuracy wins | 200 (early stopping, patience 20) | c5 to c10 |

Every number is val balanced accuracy on clean val (832 images), mean over seeds unless stated. The test set has
not been used in any round.
""")
nb.code("""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Markdown, display
pd.set_option("display.max_rows", 500, "display.max_colwidth", 80, "display.width", 200)

H = Path("history")
r3 = H / "r3_desktop_no-tie-rule_200ep"
ROUNDS = {   # round -> (results dir, artifacts dir)
    "r0": (H / "r0_mac_committed/claude_run/results", H / "r0_mac_committed/claude_run/artifacts"),
    "r1": (H / "r1_desktop_tie-rule_60ep_partial/results", None),
    "r2": (H / "r2_desktop_no-tie-rule_60ep/results", H / "r2_desktop_no-tie-rule_60ep/artifacts"),
    "r3": ((r3 / "results", r3 / "artifacts") if r3.exists() else (Path("results"), Path("artifacts"))),
}

rows = []
for rnd, (rdir, _) in ROUNDS.items():
    for f in sorted(rdir.glob("c*/*.json")):
        r = json.loads(f.read_text()); c = r["config"]; v = r["val"]
        rows.append({"round": rnd, "exp": f.parent.name, "option": re.sub(r"_s\\d{4}$", "", c["name"]), "seed": c["seed"],
                     "bal_acc": v["bal_acc"], "macro_f1": v["macro_f1"], "tumor_auc": v["tumor_auc"],
                     "best_ep": r["best_epoch"], "epochs_run": r["epochs_run"], "cap": c["epochs"],
                     "params": r["n_params"], "device": r.get("device", "?")})
runs = pd.DataFrame(rows)
# A round's results folder also holds files from earlier rounds (reused from the cache, or never revisited).
# Identical records are the same training run: keep it once, under the round that trained it.
runs["key"] = runs[["exp", "option", "seed", "bal_acc", "epochs_run"]].astype(str).agg("|".join, axis=1)
runs = runs.drop_duplicates("key", keep="first").drop(columns="key")
print(f"{len(runs)} distinct training runs, by the round that trained them:", runs.groupby("round").size().to_dict())
""")

nb.md("## 1. Decisions per round")
nb.md("""
What each round decided at each step. r1's artifacts were overwritten by r2 before history keeping existed; its
decisions are taken from `history/r1_desktop_tie-rule_60ep_partial/RECORD.md` (copied from the r1 logs).
""")
nb.code("""
def load(art, name):
    p = art / name if art else None
    return json.loads(p.read_text()) if p and p.exists() else {}

def arch_str(a):
    if not a: return None
    return (f"{'residual' if a['residual'] else 'plain'} {1 + 4 * a['depths'][0] if a['block'] == 'single' else 1 + 8 * a['depths'][0]} layers, "
            f"{a['kernel']}x{a['kernel']}, {a['downsample']}, width {a['widths'][0]}")

def final_score(rdir):
    f = rdir / "c9" / "final_s1738.json"
    return round(json.loads(f.read_text())["val"]["bal_acc"], 4) if f.exists() else None

dec = {}
for rnd, (rdir, art) in ROUNDS.items():
    i, o, a, g, t = (load(art, n) for n in ["input.json", "optim.json", "arch.json", "reg.json", "threshold.json"])
    dec[rnd] = {"input": i and f"{i['mode']}, {i['resize']}, {i['size']} px, {i['normalize']} norm",
                "optimiser / lr": o and f"{o['optimizer']} {o['lr']:g}", "batch size": o.get("batch_size"),
                "lr schedule": o.get("schedule"), "loss": o.get("loss"),
                "small candidate": arch_str(a.get("small")), "deep candidate": arch_str(a.get("deep")),
                "winner": g.get("winner"), "augmentation": g.get("aug"), "weight decay": g.get("weight_decay"),
                "dropout": g.get("dropout"), "final model val bal. acc (seed 1738)": final_score(rdir),
                "tumour threshold": t and round(t["threshold"], 3)}
dec["r1"] = {"input": "gray, crop, 128 px, global norm", "optimiser / lr": "adam 0.03", "batch size": 32,
             "lr schedule": "none", "loss": "ce", "small candidate": "plain 5 layers, 3x3, maxpool, width 16",
             "deep candidate": "residual 17 layers, 3x3, maxpool, width 16", "winner": "small", "augmentation": "none",
             "weight decay": 0.0, "dropout": 0.0, "final model val bal. acc (seed 1738)": 0.932, "tumour threshold": 0.405}
dec["r0"]["optimiser / lr"] = dec["r0"]["optimiser / lr"] + " (c4 default; c6 unfinished on the Mac)"
D = pd.DataFrame(dec)[list(ROUNDS)].replace({"": None})
display(D)
""")

nb.md("## 2. Final model score per round")
nb.code("""
prog = {rnd: load(art, "progress.json") for rnd, (_, art) in ROUNDS.items()}
prog["r1"] = {"c4 initial": 0.8956, "c5 input": 0.8956, "c6 optimisation": 0.934, "c8 regularised small": 0.9352,
              "c8 regularised deep": 0.9486}
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
fs = {r: D.loc["final model val bal. acc (seed 1738)", r] for r in ROUNDS}
fs = {r: v for r, v in fs.items() if v is not None and not pd.isna(v)}
ax[0].bar(list(fs), list(fs.values()), color="C0"); ax[0].set(ylim=(0.85, 1), ylabel="val balanced accuracy",
          title="final model (seed 1738) per round"); ax[0].grid(alpha=0.3, axis="y")
for k, v in fs.items(): ax[0].text(k, v + 0.003, f"{v:.3f}", ha="center")
for rnd, p in prog.items():
    if p: ax[1].plot(list(p), list(p.values()), "o-", label=rnd)
ax[1].set(ylabel="mean val balanced accuracy of the chosen configuration", title="progress through the steps")
ax[1].tick_params(axis="x", rotation=30); ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout(); plt.show()
""")

nb.md("## 3. Every configuration tried, per notebook")
nb.md("""
One row per configuration (run name without the seed), one column block per round: mean val balanced accuracy,
number of seeds, and the mean number of epochs actually trained. Each run is listed under the round that trained
it; a later round that reused it from the cache shows an empty cell. Run names encode the choices made earlier in
the same round, so the same question can appear under different names in different rounds (for example
`k3s3_maxpool_plain17` in r1 and `k5s5_avgpool_plain17` in r2).
""")
nb.code("""
agg = (runs.groupby(["exp", "option", "round"])
           .agg(bal_acc=("bal_acc", "mean"), sd=("bal_acc", "std"), n=("seed", "size"), epochs=("epochs_run", "mean"))
           .round(4).reset_index())
for exp in sorted(agg.exp.unique()):
    t = agg[agg.exp == exp].pivot(index="option", columns="round", values=["bal_acc", "n", "epochs"])
    t = t.swaplevel(axis=1).sort_index(axis=1, level=0)
    display(Markdown(f"### {exp}"))
    display(t)
""")

nb.md("## 4. Round records (decisions, outcomes and observations written per round)")
nb.code("""
for rec in sorted(H.glob("r*/RECORD.md")):
    display(Markdown(rec.read_text(encoding="utf-8")))
""")

nb.md("## 5. Every single run")
nb.code("""
display(runs.sort_values(["round", "exp", "option", "seed"]).reset_index(drop=True).round(4))
""")

nb.write("history_all_rounds.ipynb")
