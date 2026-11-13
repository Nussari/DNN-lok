import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from d4_text import T

nb = NB()
nb.md("""
# d4 · Pruning: zero out weights, or remove channels?

Two ways to remove parameters from a trained student, each followed by fine-tuning:

* **Unstructured (magnitude) pruning.** Set the smallest |w| to zero, globally across all layers, so layers that
  matter less lose more. The model keeps its shape; the zeros only save bytes if stored sparsely (d1: 1-bit mask
  plus nonzero values) and need sparse kernels to run faster. Sparsity is ramped up gradually over 10 epochs
  (Zhu & Gupta 2017) while training continues, then held while fine-tuning. Levels: 50, 75, 90, 95 %.
* **Structured pruning (channel slimming).** Remove whole channels, keeping those with the largest BatchNorm scale
  |γ| (Liu et al. 2017), then fine-tune. The result is simply a narrower dense model, so it runs faster
  everywhere. The real question is whether **slimming a wider trained model beats training the narrow model
  directly** (d2/d3), so it is compared exactly against that.

**Starting points** (from d3): the two reference students (smallest at or above 0.95 and at or above 0.85) and
the next larger frontier model above the 0.95 one (the teacher's own layout, 1,080 kB, distilled), each
trained the way d3 found best for it.

**Fine-tuning recipe**: as the student was trained (with the teacher if it was distilled), but at **lr 0.003**
(a tenth of the training lr: the start is already a good model, and lr 0.03 would throw much of it away),
30 epochs, early stopping with patience 10 counted only once the target sparsity is reached.

**Control.** Every starting point is also fine-tuned with exactly this recipe *without* pruning ("0 %, fine-tuned
only"). Pruning is judged against that, not against the original model, so 30 extra epochs at a lower learning
rate are not credited to pruning.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from dataclasses import replace
from lib import compress as C, dreport as R
from lib.train import RESULTS
from lib.data import ARTIFACTS
D3 = json.loads((ARTIFACTS / "d3_students.json").read_text())
KD = D3["kd"]
S = pd.DataFrame(D3["students"]).set_index("name").sort_values("bytes")
S = S[S.run.notna()]          # a start needs stored weights for all 3 seeds
W = lambda w0: (w0, 2 * w0, 4 * w0, 8 * w0)
def cfg_of(name, exp, newname, **kw):
    a = S.loc[name]
    c = R.student(exp, newname, block=a.block, kernel=int(a.kernel), stem_kernel=int(a.kernel), widths=W(int(a.w0)))
    return replace(c, **kw)
def runs_of(name):
    exp, n = S.loc[name, "run"].split("/")
    return [json.loads((RESULTS / exp / f"{n}_s{s}.json").read_text()) for s in R.SEEDS3]
hi = S[S.bal_acc >= R.FLOORS["high-performance"]]
lo = S[S.bal_acc >= R.FLOORS["tiny"]]
starts = []
if len(hi): starts.append(hi.index[0])
bigger = S[S.bytes > (S.loc[starts[0], "bytes"] if starts else 0)]
if len(bigger): starts.append(bigger.index[0])
if len(lo) and lo.index[0] not in starts: starts.append(lo.index[0])
print("starting points:"); display(S.loc[starts, ["block", "kernel", "w0", "method", "bytes", "bal_acc"]])
FT = dict(lr=0.003, epochs=30, patience=10)
def ft_extra(name):   # fine-tune the way the student was trained
    return {"kd": KD} if S.loc[name, "method"] == "distilled" else {}
""")
nb.md("## 1. Unstructured magnitude pruning")
nb.code("""
LEVELS = [0.5, 0.75, 0.9, 0.95]
unst = {}
for st in starts:
    unst[st] = {"0 %": runs_of(st)}
    # control: the same fine-tuning without pruning, so that any gain is not mistaken for an effect of pruning
    base0 = cfg_of(st, "d4", f"{st}_{S.loc[st, 'method'][:4]}_prune0", **FT)
    unst[st]["0 %, fine-tuned only"] = [R.run(R.with_src(base0, S.loc[st, "run"], "init", s, **ft_extra(st),
                                                         prune={"sparsity": 0.0, "ramp": 10})) for s in R.SEEDS3]
    for sp in LEVELS:
        base = cfg_of(st, "d4", f"{st}_{S.loc[st, 'method'][:4]}_prune{int(sp * 100)}", **FT)
        unst[st][f"{int(sp * 100)} %"] = [R.run(R.with_src(base, S.loc[st, "run"], "init", s, **ft_extra(st),
                                                          prune={"sparsity": sp, "ramp": 10})) for s in R.SEEDS3]
tabs_u = {st: R.table(v) for st, v in unst.items()}
for st, tt in tabs_u.items():
    print(st); display(tt[["sparsity", "kB", "x smaller than teacher", "bal_acc", "sd", "tumor_auc"]])
""")
nb.code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for i, (st, tt) in enumerate(tabs_u.items()):
    ax[0].errorbar(tt.sparsity.astype(float) * 100, tt.bal_acc.astype(float), yerr=tt.sd.astype(float).fillna(0), marker="o",
                   capsize=3, label=st, color=f"C{i}")
ax[0].set(xlabel="sparsity (% of weights removed)", ylabel="val balanced accuracy", title="accuracy vs sparsity")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
for fl in R.FLOORS.values(): ax[0].axhline(fl, ls="--", color="grey", lw=1)
R.tradeoff_plot({f"{st} pruned": tt for st, tt in tabs_u.items()}, "unstructured pruning: size vs accuracy", ax=ax[1])
plt.tight_layout(); plt.show()
""")
nb.md(T["unstructured"])
nb.md("## 2. Structured pruning (channel slimming) vs training the narrow model directly")
nb.code("""
REG = {}   # label -> (runs, stored record "exp/name" without seed): every model of this notebook, for the frontier
for st in S.index:
    REG[f"{st} ({S.loc[st, 'method']})"] = (runs_of(st), S.loc[st, "run"])
for st, v in unst.items():
    REG[f"{st} fine-tuned, not pruned"] = (v["0 %, fine-tuned only"], f"d4/{st}_{S.loc[st, 'method'][:4]}_prune0")
    for sp in LEVELS:
        REG[f"{st} pruned {int(sp * 100)} %"] = (v[f"{int(sp * 100)} %"], f"d4/{st}_{S.loc[st, 'method'][:4]}_prune{int(sp * 100)}")
struct = {}
for st in starts[:2]:
    w0 = int(S.loc[st, "w0"])
    fam = f"{S.loc[st, 'block']}_k{int(S.loc[st, 'kernel'])}"
    for f in [2, 4]:
        nw = w0 // f
        if nw < 1: continue
        sname = f"{st}_{S.loc[st, 'method'][:4]}_slim_w{nw}"
        base = cfg_of(st, "d4", sname, widths=W(nw), **FT)
        sl = [R.run(R.with_src(base, S.loc[st, "run"], "slim", s, **ft_extra(st))) for s in R.SEEDS3]
        # the same narrow architecture trained directly: labels only (d2) and distilled (d3 setting)
        dname = f"{fam}_w{nw}_kdT{KD['T']}a{KD['alpha']}"
        opts = {f"slimmed from w{w0}, fine-tuned": (sl, f"d4/{sname}"),
                "trained directly, labels only": (R.run_seeds(cfg_of(st, "d2", f"{fam}_w{nw}", widths=W(nw))), f"d2/{fam}_w{nw}"),
                "trained directly, distilled": (R.run_seeds(cfg_of(st, "d3", dname, widths=W(nw), extra={"kd": KD})), f"d3/{dname}")}
        struct[(st, nw)] = opts
        for k, v in opts.items():
            REG[f"{fam}_w{nw}: {k}"] = v
rows = []
for (st, nw), opts in struct.items():
    tt = R.table({k: v[0] for k, v in opts.items()})
    for k, r in tt.iterrows():
        rows.append({"from": st, "to width": nw, "how": k, "kB": r.kB, "bal_acc": r.bal_acc, "sd": r.sd})
TS = pd.DataFrame(rows); display(TS.round(4))
""")
nb.md(T["structured"])
nb.md("## 3. Frontier after pruning")
nb.code("""
TA = R.table({k: v[0] for k, v in REG.items()})
F = R.pareto(TA)
display(F[["kB", "x smaller than teacher", "params", "sparsity", "bal_acc", "sd"]].round(4))
kind = lambda k: "unstructured pruned" if " pruned " in k else ("slimmed / trained narrow" if ": " in k else "students (d3)")
grp = {g: TA.loc[[k for k in TA.index if kind(k) == g]] for g in ["students (d3)", "unstructured pruned", "slimmed / trained narrow"]}
R.tradeoff_plot(grp, "after d4: every student and every pruned model",
                teacher_acc=json.loads((ARTIFACTS / "compress_rules.json").read_text())["teacher"]["bal_acc"])
plt.tight_layout(); plt.show()
front = [{"label": k, "run": REG[k][1], "bytes": int(F.loc[k, "bytes"]), "bal_acc": float(F.loc[k, "bal_acc"]),
          "sparsity": float(F.loc[k, "sparsity"])} for k in F.index]
_ = (ARTIFACTS / "d4_frontier.json").write_text(json.dumps(front, indent=1))
""")
nb.md(T["frontier"])
nb.write("d4_pruning.ipynb")
