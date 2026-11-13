import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from d2_text import T

nb = NB()
nb.md("""
# d2 · Student architecture: how small can a model trained on the data alone get?

Starting point from the c-notebooks: the teacher's layout (5 conv layers, BatchNorm, average pooling, 4 stages
that halve the resolution, global average pooling) and its training recipe. c7 showed 5x5 kernels beat 3x3, but
d1 showed the cost of a conv grows with kernel area and the square of the width. So the students vary the three
levers that set the size:

| lever | options |
|---|---|
| width (first stage; the others double) | 32, **16 (teacher)**, 8, 4, 2 |
| kernel | **5x5 (teacher)**, 3x3 |
| block | **plain conv (teacher)**, depthwise-separable (depthwise k x k + pointwise 1x1) |

15 students, 3 seeds each, trained on the labels only (no teacher). Distillation is d3. The teacher's own
architecture is the c9 runs (no retraining).
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, ".")
from lib import compress as C, dreport as R
from lib.train import RESULTS, build_model
from lib.data import ARTIFACTS
W = lambda w0: (w0, 2 * w0, 4 * w0, 8 * w0)
GRID = {("single", 5): [8, 4, 2], ("single", 3): [16, 8, 4, 2],
        ("dwsep", 5): [32, 16, 8, 4, 2], ("dwsep", 3): [16, 8, 4]}
runs = {}
for (block, k), ws in GRID.items():
    for w0 in ws:
        cfg = R.student("d2", f"{block}_k{k}_w{w0}", block=block, kernel=k, stem_kernel=k, widths=W(w0))
        runs[(block, k, w0)] = R.run_seeds(cfg)
# the teacher's architecture trained on the data = the c9 runs (same recipe); sizes from the architecture
tsize = C.size_report(build_model(R.teacher_cfg(), 1))
runs[("single", 5, 16)] = [dict(json.loads((RESULTS / "c9" / f"final_s{s}.json").read_text()), size=tsize) for s in R.SEEDS3]
print(len(runs), "architectures,", sum(len(v) for v in runs.values()), "runs")
""")
nb.md("## 1. Every student")
nb.code("""
fam = lambda b, k: f"{'plain' if b == 'single' else 'depthwise-sep.'} {k}x{k}"
t = R.table({f"{fam(b, k)}, w{w0}": rs for (b, k, w0), rs in runs.items()})
t["family"] = [fam(b, k) for (b, k, w0) in runs]; t["width"] = [w0 for (b, k, w0) in runs]
t = t.sort_values("bytes")
display(t[["family", "width", "params", "kB", "x smaller than teacher", "bal_acc", "sd", "tumor_auc", "no_tumor_rec"]])
""")
nb.code("""
groups = {f: t[t.family == f] for f in t.family.unique()}
ax = R.tradeoff_plot(groups, "d2: students trained on the labels only", teacher_acc=float(t.loc["plain 5x5, w16", "bal_acc"]))
for i, g in enumerate(groups.values()):   # connect each family's widths (same colours as the points)
    g = g.sort_values("bytes"); ax.plot(g.kB.astype(float), g.bal_acc.astype(float), "-", lw=0.8, alpha=0.5, color=f"C{i}")
plt.tight_layout(); plt.show()
""")
nb.md(T["all"])
nb.md("## 2. The frontier")
nb.code("""
F = R.pareto(t)
display(F[["family", "width", "params", "kB", "x smaller than teacher", "bal_acc", "sd"]])
for name, fl in R.FLOORS.items():
    k = R.pick(t, fl); print(f"smallest >= {fl} ({name}):", k, "" if k is None else f"{t.loc[k, 'kB']:.1f} kB, {t.loc[k, 'bal_acc']:.4f}")
_ = (ARTIFACTS / "d2_frontier.json").write_text(json.dumps(
    [{"name": f"{b}_k{k}_w{w0}", "block": b, "kernel": k, "w0": w0} for (b, k, w0) in runs if f"{fam(b, k)}, w{w0}" in F.index]))
""")
nb.md(T["frontier"])
nb.write("d2_students.ipynb")
