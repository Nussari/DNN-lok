"""Figures for the report, at column width. Run from the repository root: python report/make_figures.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / "artifacts", Path(__file__).resolve().parent
DATA = ROOT / "data" / "bdneuro"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42})
W = 3.25                                                         # column width, inches

# Fig 1: val balanced accuracy of the chosen configuration after each step
p = json.loads((ART / "progress.json").read_text())
final = json.loads((ROOT / "results" / "c9" / "final_s1738.json").read_text())["val"]["bal_acc"]
steps = ["first\nCNN", "input", "optimi-\nzation", "archi-\ntecture", "regulari-\nzation", "final\nmodel"]
small = [p["c4 initial"], p["c5 input"], p["c6 optimisation"], p["c7 small arch"], p["c8 regularised small"]]
fig, ax = plt.subplots(figsize=(W, 1.65))
ax.plot(range(5), small, "o-", color="C0", ms=3.5, label="small candidate (plain, 5 layers)")
ax.plot([3, 4], [p["c7 deep arch"], p["c8 regularised deep"]], "s--", color="C1", ms=3.5,
        label="deep candidate (residual, 17 layers)")
ax.scatter([5], [final], marker="*", s=45, color="C3", zorder=3, label=f"final model, seed 1738 ({final:.3f})")
for x, v in enumerate(small):
    ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, -9), ha="center", fontsize=5.5)
ax.set_xticks(range(6), steps)
ax.set(ylabel="val balanced accuracy", ylim=(0.86, 0.985), xlim=(-0.3, 5.3))
ax.grid(alpha=0.3); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(pad=0.2); fig.savefig(OUT / "fig_progress.pdf"); plt.close(fig)

# Fig 2: clean-test balanced accuracy vs stored size
t = json.loads((ART / "test_results.json").read_text())
sizes = {"large": 1105488 / 1024, "18.8 kB": 18.79, "9.3 kB": 9.29, "1.4 kB": 1.41}
fig, ax = plt.subplots(figsize=(W, 1.5))
for i, (n, kb) in enumerate(sizes.items()):
    r = t[f"{n} | clean"]
    lo, hi = (float(v) for v in r["95 % CI"].strip("[]").split(","))
    ax.errorbar(kb, r["bal. acc"], yerr=[[r["bal. acc"] - lo], [hi - r["bal. acc"]]], fmt="o", ms=3.5,
                capsize=2, color=f"C{i}")
    off = {"large": (-6, -3, "right"), "18.8 kB": (6, 3, "left"), "9.3 kB": (-6, -3, "right"), "1.4 kB": (6, -3, "left")}[n]
    ax.annotate(f"{n}: {r['bal. acc']:.3f}", (kb, r["bal. acc"]), textcoords="offset points", xytext=off[:2],
                ha=off[2], va="center", fontsize=6)
ax.set(xscale="log", xlabel="stored size (kB, log scale)", ylabel="clean test bal. acc.", ylim=(0.76, 1.0),
       xlim=(0.9, 2500))
ax.grid(alpha=0.3)
fig.tight_layout(pad=0.2); fig.savefig(OUT / "fig_size.pdf"); plt.close(fig)

# Fig 3: example errors on clean test (as stored)
ex = [("test/glioma/glioma_test_00322.jpg", "missed: glioma\n(CT-like scan)"),
      ("test/glioma/glioma_test_00172.jpg", "missed: glioma\n(FLAIR-like)"),
      ("test/no_tumor/no_tumor_test_00103.jpg", "false alarm\n(white borders)"),
      ("test/no_tumor/no_tumor_test_00007.jpg", "false alarm\n(T2, tilted)"),
      ("test/glioma/glioma_test_00010.jpg", "typical, correct\n(T1 + contrast)")]
fig, axes = plt.subplots(1, 5, figsize=(W, 1.05))
for a, (f, lab) in zip(axes, ex):
    with Image.open(DATA / f) as im:
        a.imshow(im.convert("L"), cmap="gray")
    a.set_title(lab, fontsize=5.5); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.4)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.74, wspace=0.08); fig.savefig(OUT / "fig_errors.pdf", dpi=300); plt.close(fig)
print("figures written to", OUT)
