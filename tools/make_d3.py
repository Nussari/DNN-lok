import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from d3_text import T

nb = NB()
nb.md("""
# d3 · Distillation vs training on the labels alone

**Knowledge distillation** (Hinton et al. 2015): the student is trained to match the teacher's softened output
distribution as well as the labels,

  loss = α · T² · KL( softmax(teacher / T) ‖ softmax(student / T) ) + (1 − α) · label loss.

The teacher's probabilities carry more than the label: *how* glioma-like a meningioma looks, which images are
ambiguous. T (temperature) softens them so that small probabilities still carry signal; α weighs teacher against
labels. The teacher sees exactly the same augmented batch as the student.

**Plan.**
1. Tune T ∈ {2, 4, 8} and α ∈ {0.5, 0.9} on two reference students from d2: the smallest with mean ≥ 0.95 and the
   smallest with mean ≥ 0.85 (the two floors). Highest mean (over both references) wins.
2. Train every d2 frontier architecture with that setting, and compare each with the same architecture trained on
   the labels only (d2): same seeds, same recipe, only the loss differs.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from dataclasses import replace
from lib import compress as C, dreport as R
from lib.train import RESULTS, build_model
from lib.data import ARTIFACTS
FRONT = json.loads((ARTIFACTS / "d2_frontier.json").read_text())
W = lambda w0: (w0, 2 * w0, 4 * w0, 8 * w0)
arch_cfg = lambda a, exp, name: R.student(exp, name, block=a["block"], kernel=a["kernel"], stem_kernel=a["kernel"], widths=W(a["w0"]))
TEACHER_ARCH = "single_k5_w16"
tsize = C.size_report(build_model(R.teacher_cfg(), 1))

def plain_runs(a):
    if a["name"] == TEACHER_ARCH:   # the teacher's own architecture trained on labels = c9
        return [dict(json.loads((RESULTS / "c9" / f"final_s{s}.json").read_text()), size=tsize) for s in R.SEEDS3]
    return R.run_seeds(arch_cfg(a, "d2", a["name"]))

plain = {a["name"]: plain_runs(a) for a in FRONT}
tp = R.table(plain)
refs = {}
for label, fl in R.FLOORS.items():
    k = R.pick(tp, fl)
    refs[label] = k if k is not None else tp.bal_acc.astype(float).idxmax()
print("d2 frontier:", list(plain)); print("reference students:", refs)
""")
nb.md("## 1. Temperature and weight")
nb.code("""
GRID = [(Tk, a) for Tk in [2, 4, 8] for a in [0.5, 0.9]]
arch = {a["name"]: a for a in FRONT}
sweep = {}
for label, ref in refs.items():
    sweep[label] = {"labels only": plain[ref]}
    for Tk, a in GRID:
        cfg = replace(arch_cfg(arch[ref], "d3", f"{ref}_kdT{Tk}a{a}"), extra={"kd": {"T": Tk, "alpha": a}})
        sweep[label][f"T={Tk}, a={a}"] = R.run_seeds(cfg)
tabs = {label: R.table(s) for label, s in sweep.items()}
for label, tt in tabs.items():
    print(f"reference '{label}': {refs[label]}"); display(tt[["kB", "bal_acc", "sd", "tumor_auc", "no_tumor_rec"]])
mean_kd = pd.concat({l: tt.bal_acc.astype(float) for l, tt in tabs.items()}, axis=1).drop("labels only")
mean_kd["mean of both"] = mean_kd.mean(axis=1)
display(mean_kd.round(4))
best = mean_kd["mean of both"].idxmax()
KD = dict(T=int(best.split(",")[0].split("=")[1]), alpha=float(best.split("a=")[1]))
print("chosen:", KD)
fig, axes = plt.subplots(1, len(tabs), figsize=(6 * len(tabs), 3.6))
for ax, (label, tt) in zip(np.atleast_1d(axes), tabs.items()):
    for a, mk in [(0.5, "o"), (0.9, "s")]:
        ks = [f"T={Tk}, a={a}" for Tk in [2, 4, 8]]
        ax.errorbar([2, 4, 8], tt.loc[ks, "bal_acc"].astype(float), yerr=tt.loc[ks, "sd"].astype(float), marker=mk, capsize=3, label=f"alpha {a}")
    ax.axhline(float(tt.loc["labels only", "bal_acc"]), color="k", ls="--", label="labels only")
    ax.set(xscale="log", xticks=[2, 4, 8], xticklabels=["2", "4", "8"], xlabel="temperature T", ylabel="val balanced accuracy",
           title=f"{label} reference: {refs[label]}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
nb.md(T["sweep"])
nb.md("## 2. Distillation vs labels only, at every size")
nb.code("""
kd = {}
for a in FRONT:
    cfg = replace(arch_cfg(a, "d3", f"{a['name']}_kdT{KD['T']}a{KD['alpha']}"), extra={"kd": KD})
    kd[a["name"]] = R.run_seeds(cfg)
tk, tl = R.table(kd), R.table(plain)
cmp = pd.DataFrame({"kB": tl.kB, "labels only": tl.bal_acc, "distilled": tk.bal_acc,
                    "gain (pp)": (tk.bal_acc.astype(float) - tl.bal_acc.astype(float)) * 100,
                    "sd labels": tl.sd, "sd distilled": tk.sd}).sort_values("kB")
# paired by seed: same seed, same initialisation and data order; only the loss differs
cmp["seeds improved"] = [sum(k["val"]["bal_acc"] > p["val"]["bal_acc"] for k, p in zip(kd[n], plain[n])) for n in cmp.index]
display(cmp.round(4))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
R.tradeoff_plot({"labels only (d2)": tl, f"distilled (T={KD['T']}, a={KD['alpha']})": tk}, "distillation vs labels only",
                ax=ax[0], teacher_acc=json.loads((ARTIFACTS / "compress_rules.json").read_text())["teacher"]["bal_acc"])
for n in cmp.index:
    ax[0].plot([tl.loc[n, "kB"]] * 2, [tl.loc[n, "bal_acc"], tk.loc[n, "bal_acc"]], color="grey", lw=0.8)
ax[1].bar(range(len(cmp)), cmp["gain (pp)"], color=["C2" if g > 0 else "C3" for g in cmp["gain (pp)"]])
ax[1].set_xticks(range(len(cmp)), [f"{n}\\n{kb:.1f} kB" for n, kb in zip(cmp.index, cmp.kB)], rotation=30, ha="right", fontsize=8)
ax[1].axhline(0, color="k", lw=0.8); ax[1].set(ylabel="distilled minus labels only (pp)", title="gain from distillation, by size")
ax[1].grid(alpha=0.3, axis="y"); plt.tight_layout(); plt.show()
""")
nb.md(T["compare"])
nb.md("## 3. Decision: how each student is trained from here on")
nb.code("""
best_student = {n: ("distilled" if float(tk.loc[n, "bal_acc"]) > float(tl.loc[n, "bal_acc"]) else "labels only") for n in cmp.index}
out = [{**a, "method": best_student[a["name"]], "kd": KD if best_student[a["name"]] == "distilled" else None,
        "run": f"d3/{a['name']}_kdT{KD['T']}a{KD['alpha']}" if best_student[a["name"]] == "distilled"
               else (f"d2/{a['name']}" if a["name"] != TEACHER_ARCH else None),
        "bal_acc": float(max(tk.loc[a["name"], "bal_acc"], tl.loc[a["name"], "bal_acc"])),
        "bytes": int(tl.loc[a["name"], "bytes"])} for a in FRONT]
_ = (ARTIFACTS / "d3_students.json").write_text(json.dumps({"kd": KD, "students": out}, indent=1))
pd.DataFrame(out)[["name", "method", "bytes", "bal_acc"]]
""")
nb.md(T["decision"])
nb.write("d3_distillation.ipynb")
