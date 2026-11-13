import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from d6_text import T

nb = NB()
nb.md("""
# d6 · Everything combined: the size-accuracy frontier and the final models

Every model made in d2-d5 (students, distilled students, pruned, slimmed, quantized; 3 seeds each) on one plot,
the Pareto frontier, and the final models:

* **high-performance**: smallest model with mean val balanced accuracy ≥ 0.95 (rule fixed in d1);
* **high-performance, 1 pp margin**: smallest model with mean val balanced accuracy ≥ 0.96, added after this
  notebook showed that the 0.95 pick clears its floor by only 0.0006 (decided on val, before the test evaluation);
* **tiny**: smallest model with mean val balanced accuracy ≥ 0.85 (rule fixed in d1).

The teacher is the large model from c9; its val balanced accuracy is 0.968 for the seed-1738 model it uses (0.969
as a 3-seed mean). Compressed models are compared by their 3-seed means. For each pick, the path from the
teacher's architecture shows what every step cost in accuracy and saved in bytes.
""")
nb.code("""
import sys, json, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from lib import dreport as R
from lib.train import RESULTS
from lib.data import ARTIFACTS, CLASSES
from lib.data import NO_TUMOR
from lib.evaluation import bootstrap_ci
TEACH = json.loads((ARTIFACTS / "compress_rules.json").read_text())["teacher"]
recs = {}
for exp in ["d2", "d3", "d4", "d5"]:
    for f in sorted((RESULTS / exp).glob("*.json")):
        r = json.loads(f.read_text())
        if "size" not in r: continue
        recs.setdefault(f"{exp}/{re.sub(r'_s\\d{4}$', '', r['config']['name'])}", []).append(r)
recs = {k: v for k, v in recs.items() if len(v) == 3}
def kind(k, r):
    e = r["config"]["extra"]
    if e.get("ptq") == "int8": return "real int8"
    if e.get("ptq") or e.get("wbits"): return "quantized (simulated)"
    if k.endswith("_ft32") or (e.get("prune") and e["prune"]["sparsity"] == 0): return "fine-tuned only (control)"
    if e.get("prune"): return "unstructured pruned"
    if e.get("slim"): return "slimmed"
    if e.get("kd"): return "distilled student"
    return "student, labels only"
TA = R.table(recs); TA["kind"] = [kind(k, v[0]) for k, v in recs.items()]
print(len(TA), "compressed configurations,", 3 * len(TA), "models"); TA.groupby("kind").size()
""")
nb.code("""
fig, ax = plt.subplots(figsize=(12, 6.5))
sim = TA[TA.kind != "real int8"]   # real int8 duplicates the simulated 8-bit points (d5); left out of the frontier
R.tradeoff_plot({k: sim[sim.kind == k] for k in ["student, labels only", "distilled student", "fine-tuned only (control)",
                                                 "slimmed", "unstructured pruned", "quantized (simulated)"]},
                "d6: every compressed model", ax=ax, teacher_acc=TEACH["bal_acc"])
F = R.pareto(sim)
# d1 rules: smallest model >= 0.95 and >= 0.85. The 0.95 pick turned out borderline (it clears the floor by 0.0006),
# so a third pick with a 1 pp margin was added after this table was seen (on val, before any test evaluation).
PICK = {"high-performance": R.pick(sim, R.FLOORS["high-performance"]),
        "high-performance, 1 pp margin": R.pick(sim, R.FLOORS["high-performance"] + 0.01),
        "tiny": R.pick(sim, R.FLOORS["tiny"])}
for name, k in PICK.items():
    if k: ax.scatter([sim.loc[k, "kB"]], [sim.loc[k, "bal_acc"]], s=250, facecolor="none", edgecolor="r", lw=2, zorder=6)
plt.tight_layout(); plt.show()
display(F[["kind", "kB", "x smaller than teacher", "params", "sparsity", "wbits", "bal_acc", "sd"]].round(4))
""")
nb.md(T["frontier"])
nb.md("## The final models")
nb.code("""
def path(ref):
    \"\"\"Follow init/slim/src links from a record back to the trained student, then the teacher.\"\"\"
    out = [ref]
    while True:
        r = recs.get(out[-1]) or [json.loads((RESULTS / out[-1].split("/")[0] / f"{out[-1].split('/')[1]}_s1738.json").read_text())]
        e = r[0]["config"]["extra"]
        src = e.get("init") or e.get("slim") or e.get("src")
        if not src: break
        out.append(f"{src['exp']}/{re.sub(r'_s\\d{4}$', '', src['name'])}")
    return out[::-1]
final = {}
for name, k in PICK.items():
    if k is None: print(name, ": no model reaches the floor"); continue
    rs = recs[k]; r0 = next(r for r in rs if r["config"]["seed"] == 1738)
    d = np.load(RESULTS / k.split("/")[0] / f"{r0['config']['name']}_val.npz"); y, p = d["y"], d["prob"]
    lo, hi = bootstrap_ci(y, p.argmax(1))
    v = r0["val"]
    final[name] = {"record": k, "kind": TA.loc[k, "kind"], "stored kB": round(float(TA.loc[k, "kB"]), 2),
                   "x smaller than teacher": round(TEACH["bytes"] / float(TA.loc[k, "bytes"]), 1),
                   "parameters": int(TA.loc[k, "params"]), "sparsity": round(float(TA.loc[k, "sparsity"]), 3),
                   "weight bits": int(TA.loc[k, "wbits"]),
                   "mean val bal. acc (3 seeds)": round(float(TA.loc[k, "bal_acc"]), 4), "sd": round(float(TA.loc[k, "sd"]), 4),
                   "seed 1738 bal. acc": round(v["bal_acc"], 4), "seed 1738 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                   **{f"recall {c}": v["recall"][c] for c in CLASSES},
                   "tumour sensitivity": round(v["tumor_sens"], 4), "tumour specificity": round(v["tumor_spec"], 4),
                   "tumour AUC": round(v["tumor_auc"], 4)}
teacher_row = {"record": "c9/final (teacher)", "stored kB": round(TEACH["bytes"] / 1024, 2), "x smaller than teacher": 1.0,
               "parameters": 276628, "weight bits": 32, "seed 1738 bal. acc": TEACH["bal_acc"]}
FT = pd.DataFrame({"teacher": teacher_row, **final}).T
display(FT.T.astype(object).where(FT.T.notna(), ""))
_ = (ARTIFACTS / "compress_final.json").write_text(json.dumps(final, indent=1, default=str))
""")
nb.code("""
fig, axes = plt.subplots(1, len(final), figsize=(7 * len(final), 4.2), squeeze=False)
for ax, (name, f) in zip(axes[0], final.items()):
    steps = path(f["record"])
    kb = [TEACH["bytes"] / 1024] + [float(TA.loc[s, "kB"]) if s in TA.index else np.nan for s in steps]
    acc = [TEACH["bal_acc"]] + [float(TA.loc[s, "bal_acc"]) if s in TA.index else np.nan for s in steps]
    labels = ["teacher"] + [s.split("/")[1] for s in steps]
    ax.plot(kb, acc, "o-")
    for x, yv, l in zip(kb, acc, labels): ax.annotate(l, (x, yv), fontsize=7, xytext=(4, -10), textcoords="offset points")
    for fl in R.FLOORS.values(): ax.axhline(fl, ls="--", color="grey", lw=1)
    ax.set(xscale="log", xlabel="stored kB (log)", ylabel="mean val balanced accuracy", title=f"path to the {name} model")
    ax.grid(alpha=0.3, which="both"); ax.invert_xaxis()
    print(name, ":", " -> ".join(labels))
plt.tight_layout(); plt.show()
""")
nb.md(T["final"])
nb.write("d6_final.ipynb")
