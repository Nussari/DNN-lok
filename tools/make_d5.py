import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from d5_text import T

nb = NB()
nb.md("""
# d5 · Quantization: 8, 4 and 2 bits

Every model on the d4 frontier is quantized: its weights are stored with *b* bits instead of 32, plus one fp32
scale per output channel (d1).

| method | what happens | cost |
|---|---|---|
| **post-training (PTQ)** | round the trained weights to the grid | none, seconds |
| **quantization-aware training (QAT)** | fine-tune with the rounding in the forward pass (straight-through gradient), so the weights settle where rounding hurts least | 20 epochs at lr 0.003 |
| **real int8 (PyTorch FX, x86)** | weights *and* activations in int8, run by PyTorch's quantized CPU kernels | calibration on 1,280 train images |

Bit widths: **8** (255 levels: 127 on each side of zero), **4** (15 levels), **2** (3 levels: −1, 0, +1 times a scale, i.e. ternary).

**Control.** QAT fine-tunes for 20 more epochs at a lower learning rate, which can help on its own. So every model
is also fine-tuned with the same recipe in fp32 ("fp32 fine-tuned"), and the cost of quantization is measured
against that. QAT keeps the pruning pattern of a pruned model (its zeros stay zero) and trains with the teacher if the model
was distilled. Sub-8-bit is simulated: the accuracy is exactly that of a model whose weights take only those
values, but no PyTorch kernel runs 4- or 2-bit weights, so there is no latency for them.

Real int8 also checks the simulation: its accuracy should match simulated 8-bit PTQ closely (it additionally
quantizes the activations).
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from dataclasses import replace
from lib import compress as C, dreport as R
from lib.train import RESULTS, Config
from lib.data import ARTIFACTS
FRONT = [f for f in json.loads((ARTIFACTS / "d4_frontier.json").read_text()) if f["run"]]
def src_runs(ref):
    exp, n = ref.split("/")
    return [json.loads((RESULTS / exp / f"{n}_s{s}.json").read_text()) for s in R.SEEDS3]
def qat_cfg(ref, bits):
    \"\"\"bits=None: the control, the same fine-tuning in fp32.\"\"\"
    exp, n = ref.split("/")
    c = src_runs(ref)[0]["config"]
    c = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.items()})
    keep = {k: c.extra[k] for k in ["kd"] if k in c.extra}
    sp = src_runs(ref)[0]["size"]["sparsity"]
    if sp > 0.01: keep["prune"] = {"sparsity": round(sp, 4), "ramp": 0}   # hold the existing zeros
    if bits is None:
        return replace(c, exp="d5", name=f"{n}_ft32", lr=0.003, epochs=20, patience=10, extra=keep)
    return replace(c, exp="d5", name=f"{n}_qat{bits}", lr=0.003, epochs=20, patience=10, extra={**keep, "wbits": bits})
print(len(FRONT), "frontier models from d4:"); display(pd.DataFrame(FRONT))
""")
nb.md("## 1. Post-training and quantization-aware, every frontier model")
nb.code("""
Q = {}
for f in FRONT:
    exp, n = f["run"].split("/")
    Q[(f["label"], "fp32")] = src_runs(f["run"])
    base = qat_cfg(f["run"], None)   # control: QAT's fine-tuning without quantization
    Q[(f["label"], "fp32 fine-tuned")] = [R.run(R.with_src(base, f["run"], "init", s, **base.extra)) for s in R.SEEDS3]
    for b in [8, 4, 2]:
        Q[(f["label"], f"PTQ {b}-bit")] = [R.ptq(exp, f"{n}_s{s}", "d5", b) for s in R.SEEDS3]
    for b in [8, 4, 2]:
        base = qat_cfg(f["run"], b)
        Q[(f["label"], f"QAT {b}-bit")] = [R.run(R.with_src(base, f["run"], "init", s, **base.extra)) for s in R.SEEDS3]
    Q[(f["label"], "real int8")] = [R.ptq(exp, f"{n}_s{s}", "d5", "int8") for s in R.SEEDS3]
TQ = R.table({f"{m} | {q}": v for (m, q), v in Q.items()})
TQ["model"] = [m for (m, q) in Q]; TQ["quant"] = [q for (m, q) in Q]
P = TQ.pivot(index="model", columns="quant", values="bal_acc").astype(float)
P = P[["fp32", "fp32 fine-tuned", "PTQ 8-bit", "QAT 8-bit", "real int8", "PTQ 4-bit", "QAT 4-bit", "PTQ 2-bit", "QAT 2-bit"]]
# the cost of quantization itself: QAT against the same fine-tuning in fp32 (not against the original model)
display((P[["QAT 8-bit", "QAT 4-bit", "QAT 2-bit"]].sub(P["fp32 fine-tuned"], axis=0) * 100).round(2)
        .rename(columns=lambda c: c + " minus fp32 fine-tuned (pp)"))
P.insert(0, "fp32 kB", TQ[TQ.quant == "fp32"].set_index("model").kB.astype(float))
display(P.sort_values("fp32 kB").round(4))
""")
nb.code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
order = P.sort_values("fp32 kB").index
for i, q in enumerate(["fp32", "PTQ 8-bit", "QAT 8-bit", "PTQ 4-bit", "QAT 4-bit", "PTQ 2-bit", "QAT 2-bit"]):
    ax[0].plot(range(len(order)), P.loc[order, q], marker="o" if "PTQ" in q or q == "fp32" else "s",
               ls="--" if "PTQ" in q else "-", label=q, color={"fp32": "k"}.get(q, f"C{(i + 1) // 2}"))
ax[0].set_xticks(range(len(order)), [f"{m[:28]}\\n{kb:.1f} kB" for m, kb in zip(order, P.loc[order, "fp32 kB"])], rotation=30, ha="right", fontsize=7)
ax[0].set(ylabel="val balanced accuracy", title="accuracy per frontier model and quantization"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)
for fl in R.FLOORS.values(): ax[0].axhline(fl, ls=":", color="grey")
R.tradeoff_plot({q: TQ[TQ.quant == q] for q in ["fp32", "QAT 8-bit", "QAT 4-bit", "QAT 2-bit", "PTQ 4-bit"]},
                "quantized models: size vs accuracy", ax=ax[1])
plt.tight_layout(); plt.show()
""")
nb.md(T["quant"])
nb.md("## 2. Real int8 vs simulated 8-bit, and latency")
nb.code("""
rows = []
for f in FRONT:
    ri = Q[(f["label"], "real int8")]
    rows.append({"model": f["label"], "simulated 8-bit PTQ": np.mean([r["val"]["bal_acc"] for r in Q[(f["label"], "PTQ 8-bit")]]),
                 "real int8": np.mean([r["val"]["bal_acc"] for r in ri]),
                 "analytic 8-bit kB": np.mean([r["size"]["bytes"] for r in ri]) / 1024,
                 "torch.save int8 kB": np.mean([r["size"]["serialized_bytes"] for r in ri]) / 1024,
                 "fp32 ms": np.mean([r["latency"]["fp32_ms"] for r in ri]), "int8 ms": np.mean([r["latency"]["int8_ms"] for r in ri])})
RI = pd.DataFrame(rows).set_index("model"); RI["int8 - sim (pp)"] = (RI["real int8"] - RI["simulated 8-bit PTQ"]) * 100
display(RI.round(4))
""")
nb.md(T["int8"])
nb.md("## 3. Decision and frontier after quantization")
nb.code("""
best_q = {}
for f in FRONT:   # per model and bit width: QAT or PTQ, highest mean
    for b in [8, 4, 2]:
        pt, qa = TQ.loc[f"{f['label']} | PTQ {b}-bit"], TQ.loc[f"{f['label']} | QAT {b}-bit"]
        best_q[(f["label"], b)] = "QAT" if float(qa.bal_acc) > float(pt.bal_acc) else "PTQ"
print("QAT better than PTQ in", sum(v == "QAT" for v in best_q.values()), "of", len(best_q), "model/bit-width pairs")
keep = {k: v for (m, q), v in Q.items() for k in [f"{m} | {q}"] if q != "real int8"}
TK = R.table(keep); F5 = R.pareto(TK)
display(F5[["kB", "x smaller than teacher", "sparsity", "wbits", "bal_acc", "sd"]].round(4))
def ref_of(m, q):
    f = next(x for x in FRONT if x["label"] == m); exp, n = f["run"].split("/")
    if q == "fp32": return f["run"]
    b = q.split()[1].split("-")[0]
    return f"d5/{n}_qat{b}" if q.startswith("QAT") else f"d5/{n}_ptq{b}"
front = [{"label": k, "run": ref_of(*k.split(" | ")), "bytes": int(F5.loc[k, "bytes"]), "bal_acc": float(F5.loc[k, "bal_acc"])} for k in F5.index]
_ = (ARTIFACTS / "d5_frontier.json").write_text(json.dumps(front, indent=1))
""")
nb.md(T["decision"])
nb.write("d5_quantization.ipynb")
