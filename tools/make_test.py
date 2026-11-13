import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from test_text import T

nb = NB()
nb.md("""
# Test evaluation: the four final models

The single evaluation on the held-out test set, with every model and decision frozen beforehand: the large model
(c9) and the three compressed models (d6), each the seed-1738 run, and the tumour decision rule from c10
(argmax, except a cutoff of 0.283 on P(tumour) for the 1.4 kB model).

Two test sets: the **official** test split (889 images) and the **clean** test set (833), which leaves out the
test images that are near-copies of training images (c1). Clean test is the fair measure of generalisation; the
official split is what published results use.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
sys.path.insert(0, "..")
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from lib.train import Config, build_model, prepare_data, predict, DEVICE, RESULTS, CKPT
from lib.data import ARTIFACTS, CLASSES, NO_TUMOR, load_split, GPUSplit
from lib.evaluation import bootstrap_ci, tumor_score

PICKS = json.loads((ARTIFACTS / "compress_final.json").read_text())        # the compressed models chosen in d6
MODELS = {"large": "c9/final_s1738", **{f"{p['stored kB']:.1f} kB": f"{p['record']}_s1738" for p in PICKS.values()}}
CUTOFF = json.loads((ARTIFACTS / "threshold.json").read_text())["decisions"]
TEACHER_BYTES = json.loads((ARTIFACTS / "compress_rules.json").read_text())["teacher"]["bytes"]

def macs(net, size):
    total = 0
    def hook(m, i, o):
        nonlocal total
        if isinstance(m, torch.nn.Conv2d):
            total += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            total += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in net.modules()]
    with torch.no_grad():
        net.cpu().eval()(torch.zeros(1, 1, size, size))
    for h in hs: h.remove()
    return total

probs, info = {}, {}
for name, run in MODELS.items():
    rec = json.loads((RESULTS / f"{run}.json").read_text())
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    train, _ = prepare_data(cfg)                                   # train mean/std, as in training
    net = build_model(cfg, train.X.shape[1])
    net.load_state_dict(torch.load(CKPT / f"{run}.pt", map_location="cpu"))
    info[name] = {"stored kB": (rec["size"]["bytes"] if "size" in rec else TEACHER_BYTES) / 1024,
                  "parameters": sum(p.numel() for p in net.parameters()), "MMACs": macs(net, cfg.size) / 1e6,
                  "val": np.load(RESULTS / f"{run}_val.npz")}
    net.to(DEVICE)
    for clean in [False, True]:
        te = GPUSplit(load_split("test", cfg.mode, cfg.resize, cfg.size, clean=clean), DEVICE, train.mean, train.std,
                      cfg.normalize == "per_image")
        probs[name, clean] = (te.y.cpu().numpy(), predict(net, te))
print({k: len(v[0]) for k, v in list(probs.items())[:2]})
""")

nb.md("## 1. Scores")
nb.code("""
def decide(prob, cutoff):
    if cutoff is None:
        return prob.argmax(1)
    p = prob.copy(); p[:, NO_TUMOR] = -1
    return np.where(tumor_score(prob) >= cutoff, p.argmax(1), NO_TUMOR)

rows = {}
for (name, clean), (y, prob) in probs.items():
    pred = decide(prob, CUTOFF[name]); tum = y != NO_TUMOR
    lo, hi = bootstrap_ci(y, pred)
    r4 = lambda v: round(float(v), 4)
    rows[(name, "clean" if clean else "official")] = {
        "bal. acc": r4(balanced_accuracy_score(y, pred)), "95 % CI": f"[{lo:.3f}, {hi:.3f}]",
        "accuracy": r4(accuracy_score(y, pred)), "macro-F1": r4(f1_score(y, pred, average="macro")),
        **{f"recall {c}": r4((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)},
        "tumour sens.": r4((pred[tum] != NO_TUMOR).mean()), "missed tumours": int((pred[tum] == NO_TUMOR).sum()),
        "tumour spec.": r4((pred[~tum] == NO_TUMOR).mean()), "false alarms": int((pred[~tum] != NO_TUMOR).sum()),
        "tumour AUC": r4(roc_auc_score(tum, tumor_score(prob)))}
S = pd.DataFrame(rows).T
display(S)
# val score of the same model under the same decision rule, for comparison
for name in info:
    v = info[name].pop("val")
    info[name]["val bal. acc"] = balanced_accuracy_score(v["y"], decide(v["prob"], CUTOFF[name]))
""")
nb.code("""
summary = pd.DataFrame(info).T
summary["test bal. acc (clean)"] = [S.loc[(n, "clean"), "bal. acc"] for n in summary.index]
summary["test bal. acc (official)"] = [S.loc[(n, "official"), "bal. acc"] for n in summary.index]
summary["test - val (clean)"] = summary["test bal. acc (clean)"] - summary["val bal. acc"]
display(summary.astype(float).round({"stored kB": 1, "MMACs": 1, "parameters": 0}).round(4))
y, prob = probs["large", True]
print("large model, clean test, confusion matrix (rows = true class):")
display(pd.DataFrame(confusion_matrix(y, decide(prob, None), labels=range(4)),
                     index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES]))
_ = (ARTIFACTS / "test_results.json").write_text(json.dumps(
    {f"{n} | {s}": {k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()} for (n, s), r in rows.items()}, indent=1))
""")

nb.md("## 2. Size and compute vs score")
nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, col, label in [(axes[0], "stored kB", "stored size (kB, log scale)"), (axes[1], "MMACs", "multiply-adds per image (millions, log scale)")]:
    for i, n in enumerate(summary.index):
        x = summary.loc[n, col]
        lo, hi = (float(v) for v in S.loc[(n, "clean"), "95 % CI"].strip("[]").split(","))
        t = summary.loc[n, "test bal. acc (clean)"]
        ax.errorbar(x, t, yerr=[[t - lo], [hi - t]], fmt="o", color=f"C{i}", capsize=4, ms=8, label=f"{n}: test (clean)")
        ax.scatter(x, summary.loc[n, "val bal. acc"], marker="x", s=60, color=f"C{i}", label=f"{n}: val")
        ax.annotate(n, (x, t), textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax.set(xscale="log", xlabel=label, ylabel="balanced accuracy"); ax.grid(alpha=0.3)
axes[0].set_title("score vs stored size"); axes[1].set_title("score vs compute")
axes[1].legend(fontsize=7, loc="lower right", ncol=2)
plt.tight_layout(); plt.show()
""")
nb.md(T["results"])
nb.write("test_evaluation.ipynb")
