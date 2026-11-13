import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from errors_text import T

nb = NB()
nb.md("""
# Error analysis: where the four final models fail on clean test

Run after `test_evaluation.ipynb`, with every model and decision frozen: nothing here feeds back into a choice. It
asks two questions:

1. **Do the models fail on the same images?** If the compressed models miss the same tumours as the large model,
   those images are hard in themselves; if not, the errors come from the individual models.
2. **What do the failures look like?** Every missed tumour is shown as stored and as the model sees it (cropped to
   the head, grayscale, 128 px), next to correctly detected tumours of the same classes for comparison.

Predictions use the tumour decision rule from c10 (argmax, except a cutoff on P(tumour) for the 1.4 kB model), as
in the test evaluation.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from PIL import Image
sys.path.insert(0, "..")
from lib.train import Config, build_model, prepare_data, predict, DEVICE, RESULTS, CKPT
from lib.data import ARTIFACTS, CLASSES, NO_TUMOR, SEED, DATA_ROOT, load_split, GPUSplit
from lib.evaluation import tumor_score

PICKS = json.loads((ARTIFACTS / "compress_final.json").read_text())        # the compressed models chosen in d6
RUNS = {"large": "c9/final_s1738", **{f"{p['stored kB']:.1f} kB": f"{p['record']}_s1738" for p in PICKS.values()}}
RUNS = dict(sorted(RUNS.items(), key=lambda kv: -float(kv[0].split()[0]) if kv[0] != "large" else -1e9))
CUTOFF = json.loads((ARTIFACTS / "threshold.json").read_text())["decisions"]
NAMES = list(RUNS)

def decide(prob, cutoff):
    if cutoff is None:
        return prob.argmax(1)
    p = prob.copy(); p[:, NO_TUMOR] = -1
    return np.where(tumor_score(prob) >= cutoff, p.argmax(1), NO_TUMOR)

prob, pred = {}, {}
for name, run in RUNS.items():
    rec = json.loads((RESULTS / f"{run}.json").read_text())
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
    train, _ = prepare_data(cfg)
    net = build_model(cfg, train.X.shape[1])
    net.load_state_dict(torch.load(CKPT / f"{run}.pt", map_location="cpu"))
    test = load_split("test", cfg.mode, cfg.resize, cfg.size, clean=True)
    te = GPUSplit(test, DEVICE, train.mean, train.std, cfg.normalize == "per_image")
    prob[name] = predict(net.to(DEVICE), te)
    pred[name] = decide(prob[name], CUTOFF[name])
y, paths, X = te.y.cpu().numpy(), te.paths, test["X"]          # identical for every model (same input settings)
tum = y != NO_TUMOR
print(f"clean test: {len(y)} images, {tum.sum()} tumours, {(~tum).sum()} no_tumor; models: {NAMES}")
""")

nb.md("## 1. Do the models fail on the same images?")
nb.md("""
Three kinds of error, per model: a **missed tumour** (tumour predicted as no_tumor), a **false alarm** (no_tumor
predicted as a tumour) and a **wrong tumour type** (right that there is a tumour, wrong class). For each pair of
models the table counts the images both get wrong. As a reference, it also gives the count expected if the two
models erred **independently** of each other: n_a · n_b / N, with N the number of images the error can occur on.
Many more shared errors than that means the errors sit in the images, not in the models.
""")
nb.code("""
KINDS = {"missed tumour": (tum, lambda p: p == NO_TUMOR),
         "false alarm": (~tum, lambda p: p != NO_TUMOR),
         "wrong tumour type": (tum, lambda p: (p != NO_TUMOR) & (p != y)),
         "any error": (np.ones_like(tum), lambda p: p != y)}
ERR = {k: {n: np.flatnonzero(elig & f(pred[n])) for n in NAMES} for k, (elig, f) in KINDS.items()}

for kind, (elig, _) in KINDS.items():
    N, sets = int(elig.sum()), {n: set(ERR[kind][n]) for n in NAMES}
    shared = pd.DataFrame([[len(sets[a] & sets[b]) for b in NAMES] for a in NAMES], index=NAMES, columns=NAMES)
    expect = pd.DataFrame([[len(sets[a]) * len(sets[b]) / N for b in NAMES] for a in NAMES], index=NAMES, columns=NAMES)
    table = shared.astype(str) + " (" + expect.round(1).astype(str) + ")"
    for n in NAMES:
        table.loc[n, n] = str(len(sets[n]))
    print(f"{kind} (of {N} images): diagonal = errors per model; off-diagonal = shared (expected if independent)")
    display(table)
""")
nb.code("""
def n_wrong(kind):                                             # per image: how many models make this error
    return sum(np.isin(np.arange(len(y)), ERR[kind][n]) for n in NAMES)
counts = pd.DataFrame({kind: np.bincount(n_wrong(kind), minlength=len(NAMES) + 1)[1:] for kind in KINDS},
                      index=range(1, len(NAMES) + 1))
counts.index.name = "number of models that get it wrong"
display(counts.T)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.imshow([[len(set(ERR["any error"][a]) & set(ERR["any error"][b])) / len(set(ERR["any error"][a]) | set(ERR["any error"][b]))
            for b in NAMES] for a in NAMES], vmin=0, vmax=1, cmap="Blues")
for i, a in enumerate(NAMES):
    for j, b in enumerate(NAMES):
        sa, sb = set(ERR["any error"][a]), set(ERR["any error"][b])
        ax.text(j, i, f"{len(sa & sb) / len(sa | sb):.2f}", ha="center", va="center", fontsize=9)
ax.set(xticks=range(len(NAMES)), xticklabels=NAMES, yticks=range(len(NAMES)), yticklabels=NAMES,
       title="overlap of all errors (shared / either), clean test")
plt.tight_layout(); plt.show()
""")
nb.md(T["overlap"])

nb.md("## 2. The missed tumours")
nb.code("""
missed = sorted(set(np.concatenate([ERR["missed tumour"][n] for n in NAMES])),
                key=lambda i: (-sum(pred[n][i] == NO_TUMOR for n in NAMES), y[i], paths[i]))
def stored_info(i):
    with Image.open(DATA_ROOT / paths[i]) as im:
        a = np.asarray(im.convert("RGB")).astype(int)
        return {"stored size": f"{im.size[0]}x{im.size[1]}", "format": im.format,
                "colour": "yes" if np.abs(a - a.mean(2, keepdims=True)).max() > 10 else "no"}
M = pd.DataFrame([{"#": k, "file": paths[i].split("/")[-1], "true class": CLASSES[y[i]],
                   "missed by": ", ".join(n for n in NAMES if pred[n][i] == NO_TUMOR),
                   **{f"P(tumour) {n}": round(float(tumor_score(prob[n])[i]), 3) for n in NAMES}, **stored_info(i)}
                  for k, i in enumerate(missed, 1)]).set_index("#")
display(M)
""")
nb.md("""
Each missed tumour, as stored (top) and as the models see it (bottom: head crop, grayscale, 128 px). The number
matches the table above.
""")
nb.code("""
def show(idx, labels, ncol=6):
    nrow = -(-len(idx) // ncol)
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(2.3 * ncol, 4.9 * nrow))
    for a in axes.flat:
        a.axis("off")
    for k, (i, lab) in enumerate(zip(idx, labels)):
        r, c = 2 * (k // ncol), k % ncol
        with Image.open(DATA_ROOT / paths[i]) as im:
            axes[r, c].imshow(im.convert("RGB"))
        axes[r, c].set_title(lab, fontsize=8)
        axes[r + 1, c].imshow(X[i, 0], cmap="gray", vmin=0, vmax=255)
    plt.tight_layout(); plt.show()

short = {n: n.replace(" kB", "k") for n in NAMES}
show(missed, [f"#{k} {CLASSES[y[i]]}\\nmissed: {', '.join(short[n] for n in NAMES if pred[n][i] == NO_TUMOR)}"
              for k, i in enumerate(missed, 1)])
""")
nb.md("""
For comparison: tumours that **every** model detects, drawn at random (seed 1738) with the same number per class
as the missed tumours.
""")
nb.code("""
rng = np.random.default_rng(SEED)
caught = np.flatnonzero(tum & np.all([pred[n] != NO_TUMOR for n in NAMES], axis=0))
ctrl = np.concatenate([rng.choice(caught[y[caught] == c], n, replace=False)
                       for c, n in zip(*np.unique(y[missed], return_counts=True))])
display(pd.DataFrame([{"file": paths[i].split("/")[-1], "true class": CLASSES[y[i]], **stored_info(i)} for i in ctrl]))
show(ctrl, [f"{CLASSES[y[i]]}\\ndetected by all" for i in ctrl])
""")
nb.md(T["missed"])

nb.md("## 3. The false alarms of the three larger models")
nb.md("""
No-tumour scans that the large, 18.8 kB or 9.3 kB model calls a tumour. (The 1.4 kB model's false alarms are
not shown: its cutoff trades specificity for sensitivity by design, see c10.)
""")
nb.code("""
big = NAMES[:3]
fa = sorted(set(np.concatenate([ERR["false alarm"][n] for n in big])),
            key=lambda i: (-sum(pred[n][i] != NO_TUMOR for n in big), paths[i]))
display(pd.DataFrame([{"#": k, "file": paths[i].split("/")[-1],
                       "flagged by": ", ".join(n for n in big if pred[n][i] != NO_TUMOR),
                       **{f"predicted ({n})": CLASSES[pred[n][i]] for n in big}, **stored_info(i)}
                      for k, i in enumerate(fa, 1)]).set_index("#"))
show(fa, [f"#{k} flagged: {', '.join(short[n] for n in big if pred[n][i] != NO_TUMOR)}" for k, i in enumerate(fa, 1)])
""")
nb.md(T["false_alarms"])
nb.md(T["summary"])
nb.write("error_analysis.ipynb")
