import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c1_text import T  # conclusion texts, filled in after looking at the results

nb = NB()
nb.md("""
# c1 · Step 1: Data exploration

Independent run (Claude). Every section answers one question and ends with **finding → decision**.
Decisions that need a model to test are only *flagged* here and tested in later notebooks.

| # | Question | Feeds |
|---|---|---|
| 1 | What did we download? | loader (c3) |
| 2 | Class distribution | metric (c2), loss (c6) |
| 3 | What do the images look like? | augmentation (c8) |
| 4 | Image properties, and do they differ per class? | preprocessing (c3, c5) |
| 5 | Pixel values | normalisation (c3) |
| 6 | Near-duplicates across splits | evaluation set (c2), report |
""")
nb.code("""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, ".")
from lib.data import DATA_ROOT, ARTIFACTS, SPLITS, CLASSES, SEED, list_split
rng = np.random.default_rng(SEED)
pd.set_option("display.precision", 3)
print(sorted(p.name for p in DATA_ROOT.iterdir()))
""")

nb.md("""
## 1. Inventory: one row per image
Data was downloaded and checksum-verified by the user's `s1_data-explore.ipynb` (same zip, SHA-256 checked).
One pass over all files: header metadata, a check whether "RGB" images really contain colour, and a
128×128 grayscale thumbnail reused below.
""")
nb.code("""
rows, thumbs = [], []
for split in SPLITS:
    for p, c in list_split(split):
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
            rows.append(dict(path=str(p.relative_to(DATA_ROOT)), split=split, cls=CLASSES[c], w=im.width,
                             h=im.height, mode=im.mode, ext=p.suffix.lower(), kb=p.stat().st_size / 1024,
                             chroma=int(np.abs(rgb - rgb.mean(2, keepdims=True)).max())))
            thumbs.append(np.asarray(im.convert("L").resize((128, 128), Image.BILINEAR)))
df = pd.DataFrame(rows)
df["cls"] = pd.Categorical(df.cls, CLASSES); df["split"] = pd.Categorical(df.split, SPLITS)
X = np.stack(thumbs)
print(df.shape, "| extensions:", df.ext.value_counts().to_dict(), "| unreadable files: 0")
""")

nb.md("## 2. Class distribution")
nb.code("""
counts = pd.crosstab(df.cls, df.split, margins=True, margins_name="total")
display(counts)
display(pd.crosstab(df.cls, df.split, normalize="columns").mul(100).round(1).rename(columns=lambda c: c + " %"))
tr = counts.loc[CLASSES, "train"]
print(f"imbalance ratio {tr.max() / tr.min():.2f} | majority-class accuracy on val "
      f"{(df[df.split == 'val'].cls == 'glioma').mean():.3f}")
""")
nb.md(T["dist"])

nb.md("## 3. What do the images look like?")
nb.code("""
fig, axes = plt.subplots(4, 8, figsize=(16, 8.6))
for r, cls in enumerate(CLASSES):
    sub = df[(df.split == "train") & (df.cls == cls)]
    for c, i in enumerate(rng.choice(sub.index, 8, replace=False)):
        axes[r, c].imshow(X[i], cmap="gray"); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, c].set_xlabel(f"{df.w[i]}x{df.h[i]} {df['mode'][i]}", fontsize=7)
        if c == 0: axes[r, c].set_ylabel(cls, fontsize=12)
plt.suptitle("8 random train images per class (original size + colour mode)"); plt.tight_layout(); plt.show()
""")
nb.code("""
sub = df[(df.split == "train") & (df.cls == "no_tumor")]
fig, axes = plt.subplots(3, 8, figsize=(16, 6.6))
for ax, i in zip(axes.ravel(), rng.choice(sub.index, 24, replace=False)):
    ax.imshow(X[i], cmap="gray"); ax.axis("off"); ax.set_title(f"{df.w[i]}x{df.h[i]}", fontsize=7)
plt.suptitle("no_tumor close-up: 24 random train images"); plt.tight_layout(); plt.show()
""")
nb.md(T["look"])

nb.md("""
## 4. Image properties, and do they differ per class?
A CNN learns *any* systematic difference between classes. If a property such as resolution or colour mode
differs per class, the model can use it instead of anatomy, and because the same bias is in val and test,
the score would not reveal it.
""")
nb.code("""
df["res"] = (df.w.astype(str) + "x" + df.h.astype(str)).where(lambda s: s.isin(["512x512", "224x224"]), "other")
df["square"] = df.w == df.h
df["has_colour"] = df.chroma > 2
print("resolution (row %)"); display(pd.crosstab(df.cls, df.res, normalize="index").mul(100).round(1))
print("stored colour mode (row %)"); display(pd.crosstab(df.cls, df["mode"], normalize="index").mul(100).round(1))
display(df.groupby("cls", observed=True)[["square", "has_colour", "kb"]].mean())
print(f"{df[['w', 'h']].drop_duplicates().shape[0]} distinct resolutions")
""")
nb.code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for cls in CLASSES:
    s = df[df.cls == cls]
    ax[0].scatter(s.w, s.h, s=8, alpha=0.4, label=cls)
    ax[1].hist(s.kb, bins=60, range=(0, df.kb.quantile(0.99)), histtype="step", lw=1.5, label=cls)
ax[0].set(xscale="log", yscale="log", xlabel="width", ylabel="height", title="original resolution")
ax[1].set(xlabel="file size (KB)", title="file size"); ax[0].legend(); ax[1].legend()
plt.tight_layout(); plt.show()
""")
nb.md("""
**Test: metadata-only classifier.** How well can the label be predicted from width, height, aspect ratio,
file size and colour flags alone, *without any pixels*? Chance is 0.25 balanced accuracy.
""")
nb.code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
meta = pd.DataFrame({"w": df.w, "h": df.h, "aspect": df.w / df.h, "kb": df.kb,
                     "is_L": (df["mode"] == "L").astype(int), "has_colour": df.has_colour.astype(int)})
tr_m, va_m = df.split == "train", df.split == "val"
rf = RandomForestClassifier(300, class_weight="balanced", random_state=SEED).fit(meta[tr_m], df.cls[tr_m])
pred = rf.predict(meta[va_m])
print(f"metadata-only balanced accuracy on val: {balanced_accuracy_score(df.cls[va_m], pred):.3f}")
print(classification_report(df.cls[va_m], pred, digits=3))
""")
nb.md(T["props"])

nb.md("## 5. Pixel values")
nb.code("""
Xf = X.astype(np.float32) / 255
df["mean_px"] = Xf.mean((1, 2)); df["frac_black"] = (X < 10).mean((1, 2))
display(df.groupby("cls", observed=True)[["mean_px", "frac_black"]].agg(["mean", "std"]))
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, cls in zip(axes, CLASSES):
    m = ((df.cls == cls) & (df.split == "train")).to_numpy()
    ax.imshow(Xf[m].mean(0), cmap="gray"); ax.set_title(f"{cls}: mean image"); ax.axis("off")
plt.tight_layout(); plt.show()
""")
nb.md(T["pixels"])

nb.md("""
## 6. Near-duplicates across splits (leakage)
For every val/test image: Pearson correlation of 32×32 thumbnails with its most similar **train** image.
(A 64-bit perceptual hash does not work here: the black background makes unrelated scans hash alike.)
Options for what to do with near-duplicates:

| Option | Pro | Con |
|---|---|---|
| A. keep everything | official split, comparable | memorised copies inflate val/test, favour overfitting |
| B. drop near-duplicates from val/test | honest generalisation estimate | slightly smaller eval sets |
| C. B for model selection, report test both ways | honest selection + comparability | two test numbers to explain |

The threshold is chosen by **looking at pairs** in correlation bins below.
""")
nb.code("""
def vecs(size=32):
    v = np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR), np.float32).ravel() for x in X])
    v -= v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
V = vecs()
itr = np.where(df.split == "train")[0]
near = []
for split in ["val", "test"]:
    ib = np.where(df.split == split)[0]
    C = V[itr] @ V[ib].T
    near.append(pd.DataFrame({"i": itr[C.argmax(0)], "j": ib, "corr": C.max(0), "split": split}))
near = pd.concat(near, ignore_index=True)
near["same_label"] = df.cls.values[near.i] == df.cls.values[near.j]
bins = [0.9, 0.95, 0.97, 0.98, 0.99, 1.001]
display(pd.crosstab(pd.cut(near["corr"], bins, right=False), near.split, margins=True))
""")
nb.code("""
fig, axes = plt.subplots(2 * 4, 6, figsize=(12, 16.5))
for b, (lo, hi) in enumerate([(0.99, 1.01), (0.98, 0.99), (0.97, 0.98), (0.95, 0.97)]):
    sel = near[(near["corr"] >= lo) & (near["corr"] < hi)]
    sel = sel.sample(min(6, len(sel)), random_state=SEED)
    for k in range(6):
        for r in range(2):
            ax = axes[2 * b + r, k]; ax.axis("off")
            if k < len(sel):
                idx = int(sel.iloc[k].i if r == 0 else sel.iloc[k].j)
                ax.imshow(X[idx], cmap="gray")
                ax.set_title(f"{df.split[idx]}/{df.cls[idx]}" + (f"  r={sel.iloc[k]['corr']:.3f}" if r == 0 else ""), fontsize=7)
    axes[2 * b, 0].text(-0.3, 0, f"r in [{lo}, {min(hi, 1)})", transform=axes[2 * b, 0].transAxes, fontsize=10,
                        rotation=90, va="top")
plt.suptitle("Random pairs per correlation bin (top: train, below: val/test)")
plt.subplots_adjust(wspace=0.05, hspace=0.3, top=0.95, left=0.05); plt.show()
""")
nb.md(T["dups"])
nb.code("""
THRESH = 0.98
flag = near[near["corr"] >= THRESH]
print(flag.groupby("split").size().to_dict(), "flagged | label mismatches:", (~flag.same_label).sum())
display(pd.crosstab(df.cls.values[flag.j], flag.split))
ARTIFACTS.mkdir(exist_ok=True)
_ = (ARTIFACTS / "near_duplicates.json").write_text(json.dumps(
    {"threshold": THRESH, "method": "max Pearson r of 32x32 grayscale thumbnails vs train",
     "exclude": sorted(df.path.values[flag.j].tolist())}, indent=1))
""")

nb.md("## Findings → decisions")
nb.md(T["summary"])
nb.write("c1_data_exploration.ipynb")
