import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c3_text import T

nb = NB()
nb.md("""
# c3 · Step 3: Preprocessing and dataloader

Two separate parts, as planned:
1. **Preprocessing** (deterministic, run once): image → fixed-size uint8 tensor, cached to `cache/`.
   One cache per variant that c5 will compare.
2. **Dataloader** (every epoch): the whole split lives on the GPU as uint8. Each batch is converted to float,
   **augmented (train only)**, then normalised.

Code lives in `lib/data.py`. This notebook builds, checks and visualises it.
""")
nb.code("""
import sys, time
import numpy as np, torch, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from PIL import Image
from lib.data import (CLASSES, SEED, list_split, preprocess, head_mask, _bbox, build_cache, load_split,
                      GPUSplit, AUG_PRESETS, augment)
from lib.train import DEVICE
rng = np.random.default_rng(SEED)
print("device:", DEVICE)
""")

nb.md("""
## 1. Geometry options
How to turn 349 different resolutions into one square size:

| Option | What it does | Risk |
|---|---|---|
| **stretch** | resize directly | distorts non-square images (mostly no_tumor), which is a visible class cue |
| **pad** | pad to square with black, then resize | adds black bars (mostly to no_tumor), which is a class cue |
| **crop** | crop to the head bounding box, pad the rest to square, resize | depends on the head detector working |

All three are built and compared in c5. Below: the same non-square images under each option.
""")
nb.code("""
items = list_split("train")
nonsq = [p for p, c in items if (lambda im: im.width != im.height)(Image.open(p))]
pick = [nonsq[i] for i in rng.choice(len(nonsq), 5, replace=False)]
fig, axes = plt.subplots(4, 5, figsize=(13, 10.5))
for k, p in enumerate(pick):
    with Image.open(p) as im:
        axes[0, k].imshow(im.convert("L"), cmap="gray"); axes[0, k].set_title(f"original {im.width}x{im.height}", fontsize=8)
    for r, mode in enumerate(["stretch", "pad", "crop"], 1):
        axes[r, k].imshow(preprocess(p, "gray", mode, 128)[0], cmap="gray", vmin=0, vmax=255)
        axes[r, k].set_title(mode, fontsize=8)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
""")

nb.md("""
## 2. Does the head crop work?
The crop finds the largest bright connected region (blurred and thresholded, holes filled) and takes its bounding
box plus a 3 % margin. Check: the crop's share of the image area over **all** images, and the most extreme cases
by eye.
""")
nb.code("""
rows = []
for split in ["train", "val", "test"]:
    for p, c in list_split(split):
        g = np.asarray(Image.open(p).convert("L"))
        x0, y0, x1, y1 = _bbox(head_mask(g))
        rows.append((p, c, (x1 - x0) * (y1 - y0) / g.size, (x1 - x0) / max(y1 - y0, 1)))
area = np.array([r[2] for r in rows]); aspect = np.array([r[3] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].hist(area, 60); ax[0].set(title="crop area / image area", xlabel="fraction")
ax[1].hist(aspect, 60); ax[1].set(title="crop aspect ratio (w/h)", xlabel="ratio")
plt.tight_layout(); plt.show()
print(f"crop area: min {area.min():.2f}, 1st pct {np.percentile(area, 1):.2f}, median {np.median(area):.2f}")
print(f"crops < 30 % of the image: {(area < 0.3).sum()} | aspect outside [0.6, 1.6]: {((aspect < 0.6) | (aspect > 1.6)).sum()}")
""")
nb.code("""
order = np.argsort(area)
show = list(order[:8]) + list(order[-4:]) + list(rng.choice(len(rows), 4, replace=False))
fig, axes = plt.subplots(2, 16, figsize=(22, 3.4))
for k, i in enumerate(show):
    p = rows[i][0]
    g = np.asarray(Image.open(p).convert("L")); x0, y0, x1, y1 = _bbox(head_mask(g))
    axes[0, k].imshow(g, cmap="gray"); axes[0, k].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="r", lw=1))
    axes[0, k].set_title(("smallest" if k < 8 else "largest" if k < 12 else "random") + f" {area[i]:.2f}", fontsize=7)
    axes[1, k].imshow(preprocess(p, "gray", "crop", 128)[0], cmap="gray")
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
""")
nb.md(T["crop"])

nb.md("""
## 3. Build the caches (run once)
Every variant needed in c5: three geometries at 128 px, resolutions 64/192 for the chosen geometry,
and a 3-channel version. Each is preprocessed once with 4 worker processes and saved as a uint8 tensor.
""")
nb.code("""
VARIANTS = [("gray", "stretch", 128), ("gray", "pad", 128), ("gray", "crop", 128),
            ("gray", "crop", 64), ("gray", "crop", 192), ("rgb", "crop", 128)]
for mode, resize, size in VARIANTS:
    t = time.time()
    shapes = [tuple(build_cache(s, mode, resize, size)["X"].shape) for s in ["train", "val", "test"]]
    print(f"{mode:4s} {resize:7s} {size:3d}: {shapes}  ({time.time() - t:.0f}s)")
""")

nb.md("## 4. Normalisation statistics (train only)")
nb.code("""
for mode, resize, size in VARIANTS:
    X = load_split("train", mode, resize, size)["X"].float() / 255
    print(f"{mode:4s} {resize:7s} {size:3d}: mean {X.mean():.4f}  std {X.std():.4f}")
""")
nb.md(T["norm"])

nb.md("""
## 5. Dataloader checks
* shapes, dtype and value range of a normalised batch
* the weighted sampler really balances classes
* speed of one pass over train
""")
nb.code("""
tr = load_split("train", "gray", "crop", 128)
Xf = tr["X"].float() / 255
train = GPUSplit(tr, DEVICE, Xf.mean().item(), Xf.std().item())
gen = torch.Generator().manual_seed(SEED)
x, y = next(train.batches(64, shuffle=True, gen=gen))
print("batch:", tuple(x.shape), x.dtype, x.device, f"mean {x.mean():.3f} std {x.std():.3f}")

counts = torch.bincount(train.y.cpu(), minlength=4).float()
w = (1 / counts)[train.y.cpu()]
seen = torch.cat([y.cpu() for _, y in train.batches(64, gen=gen, weights=w)])
print("class share, plain shuffle   :", (counts / counts.sum()).numpy().round(3))
print("class share, weighted sampler:", (torch.bincount(seen, minlength=4) / len(seen)).numpy().round(3))

for aug in ["none", "strong"]:
    t = time.time()
    for x, y in train.batches(64, shuffle=True, gen=gen, aug=AUG_PRESETS[aug]):
        pass
    torch.mps.synchronize() if DEVICE.type == "mps" else None
    print(f"one pass over train, aug={aug}: {time.time() - t:.2f}s")
""")

nb.md("""
## 6. Augmentation preview
Augmentation is implemented now but only switched on in c8 (regularisation). Here it is only checked by eye:
every augmented image must still look like a plausible scan of the same class.
""")
nb.code("""
idx = torch.tensor(rng.choice(len(train), 8, replace=False))
x0 = train.X[idx.to(DEVICE)].float() / 255
fig, axes = plt.subplots(5, 8, figsize=(16, 10.5))
for r, name in enumerate(["none", "geometric", "intensity", "mild", "strong"]):
    xa = augment(x0, AUG_PRESETS[name], torch.Generator().manual_seed(r)).cpu()
    for k in range(8):
        axes[r, k].imshow(xa[k, 0], cmap="gray", vmin=0, vmax=1); axes[r, k].axis("off")
    axes[r, 0].set_title(name, loc="left", fontsize=11)
plt.tight_layout(); plt.show()
print({k: v for k, v in AUG_PRESETS.items() if k != "none"})
""")
nb.md(T["aug"])
nb.md("## Decisions")
nb.md(T["summary"])
nb.write("c3_preprocessing.ipynb")
