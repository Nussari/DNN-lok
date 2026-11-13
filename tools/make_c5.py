import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from nbtools import NB
from c5_text import T

nb = NB()
nb.md("""
# c5 · Step 5: Input experiments

Model and training are fixed to the c4 initial model. Only the **input** changes. Every option runs with 3 seeds
and the option with the highest mean val balanced accuracy wins (`lib.report.pick`), whatever its size or cost.
One exception, for validity rather than simplicity: geometry also weighs the shortcut probe (section 1).

The runs in this notebook were trained on Apple MPS in round r0 and reused by the reported round r2, which
retrains the chosen input on CUDA at the start of c6. Round r0 used the *tie rule* (an option replaced the current
choice only if it was better by more than 2·σ_seed·√(2/n), about 5 pp, see c2 and c4); "Other rounds" notes
refer to it.

| # | Question | Options |
|---|---|---|
| 1 | Geometry | stretch / pad / crop to head |
| 2 | Channels | 1 (grayscale) / 3 (as stored) |
| 3 | Normalisation | global train mean/std / per image |
| 4 | Resolution | 64 / 128 / 192 px |

**The shortcut problem.** c1 showed that size, framing and colour differ by class. A shortcut that is present
in val too *raises* the val score, so val accuracy alone would *reward* the leakiest preprocessing. The geometry
decision therefore also uses a **probe**: the same model trained and evaluated on images where the head is
blacked out (`probe=True`), so only background, borders and padding remain. A probe score near 0.25 means no
usable signal outside the head. Anything above that is signal the model could use instead of anatomy.
""")
nb.code("""
import sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
from dataclasses import replace
from lib.train import Config, run
from lib.report import run_seeds, compare, pick, sigma_seed, log_progress, SEEDS3, curve_test
from lib.data import ARTIFACTS, load_split
from lib.report import protocol; proto = protocol()   # c4 protocol; epoch cap = lib.report.EPOCH_CAP (60 in the reported round r2, 200 in r3)
# at c4's 60-epoch cap the crop baseline is exactly c4's seed runs; at a raised cap (round r3) c5 trains its own
BASE = (Config(exp="c4", name="seed", monitor=proto["monitor"], patience=proto["patience"], epochs=proto["epochs"]) if proto["epochs"] == 60
        else Config(exp="c5", name="crop", monitor=proto["monitor"], patience=proto["patience"], epochs=proto["epochs"]))
C5 = replace(BASE, exp="c5")
print(proto, f"sigma_seed={sigma_seed():.4f}")
""")

nb.md("## 1. Geometry: val score and shortcut probe")
nb.code("""
geo = {"stretch": run_seeds(replace(C5, name="stretch", resize="stretch")),
       "pad": run_seeds(replace(C5, name="pad", resize="pad")),
       "crop": run_seeds(BASE)}
tg = compare(geo, "crop", "geometry: val balanced accuracy"); display(tg)
probe = {g: run_seeds(replace(C5, name=f"probe_{g}", resize=g, probe=True)) for g in ["stretch", "pad", "crop"]}
tp = compare(probe, "crop", "probe (head blacked out): val balanced accuracy"); display(tp)
# validity guard (not a simplicity rule): among geometries within the c4 bar (2*sigma_seed*sqrt(2/3)) of the best val score, take the
# one with the least signal outside the head, so the model is not rewarded for learning a border/scanner shortcut
best = tg.bal_acc.astype(float).max()
ok = [g for g in geo if tg.loc[g, "bal_acc"] >= best - tg.loc[g, "noise_2se"]]
GEOM = min(ok, key=lambda g: tp.loc[g, "bal_acc"])
print("tie with best val score:", ok, "-> lowest probe score:", GEOM)
""")
nb.code("""
fig, axes = plt.subplots(3, 8, figsize=(16, 6.4))
for r, g in enumerate(["stretch", "pad", "crop"]):
    X = load_split("val", "gray", g, 128, probe=True)["X"]
    for k, i in enumerate(np.linspace(0, len(X) - 1, 8).astype(int)):
        axes[r, k].imshow(X[i, 0], cmap="gray", vmin=0, vmax=255); axes[r, k].axis("off")
    axes[r, 0].set_title(f"probe input: {g}", loc="left")
plt.tight_layout(); plt.show()
""")
nb.md(T["geometry"])

nb.md("## 2. Channels")
nb.code("""
ch = {"gray (1 ch)": geo[GEOM], "rgb (3 ch)": run_seeds(replace(C5, name="rgb", mode="rgb", resize=GEOM))}
t = compare(ch, "gray (1 ch)", "channels"); display(t)
MODE = "rgb" if pick(t, "gray (1 ch)") == "rgb (3 ch)" else "gray"; print("chosen:", MODE)
""")
nb.md(T["channels"])

nb.md("## 3. Normalisation")
nb.code("""
nm = {"global": geo[GEOM], "per image": run_seeds(replace(C5, name="perimage", normalize="per_image", resize=GEOM))}
t = compare(nm, "global", "normalisation"); display(t)
NORM = "per_image" if pick(t, "global") == "per image" else "global"; print("chosen:", NORM)
""")
nb.md(T["norm"])

nb.md("## 4. Resolution")
nb.code("""
res = {"64": run_seeds(replace(C5, name="r64", size=64, resize=GEOM)), "128": geo[GEOM],
       "192": run_seeds(replace(C5, name="r192", size=192, resize=GEOM))}
t = compare(res, "128", "resolution"); display(t)
ct = curve_test({int(k): v for k, v in res.items()})   # ordered setting: test the whole curve (rule introduced in c6)
print({k: (round(float(v), 4) if not isinstance(v, (dict, tuple, bool)) else v) for k, v in ct.items() if k != "means"})
SIZE = int(ct["best"]); print("chosen:", SIZE)
""")
nb.md(T["res"])
nb.md("## Decisions")
nb.md(T["summary"])
nb.code("""
INPUT = dict(mode=MODE, resize=GEOM, size=SIZE, normalize=NORM)   # carried into c6
_ = (ARTIFACTS / "input.json").write_text(json.dumps(INPUT))
chosen_runs = run_seeds(replace(C5, name="input_choice", **INPUT)) if INPUT != dict(mode="gray", resize="crop", size=128, normalize="global") else geo["crop"]
print(INPUT, "-> mean val bal. acc", round(log_progress("c5 input", chosen_runs), 4))
""")
nb.write("c5_input_experiments.ipynb")
