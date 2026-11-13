# Round r2: desktop (RTX 2080 Ti), highest mean wins, 60-epoch cap

Run on 2026-11-12, 18:19 to 21:32, notebooks c5 to c10. Complete record next to this file: `results/`,
`artifacts/`, the executed notebooks and `logs/`.

Rule: the option with the highest mean val balanced accuracy wins, whatever its size or cost. Sweeps take the
value with the highest mean (trend fits reported, not gating). One validity guard kept in c5: among geometries
within 2 standard errors of the best, take the one with the least signal outside the head (shortcut probe).
c7 width sweep extended with width 64 (skipped on the Mac for speed).

## Decisions and outcomes

| step | decision | mean val bal. acc |
|---|---|---|
| c5 input | gray, crop, 128 px, **per-image normalisation** | 0.883 (CUDA retrain of the choice) |
| c6 optimisation | Adam 0.03, batch 64, no schedule, **label smoothing** | 0.937 |
| c7 architecture | **5x5 kernels, avgpool**; deep = residual 17 at **width 32** (16: 0.9555, 32: 0.9623, 64: 0.9594) | small 0.959, deep 0.962 |
| c8 regularisation | small: **mild augmentation**, wd 0, dropout 0; deep: geometric aug, wd 0, dropout 0.3 | small 0.974, deep 0.967 |
| c8 winner | **small** (277k parameters) over deep (7.7M) on mean score | 0.974 |
| c9 final | lr re-check kept 0.03; seed 1738: **0.968 [95% CI 0.955, 0.979]**; accuracy 0.966, macro-F1 0.966 | |
| c10 threshold | 0.717 for tumour sensitivity >= 0.98; balanced accuracy 0.967 | |

## Observations

* **Cross-device comparison in c5.** Per-image normalisation was chosen from the Mac runs (MPS, 0.901) against
  global (0.896, also MPS). Retrained on CUDA in c6 the same per-image configuration scored 0.883. The c5 choice
  therefore rested on a cross-device comparison and is not supported by the CUDA runs. Round r3 retrains every
  c5 option on CUDA.
* Most runs reached their best epoch near the 60-epoch cap (augmented runs were still improving), which led to
  round r3 with a 200-epoch cap and early stopping.
