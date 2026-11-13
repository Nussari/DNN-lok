# Round r3: desktop (RTX 2080 Ti), highest mean wins, 200-epoch cap with early stopping

Run on 2026-11-12 21:54 to 2026-11-13 01:46, notebooks c5 to c10. Complete record next to this file:
`results/`, `artifacts/`, the executed notebooks and `logs/`. Runs that r3 replaced are also kept in
`claude_run/results/_superseded/` (304 files from r2).

Changes from r2: epoch cap raised from 60 to 200 from c5 on (`EPOCH_CAP` in `lib/report.py`); early stopping
unchanged (patience 20 on val balanced accuracy, best weights restored). c4 left untouched; c5 trains its own
baseline (`c5/crop`) instead of reusing c4's 60-epoch runs. Every c5 option retrained on CUDA.
Side effect: cosine and step LR schedules are sized to 200 epochs, so early stopping can end them before the
LR has decayed; they are at a disadvantage in c6.

## Decisions and outcomes

| step | decision | mean val bal. acc |
|---|---|---|
| c5 input | gray, crop, 128 px, per-image normalisation (crop baseline 0.895, pad 0.912 but more shortcut signal) | 0.907 |
| c6 optimisation | Adam 0.03, **batch 128**, no schedule, **focal loss** | 0.941 |
| c7 architecture | 5x5 kernels, avgpool; small = **residual 9**, deep = residual 17 at **width 16** (16: 0.9475, 32: 0.9475, 64: 0.9452; exact tie, first option kept) | small 0.952, deep 0.947 |
| c8 regularisation | small: **geometric augmentation**, wd 0, dropout 0; deep: no augmentation, wd 0, dropout 0 | small 0.968, deep 0.956 |
| c8 winner | small (residual 9, 832k parameters) | 0.968 |
| c9 lr re-check | **0.01** (0.9601) over 0.03 (0.9521) and 0.09 (0.9391) | |
| c9 final | seeds 1738 / 1739 / 1740: 0.952 / 0.959 / 0.968 (mean 0.960); seed 1738 **0.952 [95% CI 0.935, 0.966]** | |
| c10 threshold | 0.710 for tumour sensitivity >= 0.98; balanced accuracy 0.957, tumour AUC 0.997 | |

## Observations

* **The longer cap did not improve the result.** Runs used the room (77 to 87 epochs on average in c8 instead
  of the 60 cap), but the final model scored 0.952 against r2's 0.968, and the best c8 configuration 0.968
  against r2's 0.974. The confidence intervals overlap: r2 and r3 are about equal, not r3 worse.
* **Several step winners changed between r2 and r3** (batch size 64 to 128, label smoothing to focal loss, plain 5
  to residual 9, mild to geometric augmentation, final lr 0.03 to 0.01). These options lie within seed noise of
  each other, so which one "wins" a step partly depends on chance; the path through the steps then differs.
* The seed chosen in advance as the final model (1738) was the weakest of the three in r3 (0.952 vs 0.959, 0.968).
* **Why early stopping did not prevent the worse result.** Early stopping works inside one run: it ends training
  when val stops improving and restores the best epoch, so a run can never end worse than its own best epoch.
  Checked on all 188 r3 runs: the best val score within the first 60 epochs averaged 0.8822, the best overall
  0.8842 (+0.19 pp; 20 % of runs improved after epoch 60, none got worse). So the longer cap made each run
  slightly better, not worse. What it cannot do is stop the selection from taking a different path: r3's steps
  chose different (within-noise) winners than r2, ended at a different final configuration, and the final seed
  fixed in advance happened to be that configuration's weakest. The lower score comes from the selection path
  and seed noise, not from training too long.
* Best performer across all rounds remains the r2 final model (`history/best_model/r2_final_s1738.pt`).
  r3's weights were not kept.
