# Round r1: first desktop run (RTX 2080 Ti), old tie rule, 60-epoch cap

Run on 2026-11-12, 11:01 to 12:55, notebooks c6 to c10 (c4 and c5 from the Mac, round r0).

Selection rule in this round: an option replaced the current choice only if its 3-seed mean val balanced accuracy
beat the baseline by more than 2 * sigma_seed * sqrt(2/n), with sigma_seed = 0.0309 measured in c4 on the weak
lr 1e-3 model. With n = 3 that is 0.0505 (0.0619 for 2 vs 2 seeds). Otherwise "tie", and the simpler or default
option was kept. Sweeps (lr, batch size) fell back to the default unless a trend/peak fit was significant.

Why this file exists: the executed notebooks and logs of r1 were overwritten by round r2 before history keeping
was in place, and so were the result files whose run names r2 reused. 150 result files of r1 survive in
`results/` next to this file. The tables below were copied from the r1 logs during the session.

Input (from c5, r0): gray, crop to head, 128 px, global normalisation.

## c6 optimisation

Device check (c4 baseline, 3 seeds each): MPS 0.8956 (sd 0.0088), CUDA 0.8973 (sd 0.0095): tie, 3.80 vs 0.55 s/epoch.

Adam learning rate (3 seeds each, val balanced accuracy). These runs were not retrained in r1: they are the Mac
(r0) runs, reused from the cache (about 4 s/epoch, the MPS speed). r1 retrained only what the Mac had not run.

| lr | bal_acc | sd |
|---|---|---|
| 1e-4 | 0.8557 | 0.0076 |
| 3e-4 | 0.8716 | 0.0107 |
| 1e-3 (baseline) | 0.8956 | 0.0088 |
| 3e-3 | 0.9133 | 0.0052 |
| 1e-2 | 0.9218 | 0.0178 |
| 3e-2 | 0.9275 | 0.0042 |
| 1e-1 | 0.8954 | 0.0110 |

Trend fit: slope +1.83 pp/decade (95% CI 0.96 to 2.70), p_linear 0.0003, peak inside the range. Chosen lr 0.03.

Decisions: optimiser Adam 0.03 (vs SGD), batch size 32, schedule none, loss cross-entropy.
Result: {optimizer adam, lr 0.03, batch 32, schedule none, loss ce, sampler none}, mean val bal. acc 0.934.

## c7 architecture

| question | option | bal_acc | delta | verdict |
|---|---|---|---|---|
| kernel | 3x3 (baseline) | 0.9230 | 0 | |
| | 5x5 | 0.9432 | +0.0202 | tie |
| | 7x7 stem + 3x3 | 0.9139 | -0.0090 | tie |
| downsampling | maxpool (baseline) | 0.9230 | 0 | |
| | avgpool | 0.9312 | +0.0082 | tie |
| | stride | 0.8684 | -0.0546 | worse |
| depth | plain 5 (baseline) | 0.9230 | 0 | |
| | plain 9 | 0.9281 | +0.0052 | tie |
| | plain 17 | 0.8723 | -0.0507 | worse |
| | plain 33 | 0.8467 | -0.0762 | worse |
| | residual 9 | 0.9519 | +0.0289 | tie |
| | residual 17 | 0.9502 | +0.0273 | tie |
| | residual 33 | 0.9330 | +0.0100 | tie |
| | plain 5 / plain 17 / residual 17, no BatchNorm | 0.2500 | -0.6730 | worse (collapsed at epoch 1, lr 0.03 diverges without BN) |
| width (residual 17) | 16 (baseline) | 0.9502 | 0 | |
| | 32 | 0.9554 | +0.0052 | tie |

Decisions: 3x3 kernels, maxpool, width 16. Candidates for c8: small = plain 5, deep = residual 17 (width 16).

## c8 regularisation

Augmentation (val balanced accuracy, train-val gap in brackets):

| aug | small (3 seeds) | deep (2 seeds) |
|---|---|---|
| none (baseline) | 0.9352 (gap 0.061) | 0.9486 (gap 0.045) |
| geometric | 0.9469 (gap 0.020) | 0.9668 (gap 0.014) |
| mild | 0.9436 (gap 0.017) | 0.9646 (gap 0.009) |
| strong | 0.9352 (gap -0.010) | 0.9585 (gap -0.001) |

All ties under the rule. Decisions: augmentation none, weight decay 0, dropout 0 for both models;
winner small (deep not "clearly better").

## c9 final model

Learning-rate re-check kept 0.03. Final model: 5 conv layers, 100,244 parameters, 0.6 s/epoch.
Seed 1738 (fixed in advance) on clean val: balanced accuracy 0.932 [95% CI 0.913, 0.949].
Mean over 3 seeds 0.930.

(c9 first failed with `TypeError: replace() got multiple values for keyword argument 'optimizer'`;
fixed in make_c9.py by merging the settings into one dict, then rerun.)

## c10 threshold

Threshold for tumour sensitivity >= 0.98 on clean val: 0.405, balanced accuracy 0.9336. Test set not evaluated.

## Outcome

The rule's bar (0.05) was larger than any effect in c6 to c8 (observed seed sd 0.004 to 0.009), so it kept the
baseline almost everywhere. This led to round r2, where the option with the highest mean wins.
