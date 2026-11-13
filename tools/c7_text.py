T = {
    'kernel': """
**Result.** 3x3 0.936, **5x5 0.946**, 7x7 stem + 3x3 0.919. 5x5 kernels cost 2.8x the parameters (277k vs 100k)
and 20 % more time per epoch.

**Decision: 5x5 kernels** (highest mean). A larger kernel gives the shallow 5-layer model a larger receptive
field, which c5 already pointed to as its limit.

*Other rounds.* r1 measured the same direction (5x5 +2.0 pp) but kept 3x3 under the tie rule. r3 chose 5x5
(+0.2 pp).
""",
    'down': """
**Result.** maxpool 0.946, **avgpool 0.959**, strided convolution 0.929.

**Decision: average pooling** (highest mean, +1.3 pp). Strided convolution is clearly worse in every round.

*Other rounds.* r1 measured avgpool +0.8 pp and kept maxpool (tie rule). r3 chose avgpool (+0.3 pp).
""",
    'depth': """
**Result** (all with 5x5 kernels and avgpool):

| depth | plain | residual |
|---|---|---|
| 5 | **0.959** | |
| 9 | 0.945 | 0.958 |
| 17 | 0.903 | 0.956 |
| 33 | 0.823 | 0.952 |

* **Plain networks get worse with depth.** The 33-layer plain net is unstable (seed sd 0.068) and fits the
  training set worse, not better: train loss at the best epoch rises from 0.356 (plain 9) to 0.436 (17) and 0.595
  (33), while the residual nets stay at 0.352 to 0.356. That is the classic degradation problem (an optimisation
  failure, not overfitting).
* **Residual connections fix that.** Residual 9 to 33 all stay within 1 pp of the best.
* **Without BatchNorm every model collapses** to 0.25 (always predicting one class) from the first epoch. That is
  the learning rate: 0.03, tuned in c6 with BatchNorm, makes un-normalised networks diverge. It does not show
  that BatchNorm-free networks cannot work, only that they would need their own (much lower) learning rate.

**Decision.** Two candidates go to c8, because regularisation helps models that overfit, and deep models overfit
more:
* **small: plain 5 layers** (0.959; residual 9 0.958 is within 0.05 pp, plain 5 has the higher mean),
* **deep: residual 17** (the best model of 17 layers or more).

*Other rounds.* r1 found the same pattern (residual 9 best at 0.952, plain 33 worst). r3: residual 9 best (0.952),
plain 5 0.939, so r3's small candidate was residual 9.
""",
    'width': """
**Result** (residual 17): width 16 0.956, **32 0.962**, 64 0.959. Width 32 has 7.7M parameters and takes 2x the time per epoch of width 16; width 64 has
30.7M and takes 4.7x.

**Decision: width 32** for the deep candidate (highest mean).

*Other rounds.* r1 (tie rule) kept 16 (32 was +0.5 pp). r3: 16 and 32 tied exactly (0.9475), 64 0.945; the first
option (16) was kept. Width is not where the accuracy is.
""",
    'summary': """
| Question | Choice | Mean val bal. acc (3 seeds) | Runner-up |
|---|---|---|---|
| Kernel | 5x5 | 0.946 | 3x3: 0.936 |
| Downsampling | avgpool | 0.959 | maxpool: 0.946 |
| Depth (small) | plain 5 | 0.959 | residual 9: 0.958 |
| Depth (deep) | residual 17 | 0.956 | residual 33: 0.952 |
| Width (deep) | 32 | 0.962 | 64: 0.959 |

**c7 result:** small candidate 0.959 (277k parameters), deep candidate 0.962 (7.7M parameters), from 0.937 after
c6 (the c7 rerun of the c6 configuration scored 0.936). Kernel size and pooling gave the gains. **Depth did not
help in this round:** every residual net scored at or below plain 5 (residual 9 0.958, 17 0.956, 33 0.952 vs
0.959), and plain nets got worse with depth. The deep candidate's 0.962 comes from width 32, not from depth.
(In r1 residual 9 was 2.9 pp above plain 5, so the depth result is not stable across rounds.)

**Across rounds.** r1 (tie rule) measured the same gains (5x5, avgpool, residual) but kept the baseline at every
step: 0.923 small, 0.950 deep. r3 (200-epoch cap) made the same kernel and pooling choices, small candidate
residual 9 (0.952), deep residual 17 at width 16 (0.947).
""",
}
