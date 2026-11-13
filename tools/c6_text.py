T = {
    'lr': """
**Result.** Val balanced accuracy is highest at 0.03 and drops at 0.1; below 0.03 the curve rises overall but
not monotonically (1e-3 is below 3e-4):
1e-4 0.878, 3e-4 0.892, 1e-3 0.883, 3e-3 0.911, 1e-2 0.921, **3e-2 0.937**, 1e-1 0.917.
The trend across all 21 runs is clear (+1.75 pp per decade, 95 % CI 0.8 to 2.7, p = 0.0009).

**Is it real or a lucky best epoch?** A higher LR could give a jumpier val curve whose maximum is higher by
chance. The check says no: the mean of the last 10 epochs (no selection at all) also peaks at 0.03 (0.856),
and the curve at 0.03 is *less* jumpy than at 1e-3 (sd of epoch-to-epoch change 0.106 vs 0.160).

**Decision: Adam, lr 0.03** (highest mean, 5.4 pp over the c4 default 1e-3). 0.03 is high for Adam; it works here
because every model has BatchNorm. Without BatchNorm it diverges (c7).

*Other rounds.* r0/r1 (global normalisation, MPS runs): same shape, 0.03 best at 0.928. r3 (200 epochs): 0.03
best at 0.936.
""",
    'opt': """
**Result.** Best Adam (0.03): 0.937. Best SGD with Nesterov momentum (0.1, from its own LR sweep; 0.1 was at the edge, so the sweep was extended to
0.3, which scored 0.923): 0.928, with
lower no_tumor recall (0.910 vs 0.948).

**Decision: Adam 0.03** (highest mean).

*Other rounds.* Adam in r1 and r3 as well (r3: Adam 0.936, SGD 0.934).
""",
    'bs': """
**Result.** 32: 0.931, **64: 0.937**, 128: 0.922 (learning rate kept at 0.03). No trend across the sizes
(p = 0.59): the differences are seed noise.

**Decision: batch size 64** (highest mean).

*Other rounds.* This is one of the unstable choices: r1 picked 32, r3 picked 128 (32: 0.935, 64: 0.936,
128: 0.938). All within about 1.5 pp; batch size does not matter much in this range.
""",
    'sched': """
**Result.** none 0.937, cosine 0.937, step 0.932, reduce-on-plateau 0.926. None and cosine differ by 0.00003
(0.93706 vs 0.93709). The comparison table rounds to 4 decimals, where both are 0.9371, and `pick` then keeps
the first option listed (none). Strictly, cosine has the higher mean; a difference of 0.003 pp is meaningless.

**Decision: no schedule.**

*Other rounds.* No schedule in r1 and r3. In r3 (cap 200 epochs) cosine scored only 0.913: its decay is sized
to the 200-epoch cap, and early stopping usually ended the run long before the learning rate had decayed, so
the comparison at a raised cap is unfair to cosine and step.
""",
    'loss': """
**Result.** cross-entropy 0.937, class-weighted CE 0.934, CE + weighted sampler 0.929, **label smoothing 0.937**,
focal 0.931. Label smoothing beats plain CE by 0.03 pp, which is nothing. What differs is the per-class recall:
plain CE is strongest on glioma (0.941) and weakest on meningioma (0.872); label smoothing is more even
(glioma 0.904, meningioma 0.928) and has the best tumour AUC (0.996).

**Decision: label smoothing (0.1)** (highest mean).

**In hindsight this was most likely a mistake.** Label smoothing won by 0.03 pp, i.e. it bought nothing, and the
plan always included distilling this model into small students. Label smoothing pulls the wrong-class
probabilities together and erases much of the class-similarity information that distillation transfers
(Müller, Kornblith & Hinton 2019). Plain cross-entropy (0.937, the same score) would have been the better choice
for a teacher. Distillation indeed gave no reliable gain (d3). Retraining the teacher without it was not
tested.

*Other rounds.* Another unstable choice: r1 kept plain CE (tie rule), r3 picked focal loss (0.941 vs CE 0.938,
label smoothing 0.931). The imbalance fixes (weighting, sampler) never won: with balanced accuracy as the target
and 4 classes of reasonable size, plain training already treats the small classes well enough.
""",
    'summary': """
| # | Setting | Choice | Mean val bal. acc (3 seeds) | Runner-up |
|---|---|---|---|---|
| 1 | learning rate | Adam 0.03 | 0.937 | 0.01: 0.921 |
| 2 | optimiser | Adam | 0.937 | SGD 0.1: 0.928 |
| 3 | batch size | 64 | 0.937 | 32: 0.931 |
| 4 | schedule | none | 0.937 | cosine 0.937 (tie) |
| 5 | loss | label smoothing | 0.937 | CE 0.937 |

**c6 result: 0.937** (from 0.883 for the same input at the c4 settings). Almost all of the gain comes from the
learning rate; the other four settings move the score by less than the seed noise, which is why different rounds
picked different batch sizes and losses.

**Across rounds.** r1 (tie rule): Adam 0.03, batch 32, CE, 0.934. r2 (this run): 0.937. r3 (200-epoch cap): Adam
0.03, batch 128, focal, 0.941. The learning rate is the only robust finding.
""",
}
