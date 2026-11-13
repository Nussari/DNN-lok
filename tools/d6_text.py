T = {
    'frontier': """
**The frontier** runs from 0.54 kB (2-bit, at chance) to 272 kB (the teacher's layout with 8-bit PTQ, 0.975). The
teacher itself (1,080 kB, 0.968) is not on it, because smaller models beat it. Selected points, read from the top
(the teacher row is for reference):

| stored size | x smaller | val bal. acc | model |
|---|---|---|---|
| 1,080 kB | 1 | 0.968 | teacher (reference, not on the frontier) |
| 57 kB | 19 | 0.971 | dw-sep 5x5 w32, fine-tuned, 8-bit QAT |
| 25 kB | 43 | 0.969 | dw-sep 5x5 w32, 50 % pruned, 4-bit QAT |
| 19 kB | 57 | 0.961 | dw-sep 5x5 w32, 75 % pruned, 4-bit QAT |
| 9.3 kB | 116 | 0.951 | dw-sep 3x3 w16, 4-bit QAT |
| 3.9 kB | 278 | 0.924 | dw-sep 5x5 w8, 4-bit QAT |
| 1.4 kB | 766 | 0.858 | dw-sep 5x5 w4 (distilled), 50 % pruned, 4-bit QAT |
| 0.74 kB | 1,457 | 0.786 | dw-sep 5x5 w2, 4-bit QAT |

* **Nothing is lost down to about 25 kB** (43x smaller): those models match or beat the teacher.
* Below that the loss grows quickly. Against the teacher: −0.7 pp at 19 kB, −1.7 pp at 9.3 kB, −4.3 pp at 3.9 kB,
  −11 pp at 1.4 kB, −18 pp at 0.74 kB.
* **What each technique contributed**, in order of impact: depthwise-separable architecture (5x smaller for free,
  d2), 4-bit QAT (4.4 to 7.9x smaller stored size for almost nothing, d5), unstructured pruning (2x for larger models, d4), a short
  low-lr fine-tuning phase (+0.3 to 1.4 pp for models not fine-tuned before, d4; about 0 for models that already
  were, d5 controls). Distillation (d3) and structured pruning (d4)
  did not help.
""",
    'final': """
**High-performance model: depthwise-separable 3x3, width 16, 4-bit QAT. 9.3 kB, 116x smaller than the teacher,
mean val balanced accuracy 0.951** (teacher 0.968).
* It passes the 0.95 floor by 0.0006, with a seed sd of 0.011, and the designated seed 1738 alone scores 0.944
  [95 % CI 0.926, 0.961]. By the rule fixed in d1 it is the pick, but **it is a borderline one**: a re-run could
  land on either side of 0.95. This is why the third pick was added.
* Errors: meningioma recall 0.917 is the weakest class (teacher 0.931); tumour sensitivity 0.989, specificity 0.927.

**High-performance model with a 1 pp margin: depthwise-separable 5x5, width 32, 75 % pruned, 4-bit QAT. 18.8 kB,
57x smaller than the teacher, mean val balanced accuracy 0.961** (seed 1738: 0.961 [0.947, 0.974]).
* The smallest model with mean ≥ 0.96. The frontier point between the two high-performance picks (dw-sep 3x3 w16
  with 8-bit QAT, 15.5 kB, 0.954) clears 0.95 by only 0.4 pp.

**Tiny model: depthwise-separable 5x5, width 4, distilled, 50 % pruned, 4-bit QAT. 1.41 kB, 766x smaller than the
teacher, mean val balanced accuracy 0.858** (seed 1738: 0.853 [0.829, 0.877]).
* 1,912 parameters; half of its conv/linear weights are zero, and the rest are stored in 4 bits. It still catches 95.9 % of tumours
  (sensitivity) and identifies 89.6 % of the healthy scans, but meningioma recall drops to 0.685: the tiny model
  sacrifices the hardest distinction (meningioma vs glioma) first.
* For orientation only: the dataset authors report test accuracy 0.847 for their from-scratch CNN. That is a
  different metric on a different split; the comparison on the test set is in `test_evaluation.ipynb`.

**Paths** (plot above): the 9.3 kB model is a small architecture trained on the labels, then 4-bit QAT; the
18.8 kB model is a wider student trained on the labels, pruned 75 %, then 4-bit QAT; the 1.4 kB model is a small
student distilled from the teacher, pruned 50 %, then 4-bit QAT. Only the 1.4 kB model uses the teacher. Every step
was chosen by the d1 rules from measured alternatives.

**Caveats.**
* All scores are on val, which was used for every choice in d2-d6; they are optimistic in the same way as the c9
  score. The test set has still not been used.
* Sizes below 8 bits are computed from simulated quantization (the accuracy is exact; the storage format is the
  lean one defined in d1, not a PyTorch file). In PyTorch's own int8 format the files are 20 kB or more (d5).
""",
}
