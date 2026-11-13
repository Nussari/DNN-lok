T = {
    'frontier': """
**The frontier** runs from 0.54 kB (2-bit, at chance) to the teacher's 1,080 kB. Read from the top:

| stored size | x smaller | val bal. acc | model |
|---|---|---|---|
| 1,080 kB | 1 | 0.968 | teacher |
| 57 kB | 19 | 0.971 | dw-sep 5x5 w32, fine-tuned, 8-bit QAT |
| 25 kB | 43 | 0.969 | dw-sep 5x5 w32, 50 % pruned, 4-bit QAT |
| 19 kB | 57 | 0.961 | dw-sep 5x5 w32, 75 % pruned, 4-bit QAT |
| 9.3 kB | 116 | 0.951 | dw-sep 3x3 w16, 4-bit QAT |
| 3.9 kB | 278 | 0.924 | dw-sep 5x5 w8, 4-bit QAT |
| 1.4 kB | 766 | 0.858 | dw-sep 5x5 w4 (distilled), 50 % pruned, 4-bit QAT |
| 0.74 kB | 1,457 | 0.786 | dw-sep 5x5 w2, 4-bit QAT |

* **Nothing is lost down to about 25 kB** (43x smaller): those models match or beat the teacher.
* Below that the loss grows quickly. Against the teacher: −0.7 pp at 19 kB, −1.7 pp at 9.3 kB, −4.4 pp at 3.9 kB,
  −11 pp at 1.4 kB, −18 pp at 0.74 kB.
* **What each technique contributed**, in order of impact: depthwise-separable architecture (5x smaller for free,
  d2), 4-bit QAT (8x smaller for almost nothing, d5), unstructured pruning (2x for larger models, d4), a short
  low-lr fine-tuning phase (+0.3 to 1.7 pp everywhere, d4/d5 controls). Distillation (d3) and structured pruning (d4)
  did not help.
""",
    'final': """
**High-performance model: depthwise-separable 3x3, width 16, 4-bit QAT. 9.3 kB, 116x smaller than the teacher,
mean val balanced accuracy 0.951** (teacher 0.968).
* It passes the 0.95 floor by 0.0006, with a seed sd of 0.011, and the designated seed 1738 alone scores 0.944
  [95 % CI 0.926, 0.961]. By the rule fixed in d1 it is the pick, but **it is a borderline one**: a re-run could
  land on either side of 0.95. The next frontier model, **dw-sep 5x5 w32, 75 % pruned, 4-bit (18.8 kB, 57x smaller,
  0.961)**, clears the floor with margin and is the safer choice if the floor must hold.
* Errors: meningioma recall 0.917 is the weakest class (teacher 0.931); tumour sensitivity 0.989, specificity 0.927.

**Tiny model: depthwise-separable 5x5, width 4, distilled, 50 % pruned, 4-bit QAT. 1.41 kB, 766x smaller than the
teacher, mean val balanced accuracy 0.858** (seed 1738: 0.853 [0.829, 0.877]).
* 1,912 parameters; half of its conv/linear weights are zero, and the rest are stored in 4 bits. It still catches 95.9 % of tumours
  (sensitivity) and identifies 89.6 % of the healthy scans, but meningioma recall drops to 0.685: the tiny model
  sacrifices the hardest distinction (meningioma vs glioma) first.
* For comparison: the dataset authors' from-scratch CNN scores 0.847 accuracy on the test set. This model is
  in the same range at 1.4 kB.

**Paths from the teacher** (plot above): high-performance = train the small architecture directly, then 4-bit
QAT. Tiny = train small with distillation, prune 50 %, then 4-bit QAT. Every step of both paths was chosen by the
d1 rules from measured alternatives.

**Caveats.**
* All scores are on val, which was used for every choice in d2-d6; they are optimistic in the same way as the c9
  score. The test set has still not been used.
* Sizes below 8 bits are computed from simulated quantization (the accuracy is exact; the storage format is the
  lean one defined in d1, not a PyTorch file). In PyTorch's own int8 format the files are 20 kB or more (d5).
""",
}
