T = {
    'unstructured': """
**Result** (mean val balanced accuracy, 3 seeds):

| start | original | fine-tuned only (control) | 50 % | 75 % | 90 % | 95 % |
|---|---|---|---|---|---|---|
| dw-sep 5x5 w32, 207 kB | 0.964 | **0.971** | 0.970 (111 kB) | 0.960 (60 kB) | 0.862 (30 kB) | 0.469 (20 kB) |
| teacher layout (distilled), 1,080 kB | 0.971 | **0.974** | 0.974 (574 kB) | 0.969 (304 kB) | 0.926 (143 kB) | 0.774 (89 kB) |
| dw-sep 5x5 w4 (distilled), 7.1 kB | 0.869 | **0.883** | 0.858 (4.0 kB) | 0.565 (2.3 kB) | 0.348 | 0.250 (chance) |

* **The control matters.** Without it, 50 % pruning would look like it *improves* the 207 kB model (0.964 to 0.970).
  The control shows that gain comes from the 30 extra epochs at lr 0.003, not from pruning: fine-tuning alone gives
  0.971. Measured against the control, 50 % pruning costs 0.0 to 0.1 pp on the two larger models and 2.5 pp on
  the 7 kB one.
* **Larger models tolerate more pruning.** Against their fine-tuned controls, at 75 % the 207 kB model loses 1.1 pp
  and the 1,080 kB model 0.5 pp, and both collapse only at 90-95 %; the 7 kB model already collapses at 75 % (sd 0.27: some seeds keep working, others do not).
  A model with few weights has little redundancy to remove.
* **Size gain is smaller than the sparsity.** Sparse storage needs a 1-bit mask over every position, so 50 %
  sparsity saves 46 %, and 90 % saves 86 %.
* **Side finding.** A second phase at a tenth of the learning rate adds 0.3 to 1.4 pp to every model. The c-series
  recipe (constant lr, early stopping) ends slightly too early; a step-down at the end would likely help the
  teacher too. (c6 tested step decay at fixed epochs, not "train, then fine-tune at lr/10".)
""",
    'structured': """
**Result.** Slimming a wider trained model and fine-tuning it is **worse than training the narrow model directly**
in 3 of 4 cases, by a lot when the cut is large:

| target | slimmed and fine-tuned | trained directly (labels) | trained directly (distilled) |
|---|---|---|---|
| dw-sep 5x5 w16 (from w32), 61 kB | 0.936 | 0.942 | 0.944 |
| dw-sep 5x5 w8 (from w32), 20 kB | 0.853 | **0.916** | 0.898 |
| plain 5x5 w8 (from w16), 271 kB | **0.957** | 0.953 | 0.954 |
| plain 5x5 w4 (from w16), 68 kB | 0.885 | 0.929 | **0.930** |

Removing 75 % of the channels (w32 to w8, w16 to w4) loses 4 to 6 pp against training that width from scratch;
removing half is roughly a tie. The fine-tuning budget (30 epochs at lr 0.003) is probably too small to recover
from a large cut, but the comparison with direct training is what matters for this task: **for a narrower model,
train it directly.** Structured pruning is dropped.
""",
    'frontier': """
**Frontier after d4.** Unstructured pruning adds two points that matter:
* **dw-sep 5x5 w32, 75 % pruned: 60 kB, 0.960**, above the 0.95 floor (the dense frontier had nothing between 52 kB
  at 0.948 and 207 kB);
* **dw-sep 5x5 w4, 50 % pruned: 4.0 kB, 0.858**, above the 0.85 floor (the dense frontier's smallest model above it
  was 7.1 kB).

The fine-tuned controls also join the frontier (fine-tuning without pruning is a legitimate way to get a better
model of the same size). The collapsed models (the 7 kB model pruned 75, 90 and 95 %: 0.565, 0.348 and chance) are on the frontier
only because nothing smaller exists; they are kept for completeness. All frontier models go to d5.
""",
}
