# Experiment history

The pipeline (c1 to c10) was run in four rounds. The notebooks in `notebooks/` and the runs in `results/` are the
reported round, **r2**. This folder keeps the runs of the other rounds (`rounds/r0`, `rounds/r1`, `rounds/r3`:
run records and, where they exist, the decision files) and this summary of what was tried, what was decided and
why. `notebooks/history_all_rounds.ipynb` compares every configuration of every round side by side.

All scores are balanced accuracy on clean val (832 images), mean of 3 seeds (2 for the deep candidates in c8 and
c9) unless stated.

## Rounds

| round | selection rule | epoch cap | device | steps | final model (seed 1738) |
|---|---|---|---|---|---|
| r0 | tie rule | 60 | Apple M1 (MPS) | c1 to c6; c7 partly | none |
| r1 | tie rule | 60 | RTX 2080 Ti (CUDA) | c6 to c10 | 0.932 [0.913, 0.949] |
| **r2** | highest mean wins | 60 | RTX 2080 Ti (CUDA) | c5 to c10 | **0.968 [0.955, 0.979]** |
| r3 | highest mean wins | 200 | RTX 2080 Ti (CUDA) | c5 to c10 | 0.952 [0.935, 0.966] |

The tie rule: an option replaced the current choice only if its mean beat it by more than 2·σ_seed·√(2/n)
(σ_seed = 0.031 from c4, so 5.05 pp for 3 vs 3 seeds); otherwise the simpler or default option was kept.

### r0: the first pass
* c1 to c4 fixed the data handling (head crop, near-duplicate removal, clean val/test), the metric (balanced
  accuracy), the early-stopping protocol (val balanced accuracy, patience 20, cap 60) and the seed noise.
* **The pairwise rule was too blunt for sweeps.** In c6 every learning rate tied with the default 1e-3, yet the
  sweep rose steadily: +3.5 pp per decade of learning rate (p = 5e-7), with the best value at the edge of the
  sweep. A curve test was added for ordered settings (regression over all runs, sweep extended while the best
  value is at the edge). It moved the learning rate from 1e-3 to 0.03, the largest single gain of the project.
* The c7 runs already made at lr 1e-3 (32 runs) were discarded as a basis for decisions; they are kept in
  `rounds/r0/results/c7`.

### r1: the tie rule on the GPU
Same rule, c6 to c10. The c4 baseline retrained on CUDA scored 0.897 vs 0.896 on MPS (sd 0.009), so runs from the
two devices are comparable.

| step | measured (mean val balanced accuracy) | kept |
|---|---|---|
| c6 | Adam lr: 1e-3 0.896, 3e-3 0.913, 1e-2 0.922, **3e-2 0.928**, 1e-1 0.895 | Adam 0.03 (curve test), batch 32, no schedule, cross-entropy: 0.934 |
| c7 kernel | 3x3 0.923, 5x5 0.943, 7x7 stem 0.914 | 3x3 (tie) |
| c7 downsampling | maxpool 0.923, avgpool 0.931, stride 0.868 | maxpool |
| c7 depth | plain 5 0.923, plain 9 0.928, plain 17 0.872, plain 33 0.847; residual 9 0.952, 17 0.950, 33 0.933; no BatchNorm: 0.25 (diverges at lr 0.03) | plain 5 (small), residual 17 (deep) |
| c7 width | residual 17: 16 0.950, 32 0.955 | 16 |
| c8 augmentation | small: none 0.935, geometric 0.947, mild 0.944, strong 0.935; deep: none 0.949, geometric 0.967 | none (tie) |
| c9 | lr re-check kept 0.03 | final 0.932 |

**Outcome: the tie rule was miscalibrated.** σ_seed = 0.031 was measured on the untuned c4 model; once tuned, the
seed sd was 0.004 to 0.009. A 5 pp bar was larger than every effect in c6 to c8, so the baseline was kept almost
everywhere, including gains that were consistent across seeds (5x5 kernels, residual blocks, augmentation).
r1's executed notebooks were not kept; its runs are in `rounds/r1/results`.

### r2: highest mean wins (reported)
**Decision:** the goal of the large model is the best possible score regardless of size, so the option with the
highest mean wins. Sweeps take the value with the highest mean. One exception: in c5 geometry, the guard that
prefers the input with the least signal outside the head stays, because it protects validity, not simplicity.
Width 64 was added to c7.

| step | decision | mean val balanced accuracy |
|---|---|---|
| c5 input | grayscale, crop to head, 128 px, per-image normalisation | 0.883 (retrained on CUDA) |
| c6 optimisation | Adam 0.03, batch 64, no schedule, label smoothing | 0.937 |
| c7 architecture | 5x5 kernels, average pooling; small = plain 5 layers, deep = residual 17 at width 32 | 0.959 / 0.962 |
| c8 regularisation | small: mild augmentation; deep: geometric augmentation, dropout 0.3 | 0.974 / 0.967 |
| c8 winner | small (277k parameters) over deep (7.7M) | 0.974 |
| c9 final | lr 0.03 kept; seed 1738 | 0.968 [0.955, 0.979] |

Per-image normalisation was chosen in c5 from Apple-M1 runs (0.901 vs 0.896 for global; r0 had kept global under
the tie rule from the same runs); retrained on CUDA the same configuration scored 0.883. r3, with every c5 option retrained on CUDA, chose per-image normalisation again.

### r3: a 200-epoch cap
Most r2 runs reached their best epoch near the 60-epoch cap, so r3 raised the cap to 200 (early stopping
unchanged). r3 chose differently at several steps (batch 128, focal loss, residual 9 as the small model,
geometric augmentation, final lr 0.01) and ended at 0.952.

**Why the longer cap did not help.** Early stopping works within a run: across all r3 runs the best score in the
first 60 epochs averaged 0.8822 and the best overall 0.8842, so the extra epochs made individual runs slightly
better, never worse. The lower final score comes from the selection path: the options at several steps lie within
seed noise of each other, r3 picked different ones, and the seed fixed in advance (1738) was the weakest of its
final configuration (0.952 vs 0.959 and 0.968). r2 and r3 are equal within their confidence intervals.

**Decision:** r2 is the reported pipeline (`EPOCH_CAP = 60` in `lib/report.py`; 200 reproduces r3). This was
decided on val, before the test evaluation.

## Compression (d1 to d6)

* **Set-up decisions:** the teacher is the r2 final model alone; size means stored bytes (parameters and compute
  reported alongside); two targets, the smallest model with mean val balanced accuracy ≥ 0.95 and the smallest with
  ≥ 0.85; simulated 8/4/2-bit quantization plus real int8.
* **A third pick.** The 0.95 pick (9.3 kB) cleared its floor by only 0.0006, so after d6 a third model was added:
  the smallest with mean ≥ 0.96 (18.8 kB). Decided on val, before the test evaluation; on test the 9.3 kB model
  scores 0.937 and the 18.8 kB model 0.950.
* **Controls.** The first pruning and 8-bit QAT runs seemed to *improve* accuracy (for example 0.964 to 0.970 at
  50 % pruning). Fine-tune-only controls (same epochs and learning rate, no pruning or quantization) showed that
  the gain comes from the extra low-lr fine-tuning (+0.3 to 1.4 pp for models not fine-tuned before). All pruning
  and quantization results are measured against these controls.
* **What worked:** depthwise-separable blocks (5x smaller at no cost), 4-bit quantization-aware training (4.4 to
  7.9x smaller stored size, almost free above 50 kB), unstructured pruning (50 % free, 75 % costs 0.5 to 1 pp on
  the larger students).
* **What did not:** distillation (3 of 8 students gained under 1 pp, others lost up to 2.8 pp), structured
  pruning, 2-bit weights (every model collapses), real int8 in PyTorch (1.3 to 4.9x slower than fp32 on CPU at
  these sizes, and less accurate than simulated 8-bit for the smallest models).

## Decisions revised after the fact

* **Tumour decision rule.** c2 fixed "the highest cutoff with tumour sensitivity ≥ 0.98" before any model existed,
  expecting argmax to fall short of 0.98. The r2 model already reaches 0.992 at argmax, so the rule as worded
  *raised* the cutoff (10 missed tumours instead of 6, to save one false alarm). It was restated as a floor:
  argmax, and a cutoff only when argmax misses 0.98. Decided on val, before the test evaluation.
* **Label smoothing was most likely a mistake.** It won c6 by 0.03 pp over plain cross-entropy, i.e. it bought
  the teacher nothing, and label smoothing removes much of the class-similarity information distillation relies on
  (Müller, Kornblith & Hinton 2019). Distillation then gave no reliable gain. Since distillation was planned from
  the start, plain cross-entropy would have been the better choice. Not retested.

## Test results

The four final models were evaluated once on the test set, after every decision above
(`notebooks/test_evaluation.ipynb`). Clean test balanced accuracy: large 0.959, 18.8 kB 0.950, 9.3 kB 0.937,
1.4 kB 0.807; 0.5 to 1.1 pp below the val score of the same model under the same decision rule.

## Published results on this dataset

BDNeuro-MRI v7 (Mendeley Data, DOI 10.17632/zwr4ntf94j.7). Results reported on the official test split (889):

| source | model | pretrained | accuracy |
|---|---|---|---|
| dataset authors | CNN | no | 0.847 |
| dataset authors | ViT | no | 0.826 |
| dataset authors | hybrid CNN-ViT | no | 0.819 |
| dataset authors | ResNet50 / DenseNet121 | frozen ImageNet | 0.803 / 0.871 |
| dataset authors | ResNet50 / DenseNet121 | fine-tuned ImageNet | 0.971 / 0.983 |
| **this project** | large model (from scratch) | no | **0.962** |
| **this project** | 1.4 kB model | no | 0.835 |

Caveats: the authors report single runs and an unspecified F1. arXiv 2611.11793 matched about 72 % of the
BDNeuro images to the public Kaggle brain-tumour collection, so ImageNet- or Kaggle-pretrained models may have
seen them. The official test split keeps 56 near-copies of training images; on the clean test set (833) our
scores are 0.2 to 0.4 pp lower.

Sources: https://data.mendeley.com/datasets/zwr4ntf94j/7 ,
https://github.com/irfanulkabirhira/A-Bangladeshi-Clinical-Brain-Tumor-MRI-Dataset-for-Four-Class-Deep-Learning-Classification ,
https://arxiv.org/html/2611.11793

## Not tested

* A teacher trained without label smoothing, and distillation from it.
* A final low-learning-rate phase in the large model's recipe (it added 0.3 to 1.4 pp to every compressed model
  that had not been fine-tuned before).
* Zeroing the background below the head threshold, to remove the remaining signal outside the head (c5 probe).
