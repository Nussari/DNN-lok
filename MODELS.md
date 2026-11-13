# The four models

One large model (the classifier pipeline, c1 to c10; it is the *teacher* in the compression notebooks) and three
compressed versions of it (d1 to d6). Each is the seed-1738 run of its configuration. Test scores come from `notebooks/test_evaluation.ipynb`, val scores from the
notebooks that chose the models.

## Overview

| | Large | Compressed 18.8 kB | Compressed 9.3 kB | Compressed 1.4 kB |
|---|---|---|---|---|
| d6 pick | (teacher) | smallest with val ≥ 0.96 (0.95 floor + 1 pp margin) | smallest with val ≥ 0.95 | smallest with val ≥ 0.85 |
| architecture | plain CNN, 5 conv layers, 5x5, avgpool, widths 16-128 | depthwise-separable, 5x5, avgpool, widths 32-256 | depthwise-separable, 3x3, avgpool, widths 16-128 | depthwise-separable, 5x5, avgpool, widths 4-32 |
| parameters | 276,628 | 53,796 | 13,588 | 1,912 |
| nonzero weights | 100 % | 25 % (75 % pruned) | 84 % | 50 % (50 % pruned) |
| weight bits | 32 | 4 | 4 | 4 |
| **stored size** | **1,080 kB** (1,105,488 B) | **18.8 kB** (19,236 B) | **9.3 kB** (9,512 B) | **1.4 kB** (1,444 B) |
| x smaller than large | 1 | 57 | 116 | 766 |
| multiply-adds per image (128x128) | 72 M | 28 M | 5.8 M | 2.4 M |

Stored size = BatchNorm folded into the convolutions, *b*-bit weights plus fp32 per-channel scales and biases,
sparse layers as a 1-bit mask plus the nonzero values when that is smaller (d1). 1 kB = 1,024 bytes.

## Test results

Clean test set (833 images: 309 glioma, 218 meningioma, 105 no_tumor, 201 pituitary; near-copies of training
images removed) and the official test split (889).

| | Large | 18.8 kB | 9.3 kB | 1.4 kB |
|---|---|---|---|---|
| **balanced accuracy, clean test** [95 % CI] | **0.959** [0.942, 0.973] | **0.950** [0.934, 0.966] | **0.937** [0.918, 0.953] | **0.807** [0.779, 0.835] |
| balanced accuracy, official test | 0.961 | 0.954 | 0.939 | 0.811 |
| accuracy, official test | 0.962 | 0.956 | 0.938 | 0.835 |
| macro-F1, clean test | 0.955 | 0.949 | 0.934 | 0.818 |
| recall glioma / meningioma / no_tumor / pituitary | 0.951 / 0.940 / 0.943 / 1.000 | 0.951 / 0.940 / 0.924 / 0.985 | 0.926 / 0.922 / 0.924 / 0.975 | 0.900 / 0.697 / 0.686 / 0.945 |
| tumour sensitivity (missed of 728) | 0.988 (9) | 0.989 (8) | 0.989 (8) | 0.989 (8) |
| tumour specificity (false alarms of 105) | 0.943 (6) | 0.924 (8) | 0.924 (8) | 0.686 (33) |
| tumour AUC | 0.997 | 0.997 | 0.994 | 0.972 |
| change from val (seed 1738, same decision rule) | −0.9 pp | −1.1 pp | −0.7 pp | −0.5 pp |

## Validation results

Clean val (832 images). Val was used for every choice, so these scores are optimistic; the test scores above are
the unbiased ones.

| | Large | 18.8 kB | 9.3 kB | 1.4 kB |
|---|---|---|---|---|
| balanced accuracy, mean of 3 seeds | 0.969 (sd 0.001) | 0.961 (sd 0.013) | 0.951 (sd 0.011) | 0.858 (sd 0.006) |
| seeds 1738 / 1739 / 1740 | 0.968 / 0.968 / 0.969 | 0.961 / 0.973 / 0.948 | 0.944 / 0.964 / 0.944 | 0.853 / 0.856 / 0.864 |
| seed 1738, 95 % CI | 0.968 [0.955, 0.979] | 0.961 [0.947, 0.974] | 0.944 [0.926, 0.961] | 0.853 [0.829, 0.877] |
| tumour AUC | 0.998 | 0.994 | 0.996 | 0.973 |
| balanced accuracy with the decision rule (seed 1738) | 0.968 | 0.961 | 0.944 | 0.812 |

## Tumour decision rule (c10)

**Argmax; only if tumour sensitivity on clean val is below 0.98, the highest cutoff on P(tumour) that reaches
0.98.** This gives argmax for the large, 18.8 kB and 9.3 kB models, and a cutoff of 0.283 for the 1.4 kB model
(on val: missed tumours 30 to 14 of 736, false alarms 10 to 31 of 96). All test scores above use this rule.

## How each was made

* **Large:** grayscale, cropped to the head, 128 px, per-image normalisation; Adam lr 0.03, batch 64, no schedule,
  label smoothing 0.1, mild augmentation, early stopping on val balanced accuracy (patience 20, cap 60 epochs).
* **18.8 kB:** depthwise-separable student trained on the labels (d2), 75 % unstructured magnitude pruning with
  fine-tuning (d4), 4-bit quantization-aware training (d5).
* **9.3 kB:** depthwise-separable student trained on the labels (d2), 4-bit quantization-aware training (d5).
* **1.4 kB:** depthwise-separable student distilled from the large model (d3), 50 % pruning (d4), 4-bit
  quantization-aware training (d5). Distillation gave no reliable gain overall (d3), but for this architecture the
  distilled student had the higher mean, so d3's rule kept it.

## Files

| | weights | run record (config, history, val metrics) |
|---|---|---|
| Large | `checkpoints/c9/final_s1738.pt` | `results/c9/final_s1738.json` |
| 18.8 kB | `checkpoints/d5/dwsep_k5_w32_labe_prune75_qat4_s1738.pt` | `results/d5/dwsep_k5_w32_labe_prune75_qat4_s1738.json` |
| 9.3 kB | `checkpoints/d5/dwsep_k3_w16_qat4_s1738.pt` | `results/d5/dwsep_k3_w16_qat4_s1738.json` |
| 1.4 kB | `checkpoints/d5/dwsep_k5_w4_dist_prune50_qat4_s1738.pt` | `results/d5/dwsep_k5_w4_dist_prune50_qat4_s1738.json` |

Run names encode how a model was made: `dwsep` / `single` = block type, `k5` = kernel size, `w32` = base width,
`labe` / `dist` = trained on the labels / distilled from the teacher, `prune75` = 75 % pruned, `qat4` = 4-bit
quantization-aware training, `_s1738` = seed.

The weights are standard fp32 PyTorch state dicts. In the compressed models every output channel's weights
already lie on the 4-bit grid (at most 15 distinct values) and pruned weights are exactly zero, so they reproduce
the quantized models exactly. Build a model from its run record's `config` with `lib.train.build_model` and load
the state dict (see the README).

## Limitations

* The large model was trained with label smoothing, which gained only 0.03 pp over plain cross-entropy in c6 and
  most likely weakened distillation (d3). For a model meant to be distilled, plain cross-entropy is the better
  choice; this was not tested.
* The 9.3 kB model was chosen as the smallest model above 0.95 on val (mean 0.951); on test it scores 0.937. The
  18.8 kB model, picked with a 1 pp margin for that reason, holds 0.950 on test.
* 4-bit weights are simulated: PyTorch has no 4-bit CPU kernels, so the stored size is the deployment size, but
  inference runs in fp32.
