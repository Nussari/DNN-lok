T = {
    'lr': """
**Result.** On the final configuration: lr 0.01 0.965, **0.03 0.969**, 0.09 0.952. No significant trend
(p = 0.13); 0.03 has the highest mean.

**Decision: keep lr 0.03.** The x1 runs are the final model's runs, so nothing is retrained.

**Why 0.969 here and 0.974 in c8 for the same configuration.** The x1 runs have exactly the c8 winner's
configuration and seeds, but were trained again: c8 gave 0.968 / 0.980 / 0.975 (mean 0.974), c9 gives 0.968 /
0.968 / 0.969 (mean 0.969). GPU training is not bit-for-bit reproducible here (cuDNN chooses its algorithms at run
time), so retraining the same configuration moves the mean by about 0.5 pp. This is not a loss between steps; it
is a direct measurement of how much the c8 ranking itself can move.

*Other round.* r3 (200-epoch cap, residual 9 model) moved to 0.01 (0.960 vs 0.952 at 0.03).
""",
    'final': """
**The final model:** plain CNN, 5 conv layers with 5x5 kernels, BatchNorm, average pooling, widths 16-32-64-128,
277k parameters. Input: grayscale, cropped to the head, 128 px, per-image normalisation, mild augmentation.
Training: Adam lr 0.03, batch 64, label smoothing 0.1, no schedule, early stopping on val balanced accuracy
(patience 20, cap 60 epochs, best epoch 48).

**Result (seed 1738, fixed in advance):** val balanced accuracy **0.968 [95 % CI 0.955, 0.979]**, macro-F1 0.965,
tumour AUC 0.998. Plain accuracy 0.966. The other two seeds give 0.968 and 0.969: this configuration is
stable (sd 0.001).

| class | recall | 95 % CI |
|---|---|---|
| glioma | 0.971 | 0.946 to 0.985 |
| meningioma | 0.931 | 0.889 to 0.957 |
| no_tumor | 0.979 | 0.927 to 0.994 |
| pituitary | 0.990 | 0.966 to 0.997 |

Meningioma is the weakest class; most of its errors are predicted as glioma (9 of 15), and glioma's errors go to
meningioma (6 of 9). The glioma/meningioma confusion noted in c4 is still the main error, at a much lower level.

**Progress:** c4 0.896, c5 0.883 (same input retrained on CUDA), c6 0.937, c7 0.959 / 0.962, c8 0.974,
final 0.968 (the same configuration as c8's 0.974, retrained: see the note in section 1).

**Across rounds.** r1 (tie rule, 60 epochs): 0.932 [0.913, 0.949]. r2 (this run): 0.968. r3 (200-epoch cap):
0.952 [0.935, 0.966], seeds 0.952 / 0.959 / 0.968. r2 is reported because it has the highest val score; its
interval overlaps r3's. Why r3 ended lower is analysed in the history notebook: early stopping made each r3 run
slightly better, but the steps chose a different path of within-noise winners, and the pre-fixed seed was that
configuration's weakest.

**Comparison with published results on this dataset** (BDNeuro-MRI v7, same official split, test set n = 889):
the dataset authors report accuracy 0.847 for a CNN trained from scratch, 0.826 for a ViT and 0.80 to 0.87 for
frozen ImageNet ResNet50/DenseNet121; fine-tuned ImageNet models reach 0.971 (ResNet50) and 0.983
(DenseNet121). Our from-scratch model (0.966 accuracy on val) is far above the published from-scratch baselines
and in the range of the fine-tuned pretrained models. This comparison is not fair in either direction:
* **ours is optimistic**: the score is on val, which was used to choose among several hundred configurations;
  the published numbers are single test evaluations. Only the final test evaluation is comparable.
* **theirs is optimistic**: our clean val has 60 copies of train images removed; the official test set has not,
  and near-duplicates of training images inflate a test score. About 72 % of the BDNeuro images also occur in the
  public Kaggle brain-tumour collection (arXiv 2611.11793), so ImageNet- or Kaggle-pretrained models may have seen
  them.
* The authors report accuracy and an unspecified F1 from single runs; we report balanced accuracy (for this model
  plain accuracy is 0.966, balanced 0.968, so the metric barely matters here).
""",
}
