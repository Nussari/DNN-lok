T = {
    "results": """
**Result.** Balanced accuracy on the clean test set (833 images), with the val score of the same model under the
same decision rule for comparison:

| model | stored size | multiply-adds | val | **test (clean)** | test (official, 889) |
|---|---|---|---|---|---|
| large | 1,080 kB | 72 M | 0.968 | **0.959** [0.942, 0.973] | 0.961 |
| 18.8 kB | 18.8 kB | 28 M | 0.961 | **0.950** [0.934, 0.966] | 0.954 |
| 9.3 kB | 9.3 kB | 5.8 M | 0.944 | **0.937** [0.918, 0.953] | 0.939 |
| 1.4 kB | 1.4 kB | 2.4 M | 0.812 | **0.807** [0.779, 0.835] | 0.811 |

* **The val scores were optimistic, as expected, but only slightly.** Every model loses 0.5 to 1.1 pp from val to
  test, inside its confidence interval: selecting among hundreds of configurations on val cost about 1 pp.
* **The order of the models is the same as on val**, and the 57x smaller 18.8 kB model stays within 0.9 pp of the
  large one. The 9.3 kB model ends below the 0.95 floor it was chosen for on val (0.937); the 18.8 kB model, picked
  with a 1 pp margin for exactly this risk, holds it (0.950).
* **Tumour detection holds up**: on clean test every model catches 98.8 to 98.9 % of tumours (8 or 9 missed out
  of 728). The cost
  differs: 6 false alarms out of 105 healthy scans for the large model, 8 for the 18.8 and 9.3 kB models, and 33
  for the 1.4 kB model, whose cutoff trades specificity for sensitivity by design.
* **Official vs clean test** differ by at most 0.4 pp: the 56 near-copies of training images in the official split
  barely inflate the score.
* The main remaining error is still glioma vs meningioma (12 of the large model's 34 errors on clean test).
* For comparison, the dataset authors report test accuracy 0.847 for a CNN trained from scratch and 0.971 / 0.983
  for fine-tuned ImageNet ResNet50 / DenseNet121 (official split). The large model reaches accuracy 0.962 on the
  same split without pretraining, and the 1.4 kB model (accuracy 0.835) is close to their from-scratch CNN.
""",
}
