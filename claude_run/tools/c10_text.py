T = {
    'roc': """
**Result.** Tumour vs no-tumour on clean val (736 tumour, 96 no-tumour images), score = 1 - P(no_tumor):

| cutoff | threshold | sensitivity | missed tumours | specificity | false alarms |
|---|---|---|---|---|---|
| argmax (no threshold) | | 0.992 [0.982, 0.996] | 6 | 0.979 [0.927, 0.994] | 2 |
| **rule: sensitivity >= 0.98** | 0.717 | 0.985 [0.973, 0.992] | 11 | 0.990 [0.943, 0.998] | 1 |
| Youden's J | 0.780 | 0.977 | 17 | 1.000 | 0 |
| cost 10:1 | 0.477 | 0.996 [0.988, 0.999] | 3 | 0.958 [0.898, 0.984] | 4 |

Tumour AUC 0.998.

**Finding: the rule lowers sensitivity here.** The rule from c2 takes the *highest* cutoff that still reaches
sensitivity 0.98. It was written expecting argmax to fall short of 0.98, so that the cutoff would move down and
catch more tumours. This model already reaches 0.992 at argmax, so the rule moves the cutoff *up*: it misses
5 more tumours (11 instead of 6) to save 1 false alarm. For a screening question that is the wrong trade.
The rule is kept as it was fixed in advance; the cost-10:1 cutoff (3 missed, 4 false alarms) or plain argmax
are the alternatives to consider before the test evaluation.

*Other rounds.* r1 (weaker model): threshold 0.405, a much lower cutoff, which calls more images tumour.
r3: threshold 0.710, the same situation as here.
""",
    'effect': """
**Result.** On the 4-class problem the threshold changes almost nothing: balanced accuracy 0.968 with argmax,
0.967 with the 0.717 cutoff. The cutoff only moves images between "no_tumor" and the most likely tumour class:
5 more tumours called no_tumor (glioma 2 to 5; pituitary 0 to 2; meningioma unchanged at 4), and 1 fewer no_tumor
image called a tumour (2 to 1).
Accuracy figures elsewhere in this project are all argmax.

The test set has not been used. `RUN_TEST = False` until the final decisions (including the cutoff above) are
fixed; then the frozen model is evaluated once.
""",
}
