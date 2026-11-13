T = {
    'roc': """
**Result.** Tumour vs no-tumour on clean val (736 tumour, 96 no-tumour images), score = 1 - P(no_tumor):

| cutoff | threshold | sensitivity | missed tumours | specificity | false alarms |
|---|---|---|---|---|---|
| argmax (no threshold) | | 0.992 [0.982, 0.996] | 6 | 0.979 [0.927, 0.994] | 2 |
| **rule: sensitivity >= 0.98** | 0.717 | 0.986 [0.975, 0.993] | 10 | 0.990 [0.943, 0.998] | 1 |
| Youden's J | 0.780 | 0.978 | 16 | 1.000 | 0 |
| cost 10:1 | 0.477 | 0.997 [0.990, 0.999] | 2 | 0.958 [0.898, 0.984] | 4 |

Tumour AUC 0.998. A cutoff t calls an image tumour when P(tumour) >= t (the convention of `roc_curve`).

**Finding: the rule lowers sensitivity here.** The rule from c2 takes the *highest* cutoff that still reaches
sensitivity 0.98. It was written expecting argmax to fall short of 0.98, so that the cutoff would move down and
catch more tumours. This model already reaches 0.992 at argmax, so the rule moves the cutoff *up*: it misses
4 more tumours (10 instead of 6) to save 1 false alarm. For a screening question that is the wrong trade.
The c2 wording is therefore restated in section 3 (floor, not target). The cost-10:1 cutoff (2 missed,
4 false alarms) is not used: it is the minimum of a cost curve fitted to a handful of val images.

*Other rounds.* r1 (weaker model): threshold 0.405, a much lower cutoff, which calls more images tumour.
r3: threshold 0.710, the same situation as here.
""",
    'effect': """
**Result.** On the 4-class problem the threshold changes almost nothing: balanced accuracy 0.968 with argmax,
0.968 with the 0.717 cutoff. The cutoff only moves images between "no_tumor" and the most likely tumour class:
4 more tumours called no_tumor (glioma 2 to 4; pituitary 0 to 2; meningioma unchanged at 4), and 1 fewer no_tumor
image called a tumour (2 to 1).
Accuracy figures elsewhere in this project are all argmax.

The test set is not used in this notebook (see `test_evaluation.ipynb`).
""",
    'decision': """
**Decision: argmax for the large model and both high-performance compressed models; cutoff 0.283 for the
1.4 kB model.** The three larger models already catch at least 98.9 % of tumours at argmax, so the floor does not
apply and no cutoff is fitted to val. The 1.4 kB model catches only 95.9 % at argmax (30 of 736 tumours missed);
lowering its cutoff to 0.283 cuts the missed tumours from 30 to 14 (sensitivity 0.981) at the cost of three times
the false alarms (10 to 31 of 96) and 4 pp of balanced accuracy (0.853 to 0.812). That is the trade the floor was written for: a missed
tumour is worse than a false alarm.

**Why the rule changed.** c2 fixed the rule before any model existed, expecting argmax to fall short of 0.98. Read
literally ("the highest cutoff reaching 0.98") it also *raises* the cutoff when argmax is already above 0.98,
which trades missed tumours for fewer false alarms, against its own purpose. The restated rule keeps the intent
(sensitivity of at least 0.98) and was decided on val only; the test set was not used.
""",
}
