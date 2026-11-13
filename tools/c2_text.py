T = {}
T["dumb"] = """
**Result.** (Random guessers are averaged over 200 seeds.)
* **Accuracy** gives the *do-nothing* classifier its best score (0.375). That rules it out.
* **Weighted-F1** prefers proportional guessing (0.282), so it is biased by class size in the same way.
* **Balanced accuracy, MCC and κ** score every dumb classifier at exactly chance (0.25 / 0 / 0), whatever the
  strategy, so they have a fixed "no skill" reference. Macro-F1's chance level depends on the guessing strategy
  (0.14-0.25).
"""
T["placement"] = """
**Result.**
* **Accuracy and MCC barely notice where errors are.** All lines nearly overlap, so they don't protect the
  small class.
* **Balanced accuracy** is linear. A missed no_tumor image costs 312/96 ≈ 3.3x as much as a missed glioma image,
  because each error costs 1/(4·n_class). It ignores *where* the error lands: "glioma → no_tumor" and
  "glioma → meningioma" are the same line.
* **Macro-F1** also penalises errors *landing in* the small class (orange below green), through no_tumor's
  precision. Its penalty is non-linear and grows faster as no_tumor recall falls.

**Decision: balanced accuracy is the primary metric.**
1. Its fixed chance level (0.25) makes it easy to interpret.
2. It is linear and recall-based, so its noise is easy to reason about. It doesn't depend on class prevalence,
   which matches the sensitivity/ROC framing of step 10.
3. Its blind spot (it doesn't care *which* wrong class is predicted) is exactly the tumour → no_tumor error.
   That error is handled explicitly by the step-10 threshold and the reported tumour sensitivity, not by the
   selection metric.

**Secondary metrics (always reported):** macro-F1 (catches false alarms *into* small classes), per-class recall,
confusion matrix, tumour-vs-no-tumour ROC AUC, tumour sensitivity and specificity.
"""
T["noise"] = """
**Result.**
* **Per class:** no_tumor (96 images) is by far the noisiest. At 90 % recall its 95 % interval is about ±6 pp;
  the other classes are about ±3-4 pp.
* **Balanced accuracy** of one model on clean val: about ±2 pp (95 % CI half-width) at the 0.9 level.
* Comparing two models on the *same* images is less noisy than two independent intervals suggest, because the
  shared noise cancels. That is why comparisons use the **paired bootstrap** (`lib.evaluation.paired_bootstrap`).

**Decision: the "meaningful difference" rule** (seed noise σ_seed is measured in c4):
* A change is **adopted** if Δ balanced accuracy > 2·σ_seed **and** the paired bootstrap gives P(not better) < 0.05.
* Otherwise it is a **tie**. Ties go to the simpler, cheaper or more standard option.
* For decisions that shape the final model (architecture, regularisation), close calls get 2 extra seeds and
  the means are compared.

**Update (round r2, after c6 to c10 had run once with this rule).** The rule was dropped for model selection.
With σ_seed = 0.031 from c4 the bar for 3 vs 3 seeds is 5 pp, but once the models were tuned their seed sd was
0.004 to 0.009, and no change in c5 to c8 reached 5 pp: the rule kept the baseline at every step, including
changes that measured as clear gains (r1: 5x5 kernels +2.0 pp, residual connections +2.9 pp, augmentation +1.2
to +1.8 pp). The goal is the best possible model regardless of size, so from round r2 on **the option with the
highest mean val balanced accuracy wins**. The noise level is still reported next to every comparison, so a
reader can see which wins are within noise. The record of both rules is in the history notebook.
"""
T["prev"] = """
**Result.** The same classifier (sensitivity 0.98, specificity 0.90) has a PPV of 0.987 on this dataset, but
0.81 at 30 % prevalence and **0.09** at 1 %. Sensitivity and specificity do not change.

**Decision.** For the tumour decision, report **sensitivity, specificity and ROC AUC**. They transfer to other
prevalences. Precision/PPV is only used as a diagnostic inside this dataset, never as a headline number. This
dataset has 88.5 % tumours; clinical practice usually has far fewer.
"""
T["protocol"] = """
| Data | Used for | Never used for |
|---|---|---|
| **train** | fitting weights, normalisation statistics, class weights | any decision |
| **clean val** (832) | early stopping, checkpoint choice, every comparison in c4-c9, the step-10 threshold | fitting |
| **test** | one final run with everything frozen (official 889 and clean 833) | anything else |

**Not decided yet (needs a trained model):** what early stopping and checkpointing **monitor**.
* **val loss:** smooth, and rewards good probabilities (which step 10 needs). But it can rise from over-confidence
  while accuracy still improves.
* **val balanced accuracy:** the actual target, but a noisy step function that tends to pick a lucky epoch.

Both are logged every epoch. The choice is made in **c4** from the initial model's curves.

**Test set:** not used for any decision; all selection is on clean val. It is evaluated once, at the end
(`test_evaluation.ipynb`).
"""
T["severity"] = """
Options for picking the cutoff t on P(tumour) = 1 - P(no_tumor):

| Rule | Meaning | Verdict |
|---|---|---|
| argmax | predict the most likely class | implicitly treats all errors as equal |
| Youden's J | maximise sensitivity + specificity - 1 | equal cost for FN and FP, so rejected |
| min expected cost | minimise c_FN·FN + c_FP·FP for an assumed ratio | needs a cost ratio nobody here can justify precisely |
| **target sensitivity** | highest t with val tumour sensitivity ≥ target | explicit and easy to explain |

**Decision:** target tumour sensitivity **≥ 0.98** on clean val. That means at most 14 of 736 tumour images missed.
Specificity is reported at that cutoff, with Wilson intervals for both. A 10:1 cost-ratio cutoff is shown next to
it as a sensitivity analysis, so the reader can see how much the choice of rule matters.
"""
T["summary"] = """
| Decision | Choice | Evidence |
|---|---|---|
| Primary metric | **balanced accuracy** | fixed chance level; linear; protects small classes (sections 1-2) |
| Secondary metrics | macro-F1, per-class recall, confusion matrix, tumour AUC, sensitivity/specificity | section 1, section 3 |
| Not reported as headline | accuracy, weighted-F1, precision/PPV | dumb-classifier test, prevalence test |
| Meaningful difference | Δ > 2·σ_seed·√(1/n_a + 1/n_b) and paired-bootstrap P < 0.05; ties go to the simpler option (rounds r0, r1). **From r2: highest mean wins**, noise level reported | section 2 (σ_seed in c4) |
| Evaluation set | clean val for everything; test once at the end | c1, section 4 |
| Early-stopping monitor | decided in c4 from the curves | section 4 |
| Threshold rule | tumour sensitivity ≥ 0.98 on val | section 5 |
"""
