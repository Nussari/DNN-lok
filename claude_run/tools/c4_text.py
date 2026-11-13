T = {}
T["overfit"] = """
**Result.** Training loss falls to 0.003 and training balanced accuracy reaches 100 % at epoch 30 on 64 images.
The labels, loss, optimiser and data flow are wired correctly, and the model has enough capacity to memorise.
**Pass.**
"""
T["curves"] = """
**What the curves show.**
* **Train** loss drops steadily, and train balanced accuracy is about 100 % from epoch 25-30. After that the model
  has memorised the training set, with occasional spikes (Adam at a constant lr 1e-3 jumping out of the minimum).
* **Val** balanced accuracy is extremely jumpy in the first 25 epochs (0.4-0.85 between consecutive epochs; std of
  the epoch-to-epoch change is 0.145). It only settles around 0.87-0.90 **after** the training set is fitted.
  Val AUC is high early (about 0.98), so the jumpiness is mostly in the *argmax decision*, not in the ranking.
* The best val epochs come late: 31-54 for val loss, 42-60 for val balanced accuracy.

**Consequences.**
* An early-stopping rule that stops in the jumpy phase will stop on noise. This is a problem for the patience
  setting, not just the monitor (section 4).
* The constant learning rate is a likely cause of the jumpiness. Learning rate and schedule are tested in c6.
"""
T["monitor"] = """
**Result (mean balanced accuracy on the unseen half B, 5 seeds × 40 random splits).**
* **Patience matters more than the monitor.** With patience 5-10, every rule stops around epoch 7-13, in the
  jumpy phase, and reaches only 0.71-0.82. With patience 20 the rules reach 0.84-0.88; with 60 (= train 60 epochs,
  keep the best) 0.88-0.89.
* **val balanced accuracy ≥ val loss at every patience** (0.879 vs 0.840 at patience 20; 0.890 vs 0.876 at 60).
  Val loss keeps rising from over-confidence while the predictions still improve, so it stops too early.
* The 3-epoch moving average is no better than the raw balanced accuracy at patience ≥ 20, and is worse below.

**Decision: monitor = val balanced accuracy, patience 20, at most 60 epochs.** This almost matches "always train
60 epochs and keep the best" (0.879 vs 0.890) while saving time. Patience 60 would cost about 20 % more compute for
about +1 pp. The bias this rule could have (picking a lucky epoch on the same data it reports) was measured
directly by the A/B split above, and the rule still wins on data it did not see.
"""
T["seed"] = """
**Result.** Mean val balanced accuracy is 0.889 over 5 seeds, with **σ_seed = 0.031**. Four seeds land at 0.89-0.92.
Seed 1741 is the outlier (0.839): its best epoch was 12, and it then went 20 epochs without beating that lucky
early peak, so early stopping ended it at epoch 32. This is the failure mode from section 4, and it happens even
at patience 20, though much more rarely than at 5-10.

Seed 1738 (the designated seed) reaches 0.906 [0.883, 0.927]. Every class is well above the 0.25 of the dumb
baselines. Glioma and meningioma are mostly confused with each other (26 + 18 errors). no_tumor is rarely
confused with anything (9 errors in 96).

**What σ_seed means for later decisions.** With 3 seeds per option, a difference of means counts only if it is
larger than 2·σ·√(1/3 + 1/3) = **0.050**, i.e. about 5 pp of balanced accuracy. That is a high bar. Most single
changes will be "ties", and the rule then keeps the simpler or default option. This is deliberate: on 832 val
images and with this much seed noise, smaller differences cannot be told apart from luck without many more
seeds than this machine can afford.

**Update (round r2).** σ_seed was measured on this untuned model (lr 1e-3, jumpy val curve). The tuned models of
c6 to c8 have a seed sd of 0.004 to 0.009, so 5 pp was far too high a bar: in round r1 it rejected every change.
From round r2 the option with the highest mean wins (see c2). σ_seed is kept as the reported noise reference.
"""
T["summary"] = """
| Decision | Choice | Evidence |
|---|---|---|
| Sanity check | passed | memorises 64 images to 100 % (section 1) |
| Early-stopping monitor | val balanced accuracy | beats val loss at every patience on unseen half B (section 4) |
| Patience / epoch cap | 20 / 60 | patience 5-10 stops in the jumpy phase; 20 gets within 1 pp of "train 60, keep best". Round r3 raised the cap to 200 from c5 on: no gain (history notebook), so 60 stays |
| Noise level | σ_seed = 0.031 | 5 seeds (section 5) |
| Adoption rule | Δ > 2·σ·√(1/n_a + 1/n_b) (0.050 for 3 vs 3) | from c2, completed with σ_seed. Rounds r0, r1 only; from r2 the highest mean wins |
| Starting point | 0.896 (mean of seeds 1738-1740) | the baseline every later step must beat |

Open issues carried forward: constant LR 1e-3 is a likely cause of the jumpy val curve (c6), and
glioma/meningioma confusion is the main error (c7/c8).
"""
