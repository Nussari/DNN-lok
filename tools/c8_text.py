T = {
    'aug': """
**Result** (val balanced accuracy, train-val gap in brackets):

| augmentation | small (3 seeds) | deep (2 seeds) |
|---|---|---|
| none | 0.957 (0.039) | 0.960 (0.039) |
| geometric | 0.968 (0.029) | **0.961** (0.034) |
| mild | **0.974** (0.020) | 0.961 (0.026) |
| strong | 0.959 (0.022) | 0.959 (0.027) |

Augmentation is the largest single gain for the small model (+1.7 pp with mild) and halves its train-val gap.
The deep model barely moves (all four within 0.3 pp), although augmentation does shrink its train-val gap.
Augmented runs used nearly the full 60 epochs (small mild 59.3 on average, deep mild 54.0). Round r3 allowed up
to 200: within a run the extra epochs added only 0.2 pp on average (history notebook), and r3's best regularised
small model (a residual 9 net, so not the same model) reached 0.968, not more than here. So the 60-epoch cap is
not what limits them.

**Decision: small: mild; deep: geometric** (highest mean each; for the deep model the options are a tie in all
but name).

*Other rounds.* r1 (tie rule) measured +1.2 pp (small) and +1.8 pp (deep) for geometric and kept no augmentation.
r3 (200-epoch cap) chose geometric for small (0.968) and none for deep (0.956).
""",
    'wd': """
**Result.** small: 0 **0.974**, 0.01 0.953, 0.1 0.827. deep: 0 **0.961**, 0.01 0.950, 0.1 0.858.

**Decision: no weight decay** for both. Weight decay was tested as decoupled weight decay: for this test the code switches
Adam to **AdamW** (the table title says so). It hurts at both strengths, badly at 0.1. With the high learning
rate the shrinkage per step (lr × wd = 0.003 at wd 0.1) is large; AdamW at much smaller weight decay was not
tried.

*Other rounds.* No weight decay in every round.
""",
    'dropout': """
**Result.** small: 0 **0.974**, 0.3 0.963. deep: 0 0.961, 0.3 **0.967**.

**Decision: small: no dropout; deep: dropout 0.3** (highest mean each). Dropout helps only the large model, which
is the expected direction.

*Other rounds.* r1 and r3: no dropout for either model.
""",
    'final': """
**Result.** Regularised small 0.974 (3 seeds, 277k parameters, 0.7 s/epoch) vs regularised deep 0.967 (2 seeds,
7.7M parameters, 3.9 s/epoch). Per-class recall: small is better on no_tumor (0.972 vs 0.943), the deep model
slightly better on meningioma (0.970 vs 0.963).

**Decision: small** (highest mean). The selection rule ignores size, but the small model also happens to be
28x smaller and 5x faster.
""",
    'summary': """
| | small (plain 5) | deep (residual 17, width 32) |
|---|---|---|
| augmentation | mild | geometric |
| weight decay | 0 | 0 |
| dropout | 0 | 0.3 |
| mean val bal. acc | **0.974** | 0.967 |

**Winner: small** (0.974, from 0.959 after c7). Augmentation gave the gain; weight decay hurt; dropout helped only
the large model.

**Across rounds.** r1 (tie rule): nothing adopted, small 0.935, deep 0.949. r3 (200-epoch cap): small (residual
9, geometric augmentation) 0.968, deep (residual 17, width 16, no augmentation) 0.956. In every round the small
model won.
""",
}
