T = {
    'sweep': """
**Result.** On the 207 kB reference (dw-sep 5x5 w32, labels only 0.964) every setting lands between 0.960 and
0.965: distillation neither helps nor hurts. On the 7 kB reference (dw-sep 5x5 w4, labels only 0.862) the settings
range from 0.853 to 0.869, with seed sds of 0.009 to 0.024, so the differences are mostly noise. T = 4, α = 0.5 has
the highest mean over both references (0.916) and is used from here on. No setting beats labels-only on both
references: T = 4, α = 0.5 gains 0.7 pp on the 7 kB student and ties on the 207 kB one (0.9642 vs 0.9643).
""",
    'compare': """
**Result: distillation does not reliably help here.** It beats labels-only for 3 of the 8 architectures, by
+0.3 to +0.9 pp, and loses for 4, by up to 2.8 pp (dw-sep 5x5 w2, 0 of 3 seeds improved) and 1.8 pp (dw-sep 5x5
w8, 0 of 3). Most differences are within one or two seed sds. There is no trend with size: neither the smallest
nor the largest students gain systematically.

**Why so little?** The textbook gain from distillation comes from the teacher's "dark knowledge": the relative
probabilities it gives the wrong classes. Two things limit that here:
* **The teacher was trained with label smoothing** (chosen in c6). Label smoothing pushes all wrong-class
  probabilities towards the same value, which erases much of the class-similarity information distillation
  transfers (Müller, Kornblith & Hinton 2019, *When does label smoothing help?*).
* **The task leaves little to transfer.** With 4 classes and a teacher at 0.97, the soft targets mostly repeat
  the labels; the one structured confusion (glioma vs meningioma) is something the students already learn.

The one consistent gain is **self-distillation** of the teacher's own architecture (0.969 to 0.971, 3 of 3 seeds),
a known effect, but it does not make anything smaller.

A teacher trained without label smoothing might distil better. That was not tested: it would mean retraining the
teacher, and the task was to compress this one.
""",
    'decision': """
**Decision.** Each architecture keeps the training method with the higher mean: distilled for dw-sep 3x3 w4,
dw-sep 5x5 w4 and the teacher layout, labels only for the other five. These are the starting points for pruning
(d4). Distillation stays in the fine-tuning of a pruned or quantized model only if that model was distilled.
""",
}
