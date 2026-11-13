T = {}
T["geometry"] = """
**Val score.** stretch 0.882, pad 0.903, crop 0.896 (3 seeds each, sd about 0.009). pad has the highest mean,
by 0.8 pp over crop, well inside the seed noise.

**Probe (head blacked out).** No geometry is close to the 0.25 chance level:
* stretch 0.797, pad 0.787, crop 0.738.

The probe pictures show where that signal comes from. Masking sets the head to exactly 0, but the background
around it is not exactly 0: there is faint sensor noise, JPEG blocks and scanner borders. So the model still sees
the **outline and size of the head** and the **texture of the background**. Both differ by class, because the
no_tumor images come from other sources. Even crop leaves 0.74. Cropping removes most of the border and the
framing, but not the outline.

**Decision: crop to head.** Geometry is the one place where the highest mean does not decide on its own: among
the geometries within the c4 bar of the best val score (2·σ_seed·√(2/3) = 5 pp; all three here), the one with
the least signal outside the head wins. That bar uses σ_seed = 0.031 from the untuned c4 model and is generous:
with the seed sd observed here (about 0.009), stretch (2.1 pp below pad) would not count as within noise. The
pick does not change, because crop is within 0.7 pp of pad either way. This is a validity guard, not a preference for simplicity: pad's 0.8 pp could
come from the border shortcut, and a model that leans on scanner borders would not transfer.

**Caveat carried forward.** A probe of 0.74 means that shortcut signal exists in the inputs. It does *not* say
how much the real model uses it, because that model also sees the head, which is the stronger signal. One way to
remove more of it is to zero everything below the head threshold, so the background becomes truly black. That
was not tested here; it is noted as future work.

*Earlier and later rounds.* r0 (tie rule) also chose crop. r3 (200-epoch cap, all on CUDA): stretch 0.882,
pad 0.912, crop 0.895; crop again, by the same guard.
"""
T["channels"] = """
**Result.** Grayscale 0.896 vs RGB 0.877. RGB is lower, 3x noisier across seeds (sd 0.026 vs 0.009) and 15 %
slower. RGB does have a higher no_tumor recall (0.955 vs 0.931). That fits c1: about 4 % of the no_tumor images
are in real colour, which is a shortcut only that class has.

**Decision: grayscale** (highest mean). It also removes a shortcut instead of offering it to the model.

*Other rounds.* Same decision in r0 and r3 (r3: gray 0.895 vs RGB 0.887).
"""
T["norm"] = """
**Result.** Per-image standardisation 0.901 vs global train mean/std 0.896 (+0.5 pp, inside the seed noise).

**Decision: per-image** (highest mean). It is the candidate fix for scanners with different brightness.

**Caveat (found later).** These runs were trained on Apple MPS. Retrained on CUDA as the c6
starting point, the same per-image configuration scored 0.883, below the 0.896 of global. So the choice rested
on a cross-device comparison of two near-equal options. Round r3 retrained every option on CUDA and per-image
won again (0.904 vs global 0.895; 0.907 when retrained as the c6 starting point), so the choice holds, but the difference is small either way.

*Earlier round.* r0 (tie rule) kept global: +0.5 pp was far below the 5 pp the rule required.
"""
T["res"] = """
**Result.**
* 64 px: 0.886, 3x faster per epoch.
* 128 px: 0.896.
* 192 px: 0.846, and 2.4x slower. Its runs also stopped earlier (42 epochs on average).

**Why more pixels did not help this model.** The model has only 4 pooling stages, 3×3 kernels and global average
pooling. At 192 px each final feature sees a smaller fraction of the brain, and GAP averages more background
positions into the decision. A model needs depth and receptive field to match the resolution, which c7 tests.
So this result is about *this small model*, not about resolution in general.

**Curve test** (all 9 runs, val balanced accuracy against log resolution): significant peak (quadratic term
p = 0.018) at 128 px. The fit is reported as evidence; the value with the highest mean is chosen.

**Decision: 128 px** (highest mean, and the peak of the curve).

*Other rounds.* 128 px in r0 and r3 as well (r3: 64 px 0.892, 192 px 0.880).
"""
T["summary"] = """
| Question | Choice | Result (val bal. acc, 3 seeds) | Reason |
|---|---|---|---|
| Geometry | crop to head | 0.896 (stretch 0.882, pad 0.903) | within the guard's bar of the best (0.7 pp below pad); least shortcut signal (probe 0.74 vs 0.79-0.80) |
| Channels | grayscale | 0.896 vs 0.877 RGB | highest mean; colour is a no_tumor-only shortcut |
| Normalisation | per image | 0.901 vs 0.896 global (0.883 when retrained on CUDA) | highest mean (small; see the cross-device caveat) |
| Resolution | 128 px | 64: 0.886, 192: 0.846 | highest mean, peak of the curve |

**Selection rule.** From this round (r2) on, the option with the highest mean val balanced accuracy wins, whatever
its size or cost. The first rounds (r0 and r1) used the c2/c4 tie rule: a change counted
only if it beat the current choice by more than 2·σ_seed·√(2/3) = 5 pp, otherwise the simpler option was kept.
That bar turned out to be larger than every effect in c5 to c8 (observed seed sd 0.004 to 0.009 once the
models were tuned, against the σ_seed = 0.031 measured on the weak c4 model), so it kept the baseline almost
everywhere. r0 chose global normalisation for that reason.

The chosen input (per-image) differs from the c4 starting point, so c6 retrains its baseline on it: 0.883 on CUDA.
The input format is not where the accuracy is: every option lies within about 2 pp, except 192 px.
"""
