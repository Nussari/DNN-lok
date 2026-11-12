T = {}
T["geometry"] = """
**Val score.** All three geometries tie: stretch 0.882, pad 0.903, crop 0.896. The largest gap (2.1 pp) is well
below the 5 pp needed.

**Probe (head blacked out).** No geometry is close to the 0.25 chance level:
* stretch 0.797, pad 0.787, crop 0.738;
* stretch is significantly above crop, and pad is borderline (4.9 pp vs 5.0 needed).

The probe pictures show where that signal comes from. Masking sets the head to exactly 0, but the background
around it is not exactly 0: there is faint sensor noise, JPEG blocks and scanner borders. So the model still sees
the **outline and size of the head** and the **texture of the background**. Both differ by class, because the
no_tumor images come from other sources. Even crop leaves 0.74. Cropping removes most of the border and the
framing, but not the outline.

**Decision: crop to head.** By the rule set before the run: crop ties for the best val score and has the lowest
probe score. In effect, it gives the same accuracy with the least room for shortcuts.

**Caveat carried forward.** A probe of 0.74 means that shortcut signal exists in the inputs. It does *not* say
how much the real model uses it, because that model also sees the head, which is the stronger signal. One way to
remove more of it is to zero everything below the head threshold, so the background becomes truly black. That
was not tested here; it is noted as future work.
"""
T["channels"] = """
**Result.** Grayscale 0.896 vs RGB 0.877: a tie, with RGB slightly lower, 3x noisier across seeds (sd 0.026 vs
0.009) and 15 % slower. RGB does have a higher no_tumor recall (0.955 vs 0.931). That fits c1: about 4 % of the
no_tumor images are in real colour, which is a shortcut only that class has.

**Decision: grayscale.** It ties, it is the simpler input (MRI is single-channel by nature), and it removes
one shortcut instead of offering it to the model.
"""
T["norm"] = """
**Result.** Per-image standardisation 0.901 vs global train mean/std 0.896: a tie (+0.5 pp, needed 5 pp).

**Decision: global (the default).** Per-image normalisation was the candidate fix for scanners with different
brightness. It made no measurable difference here, so it does not earn the extra complexity.
"""
T["res"] = """
**Result.**
* 64 px: 0.886 (tie), 3x faster per epoch.
* 128 px: 0.896.
* 192 px: 0.846, worse by 5.0 pp (just below the 5.05 pp threshold, so formally a tie) and 2.4x slower. Its runs
  also stopped earlier (42 epochs on average).

**Why more pixels did not help this model.** The model has only 4 pooling stages, 3×3 kernels and global average
pooling. At 192 px each final feature sees a smaller fraction of the brain, and GAP averages more background
positions into the decision. A model needs depth and receptive field to match the resolution, which c7 tests.
So this result is about *this small model*, not about resolution in general. That matters for the desktop
scale-up: higher resolution only makes sense together with a deeper network, and even then is not supported by
this evidence.

**Curve test** (rule for ordered settings, introduced in c6 after the pairwise rule proved too blunt for sweeps):
across all 9 runs the curve has a significant peak (quadratic term p = 0.018) at 128 px.

**Decision: 128 px.** It is the peak of the curve and the default. 64 px is cheaper and only 1 pp worse, but the
deeper models in c7 need room for more downsampling stages.
"""
T["summary"] = """
| Question | Choice | Result (val bal. acc, 3 seeds) | Reason |
|---|---|---|---|
| Geometry | crop to head | 0.896 (stretch 0.882, pad 0.903: tie) | ties on val, least shortcut signal (probe 0.74 vs 0.79-0.80) |
| Channels | grayscale | 0.896 vs 0.877 RGB (tie) | simpler; colour is a no_tumor-only shortcut |
| Normalisation | global mean/std | 0.896 vs 0.901 per image (tie) | default; no measurable gain |
| Resolution | 128 px | 64: 0.886, 192: 0.846 | 192 hurts this small model; 128 leaves room for depth |

The input is the same as the c4 starting point, so c6 reuses the c4 seed runs as its baseline (0.896).
Every choice was a tie except the probe comparison. With this much seed noise, the input format is not where
the accuracy is.
"""
