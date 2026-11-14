T = {}
T["crop"] = """
**Result.**
* The first version of the detector kept only the *largest* bright region. When the skull ring was broken into
  pieces, it cropped a fragment instead (an eye socket, part of the brain). That is dangerous: a crop that cuts out
  the tumour silently changes the label.
* The fixed version (union of all sizeable regions, `lib.data.head_mask`) leaves 2 crops below 30 % of the image.
  The smallest cases above are now all genuinely small heads in large frames, correctly cropped. Median crop is
  79 % of the image.

**Decision.** The crop option is reliable enough to enter the c5 comparison.
"""
T["norm"] = """
The statistics barely differ between variants (mean 0.17-0.19, std about 0.18). Each run computes its own from
its train split (`lib.train.prepare_data`), so val and test never leak into them. The alternative, per-image
standardisation, is a c5 option.
"""
T["aug"] = """
**Check by eye.**
* *geometric* and *mild* produce plausible scans (small tilt, shift, zoom, left/right mirror).
* *strong* sometimes pushes part of the head out of frame and visibly adds noise, so it is the upper bound to test.

**Speed.** One pass over train takes 0.1 s without augmentation and 2 s with *strong*, so augmentation costs
about 2 s per epoch.
"""
T["summary"] = """
| Decision | Choice | Why |
|---|---|---|
| Where preprocessing runs | once, cached as uint8 tensors per variant (`cache/`) | deterministic; re-doing it every epoch wastes time |
| Where augmentation runs | on the GPU, per batch, train only | must differ every epoch; val/test stay untouched |
| Geometry candidates | stretch / pad / crop, all cached | chosen in c5 |
| Head crop | union-of-regions detector | the largest-region version cropped away anatomy |
| Normalisation | train mean/std per variant (per-image as a c5 option) | no val/test statistics leak in |
| Class balancing | weighted sampler implemented and verified (0.37/0.25/0.13/0.25 → about 0.25 each) | used only if c6 says so |
| Augmentation presets | none / geometric / intensity / mild / strong | tested in c8 |
"""
