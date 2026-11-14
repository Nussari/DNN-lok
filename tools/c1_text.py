T = {}
T["dist"] = """
**Finding.** The split is stratified exactly 70/15/15 per class. Imbalance ratio 2.78 (glioma 36.9 %,
no_tumor 13.2 %). Always predicting glioma gives 0.369 accuracy.

**Decisions.**
* **Keep the official split.** Options were keep / re-split by image / re-split by patient. A patient-level
  re-split is impossible (no patient IDs), and an image-level re-split gains nothing and loses comparability.
* Imbalance is handled where it belongs: metric choice in **c2**, loss/sampler experiment in **c6**.
  Val and test are never rebalanced.
"""
T["look"] = """
**Findings.**
* All three views (axial, coronal, sagittal) appear in every class.
* The tumour classes look like T1 contrast-enhanced MRI. **no_tumor is a mix**: T2 (bright CSF), FLAIR,
  non-contrast T1, **CT scans**, odd crops and watermarked images. So the classes differ not only in pathology
  but in *how the images were acquired*.

**Decisions.**
* Horizontal flip is anatomically plausible for all three views (left/right symmetry; sagittal images face both
  ways in the data) → augmentation candidates for **c8**.
* The modality mix in no_tumor is a label-correlated difference that preprocessing cannot remove (it is the
  image content). It is a **shortcut risk** that is measured with a probe in **c5** and reported as a limitation.
"""
T["props"] = """
**Findings.**
* no_tumor: 64 % have a non-standard resolution, 45 % are non-square, 99 % are stored as RGB and 4 % contain
  real colour. Tumour classes: 62-66 % are 512×512, ≥ 93 % square, none contain colour.
* **Metadata-only random forest: 0.549 balanced accuracy on val** (chance 0.25), no_tumor recall 0.71,
  without seeing a single pixel.

**Decisions.**
* **Convert everything to one grayscale channel.** No tumour image contains colour, so nothing is lost, and the
  colour/mode cue disappears. (Grayscale vs 3 channels is still tested in **c5**, as planned.)
* **Resize every image to one fixed square size.** How to get there is a real choice: *stretch* distorts the
  non-square (mostly no_tumor) images, *pad* adds black bars (mostly to no_tumor). Both leave a class cue.
  A third option, *crop to the head then pad*, removes most framing differences. → tested in **c5** with a
  shortcut probe, because val accuracy alone cannot detect a shortcut that is present in val too.
"""
T["pixels"] = """
**Findings.** no_tumor images are brighter on average (0.234 vs 0.14-0.19) and the black-background share
differs by class (30-47 %). Part of this is the T2/FLAIR content, part is framing.

**Decisions.**
* Normalise with the **train** mean/std (computed in c3, never on val/test).
* Two options that could remove part of the brightness/framing cue are tested in **c5**: cropping to the head
  bounding box, and per-image normalisation instead of a global one.
"""
T["dups"] = """
**What the bins show.**
* **r ≥ 0.98:** visually the *same image* (same slice; noise added, re-cropped or re-exported). Clear duplicates.
* **0.95-0.98:** mostly the *same patient and tumour* at a neighbouring slice. This is patient-level leakage.
* below 0.95: increasingly different scans.

no_tumor is over-represented among the duplicates: 22 of 118 val no_tumor images (19 %) have a train copy at
r ≥ 0.98, versus 7 % of val overall. That would inflate no_tumor recall in particular.

**Decision: option C with threshold 0.98.**
* 60 val and 56 test images that are clearly copies of a train image are excluded from the **clean** val/test sets
  (`artifacts/near_duplicates.json`). Model selection uses clean val (832 images, 96 no_tumor).
* The test set will be reported both ways (official and clean).
* Patient-level leakage (0.95-0.98 and below) **cannot** be removed without patient IDs. It inflates absolute
  scores for every model, so it is a stated limitation, not something the comparisons can fix.
"""
T["summary"] = """
| Finding | Decision | Where |
|---|---|---|
| Official split is stratified 70/15/15 | keep it; never rebalance val/test | all |
| Imbalance 2.78x, majority accuracy 0.369 | accuracy is not the metric; loss weighting is an experiment | c2, c6 |
| Size, aspect, colour mode differ by class (metadata alone: 0.549 bal. acc.) | 1-channel grayscale, one fixed square size; geometry chosen with a shortcut probe | c3, c5 |
| no_tumor mixes MRI sequences and CT | cannot be preprocessed away: probe + limitation | c5, report |
| Brightness and background share differ by class | train-only normalisation; test head-crop and per-image normalisation | c3, c5 |
| 116 val/test images are copies of train images (r ≥ 0.98) | excluded from *clean* val/test; select on clean val; report test both ways | c2 onward |
| Neighbouring slices of the same patient across splits | not fixable without IDs: limitation | report |
| All views present; left/right symmetric anatomy | horizontal flip, small rotation/shift/scale as augmentation candidates | c8 |
"""
