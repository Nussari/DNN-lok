T = {
    "overlap": """
* **The models share errors far more often than chance.** The large and 18.8 kB models both miss 4 tumours where
  0.1 would be expected if they erred independently, and share 17 errors of any kind where 1.6 would be expected.
  Part of the difficulty therefore lies in the images themselves.
* **Still, most errors belong to one model.** Of the 19 tumours missed by at least one model, 11 are missed by
  only one and 2 by all four; of the 172 images that any model gets wrong, 118 are wrong for one model only. The
  overlap of two models' errors (shared / either) is 0.17 to 0.35.
* The two larger compressed models overlap most with each other (0.35), the 1.4 kB model least with the others.
  Because the errors differ, combining the models (for example averaging their probabilities) could remove some
  of them; this was not tested.
""",
    "missed": """
**What the missed tumours have in common** (judged by eye, not measured):

* **They come from a mix of scan types and equipment.** #1 and #7 look like CT rather than MRI, #5 and #18 like
  FLAIR or T2-weighted MRI, #2 like T1 without contrast agent. The correctly detected tumours are almost all of
  one kind (T1-weighted with contrast agent).
* **Their contrast varies widely** compared with the rest of the data set: some are very dark (#14), some very
  bright (#11), some flat and grey (#1, #7).
* **Framing and zoom are irregular.** Some are cut very close or show an unusual part of the head (#3, #4: skull
  base and cerebellum), others show a small head in a large frame (#16), and some are side views (#13, #14).
* **The correctly detected tumours look alike**: very similar contrast, layout and position of the head in the
  frame.
""",
    "false_alarms": """
**What the false alarms have in common** (judged by eye, not measured):

* **Contrast differs from the typical scan**: T2-weighted images with bright fluid (#7, #9, #11), a CT-like scan
  (#8), a grey instead of black background (#5), heavy noise (#6).
* **Cropping goes wrong or the head sits oddly in the frame.** #12 has white margins around the scan, so the head
  crop kept the whole image and the model sees white bars next to the head; #1 and #3 are cut off at the edge.
* **Irregular positioning**: tilted or low slices (#7, #9, #13) and side or front views (#1 to #4).
""",
    "summary": """
## Summary

Errors are spread across the models rather than concentrated on a few impossible images, but they are not
random either: **both kinds of error happen on images that look different from the bulk of the data set**, in scan
type, contrast, framing or borders. The correctly classified images are much more uniform. This points to a
model that has learnt the typical appearance of each class in this data set, and generalises poorly to scans
that depart from it.

It matches the shortcut risk found in c1: the tumour classes are almost all T1-weighted MRI with contrast agent,
while no_tumor mixes T2, FLAIR, non-contrast T1, CT and odd crops. A model can therefore partly learn "atypical
scan means no tumour". The errors look like exactly that: tumours on CT, FLAIR or T2 scans are called no_tumor,
and healthy scans that look like the typical tumour images are called tumours.

**Stronger augmentation** (more contrast, zoom and rotation variation) was tried in c8 and dropped, because it
did not raise the val score. Val has the same mix as training, dominated by typical scans, so that result does
not cover scans like the ones above. A model that has to handle images with this much variation would most likely
need it, and data from more sources.

Next steps this suggests (not tested): labelling the scan type per image, to measure how it is distributed over
the classes and over the errors; a head crop that handles bright borders; stronger augmentation evaluated on
atypical scans specifically.
""",
}
