T = {
    'all': """
**Result.** 16 architectures from 1,080 kB (the teacher's own layout) down to 2.9 kB, all trained on the labels
with the teacher's recipe.

* **Accuracy falls slowly at first, then fast.** Down to about 200 kB nothing is lost (depthwise-separable 5x5,
  width 32: 0.964, teacher layout 0.969). From there each halving of the width costs more: 0.948 at 52 kB,
  0.916 at 20 kB, 0.862 at 7 kB, 0.795 at 2.9 kB.
* **Depthwise-separable blocks win at every size.** At about the same size they beat a plain conv clearly:
  dw-sep 5x5 w32 0.964 at 207 kB vs plain 5x5 w8 0.953 at 271 kB; dw-sep 3x3 w16 0.948 at 52 kB vs plain 3x3 w4
  0.906 at 25 kB and plain 5x5 w4 0.929 at 68 kB. Splitting a k x k conv into a per-channel k x k filter and a 1x1
  channel mixer costs k²·c + c·c' weights instead of k²·c·c', and for these images that loses little.
* **Kernel size** matters less than the block type. Within depthwise-separable, 3x3 and 5x5 trade places
  (3x3 better at width 16, 5x5 better at width 4 and 8). In depthwise-separable blocks a 5x5 kernel is cheap
  (the k² term is per channel, not per channel pair), so the c7 finding (5x5 better) carries over at small widths.
* **Seed spread grows as models shrink** (sd up to 0.05 for plain 3x3 w2): tiny models are less stable.
* The no_tumor class suffers first (recall 0.85 at 2.9 kB vs 0.98 for the teacher): the smallest models confuse it
  with the tumour classes more often.
""",
    'frontier': """
**Frontier.** Every model on it below the teacher's size is depthwise-separable (5x5 or 3x3, alternating).
Plain convs only appear at the top (the teacher's own layout, 0.969 at 1,080 kB), which is 5x larger than the
depthwise-separable w32 student for 0.4 pp.

**Floors, labels only:** the smallest model at or above 0.85 is depthwise-separable 5x5 w4 (7.1 kB, 0.862,
152x smaller than the teacher). Nothing between the teacher layout and dw-sep 5x5 w32 (207 kB, 0.964) is above
0.95; dw-sep 3x3 w16 misses it at 0.948 (52 kB). These two (w32 and w4) are the reference students for d3.

**Decision.** The frontier architectures go to d3, to be trained again with the teacher.
""",
}
