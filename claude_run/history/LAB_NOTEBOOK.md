# Lab notebook: desktop sessions, 2026-11-12 to 2026-11-13

Everything done on the desktop (RTX 2080 Ti), in order, including decisions, findings, problems and tests that
are not recorded anywhere else. Per-round details are in `history/r*/RECORD.md`; every training run is in
`results/` (replaced runs in `results/_superseded/`); notebook/artifact/log snapshots taken before every chain run
are in `history/auto_*`. The rounds r0-r3 are compared in `history_all_rounds.ipynb`.

## 1. Moving the project to the desktop

* Cloned `Nussari/DNN-lok` into `C:\Users\vikto\Documents\DNN-lok`.
* **Python environment**: `uv venv .venv --python 3.12` (system Python is 3.13), then
  `torch==2.6.0+cu124` from the PyTorch cu124 index and numpy, scipy, pandas, scikit-learn, matplotlib, pillow,
  nbformat, nbconvert, jupyter, ipykernel. `torch.cuda.is_available()` = True, RTX 2080 Ti.
* **Dataset download failed** with `HTTP 403 Forbidden` from Mendeley when `get_data.py` used `urllib` (Mendeley
  rejects Python's default User-Agent). Workaround used: download the zip with `curl -L` into
  `data/Brain_Tumor_MRI_Dataset_Final.zip`, then `get_data.py` verified the SHA-256 and extracted it:
  train 4159, val 892, test 889 (as expected). `get_data.py` itself was **not** changed (still fails on a fresh
  machine; fix: send a browser-like User-Agent header).
* **Notebook kernel ran the wrong Python.** The first two chain runs failed with `No module named 'torch'`:
  `nbtools.py` runs `sys.executable -m jupyter nbconvert`, and `jupyter` dispatches to the first
  `jupyter-nbconvert` on PATH, which was the global Python 3.13 install. Reinstalling the venv kernelspec did not
  help. **Fix: activate the venv (put `.venv\Scripts` first on PATH) before `run_chain.py`.** No code change.
* Device check (c6): c4 baseline on CUDA 0.8973 vs MPS 0.8956 (tie), 0.55 vs 3.80 s/epoch.

## 2. Round r1 (old tie rule, 60 epochs)

See `r1_desktop_tie-rule_60ep_partial/RECORD.md`. Bug found and fixed: `make_c9.py` passed `optimizer` to
`dataclasses.replace()` twice (`TypeError: got multiple values for keyword argument 'optimizer'`); fixed by
merging the settings into one dict. Result 0.932.

## 3. Decision: drop the tie rule (user)

User: "the point of this large model is to be as good as possible, no matter the complexity or size."
Analysis behind it: the rule's bar was 2·σ_seed·√(2/3) = 0.0505 with σ_seed = 0.031 measured on the untuned c4
model, while the tuned models' seed sd was 0.004-0.009, so no change in c6-c8 could pass and the baseline was kept
everywhere. New rule: highest mean val balanced accuracy wins. Sweeps take the best mean (trend fits reported, not
gating). Kept: c5 geometry shortcut guard (validity, not simplicity). Added width 64 to c7 (skipped on the Mac).

## 4. Published results on this dataset (research, 2026-11-12)

Dataset: **BDNeuro-MRI** (Hira, Hossain, Bithee, Sara, Hasan, Towsif, Ahmed, Uddin), Mendeley Data
DOI 10.17632/zwr4ntf94j.7 (v7, 5,941 images: 4,160 / 892 / 889). Our copy has 4,159 / 5,940 (one train image
fewer; `get_data.py` expects that). Associated paper SSRN 7370011 (not readable: 403).

| source | model | pretrained | split | metric | value |
|---|---|---|---|---|---|
| dataset authors (Mendeley v7 / GitHub) | CNN | no | official test (889) | acc / F1 | 0.8470 / 0.8457 |
| same | ViT | no | test | acc / F1 | 0.8256 / 0.8240 |
| same | hybrid CNN-ViT | no | test | acc / F1 | 0.8189 / 0.8201 |
| same | ResNet50 | frozen ImageNet | test | acc / F1 | 0.8031 / 0.8022 |
| same | DenseNet121 | frozen ImageNet | test | acc / F1 | 0.8706 / 0.8703 |
| same | ResNet50 | fine-tuned | test | acc / F1 | 0.9708 / 0.9708 |
| same | DenseNet121 | fine-tuned | test | acc / F1 | 0.9831 / 0.9831 |
| arXiv 2611.11793 | ViT-L/16 | trained on Kaggle composite | 1,675 non-duplicate images | acc / bal. acc / F1 | 94.99 / 95.21 / 94.43 % |
| same | EfficientNetV2-M | yes | same | acc / bal. acc / F1 | 92.96 / 93.55 / 91.50 % |
| arXiv 2611.00034 | InceptionResNetV2 + CBAM | ImageNet | **v1** (12,064 images) | acc | 96.73 % (not comparable) |

Caveats: arXiv 2611.11793 matched 4,266 of 5,941 images (71.8 %, pixel-verified) to the Kaggle Nickparvar
composite (Figshare + SARTAJ + Br35H), questioning the single-hospital claim; the authors' deduplication is
within classes only (patient-level leakage possible); their F1 type is not stated and results are single runs.
Context (Nickparvar dataset): from-scratch CNNs about 0.59-0.85, fine-tuned ImageNet models about 0.93-0.98;
leakage audits disagree on how much duplicates inflate scores (arXiv 2610.00421: under 2 pp; SSRN 7493561:
about 14 pp on unmatched images).
Sources: https://data.mendeley.com/datasets/zwr4ntf94j/7 ,
https://github.com/irfanulkabirhira/A-Bangladeshi-Clinical-Brain-Tumor-MRI-Dataset-for-Four-Class-Deep-Learning-Classification ,
https://arxiv.org/html/2611.11793 , https://arxiv.org/html/2611.00034 , https://arxiv.org/html/2610.00421v1 ,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7493561 . (Gathered by a research subagent; not every source
was opened directly.)

**Is the comparison fair?** (question from the user) The ROC/threshold does not affect our accuracy figures (all
argmax); the published numbers are argmax too. Unfair to us: their test set keeps near-duplicates of train images
(our clean val removed 60), and pretrained models may have seen the Kaggle copies. Unfair to them: our scores are
on val after selecting among hundreds of configurations. Only the single final test evaluation is comparable.
Metric: for the r2 final model accuracy 0.9663 vs balanced accuracy 0.9678, so accuracy vs balanced accuracy
barely matters here.

## 5. Rounds r2 and r3, and data keeping

* r2 (no tie rule, 60 epochs): final 0.968. See `r2_desktop_no-tie-rule_60ep/RECORD.md`.
* User asked for more epochs with early stopping (early stopping already existed: patience 20 on val balanced
  accuracy, best weights restored). r3 used a 200-epoch cap from c5 on: final 0.952. See
  `r3_desktop_no-tie-rule_200ep/RECORD.md` (includes why early stopping did not prevent the lower score).
* **User chose r2 as the reported pipeline**; r3 kept as a documented experiment. `EPOCH_CAP = 60` in
  `lib/report.py` (200 reproduces r3); r2's results and artifacts were restored into the working folders and the
  notebooks re-rendered from the cache with no retraining (verified: no new or superseded result files).
* **Data loss before history keeping existed.** The r2 rerun overwrote r1 result files with reused run names,
  r1's executed notebooks and logs; r3 overwrote r2's working copies (r2 had been snapshotted first). What survived
  of r1 is in `r1_*`, and its tables were recovered from the session into its RECORD.md. **Protections added:**
  `lib/train.py` moves a replaced run to `results/_superseded/` instead of overwriting; `run_chain.py` snapshots
  artifacts, notebooks and logs to `history/auto_*` before every chain run.
* **Weights policy (user):** delete all model weights except the best performer. Done: only the r2 final model is
  kept (`history/best_model/r2_final_s1738.pt`, also at `checkpoints/c9/final_s1738.pt` for c10). `train.run`
  saves weights only for c9 and the d-notebooks (exception, below).
* Notebook text rewritten for c5-c10 (c6-c10 had never been written), with update notes (not deletions) in
  c2/c4 where the tie rule and epoch cap were defined.

## 6. Open finding: the c10 threshold rule (not yet decided)

The rule fixed in c2 (highest cutoff with tumour sensitivity ≥ 0.98) **raises** the cutoff for the r2 model
(argmax already has sensitivity 0.992): it misses 11 tumours instead of 6 to save 1 false alarm. Kept as fixed;
alternatives (argmax, cost 10:1 cutoff with 3 missed / 4 false alarms) documented in c10. To decide before the
test evaluation.

## 7. Compression task (d1-d6)

* **User decisions:** teacher = r2 final model alone (not an ensemble, not the deep model); size = stored bytes
  (parameters and latency reported alongside); two final models: smallest with mean ≥ 0.95 and smallest with
  mean ≥ 0.85 ("a high performance one ... shouldn't go under 95 and one that's allowed to go down to 85");
  quantization: real int8 plus simulated 4/2-bit.
* Weights of d-runs **are** saved (exception to the weights policy): pruning/quantization start from them and the
  compressed models are the deliverable (a few kB each; `checkpoints/d*`, about 75 MB in total).
* PyTorch docs checked (2.6): FX graph mode quantization is in maintenance mode, PT2E is the prototype successor,
  `x86` is the recommended CPU backend. FX + x86 used.
* **Smoke tests before any real run** (scratch folder, 2-epoch runs; not in `results/`): teacher reproduces
  0.9678; size accounting 1,105,488 bytes vs `torch.save` 1,119,606 (folded BN + pickle overhead); unstructured
  pruning hits exactly 90 %; 4-bit weights verified on a ≤15-level grid; channel slimming to the same width
  reproduces the model exactly (max output difference 0.0) for plain and depthwise-separable models; real int8
  bal. acc 0.7086 vs fp32 0.7153 on a 2-epoch model, 84 kB vs 290 kB file, **int8 slower** (2.27 vs 1.43 ms).
* First teacher check: 74 % of the teacher's weights are in its last conv (64 to 128 channels, 5x5).
* **First d4/d5 runs lacked a control**, which made pruning and 8-bit QAT look like they *improve* accuracy (for
  example 0.964 to 0.970 at 50 % pruning). Fine-tune-only controls (same epochs and lr, no pruning/quantization)
  showed the gain is the extra low-lr fine-tuning (+0.3 to 1.7 pp for every model). Controls added to d4 and d5,
  all conclusions are measured against them.
* Distillation gave no reliable gain (3 of 8 students, under 1 pp; losses up to 2.8 pp); likely because the
  teacher was trained with label smoothing (Müller, Kornblith & Hinton 2019). Not tested: a teacher without label
  smoothing.
* **Final models** (val, optimistic): high-performance = dw-sep 3x3 w16, 4-bit QAT, 9.3 kB, 116x smaller, 0.951
  (borderline: sd 0.011, seed 1738 0.944; safer alternative dw-sep 5x5 w32, 75 % pruned, 4-bit, 18.8 kB, 0.961).
  Tiny = dw-sep 5x5 w4, distilled, 50 % pruned, 4-bit QAT, 1.41 kB, 766x smaller, 0.858 (meningioma recall 0.685).
* Real int8 in PyTorch: accuracy matches simulated 8-bit except on the tiniest models (−2 to −3.4 pp); 2-5x slower
  than fp32 on this CPU; `torch.save` files 21-287 kB (graph overhead).

## 8. Open items

1. Decide the c10 threshold rule, then run the single test evaluation (`RUN_TEST = True`) for the r2 model and
   the two compressed models.
2. Choose between the borderline 9.3 kB and the 18.8 kB high-performance model.
3. Untested ideas from the findings: a final low-lr phase for the teacher recipe (every model gained 0.3-1.7 pp
   from it); a teacher without label smoothing for distillation; AdamW (decoupled weight decay).
4. `get_data.py`: add a User-Agent header so the download works without curl.
