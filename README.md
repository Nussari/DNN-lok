# Brain-tumour MRI classification with a CNN trained from scratch

Final project for a deep neural networks course. A convolutional network is trained **from scratch** (no
pretrained weights) to classify brain MRI slices into four classes: **glioma, meningioma, pituitary tumour and no
tumour**. The model is then compressed with distillation, pruning and quantization into models of a few kilobytes.

* **Dataset:** BDNeuro-MRI v7 (Mendeley Data, DOI 10.17632/zwr4ntf94j.7), official split: 4,160 train /
  892 val / 889 test images. Licence: **CC BY 4.0**, as stated on the Mendeley Data page (the README inside the
  downloaded archive still has a placeholder licence line; the Mendeley page is the published one). The data is
  not in this repository; `tools/get_data.py` downloads it.
* **Citation:** Hira, M. I. K., Hossain, M. S., Bithee, M. A., Sara, U. S., Hasan, M. M., Towsif, A. A. &
  Ahmed, M. K. (2026). *BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning
  Classification* (Version 7). Mendeley Data. https://doi.org/10.17632/zwr4ntf94j.7
* **Metric:** balanced accuracy (the mean of the four per-class recalls). Every choice was made on **clean val**
  (832 images: val without near-copies of training images). The test set was used once, at the end.
* **Seeds:** every option was trained with seeds 1738, 1739 and 1740 (the deep candidates in c8 and c9 with the
  first two). Seed 1738 was fixed in advance as the model that is kept.

## Results

Test set, evaluated once after every decision was final (`notebooks/test_evaluation.ipynb`):

| model | stored size | parameters | multiply-adds | balanced accuracy, clean test (833) [95 % CI] | official test (889) | val |
|---|---|---|---|---|---|---|
| **large** | 1,080 kB | 276,628 | 72 M | **0.959** [0.942, 0.973] | 0.961 | 0.968 |
| compressed | 18.8 kB | 53,796 | 28 M | **0.950** [0.934, 0.966] | 0.954 | 0.961 |
| compressed | 9.3 kB | 13,588 | 5.8 M | **0.937** [0.918, 0.953] | 0.939 | 0.944 |
| compressed | 1.4 kB | 1,912 | 2.4 M | **0.807** [0.779, 0.835] | 0.811 | 0.812 |

* The large model: 5 conv layers (5x5 kernels, BatchNorm, average pooling, widths 16 to 128). It takes grayscale
  input cropped to the head at 128 px and is trained with Adam (lr 0.03), label smoothing, mild augmentation and
  early stopping.
* Every model detects 98.8 to 98.9 % of tumours on clean test (8 or 9 missed of 728).
* For comparison, the dataset authors report test accuracy 0.847 for a CNN trained from scratch and 0.971 / 0.983
  for fine-tuned ImageNet ResNet50 / DenseNet121. The large model reaches accuracy 0.962 on the same split
  without pretraining.

All scores use the tumour decision rule from c10 (argmax, except a cutoff on P(tumour) for the 1.4 kB model that
raises its tumour sensitivity to 0.98 on val). The large model loses 0.9 pp from val to test and the compressed
models 0.5 to 1.1 pp. Per-class results and the weight files of each model are in **[MODELS.md](MODELS.md)**.

## Repository layout

```
README.md          this file
MODELS.md          the four final models: test and val results, sizes, decision rule, weight files
requirements.txt
early/             the first notebooks: project plan (initial), first look at the data (s1), choice of metric (s2)
notebooks/         the pipeline, one notebook per step
                     c1-c10            the classifier
                     d1-d6             compression
                     test_evaluation   the single test evaluation of the four final models
                     error_analysis    which images the four models get wrong, and whether they share errors
                     history_all_rounds  every configuration of every round, side by side
lib/               code used by the notebooks: data, models, training, evaluation, compression
tools/             notebook generators (make_<step>.py = code, <step>_text.py = written conclusions, nbtools.py),
                   run_chain.py (builds and runs notebooks), get_data.py (downloads the dataset)
artifacts/         the decision of each step as JSON; later steps read these
results/           every training run of the reported pipeline: <step>/<run>.json (config, per-epoch history,
                   val metrics) and <run>_val.npz (val probabilities)
checkpoints/       weights of the large model (c9/) and the compression runs (d2-d5/)
history/           README.md (what was tried, what was decided and why) and the runs of the other rounds
data/, cache/      not tracked: the dataset (tools/get_data.py) and preprocessed tensors (built on first use)
```

## How to explore

1. **The plan and first look:** `early/initial.ipynb`, `early/s1_data-explore.ipynb`, `early/s2_eval_setup.ipynb`.
   These came first; c1 and c2 redo the same questions in full.
2. **The pipeline, in order.** Each notebook lists its options, tests them, and ends with a written decision and
   its reason.

| notebook | question | outcome |
|---|---|---|
| c1_data_exploration | what is in the data; shortcuts; duplicates | no_tumor images differ in size, colour and source (shortcut risk); near-duplicates removed: clean val 832, clean test 833 |
| c2_eval_setup | which metric; how much noise | balanced accuracy with bootstrap / Wilson intervals; tumour sensitivity rule |
| c3_preprocessing | input pipeline | head crop, GPU-resident data, on-GPU augmentation |
| c4_initial_model | does training work; early stopping; seed noise | overfit test passes; early stopping on val balanced accuracy, patience 20, cap 60; baseline 0.896 |
| c5_input_experiments | geometry, channels, normalisation, resolution | crop to head (least signal outside the head), grayscale, per-image normalisation, 128 px |
| c6_optimisation | learning rate, optimiser, batch size, schedule, loss | Adam lr 0.03, batch 64, no schedule, label smoothing: 0.937 |
| c7_architecture | kernel, downsampling, depth / BatchNorm / skip connections, width | 5x5 kernels, average pooling; plain 5-layer net 0.959, residual 17-layer net 0.962 |
| c8_regularisation | augmentation, weight decay, dropout, small vs deep | mild augmentation, no weight decay, no dropout; the small net wins: 0.974 |
| c9_final_model | learning-rate re-check, final model | lr 0.03 kept; seed 1738: 0.968 [0.955, 0.979] on val |
| c10_threshold | ROC and the tumour / no-tumour decision | argmax; a cutoff only when sensitivity on val falls below 0.98 |
| d1_setup | compression rules and size accounting | teacher = the c9 model; targets 0.95 and 0.85; size = stored bytes |
| d2_students | small architectures trained on the labels | depthwise-separable blocks win at every size |
| d3_distillation | does the teacher help the students | no reliable gain |
| d4_pruning | unstructured and structured pruning | 50 % pruning free, 75 % costs 0.5 to 1 pp for the larger students (against fine-tune-only controls) |
| d5_quantization | 8/4/2-bit, post-training vs quantization-aware, real int8 | 4-bit quantization-aware training is the best trade-off; 2-bit collapses |
| d6_final | size / accuracy frontier and the final picks | smallest models with val ≥ 0.95 (9.3 kB) and ≥ 0.85 (1.4 kB); a third with a 1 pp margin (18.8 kB, ≥ 0.96) because the 0.95 pick was borderline |
| test_evaluation | the four final models on the test set | see Results |
| error_analysis | do the models fail on the same images; what do the failures look like | errors are shared far more than chance, but 11 of 19 missed tumours are missed by one model only; both missed tumours and false alarms are scans that look unlike the bulk of the data (scan type, contrast, framing, borders) |

3. **`history/README.md`** explains the four rounds the pipeline went through (selection rule, epoch cap), the
   decisions that were revised and why, and what did not work. `notebooks/history_all_rounds.ipynb` shows every
   configuration of every round.
4. **Raw numbers:** any run in `results/<step>/<run>.json`; the chosen configuration of each step in `artifacts/`.

## How to replicate

### 1. Set up
```
git clone <repo url> && cd <repo>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # for CUDA, install torch from the PyTorch index first
python tools/get_data.py                                  # downloads and SHA-256-checks the dataset into data/
```
`lib/train.py` uses CUDA if available, then Apple MPS, then CPU. Keep the virtual environment activated when running
the tools below: the notebooks are executed through `jupyter`, which must come from the same environment.

### 2. Load a model and check its score (about a minute)
```python
# from the repository root
import json, torch
from lib.train import Config, build_model, prepare_data, predict, DEVICE
from lib.evaluation import evaluate

rec = json.load(open("results/c9/final_s1738.json"))
cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec["config"].items()})
train, val = prepare_data(cfg)                       # clean val, normalised as in training
net = build_model(cfg, train.X.shape[1]).to(DEVICE)
net.load_state_dict(torch.load("checkpoints/c9/final_s1738.pt", map_location=DEVICE))
print(evaluate(val.y.cpu().numpy(), predict(net, val))["bal_acc"])   # 0.9678
```
The same code loads the compressed models from the files listed in MODELS.md.

### 3. Rebuild the notebooks from the stored runs (minutes)
```
python tools/run_chain.py 1 2 3 4 5 6 7 8 9 d1 d2 d3 d4 d5 d6 10 test errors
python tools/make_history.py
```
Every training run is stored by its configuration in `results/`, so this regenerates and re-executes every
notebook without retraining. Set `CACHED_ONLY=1` to make a missing run an error instead of training it. c10 runs
after d6 because it applies the decision rule to the compressed models picked there. Logs go to `tools/_out/`.

### 4. Retrain everything from scratch (about 6 hours on an RTX 2080 Ti)
Move the stored runs, weights and decisions aside, then run the chain:
```
mkdir ../stored_runs && mv results checkpoints artifacts ../stored_runs/ && mkdir results checkpoints artifacts
python tools/run_chain.py 1 2 3 4 5 6 7 8 9 d1 d2 d3 d4 d5 d6 10 test errors
```
Each step reads the decisions of the previous steps from `artifacts/`, so the steps must run in order. The
compressed models evaluated by c10 and `test` are the ones d6 picks (`artifacts/compress_final.json`). Expect
roughly 3.5 hours for c1-c10 and 2.5 hours for d1-d6 on an RTX 2080 Ti; an Apple M1 is about 7 times slower per
epoch.

The numbers will be close to, not identical with, the ones reported: GPU training is not bit-for-bit
reproducible (retraining the same configuration moved its mean by about 0.5 pp, see c9), and several steps choose
between options that lie within seed noise of each other, so a rerun can take a slightly different path
(`history/README.md`, round r3).
