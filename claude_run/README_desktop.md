# Running the rest of claude_run on the desktop (2080 Ti)

## 1. Get the code and the data
```
git clone <repo url> lokaverk
cd lokaverk
python claude_run/tools/get_data.py      # downloads + verifies the dataset (about 230 MB) into data/
```
The repo carries `results/` and `artifacts/` (every finished run and every decision so far). `cache/` is rebuilt
from `data/` on first use, and `checkpoints/` (model weights) is not in git.

## 2. Python environment
Python 3.10+ with a CUDA build of PyTorch:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy pandas scikit-learn matplotlib pillow nbformat nbconvert jupyter ipykernel
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True NVIDIA GeForce RTX 2080 Ti`. `lib/train.py` picks CUDA automatically.

## 3. Run
From inside `lokaverk/claude_run/`:

```
python tools/run_chain.py          # c6, c7, c8, c9, c10 in order (finished runs are skipped)
python tools/run_chain.py 8 9 10   # a subset
```

Every finished training run is cached in `results/<notebook>/`. If the chain is interrupted, run the same
command again: finished runs load in milliseconds and training resumes at the first missing run. The chain
stops at the first notebook that fails; the log is in `tools/_out/c<N>_log.txt`.

## 4. Send the results back
```
git add -A && git commit -m "c6-c10 on the 2080 Ti" && git push
```
This sends the executed notebooks, `results/` and `artifacts/`. The final model's weights are in
`checkpoints/c9/` and stay on the desktop (not in git). The written conclusions are added afterwards by
re-rendering from the cache.

## Notes
* Runs stay in fp32 (no AMP), so they are comparable with the runs already made on the Mac. c6 was half done
  on the Mac: it now starts with a device check (the c4 baseline retrained on CUDA, compared with the MPS runs).
  c7 and later retrain their own baseline, so their comparisons are within one device.
* The test set is never evaluated. `c10_threshold.ipynb` has `RUN_TEST = False`.
