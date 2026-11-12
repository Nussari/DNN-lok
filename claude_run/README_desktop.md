# Running the rest of claude_run on the desktop (2080 Ti)

## 1. Copy
Copy two folders from the Mac. Keep them side by side, as on the Mac:

```
lokaverk/
  claude_run/      (everything: lib, tools, cache, results, checkpoints, artifacts, notebooks)
  data/bdneuro/    (train/val/test images)
```

`cache/` (about 0.8 GB) can be left out, because it is rebuilt from `data/` on first use. `results/` and
`artifacts/` must come along: they hold every finished run and the decisions so far.

## 2. Python environment
Python 3.10+ with a CUDA build of PyTorch:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy pandas scikit-learn matplotlib pillow nbformat nbconvert jupyter ipykernel
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True NVIDIA GeForce RTX 2080 Ti`. `lib/train.py` picks CUDA automatically.

## 3. Run
From inside `claude_run/`:

```
python tools/run_chain.py          # c6, c7, c8, c9, c10 in order
python tools/run_chain.py 8 9 10   # a subset
```

Every finished training run is cached in `results/<notebook>/`. If the chain is interrupted, run the same
command again: finished runs load in milliseconds and training resumes at the first missing run. The chain
stops at the first notebook that fails; the log is in `tools/_out/c<N>_log.txt`.

## 4. Copy back
When it finishes, copy `claude_run/` back over the Mac copy, or at least `results/`, `artifacts/`,
`checkpoints/` and the `c*.ipynb` notebooks. The written conclusions get added afterwards, by re-rendering
from the cache.

## Notes
* Runs stay in fp32 (no AMP), so they are comparable with the runs already made on the Mac. Each notebook
  compares options that were all trained on the same device, because c7 and later retrain their own baseline.
* The test set is never evaluated. `c10_threshold.ipynb` has `RUN_TEST = False`.
