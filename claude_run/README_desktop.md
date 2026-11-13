# Running the rest of claude_run on the desktop (2080 Ti)

## 1. Get the code and the data
```
git clone <repo url> lokaverk
cd lokaverk
python claude_run/tools/get_data.py      # downloads + verifies the dataset (about 230 MB) into data/
```
If the download fails with `HTTP 403` (Mendeley rejects Python's default User-Agent), download it with curl and
rerun the script, which then only verifies and extracts:
```
curl -L -o data/Brain_Tumor_MRI_Dataset_Final.zip "https://data.mendeley.com/public-files/datasets/zwr4ntf94j/files/2d24ea9a-23e3-4a95-898d-06f8cef24c29/file_downloaded"
python claude_run/tools/get_data.py
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
**Activate the virtual environment first.** Notebooks are executed with `python -m jupyter nbconvert`, which runs
the first `jupyter-nbconvert` on PATH; without activation that can be a different Python install (symptom:
`No module named 'torch'` in the log). On Windows: `.venv\Scripts\activate`.

From inside `claude_run/`:

```
python tools/run_chain.py               # c6, c7, c8, c9, c10 in order (finished runs are skipped)
python tools/run_chain.py 8 9 10        # a subset
python tools/run_chain.py d1 d2 d3 d4 d5 d6   # the compression notebooks
python tools/make_history.py            # re-render history_all_rounds.ipynb
```

Nothing is overwritten: a training run whose config changed is moved to `results/_superseded/` before it is
retrained, and every chain run first copies the notebooks, artifacts and logs to `history/auto_<time>/`.
`history/LAB_NOTEBOOK.md` records every decision and finding of the desktop sessions; `history/r*/RECORD.md`
describes each round. The epoch cap of c5-c9 is `EPOCH_CAP` in `lib/report.py` (60; 200 reproduces round r3).

Every finished training run is cached in `results/<notebook>/`. If the chain is interrupted, run the same
command again: finished runs load in milliseconds and training resumes at the first missing run. The chain
stops at the first notebook that fails; the log is in `tools/_out/c<N>_log.txt`.

## 4. Send the results back
```
git add -A && git commit -m "c6-c10 on the 2080 Ti" && git push
```
This sends the executed notebooks, `results/`, `artifacts/`, `history/`, the logs and the model weights in
`checkpoints/` (the final model in `checkpoints/c9/`, the compressed models in `checkpoints/d*/`). The dataset
(`data/`) and the preprocessing cache (`cache/`) are not in git. The written conclusions are added afterwards by
re-rendering from the cache.

## Notes
* Runs stay in fp32 (no AMP), so they are comparable with the runs already made on the Mac. c6 was half done
  on the Mac: it now starts with a device check (the c4 baseline retrained on CUDA, compared with the MPS runs).
  c7 and later retrain their own baseline, so their comparisons are within one device.
* The test set is never evaluated. `c10_threshold.ipynb` has `RUN_TEST = False`.
