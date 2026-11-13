"""Download (once), verify and extract the BDNeuro-MRI v7 dataset to <project>/data/bdneuro/{train,val,test}/<class>/.

    python tools/get_data.py
"""
import hashlib, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repository root
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "Brain_Tumor_MRI_Dataset_Final.zip"
DATASET = DATA_DIR / "bdneuro"
URL = ("https://data.mendeley.com/public-files/datasets/zwr4ntf94j/files/"
       "2d24ea9a-23e3-4a95-898d-06f8cef24c29/file_downloaded")
SHA256 = "0d8fdb6a88b2c65019c11abfd20e0693ffb7b2296955598202303840b9be0c24"

DATA_DIR.mkdir(exist_ok=True)
if not ZIP_PATH.exists():
    print("downloading ...")
    # Mendeley answers 403 to Python's default User-Agent, so present a browser-like one
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
    with urllib.request.urlopen(req) as r, open(ZIP_PATH, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
h = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
assert h == SHA256, f"bad download (sha256 {h}), delete {ZIP_PATH} and retry"
if not (DATASET / "train").exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATASET)
n = {s: sum(1 for f in (DATASET / s).rglob("*") if f.is_file()) for s in ["train", "val", "test"]}
print("ok:", n)   # expected train 4160, val 892, test 889
