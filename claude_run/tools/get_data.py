"""Download (once), verify and extract the BDNeuro-MRI v7 dataset to <project>/data/bdneuro/{train,val,test}/<class>/.

    python claude_run/tools/get_data.py
"""
import hashlib, urllib.request, zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data"
ZIP_PATH = DATA_DIR / "Brain_Tumor_MRI_Dataset_Final.zip"
ROOT = DATA_DIR / "bdneuro"
URL = ("https://data.mendeley.com/public-files/datasets/zwr4ntf94j/files/"
       "2d24ea9a-23e3-4a95-898d-06f8cef24c29/file_downloaded")
SHA256 = "0d8fdb6a88b2c65019c11abfd20e0693ffb7b2296955598202303840b9be0c24"

DATA_DIR.mkdir(exist_ok=True)
if not ZIP_PATH.exists():
    print("downloading ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
h = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
assert h == SHA256, f"bad download (sha256 {h}), delete {ZIP_PATH} and retry"
if not (ROOT / "train").exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(ROOT)
n = {s: sum(1 for _ in (ROOT / s).rglob("*.jpg")) for s in ["train", "val", "test"]}
print("ok:", n)   # expected train 4159, val 892, test 889
