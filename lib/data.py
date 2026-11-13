"""Data loading, preprocessing caches and GPU-side batching/augmentation.

Preprocessing (deterministic) is done once per variant and cached as uint8 tensors.
Augmentation (random) is done on the fly, on the GPU, for training batches only.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]          # repository root
DATA_ROOT = ROOT / "data" / "bdneuro"
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"

SPLITS = ["train", "val", "test"]
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NO_TUMOR = CLASSES.index("no_tumor")
SEED = 1738


def list_split(split: str) -> list[tuple[Path, int]]:
    return [(p, i) for i, c in enumerate(CLASSES) for p in sorted((DATA_ROOT / split / c).iterdir())]


# ---------------------------------------------------------------- geometry helpers
def head_mask(gray: np.ndarray, thr: int = 20, min_frac: float = 0.005) -> np.ndarray:
    """Boolean mask of the head: union of all bright regions larger than min_frac of the image, holes filled.

    (Taking only the largest region fails when the skull ring is broken into pieces: it then picks e.g. one
    eye socket. The union of all sizeable regions does not.)
    """
    m = ndimage.gaussian_filter(gray.astype(np.float32), 2) > thr
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones_like(gray, dtype=bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= max(min_frac * gray.size, sizes.max() * 0.02)) + 1
    return ndimage.binary_fill_holes(np.isin(lab, keep))


def _bbox(mask: np.ndarray, margin: float = 0.03):
    ys, xs = np.where(mask)
    h, w = mask.shape
    my, mx = int(margin * h), int(margin * w)
    return max(xs.min() - mx, 0), max(ys.min() - my, 0), min(xs.max() + 1 + mx, w), min(ys.max() + 1 + my, h)


def _pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new(img.mode, (s, s), 0)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def preprocess(path: Path, mode: str = "gray", resize: str = "crop", size: int = 128) -> np.ndarray:
    """Load one image -> uint8 array (C, size, size).

    mode:   'gray' (1 channel) or 'rgb' (3 channels, as stored)
    resize: 'stretch' (resize directly, distorts aspect ratio)
            'pad'     (pad to square with black, then resize)
            'crop'    (crop to head bounding box, pad to square, resize)
    """
    with Image.open(path) as im:
        img = im.convert("L" if mode == "gray" else "RGB")
    if resize == "crop":
        x0, y0, x1, y1 = _bbox(head_mask(np.asarray(img.convert("L"))))
        img = img.crop((x0, y0, x1, y1))
    if resize in ("pad", "crop"):
        img = _pad_square(img)
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr[None] if arr.ndim == 2 else np.ascontiguousarray(arr.transpose(2, 0, 1))


def _pre_one(args):
    return preprocess(*args)


def cache_path(split, mode, resize, size) -> Path:
    return CACHE / f"{split}_{mode}_{resize}_{size}.pt"


def build_cache(split: str, mode: str = "gray", resize: str = "crop", size: int = 128, workers: int = 4) -> dict:
    """Preprocess a whole split once and store it as a uint8 tensor (idempotent)."""
    path = cache_path(split, mode, resize, size)
    if path.exists():
        return torch.load(path)
    items = list_split(split)
    with ProcessPoolExecutor(workers) as ex:
        arrs = list(ex.map(_pre_one, [(p, mode, resize, size) for p, _ in items], chunksize=64))
    d = {
        "X": torch.from_numpy(np.stack(arrs)),
        "y": torch.tensor([c for _, c in items]),
        "paths": [p.relative_to(DATA_ROOT).as_posix() for p, _ in items],
    }
    CACHE.mkdir(exist_ok=True)
    torch.save(d, path)
    return d


def mask_out_head(X: torch.Tensor) -> torch.Tensor:
    """Probe input: black out the (dilated) head so only background/borders remain."""
    out = X.clone()
    for i in range(len(X)):
        m = head_mask(X[i].float().mean(0).numpy().astype(np.uint8), thr=15)
        m = ndimage.binary_dilation(m, iterations=max(2, X.shape[-1] // 32))
        out[i][:, torch.from_numpy(m)] = 0
    return out


# ---------------------------------------------------------------- split hygiene
def excluded_paths() -> set[str]:
    """val/test images flagged as near-duplicates of a train image (decided in c1)."""
    f = ARTIFACTS / "near_duplicates.json"
    if not f.exists():
        return set()
    return set(json.loads(f.read_text())["exclude"])


def load_split(split, mode="gray", resize="crop", size=128, clean=True, probe=False) -> dict:
    d = dict(build_cache(split, mode, resize, size))
    if clean and split != "train":
        ex = excluded_paths()
        keep = torch.tensor([p not in ex for p in d["paths"]])
        d = {"X": d["X"][keep], "y": d["y"][keep], "paths": [p for p, k in zip(d["paths"], keep) if k]}
    if probe:
        d["X"] = mask_out_head(d["X"])
    return d


def stratified_subset(y: torch.Tensor, frac: float, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in y.unique():
        ci = torch.where(y == c)[0]
        idx.append(ci[torch.randperm(len(ci), generator=g)[: max(1, round(frac * len(ci)))]])
    return torch.cat(idx).sort().values


# ---------------------------------------------------------------- augmentation (GPU)
AUG_PRESETS = {
    "none": {},
    "geometric": dict(rot=10, trans=0.05, scale=(0.95, 1.05), hflip=0.5),
    "intensity": dict(brightness=0.05, contrast=0.1, gamma=0.15, noise=0.0),
    "mild": dict(rot=10, trans=0.05, scale=(0.95, 1.05), hflip=0.5, brightness=0.05, contrast=0.1, gamma=0.15),
    "strong": dict(rot=20, trans=0.10, scale=(0.85, 1.15), hflip=0.5, brightness=0.10, contrast=0.2,
                   gamma=0.3, noise=0.02),
}


def augment(x: torch.Tensor, p: dict, gen: torch.Generator) -> torch.Tensor:
    """x: float (B,C,H,W) in [0,1]. Random params drawn on CPU (reproducible), applied on device."""
    if not p:
        return x
    B, dev = x.shape[0], x.device

    def U(lo, hi):
        return (torch.rand(B, generator=gen) * (hi - lo) + lo).to(dev)

    if any(k in p for k in ("rot", "trans", "scale", "hflip")):
        th = torch.deg2rad(U(-p.get("rot", 0), p.get("rot", 0)))
        s = U(*p.get("scale", (1, 1)))
        f = torch.where(torch.rand(B, generator=gen) < p.get("hflip", 0), -1.0, 1.0).to(dev)
        t = p.get("trans", 0) * 2
        tx, ty = U(-t, t), U(-t, t)
        cos, sin = torch.cos(th) / s, torch.sin(th) / s
        A = torch.stack([torch.stack([f * cos, -sin, tx], 1), torch.stack([f * sin, cos, ty], 1)], 1)
        grid = F.affine_grid(A, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if "contrast" in p or "brightness" in p:
        c = U(1 - p.get("contrast", 0), 1 + p.get("contrast", 0)).view(B, 1, 1, 1)
        b = U(-p.get("brightness", 0), p.get("brightness", 0)).view(B, 1, 1, 1)
        x = x * c + b
    if p.get("gamma"):
        g = torch.exp(U(-p["gamma"], p["gamma"])).view(B, 1, 1, 1)
        x = x.clamp(0, 1) ** g
    if p.get("noise"):
        x = x + p["noise"] * torch.randn(x.shape, generator=gen).to(dev)
    return x.clamp(0, 1)


# ---------------------------------------------------------------- batching (GPU)
class GPUSplit:
    """A whole split kept on the device as uint8; batches are normalised on the fly."""

    def __init__(self, d: dict, device, mean: float, std: float, per_image: bool = False):
        self.per_image = per_image
        self.X = d["X"].contiguous().to(device)
        self.y = d["y"].to(device)
        self.paths = d["paths"]
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.y)

    def batches(self, bs, shuffle=False, gen=None, aug=None, weights=None, idx=None):
        n = len(self) if idx is None else len(idx)
        base = torch.arange(len(self)) if idx is None else idx
        if weights is not None:  # weighted sampling with replacement, epoch size = n
            order = base[torch.multinomial(weights[base], n, replacement=True, generator=gen)]
        elif shuffle:
            order = base[torch.randperm(n, generator=gen)]
        else:
            order = base
        for i in range(0, n, bs):
            b = order[i: i + bs].to(self.X.device)
            x = self.X[b].float() / 255.0
            if aug:
                x = augment(x, aug, gen)
            if self.per_image:  # standardise every image by its own statistics
                m = x.mean((1, 2, 3), keepdim=True)
                s = x.std((1, 2, 3), keepdim=True) + 1e-6
                yield (x - m) / s, self.y[b]
            else:
                yield (x - self.mean) / self.std, self.y[b]
