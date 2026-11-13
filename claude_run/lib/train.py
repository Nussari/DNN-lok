"""Training loop with early stopping + checkpointing, cached per config.

run(cfg) trains once and stores results/<exp>/<name>.json (+ val probabilities, + checkpoint).
Calling it again with the identical config just loads the stored result, so notebooks can be
re-executed cheaply.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from .data import CLASSES, RUN_DIR, SEED, AUG_PRESETS, GPUSplit, load_split, stratified_subset
from .evaluation import evaluate
from .models import CNN, n_conv_layers, n_params

RESULTS = RUN_DIR / "results"
CKPT = RUN_DIR / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Config:
    exp: str
    name: str
    # data
    mode: str = "gray"
    resize: str = "crop"
    size: int = 128
    probe: bool = False            # mask out the head (shortcut probe)
    normalize: str = "global"      # global (train mean/std) | per_image
    train_frac: float = 1.0        # stratified subset of train
    train_n: int = 0               # >0: use only the first n (stratified) train images (sanity check)
    # model
    widths: tuple = (16, 32, 64, 128)
    depths: tuple = (1, 1, 1, 1)
    kernel: int = 3
    stem_kernel: int = 3
    block: str = "single"
    norm: bool = True
    residual: bool = False
    downsample: str = "maxpool"
    dropout: float = 0.0
    # optimisation
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    schedule: str = "none"         # none | cosine | plateau | step
    loss: str = "ce"               # ce | weighted_ce | label_smoothing | focal
    sampler: str = "none"          # none | weighted
    epochs: int = 40
    patience: int = 10
    monitor: str = "val_loss"      # val_loss | val_bal_acc
    # regularisation
    aug: str = "none"
    seed: int = SEED
    extra: dict = field(default_factory=dict)

    def key(self):
        return json.loads(json.dumps(asdict(self)))


def _same_config(stored: dict, cfg: Config) -> bool:
    """Equal, ignoring fields added to Config later, as long as the new run uses their defaults."""
    now, default = cfg.key(), Config(exp="", name="").key()
    return all(stored.get(k, default[k]) == v for k, v in now.items())


def _paths(cfg):
    d = RESULTS / cfg.exp
    return d / f"{cfg.name}.json", d / f"{cfg.name}_val.npz", CKPT / cfg.exp / f"{cfg.name}.pt"


def _supersede(cfg):
    """Never overwrite a finished run's record: move its results to results/_superseded/<exp>/<name>__<old mtime>.*,
    so every run ever made (config, history, scores, val probabilities) stays on disk. Weights are not kept."""
    jpath, npath, cpath = _paths(cfg)
    if not jpath.exists():
        return
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(jpath.stat().st_mtime))
    epath = npath.with_name(f"{cfg.name}_epochs.npz")
    for p in (jpath, npath, epath):
        if p.exists():
            suffix = p.name[len(cfg.name):]   # ".json", "_val.npz" or "_epochs.npz"
            dest = RESULTS / "_superseded" / cfg.exp / f"{cfg.name}__{stamp}{suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            p.rename(dest)
    cpath.unlink(missing_ok=True)   # old weights of the replaced run are not kept


def build_model(cfg, in_ch):
    return CNN(in_ch, len(CLASSES), cfg.widths, cfg.depths, cfg.kernel, cfg.stem_kernel, cfg.block,
               cfg.norm, cfg.residual, cfg.downsample, cfg.dropout)


def _loss_fn(cfg, class_w):
    if cfg.loss == "weighted_ce":
        return lambda z, y: F.cross_entropy(z, y, weight=class_w)
    if cfg.loss == "label_smoothing":
        return lambda z, y: F.cross_entropy(z, y, label_smoothing=0.1)
    if cfg.loss == "focal":
        def focal(z, y, g=2.0):
            logp = F.log_softmax(z, 1).gather(1, y[:, None]).squeeze(1)
            return (-(1 - logp.exp()) ** g * logp).mean()
        return focal
    return F.cross_entropy


def _optimizer(cfg, model):
    p = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(p, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.Adam(p, lr=cfg.lr, weight_decay=cfg.weight_decay)


@torch.no_grad()
def predict(model, split: GPUSplit, bs=256):
    model.eval()
    out = [F.softmax(model(x), 1) for x, _ in split.batches(bs)]
    return torch.cat(out).float().cpu().numpy()


def prepare_data(cfg):
    tr = load_split("train", cfg.mode, cfg.resize, cfg.size, probe=cfg.probe)
    va = load_split("val", cfg.mode, cfg.resize, cfg.size, probe=cfg.probe)
    Xf = tr["X"].float() / 255
    mean, std = Xf.mean().item(), Xf.std().item()
    pi = cfg.normalize == "per_image"
    return GPUSplit(tr, DEVICE, mean, std, pi), GPUSplit(va, DEVICE, mean, std, pi)


def run(cfg: Config, force=False, verbose=False) -> dict:
    jpath, npath, cpath = _paths(cfg)
    if jpath.exists() and not force:
        res = json.loads(jpath.read_text())
        if _same_config(res["config"], cfg):
            return res
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    train, val = prepare_data(cfg)
    idx = None
    if cfg.train_n:
        idx = stratified_subset(train.y.cpu(), cfg.train_n / len(train), cfg.seed)
    elif cfg.train_frac < 1:
        idx = stratified_subset(train.y.cpu(), cfg.train_frac, cfg.seed)
    ytr = train.y.cpu() if idx is None else train.y.cpu()[idx]
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    class_w = (len(ytr) / (len(CLASSES) * counts)).to(DEVICE)
    sample_w = (1.0 / counts)[train.y.cpu()] if cfg.sampler == "weighted" else None

    model = build_model(cfg, train.X.shape[1]).to(DEVICE)
    # ---- compression hooks (lib.compress): start weights, distillation teacher, pruning, quantization
    ex = cfg.extra
    from . import compress as C
    src = ex.get("init") or ex.get("slim")
    if src:
        sstate = torch.load(CKPT / src["exp"] / f"{src['name']}.pt", map_location=DEVICE)
        if ex.get("init"):
            model.load_state_dict(sstate)
        else:   # structured pruning: most important channels of a wider trained model
            scfg = json.loads((RESULTS / src["exp"] / f"{src['name']}.json").read_text())["config"]
            snet = build_model(Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in scfg.items()}),
                               train.X.shape[1]).to(DEVICE)
            snet.load_state_dict(sstate)
            model.load_state_dict(C.slim_state(snet, model))
    teacher = C.load_teacher(DEVICE) if ex.get("kd") else None
    prune = ex.get("prune")
    masks, ramp = None, (prune or {}).get("ramp", 0)
    if ex.get("wbits"):
        C.attach_weight_quant(model, ex["wbits"])
    opt = _optimizer(cfg, model)
    sched = None
    if cfg.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    elif cfg.schedule == "step":
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [cfg.epochs // 2, 3 * cfg.epochs // 4], 0.1)
    elif cfg.schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.3, patience=3)
    loss_fn = _loss_fn(cfg, class_w)
    aug = AUG_PRESETS[cfg.aug]
    yva = val.y.cpu().numpy()

    amp = bool(cfg.extra.get("amp")) and DEVICE.type == "cuda"   # mixed precision: CUDA only (desktop notebook)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    hist, best, best_state, bad = [], None, None, 0
    epoch_probs = []  # only kept if cfg.extra["save_epoch_probs"] (c4 monitor test)
    sign = 1 if cfg.monitor == "val_loss" else -1
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        if prune and (masks is None or ep <= ramp):   # gradual pruning: masks recomputed while ramping, then fixed
            masks = C.global_magnitude_masks(model, C.ramp_sparsity(prune["sparsity"], ep, ramp))
            C.apply_masks(masks)
        tl, tc, tn = 0.0, [], []
        for x, y in train.batches(cfg.batch_size, shuffle=True, gen=gen, aug=aug, weights=sample_w, idx=idx):
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp):
                z = model(x)
                loss = loss_fn(z.float(), y)
                if teacher is not None:
                    with torch.no_grad():
                        zt = teacher(x)
                    loss = C.kd_loss(z.float(), zt.float(), loss, ex["kd"]["T"], ex["kd"]["alpha"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if masks is not None:
                C.apply_masks(masks)
            tl += loss.item() * len(y)
            tc.append(z.argmax(1).cpu()); tn.append(y.cpu())
        tc, tn = torch.cat(tc).numpy(), torch.cat(tn).numpy()
        prob = predict(model, val)
        if cfg.extra.get("save_epoch_probs"):
            epoch_probs.append(prob.astype(np.float16))
        vloss = float(-np.log(prob[np.arange(len(yva)), yva] + 1e-12).mean())
        m = evaluate(yva, prob)
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], train_loss=tl / len(tn),
                   train_bal_acc=balanced_accuracy_score(tn, tc), val_loss=vloss, val_bal_acc=m["bal_acc"],
                   val_macro_f1=m["macro_f1"], val_tumor_auc=m["tumor_auc"], time=time.time() - t0)
        hist.append(row)
        score = sign * row[cfg.monitor]
        if prune and ep < ramp:   # the model only counts once it has reached the target sparsity
            pass
        elif best is None or score < best:
            best, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if verbose:
            print(f"ep {ep:3d} loss {row['train_loss']:.3f}/{vloss:.3f} bal_acc {row['train_bal_acc']:.3f}/"
                  f"{m['bal_acc']:.3f} {row['time']:.0f}s")
        if sched is not None:
            sched.step(vloss) if cfg.schedule == "plateau" else sched.step()
        if bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    if ex.get("wbits"):
        C.detach_weight_quant(model)          # weights now hold their quantized values
        best_state = copy.deepcopy(model.state_dict())
    prob = predict(model, val)
    res = {"config": cfg.key(), "history": hist, "best_epoch": best_ep, "epochs_run": len(hist),
           "val": evaluate(yva, prob), "n_params": n_params(model), "n_conv": n_conv_layers(model),
           "sec_per_epoch": (time.time() - t0) / len(hist), "device": str(DEVICE),
           "size": C.size_report(model, ex.get("wbits", 32))}
    _supersede(cfg)
    for p in (jpath, cpath):
        p.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(res, indent=1))
    np.savez_compressed(npath, prob=prob, y=yva, paths=np.array(val.paths))
    if epoch_probs:
        np.savez_compressed(npath.with_name(f"{cfg.name}_epochs.npz"), prob=np.stack(epoch_probs), y=yva)
    # weights only where a later step needs them: the final model (c10) and the compressed models (d-notebooks:
    # they are pruned/quantized further, and the tiny models are the deliverable; a few kB each)
    if cfg.exp == "c9" or cfg.exp.startswith("d"):
        torch.save(best_state, cpath)
    del train, val, model, opt
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return res


def load_probs(exp, name):
    d = np.load(RESULTS / exp / f"{name}_val.npz")
    return d["y"], d["prob"]


def summary(results: dict[str, dict]) -> pd.DataFrame:
    """One row per run: the numbers every comparison table needs."""
    rows = {}
    for name, r in results.items():
        v = r["val"]
        rows[name] = {"bal_acc": v["bal_acc"], "macro_f1": v["macro_f1"], "tumor_auc": v["tumor_auc"],
                      "no_tumor_rec": v["recall"]["no_tumor"], "best_ep": r["best_epoch"],
                      "epochs": r["epochs_run"], "params": r["n_params"], "s/ep": r["sec_per_epoch"]}
    return pd.DataFrame(rows).T
