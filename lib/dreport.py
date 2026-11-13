"""Shared helpers for the compression notebooks (d1-d6): student recipe, size/accuracy tables, Pareto frontier,
the standard trade-off plot, and cached records for evaluations that involve no training (post-training
quantization)."""
from __future__ import annotations

import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import compress as C
from .data import SEED
from .evaluation import evaluate
from .train import CKPT, DEVICE, RESULTS, Config, build_model, prepare_data, predict, run

SEEDS3 = [SEED, SEED + 1, SEED + 2]
FLOORS = {"high-performance": 0.95, "tiny": 0.85}   # fixed in d1, before any compression result


def teacher_cfg() -> Config:
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in C.teacher_config().items()})


def student(exp: str, name: str, **kw) -> Config:
    """Students use the teacher's input and training recipe (what c5-c8 found), with their own architecture."""
    base = replace(teacher_cfg(), exp=exp, name=name, extra={})
    return replace(base, **kw)


def run_seeds(cfg: Config, seeds=SEEDS3) -> list[dict]:
    return [run(replace(cfg, name=f"{cfg.name}_s{s}", seed=s)) for s in seeds]


def with_src(cfg: Config, src_name: str, key: str, seed: int, **extra) -> Config:
    """cfg for one seed that starts from (key='init') or slims (key='slim') the same seed of a stored run."""
    return replace(cfg, name=f"{cfg.name}_s{seed}", seed=seed,
                   extra={**cfg.extra, key: {"exp": src_name.split("/")[0], "name": f"{src_name.split('/')[1]}_s{seed}"},
                          **extra})


def row(rs: list[dict]) -> dict:
    b = np.array([r["val"]["bal_acc"] for r in rs])
    return {"bal_acc": b.mean(), "sd": b.std(ddof=1) if len(b) > 1 else np.nan, "n": len(b),
            "bytes": int(np.mean([r["size"]["bytes"] for r in rs])),
            "kB": np.mean([r["size"]["bytes"] for r in rs]) / 1024,
            "params": rs[0]["n_params"], "sparsity": np.mean([r["size"]["sparsity"] for r in rs]),
            "wbits": rs[0]["size"]["wbits"],
            "tumor_auc": np.mean([r["val"]["tumor_auc"] for r in rs]),
            "no_tumor_rec": np.mean([r["val"]["recall"]["no_tumor"] for r in rs])}


def table(options: dict[str, list[dict]]) -> pd.DataFrame:
    t = pd.DataFrame({k: row(v) for k, v in options.items()}).T
    t["x smaller than teacher"] = teacher_bytes() / t.bytes.astype(float)
    return t.astype({"n": int, "bytes": int, "params": int, "wbits": int}).round(4)


def teacher_bytes() -> int:
    return C.size_report(C.load_teacher(DEVICE))["bytes"]


def pareto(t: pd.DataFrame) -> pd.DataFrame:
    """Rows not beaten by a row that is both smaller (or equal) and more accurate (or equal)."""
    t = t.sort_values(["bytes", "bal_acc"], ascending=[True, False])
    keep, best = [], -1
    for k, r in t.iterrows():
        if r.bal_acc > best:
            keep.append(k)
            best = r.bal_acc
    return t.loc[keep]


def pick(t: pd.DataFrame, floor: float):
    """The headline rule (d1): the smallest model whose mean val balanced accuracy is >= floor."""
    ok = t[t.bal_acc >= floor]
    return None if ok.empty else ok.bytes.astype(float).idxmin()


def tradeoff_plot(groups: dict[str, pd.DataFrame], title: str, ax=None, teacher_acc=None, label_points=False):
    """Size (stored kB, log scale) against mean val balanced accuracy, one colour per group, floors marked."""
    ax = ax or plt.subplots(figsize=(9, 5))[1]
    for i, (g, t) in enumerate(groups.items()):
        ax.errorbar(t.kB.astype(float), t.bal_acc.astype(float), yerr=t.sd.astype(float).fillna(0), fmt="o",
                    color=f"C{i}", label=g, capsize=2, ms=5, alpha=0.85)
        if label_points:
            for k, r in t.iterrows():
                ax.annotate(str(k), (r.kB, r.bal_acc), fontsize=7, xytext=(3, 3), textcoords="offset points")
    allt = pd.concat(groups.values())
    f = pareto(allt)
    ax.step(f.kB.astype(float), f.bal_acc.astype(float), where="post", color="k", lw=1, alpha=0.5, label="Pareto frontier")
    if teacher_acc is not None:
        ax.scatter([teacher_bytes() / 1024], [teacher_acc], marker="*", s=200, color="gold", edgecolor="k",
                   zorder=5, label=f"teacher ({teacher_bytes() / 1024:.0f} kB)")
    for name, fl in FLOORS.items():
        ax.axhline(fl, ls="--", lw=1, color="grey")
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0.1, fl + 0.003, f"{name} floor {fl:.2f}", fontsize=8, color="grey")
    ax.set(xscale="log", xlabel="stored size (kB, log scale)", ylabel="val balanced accuracy (mean of seeds)", title=title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    return ax


# ------------------------------------------------------------------ post-training quantization records
_SPLITS = {}


def _splits(cfg):
    key = (cfg.mode, cfg.resize, cfg.size, cfg.normalize)
    if key not in _SPLITS:
        _SPLITS[key] = prepare_data(cfg)
    return _SPLITS[key]


def load_run_model(exp: str, name: str):
    r = json.loads((RESULTS / exp / f"{name}.json").read_text())
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in r["config"].items()})
    net = build_model(cfg, 3 if cfg.mode == "rgb" else 1)
    net.load_state_dict(torch.load(CKPT / exp / f"{name}.pt", map_location="cpu"))
    return cfg, net.eval()


def ptq(src_exp: str, src_name: str, exp: str, bits) -> dict:
    """Post-training quantization of a stored run, no retraining. bits = 8/4/2 (simulated, per-channel weights)
    or 'int8' (real PyTorch FX int8, x86 backend, CPU). Stored like a training run (results/<exp>/<name>.json)."""
    import re
    tag = f"ptq{bits}"
    name = re.sub(r"_s(\d{4})$", rf"_{tag}_s\1", src_name)   # keep the seed suffix last
    jp = RESULTS / exp / f"{name}.json"
    if jp.exists():
        return json.loads(jp.read_text())
    cfg, net = load_run_model(src_exp, src_name)
    train, val = _splits(cfg)
    yv = val.y.cpu().numpy()
    if bits == "int8":
        calib = [x.cpu() for x, _ in train.batches(64)][:20]
        q = C.int8_fx(net, calib)
        with torch.no_grad():
            prob = torch.cat([torch.softmax(q(x.cpu()), 1) for x, _ in val.batches(128)]).numpy()
        size = {**C.size_report(net, 8), "serialized_bytes": C.serialized_bytes(q)}
        lat = {"fp32_ms": C.cpu_latency_ms(net, calib[0][:1]), "int8_ms": C.cpu_latency_ms(q, calib[0][:1])}
    else:
        C.quantize_weights_ptq(net, bits)
        prob = predict(net.to(DEVICE), val)
        size = C.size_report(net, bits)
        lat = {}
    res = {"config": {**json.loads(json.dumps(cfg.key())), "exp": exp, "name": name,
                      "extra": {**cfg.extra, "ptq": bits, "src": {"exp": src_exp, "name": src_name}}},
           "history": [], "best_epoch": 0, "epochs_run": 0, "val": evaluate(yv, prob), "n_params": sum(p.numel() for p in net.parameters()),
           "n_conv": 0, "sec_per_epoch": 0, "device": "cpu" if bits == "int8" else str(DEVICE), "size": size, "latency": lat}
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(res, indent=1))
    np.savez_compressed(RESULTS / exp / f"{name}_val.npz", prob=prob, y=yv, paths=np.array(val.paths))
    return res
