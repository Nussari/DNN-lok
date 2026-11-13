"""Model compression: teacher, distillation loss, size accounting, pruning, quantization.

Everything here plugs into lib.train.run through Config.extra, so every compressed model is a cached run with the
same record (config, history, val scores) as every other run:

    extra["kd"]     = {"T": 4, "alpha": 0.9}            distil from the teacher (r2 final model)
    extra["init"]   = {"exp": "d3", "name": "..."}       start from a stored run's weights
    extra["slim"]   = {"exp": "d3", "name": "..."}       structured pruning: copy the most important channels of a
                                                         stored (wider) run into this (narrower) model
    extra["prune"]  = {"sparsity": 0.9, "ramp": 10}      unstructured magnitude pruning, ramped up over `ramp` epochs
    extra["wbits"]  = 4                                  quantization-aware training of the weights (simulated)
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import ARTIFACTS, RUN_DIR

_TEACHER = {}


# ------------------------------------------------------------------ teacher
def teacher_config() -> dict:
    """Config of the teacher: the r2 final model (artifacts/final.json)."""
    return json.loads((ARTIFACTS / "final.json").read_text())["config"]


def load_teacher(device):
    """The r2 final model (seed 1738), frozen, in eval mode. Cached per device."""
    if device not in _TEACHER:
        from .train import Config, build_model
        f = json.loads((ARTIFACTS / "final.json").read_text())
        c = {k: tuple(v) if isinstance(v, list) else v for k, v in f["config"].items()}
        net = build_model(Config(**c), 3 if c["mode"] == "rgb" else 1)
        net.load_state_dict(torch.load(RUN_DIR / f["checkpoint"].replace("\\", "/"), map_location="cpu"))
        _TEACHER[device] = net.to(device).eval()
        for p in _TEACHER[device].parameters():
            p.requires_grad_(False)
    return _TEACHER[device]


def kd_loss(student_logits, teacher_logits, hard_loss, T: float, alpha: float):
    """Hinton et al. (2015): alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * hard-label loss.
    The T^2 factor keeps the soft-target gradients on the same scale as the hard-label gradients."""
    soft = F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1), reduction="batchmean")
    return alpha * T * T * soft + (1 - alpha) * hard_loss


# ------------------------------------------------------------------ size accounting
def _fold_bn(model):
    """(weight, bias) per Conv2d/Linear after folding the following BatchNorm into it, as deployed for inference."""
    from torch.nn.utils.parametrize import ParametrizationList
    mods = [m for m in model.modules() if not isinstance(m, (ParametrizationList, nn.ModuleDict, WeightFakeQuant))]
    out = []
    for i, m in enumerate(mods):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach().float().clone()
            b = m.bias.detach().float().clone() if m.bias is not None else torch.zeros(w.shape[0], device=w.device)
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            if isinstance(nxt, nn.BatchNorm2d):
                s = nxt.weight.detach() / torch.sqrt(nxt.running_var + nxt.eps)
                w = w * s.view(-1, *([1] * (w.dim() - 1)))
                b = (b - nxt.running_mean) * s + nxt.bias.detach()
            out.append((w.cpu(), b.cpu()))
    return out


def size_report(model, wbits: int = 32) -> dict:
    """Stored size of the model as deployed, in bytes. Fixed in d1, used for every model:

    * BatchNorm is folded into the preceding conv (it is a per-channel scale and shift at inference).
    * Weights are stored at `wbits` bits. Below 32 bits, each output channel stores one fp32 scale
      (per-channel symmetric quantization). Biases stay fp32.
    * Pruned (zero) weights: stored either dense, or sparse as a 1-bit-per-position mask plus the nonzero values;
      whichever is smaller is counted.
    """
    layers = _fold_bn(model)
    n_w = sum(w.numel() for w, _ in layers)
    nnz = sum(int((w != 0).sum()) for w, _ in layers)
    n_out = sum(w.shape[0] for w, _ in layers)
    bias_bits = 32 * sum(b.numel() for _, b in layers)
    scale_bits = 32 * n_out if wbits < 32 else 0
    dense_bits = wbits * n_w
    sparse_bits = n_w + wbits * nnz
    w_bits = min(dense_bits, sparse_bits)
    total = math.ceil((w_bits + bias_bits + scale_bits) / 8)
    return {"bytes": total, "wbits": wbits, "n_weights": n_w, "nonzero": nnz, "sparsity": 1 - nnz / n_w,
            "sparse_storage": sparse_bits < dense_bits}


def serialized_bytes(model) -> int:
    """Real size of torch.save(state_dict) (includes pickle overhead and unfolded BatchNorm)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


# ------------------------------------------------------------------ unstructured pruning
def prunable(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def raw_weight(m):
    """The trainable weight tensor (under a quantization parametrization: the unquantized original)."""
    from torch.nn.utils import parametrize
    return m.parametrizations.weight.original if parametrize.is_parametrized(m, "weight") else m.weight


def global_magnitude_masks(model, sparsity: float) -> dict:
    """Global magnitude pruning: remove the `sparsity` fraction of weights with the smallest |w| across all
    conv and linear layers (layers that matter less lose more). Biases and BatchNorm are not pruned."""
    mods = prunable(model)
    allw = torch.cat([raw_weight(m).detach().abs().flatten() for m in mods])
    k = int(sparsity * allw.numel())
    if k < 1:
        return {m: torch.ones_like(raw_weight(m)) for m in mods}
    thr = allw.kthvalue(k).values
    return {m: (raw_weight(m).detach().abs() > thr).float() for m in mods}


def apply_masks(masks: dict):
    with torch.no_grad():
        for m, mask in masks.items():
            raw_weight(m).mul_(mask)


def ramp_sparsity(target: float, epoch: int, ramp: int) -> float:
    """Gradual pruning schedule (Zhu & Gupta 2017): s_t = s * (1 - (1 - t/ramp)^3), reaching s at epoch `ramp`."""
    if ramp <= 0 or epoch >= ramp:
        return target
    return target * (1 - (1 - epoch / ramp) ** 3)


# ------------------------------------------------------------------ structured pruning (channel slimming)
def slim_state(src_model, dst_model) -> dict:
    """Copy the most important channels of a trained plain (non-residual) CNN into a narrower one of the same
    layout. Importance of a channel = |gamma| of its BatchNorm (Liu et al. 2017, network slimming).
    Depthwise convs keep exactly the channels of their input. Returns the state dict for dst_model."""
    src = list(src_model.modules())
    dst = list(dst_model.modules())
    assert len(src) == len(dst), "slim_state needs the same layout"
    state, keep_in = {}, None
    names = {m: n for n, m in dst_model.named_modules()}
    for i, (s, d) in enumerate(zip(src, dst)):
        if isinstance(s, nn.Conv2d):
            if keep_in is None:
                keep_in = torch.arange(s.in_channels)
            if s.groups > 1:                                   # depthwise: channels follow the input
                keep_out = keep_in
                w = s.weight.detach()[keep_out]
            else:
                bn = src[i + 1] if isinstance(src[i + 1], nn.BatchNorm2d) else None
                imp = bn.weight.detach().abs() if bn is not None else s.weight.detach().abs().sum((1, 2, 3))
                keep_out = imp.topk(d.out_channels).indices.sort().values
                w = s.weight.detach()[keep_out][:, keep_in]
            state[names[d] + ".weight"] = w.clone()
            if s.bias is not None:
                state[names[d] + ".bias"] = s.bias.detach()[keep_out].clone()
            keep_in = keep_out
        elif isinstance(s, nn.BatchNorm2d):
            for p in ["weight", "bias", "running_mean", "running_var"]:
                state[f"{names[d]}.{p}"] = getattr(s, p).detach()[keep_in].clone()
            state[f"{names[d]}.num_batches_tracked"] = s.num_batches_tracked.clone()
        elif isinstance(s, nn.Linear):
            state[names[d] + ".weight"] = s.weight.detach()[:, keep_in].clone()
            state[names[d] + ".bias"] = s.bias.detach().clone()
    return state


# ------------------------------------------------------------------ quantization (simulated, any bit width)
class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def fake_quant_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric quantization to `bits` bits: levels -(2^(b-1)-1) .. 2^(b-1)-1 times a per-channel
    fp32 scale (max |w| of the channel). 2 bits gives {-1, 0, +1} * scale (ternary). Zeros stay zero, so it composes
    with pruning. Per-channel scaling also commutes with folding BatchNorm into the conv."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.detach().abs().flatten(1).max(1).values.clamp_min(1e-12) / qmax
    scale = scale.view(-1, *([1] * (w.dim() - 1)))
    return _RoundSTE.apply(w / scale).clamp(-qmax, qmax) * scale


class WeightFakeQuant(nn.Module):
    """Parametrization: the layer always computes with quantized weights; gradients pass straight through."""

    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def forward(self, w):
        return fake_quant_weight(w, self.bits)


def attach_weight_quant(model, bits: int):
    from torch.nn.utils import parametrize
    for m in prunable(model):
        parametrize.register_parametrization(m, "weight", WeightFakeQuant(bits))


def detach_weight_quant(model):
    """Bake the quantized values into the weights (they stay on the quantization grid)."""
    from torch.nn.utils import parametrize
    for m in prunable(model):
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


@torch.no_grad()
def quantize_weights_ptq(model, bits: int):
    """Post-training quantization of the weights, no retraining."""
    for m in prunable(model):
        m.weight.copy_(fake_quant_weight(m.weight, bits))


# ------------------------------------------------------------------ real int8 (PyTorch FX, x86 backend)
def int8_fx(model, calib_batches, qat_model=None):
    """Real int8 model via FX graph mode quantization (torch.ao.quantization.quantize_fx, backend 'x86').
    Post-training static quantization: observers are calibrated on `calib_batches` (train images, no augmentation).
    If `qat_model` is given (a model prepared with prepare_qat_fx and fine-tuned), it is converted instead.
    Runs on CPU, as the quantized kernels do."""
    import copy
    import warnings
    # FX's default x86 observers use reduce_range, which PyTorch 2.6 flags as to-be-deprecated; harmless here
    warnings.filterwarnings("ignore", message="Please use quant_min and quant_max")
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = "x86"
    if qat_model is not None:
        return convert_fx(qat_model.cpu().eval())
    m = copy.deepcopy(model).cpu().eval()
    example = (calib_batches[0][:1].cpu(),)
    prepared = prepare_fx(m, get_default_qconfig_mapping("x86"), example)
    with torch.no_grad():
        for x in calib_batches:
            prepared(x.cpu())
    return convert_fx(prepared)


@torch.no_grad()
def cpu_latency_ms(model, x1: torch.Tensor, reps: int = 200) -> float:
    """Median single-image CPU latency (1 thread) in ms."""
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model.cpu().eval()
    x1 = x1.cpu()
    for _ in range(20):
        model(x1)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x1)
        ts.append(time.perf_counter() - t0)
    torch.set_num_threads(nt)
    return float(np.median(ts) * 1000)
