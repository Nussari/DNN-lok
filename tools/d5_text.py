T = {
    'quant': """
**The cost of quantization.** QAT is measured against the same fine-tuning in fp32 (the second table above);
PTQ, which adds no training, against the original fp32 model:

| bits | post-training (PTQ) | quantization-aware (QAT) |
|---|---|---|
| **8** | within ±0.5 pp of fp32 | within ±0.3 pp of fp32 (the control), except one collapsed 90 %-pruned model (−0.8 pp): **free** |
| **4** | loses 0.8 to 35 pp; only the large models survive | **−0.0 to −0.4 pp from 52 kB up**; mostly −0.8 to −2.8 pp for 20 kB and below (one outlier, see below) |
| **2** | collapses every model to about chance (0.23 to 0.30) | −5 to −29 pp (models not already at chance) |

* **QAT beats PTQ in 37 of 42 model/bit-width pairs**, and the difference grows as the bits shrink: at 4 bits PTQ
  rounds each weight to one of 15 levels with no chance to adapt; QAT lets the other weights compensate.
* **4 bits is the sweet spot**: 4.4 to 7.9x smaller stored size than fp32 (8x for the weights alone; the fp32
  scales, biases and sparsity mask do not shrink) for almost no loss on the larger models.
* **One outlier:** dw-sep 5x5 w4, 75 % pruned, gains +15.8 pp with 4-bit QAT over its control. That is not a
  quantization effect: one seed of the control collapsed to chance (0.718 / 0.25 / 0.775) while all three QAT
  seeds trained (0.723 / 0.719 / 0.776).
* **2 bits** (ternary weights) is too coarse for every model here, even with QAT.
* Without the control, QAT 8-bit would seem to *improve* small models (for example 0.916 to 0.933 at 20 kB);
  that is the extra fine-tuning again, as in d4.
* Pruned models stay pruned under quantization (zeros map to zero), so pruning and quantization multiply.
""",
    'int8': """
**Real int8 vs simulated 8-bit.** For models from 7 kB up, real int8 (weights *and* activations in int8, PyTorch
FX) is within 0.4 pp of the simulation, except dw-sep 5x5 w8 (1.0 pp): the simulation is faithful. For some of
the smallest models it loses 2 to 3.4 pp more (not all: the 4.0 kB 50 %-pruned w4 model loses only 0.4 pp), because quantizing the *activations* of a 2- to 4-channel network
removes a large share of its information. Weight-only quantization (what the simulation does) is the better fit
for tiny models.

**Two practical caveats for real deployment in PyTorch:**
* **Latency.** int8 is **slower** than fp32 on this CPU for every model (2.2 to 5.4 ms vs 1.0 to 2.7 ms per image):
  at this size the cost of quantizing and dequantizing at the model's edges and calling int8 kernels dominates
  the arithmetic saved. int8 pays off for large models and on hardware with int8 units.
* **File size.** `torch.save` of the converted int8 model is 21 to 287 kB, far above the 1 to 272 kB of int8
  weights: it stores the FX graph and per-layer metadata. The sizes in this study assume a lean format (weights,
  scales and biases only, as in d1), which is what an embedded deployment would use.
""",
    'decision': """
**Decision.** Per model and bit width, the higher of PTQ and QAT is kept (QAT almost everywhere). The frontier after
d5 is made almost entirely of 4-bit and 8-bit QAT models, from 0.54 kB to 272 kB. It goes to d6.
""",
}
