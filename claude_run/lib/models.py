"""One configurable CNN family, from a 4-conv-layer small net to a ResNet-style deep net."""
from __future__ import annotations

import torch
from torch import nn


def conv_bn_act(cin, cout, k, stride=1, norm=True, act=True, groups=1):
    layers = [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=not norm, groups=groups)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Block(nn.Module):
    """'single': one conv.  'basic': two convs (ResNet basic block), optionally with a skip connection.
    'dwsep': depthwise-separable conv (k x k depthwise per channel, then 1x1 pointwise), as in MobileNet:
    k*k*cin + cin*cout weights instead of k*k*cin*cout."""

    def __init__(self, cin, cout, k, stride, kind, norm, residual):
        super().__init__()
        self.residual = residual and kind == "basic"
        if kind == "single":
            self.body = conv_bn_act(cin, cout, k, stride, norm)
        elif kind == "dwsep":
            self.body = nn.Sequential(conv_bn_act(cin, cin, k, stride, norm, groups=cin),
                                      conv_bn_act(cin, cout, 1, 1, norm))
        else:
            self.body = nn.Sequential(conv_bn_act(cin, cout, k, stride, norm),
                                      conv_bn_act(cout, cout, k, 1, norm, act=not self.residual))
        if self.residual:
            self.short = (nn.Identity() if cin == cout and stride == 1
                          else conv_bn_act(cin, cout, 1, stride, norm, act=False))
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.short(x)) if self.residual else self.body(x)


class CNN(nn.Module):
    """stem -> stages (each halves the resolution) -> global average pool -> dropout -> linear.

    widths/depths: channels and number of blocks per stage.
    downsample: 'maxpool' | 'avgpool' | 'stride' (first conv of the stage has stride 2).
    """

    def __init__(self, in_ch=1, n_classes=4, widths=(16, 32, 64, 128), depths=(1, 1, 1, 1), kernel=3,
                 stem_kernel=3, block="single", norm=True, residual=False, downsample="maxpool", dropout=0.0):
        super().__init__()
        strided = downsample == "stride"
        pool = {"maxpool": nn.MaxPool2d(2), "avgpool": nn.AvgPool2d(2)}.get(downsample)
        layers = [conv_bn_act(in_ch, widths[0], stem_kernel, 2 if strided else 1, norm)]
        if not strided:
            layers.append(pool)
        cin = widths[0]
        for s, (w, d) in enumerate(zip(widths, depths)):
            if s > 0 and not strided:
                layers.append({"maxpool": nn.MaxPool2d(2), "avgpool": nn.AvgPool2d(2)}[downsample])
            for b in range(d):
                stride = 2 if (strided and s > 0 and b == 0) else 1
                layers.append(Block(cin, w, kernel, stride, block, norm, residual))
                cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if residual:  # zero-init last BN of each residual branch: blocks start as identity
            for m in self.modules():
                if isinstance(m, Block) and m.residual and norm:
                    nn.init.zeros_(m.body[-1][1].weight)

    def forward(self, x):
        return self.head(self.features(x))

    def last_conv_block(self):
        return self.features[-1]


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def n_conv_layers(model) -> int:
    return sum(isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1) for m in model.modules())
