"""Model and shared feature code for the anomalous-diffusion OOD detector.

Both `train.py` and `inference.py` import from here so the network, the
hand-crafted physics features, the input channels, the D4 augmentation, and the
test-time-augmented forward pass are defined exactly once.
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def physics(x: np.ndarray) -> np.ndarray:
    """Hand-crafted physical descriptors of each trajectory.

    Log-MSD at several lags and its local slopes, velocity autocorrelations,
    speed statistics, end-to-end displacement, a gyration-tensor anisotropy, and
    early/late energy ratios -- the quantities that separate the diffusion
    families and expose out-of-distribution dynamics. Returns (N, F) float32.
    """
    x = np.asarray(x, np.float32)
    d = np.diff(x, axis=2)
    speed = np.sqrt((d * d).sum(1) + 1e-8)
    feats = []
    lags = [1, 2, 3, 4, 6, 8, 12, 16]
    msds = []
    for lag in lags:
        z = x[:, :, lag:] - x[:, :, :-lag]
        msds.append(np.log(np.mean(z * z, axis=(1, 2)) + 1e-7))
    feats.extend(msds)
    lm = np.stack(msds, 1)
    feats.extend([
        (lm[:, i + 1] - lm[:, i]) / (math.log(lags[i + 1]) - math.log(lags[i]))
        for i in range(7)
    ])
    for lag in range(1, 9):
        dot = (d[:, :, lag:] * d[:, :, :-lag]).sum(1)
        den = np.sqrt((d[:, :, lag:] ** 2).sum(1) * (d[:, :, :-lag] ** 2).sum(1) + 1e-7)
        feats.append((dot / den).mean(1))
    feats += [speed.mean(1), speed.std(1), np.quantile(speed, .1, axis=1),
              np.quantile(speed, .5, axis=1), np.quantile(speed, .9, axis=1)]
    dot = (d[:, :, 1:] * d[:, :, :-1]).sum(1)
    den = speed[:, 1:] * speed[:, :-1] + 1e-7
    c = dot / den
    feats += [c.mean(1), c.std(1), np.mean(c * c, 1)]
    end = np.sqrt(((x[:, :, -1] - x[:, :, 0]) ** 2).sum(1))
    path = speed.sum(1) + 1e-7
    feats += [end, end / path]
    centered = x - x.mean(2, keepdims=True)
    a = (centered[:, 0] ** 2).mean(1)
    b = (centered[:, 1] ** 2).mean(1)
    q = (centered[:, 0] * centered[:, 1]).mean(1)
    disc = np.sqrt((a - b) ** 2 + 4 * q * q)
    feats += [a + b, (a + b + disc) / (a + b - disc + 1e-6)]
    for p in [1, 2, 3]:
        feats.append(np.sum(speed ** p, 1))
    e = (d * d).sum(1)
    feats += [e[:, :16].mean(1) / (e[:, -16:].mean(1) + 1e-6),
              e[:, 16:32].mean(1) / (e[:, :16].mean(1) + 1e-6),
              e[:, -16:].mean(1) / (e[:, 16:32].mean(1) + 1e-6)]
    return np.nan_to_num(np.stack(feats, 1), nan=0, posinf=20, neginf=-20).astype(np.float32)


def channels(x: np.ndarray) -> np.ndarray:
    """Per-timestep sequence channels: position, velocity, speed, and time."""
    d = np.diff(x, axis=2, prepend=x[:, :, :1])
    speed = np.sqrt((d * d).sum(1, keepdims=True) + 1e-8)
    t = np.broadcast_to(np.linspace(-1, 1, 50, dtype=np.float32)[None, None, :], (len(x), 1, 50))
    return np.concatenate([x, d, speed, t], 1).astype(np.float32)


def d4(x: np.ndarray, k: int) -> np.ndarray:
    """One of the eight D4 symmetries (axis swap and per-axis sign flips)."""
    y = x.copy()
    if k & 1:
        y = y[:, ::-1, :]
    if k & 2:
        y[:, 0] *= -1
    if k & 4:
        y[:, 1] *= -1
    return y.copy()


class Block(nn.Module):
    """Dilated residual 1D-conv block with GroupNorm and GELU."""

    def __init__(self, c: int, d: int) -> None:
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.n2 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.drop = nn.Dropout(.08)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.c2(F.gelu(self.n2(self.c1(F.gelu(self.n1(x)))))))


class Net(nn.Module):
    """Multi-branch 1D-CNN fusing the sequence with the physics features.

    Outputs a normalized embedding (for density-based OOD scoring), five-way
    family logits, and a bounded anomalous-exponent regression in [0.05, 2.0].
    """

    def __init__(self, nf: int) -> None:
        super().__init__()
        self.br = nn.ModuleList([nn.Conv1d(6, 48, k, padding=k // 2) for k in (3, 5, 9)])
        self.proj = nn.Conv1d(144, 160, 1)
        self.blocks = nn.Sequential(*[Block(160, d) for d in (1, 2, 4, 8)])
        self.pm = nn.Sequential(nn.Linear(nf, 128), nn.GELU(), nn.Dropout(.12), nn.Linear(128, 128), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(448, 256), nn.GELU(), nn.Dropout(.12), nn.Linear(256, 128))
        self.cls = nn.Linear(128, 5)
        self.reg = nn.Linear(128, 1)

    def forward(self, seq: torch.Tensor, phy: torch.Tensor):
        z = self.blocks(self.proj(torch.cat([F.gelu(b(seq)) for b in self.br], 1)))
        z = torch.cat([z.mean(2), z.amax(2), self.pm(phy)], 1)
        emb = F.normalize(self.fuse(z), dim=1)
        return emb, self.cls(emb) * 8, .05 + 1.95 * torch.sigmoid(self.reg(emb).squeeze(1))


def infer(model: nn.Module, x: np.ndarray, pm: np.ndarray, ps: np.ndarray,
          batch: int = 2048, views: int = 8):
    """Test-time-augmented forward pass over `views` D4 symmetries.

    Returns averaged (embeddings, class probabilities, exponent) as numpy.
    """
    device = next(model.parameters()).device
    es, psout, aa = [], [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for s in range(0, len(x), batch):
            xb = x[s:s + batch]
            ev, pv, av = [], [], []
            for k in range(views):
                xx = d4(xb, k)
                ph = (physics(xx) - pm) / ps
                e, l, a = model(torch.from_numpy(channels(xx)).to(device),
                                torch.from_numpy(ph).to(device))
                ev.append(e)
                pv.append(l.softmax(1))
                av.append(a)
            es.append(F.normalize(torch.stack(ev).mean(0), dim=1).float().cpu())
            psout.append(torch.stack(pv).mean(0).float().cpu())
            aa.append(torch.stack(av).mean(0).float().cpu())
    return torch.cat(es).numpy(), torch.cat(psout).numpy(), torch.cat(aa).numpy()
