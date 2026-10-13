"""Time-series anomaly detection adapter: 1D-CNN autoencoder, score = reconstruction MSE, metric = ROC AUC.

Data sources
  ecg5000 : UCR ECG5000 (auto-download ~10 MB). Class 1 = normal, classes 2-5 = anomaly.
  npz     : your own data. File with X_train (normal only), X_test, y_test (1 = anomaly).
            Arrays shaped [N, L] or [N, C, L]. Use this for MIMII (pre-computed log-mel frames)
            or whatever time series the organisers hand out.
"""
from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.adapter import TaskAdapter
from core.config import ROOT
from core.runtime import as_predictor

from .common import train_loop

log = logging.getLogger(__name__)

ECG5000_URLS = [
    "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
    "http://www.timeseriesclassification.com/Downloads/ECG5000.zip",
]

DEFAULTS = {
    "source": "ecg5000",
    "data_root": "data",
    "npz_path": None,
    "normal_class": 1,
    "train_normal_frac": 0.7,   # fraction of normal samples used for training; rest go to test
    "channels": [32, 64],       # encoder widths; decoder mirrors them
    "kernel": 7,
    "batch_size": 64,
    "train_epochs": 40,
    "lr": 2e-3,
    "finetune_lr": 5e-4,
    "weight_decay": 1e-5,
    "calib_size": 256,
    "train_subset": None,
    "test_subset": None,
    "kd_alpha": 0.5,            # distill_loss: alpha*MSE(student, teacher) + (1-alpha)*MSE(student, x)
    "seed": 0,
}


class ConvAE1d(nn.Module):
    """Conv1d autoencoder: each encoder stage halves the length, each decoder stage doubles it."""

    def __init__(self, in_ch: int = 1, channels=(32, 64), k: int = 7):
        super().__init__()
        enc, c = [], in_ch
        for w in channels:
            enc += [nn.Conv1d(c, w, k, stride=2, padding=k // 2), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            c = w
        enc += [nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
        dec = []
        for w in list(channels[::-1][1:]) + [channels[0]]:
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv1d(c, w, k, padding=k // 2),
                    nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            c = w
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)
        self.out = nn.Conv1d(c, in_ch, k, padding=k // 2)

    def forward(self, x):
        return self.out(self.decoder(self.encoder(x)))


def _download(urls, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for u in urls:
        try:
            log.info("downloading %s", u)
            urllib.request.urlretrieve(u, dest)
            return dest
        except Exception as e:
            log.warning("download failed from %s: %s", u, e)
    raise RuntimeError(f"could not download {dest.name}; place it manually at {dest}")


def _read_ucr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """UCR .txt (label first, whitespace) or .ts (values..., label after ':')."""
    if path.suffix == ".txt":
        arr = np.loadtxt(path)
        return arr[:, 1:].astype(np.float32), arr[:, 0].astype(int)
    xs, ys, data = [], [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            data = True
            continue
        if data:
            *vals, lab = line.split(":")
            xs.append(np.array(vals[0].split(","), dtype=np.float32))
            ys.append(int(float(lab)))
    return np.stack(xs), np.array(ys)


def load_ecg5000(root: Path) -> tuple[np.ndarray, np.ndarray]:
    d = root / "ECG5000"
    if not any(d.glob("ECG5000_TRAIN.*")):
        z = _download(ECG5000_URLS, root / "ECG5000.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    parts = []
    for split in ("TRAIN", "TEST"):
        cand = sorted(d.rglob(f"ECG5000_{split}.txt")) or sorted(d.rglob(f"ECG5000_{split}.ts"))
        parts.append(_read_ucr(cand[0]))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


class AnomalyAdapter(TaskAdapter):
    name = "anomaly"

    def __init__(self, cfg=None, quick=False, device=None):
        super().__init__({**DEFAULTS, **(cfg or {})}, quick, device)
        self._data = None
        self.in_ch = 1
        self.seq_len = None

    # ------------------------------------------------------------------ data
    def _arrays(self):
        c = self.cfg
        root = Path(c["data_root"])
        root = root if root.is_absolute() else ROOT / root
        if c["source"] == "ecg5000":
            x, lab = load_ecg5000(root)
            y = (lab != c["normal_class"]).astype(int)
            rng = np.random.default_rng(c["seed"])
            normal, anom = np.where(y == 0)[0], np.where(y == 1)[0]
            rng.shuffle(normal)
            n_tr = int(len(normal) * c["train_normal_frac"])
            tr, te = normal[:n_tr], np.concatenate([normal[n_tr:], anom])
            return x[tr], x[te], y[te]
        if c["source"] == "npz":
            f = np.load(c["npz_path"])
            return f["X_train"].astype(np.float32), f["X_test"].astype(np.float32), f["y_test"].astype(int)
        raise ValueError(f"unknown source {c['source']}")

    def load_data(self) -> dict:
        if self._data is not None:
            return self._data
        c = self.cfg
        xtr, xte, yte = self._arrays()
        if xtr.ndim == 2:
            xtr, xte = xtr[:, None], xte[:, None]
        # z-score with training statistics, pad length to a multiple of 2**len(channels)
        mu, sd = xtr.mean(), xtr.std() + 1e-8
        # kept so preprocessing() can publish them: these two numbers are the whole normalisation,
        # and without them the exported model cannot be fed correctly by anyone but us.
        self.norm_mean, self.norm_std = float(mu), float(sd)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        m = 2 ** len(c["channels"])
        pad = (-xtr.shape[-1]) % m
        if pad:
            xtr = np.pad(xtr, ((0, 0), (0, 0), (0, pad)), mode="edge")
            xte = np.pad(xte, ((0, 0), (0, 0), (0, pad)), mode="edge")
        rng = np.random.default_rng(c["seed"])
        if c["train_subset"]:
            xtr = xtr[rng.permutation(len(xtr))[: c["train_subset"]]]
        if c["test_subset"]:
            idx = rng.permutation(len(xte))[: c["test_subset"]]
            xte, yte = xte[idx], yte[idx]
        self.in_ch, self.seq_len = xtr.shape[1], xtr.shape[2]
        xtr_t, xte_t = torch.from_numpy(xtr).float(), torch.from_numpy(xte).float()
        calib = xtr_t[rng.permutation(len(xtr_t))[: c["calib_size"]]]
        self._data = {
            "train": DataLoader(TensorDataset(xtr_t), batch_size=c["batch_size"], shuffle=True, drop_last=True),
            "val": DataLoader(TensorDataset(xte_t, torch.from_numpy(yte)), batch_size=256),
            "calib": DataLoader(TensorDataset(calib), batch_size=32),
        }
        log.info("anomaly data: train %s (normal only), test %s (%.1f%% anomalies)",
                 tuple(xtr.shape), tuple(xte.shape), 100 * yte.mean())
        return self._data

    # ------------------------------------------------------------------ model
    def build_model(self) -> nn.Module:
        self.load_data()
        torch.manual_seed(self.cfg["seed"])
        return ConvAE1d(self.in_ch, tuple(self.cfg["channels"]), self.cfg["kernel"])

    def build_student(self, channels=(16, 32), **_) -> nn.Module:
        self.load_data()
        return ConvAE1d(self.in_ch, tuple(channels), self.cfg["kernel"])

    def ignored_layers(self, model) -> list:
        return [model.out] if hasattr(model, "out") else []

    def example_input(self) -> torch.Tensor:
        return self.load_data()["val"].dataset.tensors[0][:1].clone()

    # ------------------------------------------------------------------ train / eval
    @staticmethod
    def _loss(model, batch, device):
        x = batch[0].to(device, non_blocking=True)
        return F.mse_loss(model(x), x)

    def _train(self, model, epochs, lr, tag, warmup_frac=0.0):
        # adapter_args.eval_every > 0 evaluates the task metric during training, so a curve shows
        # whether the metric is still moving - falling loss alone does not prove it is.
        every = int(self.cfg.get("eval_every", 0) or 0)
        ev = (lambda m: self.evaluate(m)["score"]) if every else None
        return train_loop(model, self.load_data()["train"], self._loss, epochs, lr, self.device,
                          weight_decay=self.cfg["weight_decay"], optimizer="adam", amp=False,
                          warmup_frac=warmup_frac, history=self.history,
                          eval_fn=ev, eval_every=every, tag=tag)

    def train_baseline(self, model):
        return self._train(model, int(self.cfg["train_epochs"]), self.cfg["lr"], "baseline")

    def finetune(self, model, epochs, lr=None):
        return self._train(model, int(epochs), lr if lr is not None else self.cfg["finetune_lr"],
                           "finetune", warmup_frac=float(self.cfg.get("finetune_warmup_frac", 0.0)))

    def distill_loss(self, student_out, teacher_out, batch):
        a = self.cfg["kd_alpha"]
        x = batch[0].to(student_out.device)
        return a * F.mse_loss(student_out, teacher_out) + (1 - a) * F.mse_loss(student_out, x)

    def anomaly_scores(self, model) -> tuple[np.ndarray, np.ndarray]:
        pred = as_predictor(model, self.device)
        scores, labels = [], []
        for x, y in self.load_data()["val"]:
            rec = pred(x)
            scores.append(((x.numpy() - rec) ** 2).mean(axis=(1, 2)))
            labels.append(y.numpy())
        return np.concatenate(scores), np.concatenate(labels)

    def evaluate(self, model) -> dict:
        from sklearn.metrics import roc_auc_score

        s, y = self.anomaly_scores(model)
        return {"metric_name": "auc", "score": float(roc_auc_score(y, s))}

    def preprocessing(self) -> dict:
        """Mirrors load_data(). The z-score constants come from the TRAINING split, so they are
        fixed properties of the deliverable, not something to recompute on new data."""
        self.load_data()      # populates norm_mean / norm_std / in_ch / seq_len
        m = 2 ** len(self.cfg["channels"])
        return {"layout": "NCL", "color": None, "dtype": "float32", "scale": None,
                "mean": self.norm_mean, "std": self.norm_std,
                "resize": {"to": self.seq_len, "mode": "edge-pad",
                           "note": f"right-pad the sequence with its edge value to a multiple of {m}"},
                "steps": [f"z-score with the TRAINING statistics mean={self.norm_mean:.6g}, "
                          f"std={self.norm_std:.6g} (scalars, not per-channel)",
                          f"right-pad with the edge value to length {self.seq_len} "
                          f"(a multiple of {m})",
                          f"arrange as NCL with {self.in_ch} channel(s)"],
                "output_note": "the model reconstructs its input; the anomaly score is the "
                               "per-sample mean squared error between output and input"}

    def describe(self) -> dict:
        d = self.load_data()
        return {"task": self.name, "quick": self.quick, "source": self.cfg["source"],
                "channels": self.cfg["channels"], "in_ch": self.in_ch, "seq_len": self.seq_len,
                "n_train": len(d["train"].dataset), "n_test": len(d["val"].dataset)}
