"""Object detection adapter: ultralytics YOLO (v8 / 11) on a YOLO-format dataset, metric = COCO mAP@[.5:.95].

Data: any YOLO-format folder (images/<split>/*.jpg + labels/<split>/*.txt with
"cls cx cy w h" normalised). Default: coco128 (auto-download, 128 COCO images).
coco128 has a single split, so it is divided into train/val by `val_fraction`.

Export / deployment contract (what goes into ONNX):
  input  : (B, 3, S, S) RGB in [0, 1], letterboxed to S = img_size
  output : (B, 4 + nc, A) decoded boxes (cx, cy, w, h in input pixels) + class scores
NMS is NOT in the graph: it runs here in evaluate() (torchvision batched_nms), as it would in
the deployed app. That keeps the ONNX graph plain conv ops -> clean INT8 quantization.
The Detect head is never pruned (ignored_layers), and its box-decoding tail is kept in FP32
automatically by core.quantize's output-tail rule (quantize.auto_keep_fp32, on by default) -
without it one INT8 scale is shared by pixel boxes and 0..1 scores and mAP halves.
"""
from __future__ import annotations

import contextlib
import io
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from core.adapter import TaskAdapter
from core.config import ROOT
from core.runtime import as_predictor

from .common import train_loop
from .ultralytics_compat import make_prunable

log = logging.getLogger(__name__)

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

DEFAULTS = {
    "data_root": "data/coco128",
    "images": "images/train2017",
    "labels": "labels/train2017",
    "val_images": None,          # separate val split (YOLO format); None -> split by val_fraction
    "val_labels": None,
    "val_fraction": 0.5,
    "nc": 80,
    "weights": "weights/detection/yolov8n.pt",  # pretrained checkpoint (auto-download if missing)
    "model_cfg": None,           # e.g. "yolo11n.yaml" to build an architecture instead of loading weights
    "img_size": 320,
    "batch_size": 16,
    "train_epochs": 0,           # the COCO-pretrained weights are the baseline
    "lr": 1e-3,
    # 2e-4 degraded even the UNPRUNED yolov8n on 64 images (0.407 -> 0.28 after 10 epochs, 4 seeds);
    # 2e-5 keeps it (0.403 / 0.380 after 3 / 10 epochs).
    "finetune_lr": 2e-5,
    "weight_decay": 5e-4,
    # pruning: never cut these top-level blocks (model.model[i]). The 16/32-channel stem/early C2f
    # layers of YOLOv8n are where any channel removal is catastrophic.
    "prune_protect_layers": [0, 1, 2, 3, 4],
    # BN re-calibration after pruning runs at the PRETRAINING resolution: yolov8n.pt's BN statistics
    # belong to 640 px; re-estimating them at 320 px alone costs ~0.12 mAP on the unpruned model.
    "bn_recal_img_size": 640,
    "conf_thres": 0.001,         # mAP evaluation settings (ultralytics val defaults)
    "iou_thres": 0.7,
    "max_det": 300,
    "calib_size": 64,
    "seed": 0,
}


# ----------------------------------------------------------------------------- data
def letterbox(img: Image.Image, size: int):
    """Resize keeping aspect ratio and pad to size x size (pad value 114).

    Returns (array, sx, sy, padx, pady, w0, h0) with the ACTUAL scale factors of the rounded resize,
    so labels map exactly onto the pixels (using the float ratio misplaced boxes by up to 0.5 px).
    """
    w0, h0 = img.size
    r = size / max(w0, h0)
    nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (padx, pady))
    return np.asarray(canvas), nw / w0, nh / h0, padx, pady, w0, h0


class YoloFolder(Dataset):
    """Returns (image float CHW in [0,1], targets [n, 5] = cls, cx, cy, w, h normalised to the letterbox)."""

    def __init__(self, files: list[Path], labels_dir: Path, images_dir: Path, size: int, augment: bool = False):
        self.files, self.labels_dir, self.images_dir = files, labels_dir, images_dir
        self.size, self.augment = size, augment
        self._cache: dict = {}

    def __len__(self):
        return len(self.files)

    def _load(self, i):
        if i not in self._cache:
            f = self.files[i]
            arr, sx, sy, px, py, w0, h0 = letterbox(Image.open(f), self.size)
            lab = self.labels_dir / f.relative_to(self.images_dir).with_suffix(".txt")
            t = np.zeros((0, 5), np.float32)
            if lab.exists():
                rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()]
                if rows:
                    t = np.array(rows, np.float32)[:, :5]
                    t[:, 1] = (t[:, 1] * w0 * sx + px) / self.size
                    t[:, 2] = (t[:, 2] * h0 * sy + py) / self.size
                    t[:, 3] = t[:, 3] * w0 * sx / self.size
                    t[:, 4] = t[:, 4] * h0 * sy / self.size
            self._cache[i] = (arr, t)
        return self._cache[i]

    def __getitem__(self, i):
        arr, t = self._load(i)
        arr, t = arr.copy(), t.copy()
        if self.augment and random.random() < 0.5:  # horizontal flip
            arr = arr[:, ::-1].copy()
            t[:, 1] = 1.0 - t[:, 1]
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return x, torch.from_numpy(t)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = [torch.cat([torch.full((len(t), 1), i, dtype=torch.float32), t], 1) for i, (_, t) in enumerate(batch)]
    return imgs, torch.cat(tgts) if tgts else torch.zeros(0, 6)   # targets: [n, 6] = batch_idx, cls, cx, cy, w, h


# ----------------------------------------------------------------------------- adapter
class DetectionAdapter(TaskAdapter):
    name = "detection"

    def __init__(self, cfg=None, quick=False, device=None):
        super().__init__({**DEFAULTS, **(cfg or {})}, quick, device)
        self._data = None

    def _path(self, p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    def _ensure_data(self):
        root = self._path(self.cfg["data_root"])
        if not (root / self.cfg["images"]).exists() and root.name == "coco128":
            from ultralytics.utils.downloads import safe_download

            safe_download(COCO128_URL, dir=root.parent, unzip=True, delete=False)
        return root

    def load_data(self) -> dict:
        if self._data is not None:
            return self._data
        c, root = self.cfg, self._ensure_data()
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_dir, lab_dir = root / c["images"], root / c["labels"]
        files = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() in exts)
        if c["val_images"]:
            vimg, vlab = root / c["val_images"], root / c["val_labels"]
            val_files = sorted(f for f in vimg.rglob("*") if f.suffix.lower() in exts)
            train = YoloFolder(files, lab_dir, img_dir, c["img_size"], augment=True)
            val = YoloFolder(val_files, vlab, vimg, c["img_size"])
        else:
            rng = random.Random(c["seed"])
            idx = list(range(len(files)))
            rng.shuffle(idx)
            n_val = int(len(files) * c["val_fraction"])
            val = YoloFolder([files[i] for i in sorted(idx[:n_val])], lab_dir, img_dir, c["img_size"])
            train = YoloFolder([files[i] for i in sorted(idx[n_val:])], lab_dir, img_dir, c["img_size"], augment=True)
        calib = YoloFolder(train.files[: c["calib_size"]], train.labels_dir, train.images_dir, c["img_size"])
        self._data = {
            "train": DataLoader(train, batch_size=c["batch_size"], shuffle=True, collate_fn=collate, drop_last=False),
            "val": DataLoader(val, batch_size=c["batch_size"], shuffle=False, collate_fn=collate),
            "calib": DataLoader(calib, batch_size=8, shuffle=False, collate_fn=collate),
        }
        log.info("detection data: %d train / %d val images at %d px", len(train), len(val), c["img_size"])
        return self._data

    # ------------------------------------------------------------------ model
    def build_model(self) -> nn.Module:
        c = self.cfg
        if c["model_cfg"]:
            from ultralytics.nn.tasks import DetectionModel

            model = DetectionModel(c["model_cfg"], nc=c["nc"], verbose=False)
            if c["weights"] and self._path(c["weights"]).exists():
                model.load(torch.load(self._path(c["weights"]), map_location="cpu", weights_only=False))
            return model
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.downloads import attempt_download_asset

        w = self._path(c["weights"])
        w.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(w))
        model, _ = load_checkpoint(str(w), device="cpu")
        if model.model[-1].nc != c["nc"]:
            raise ValueError(f"checkpoint has nc={model.model[-1].nc}, dataset nc={c['nc']}: "
                             "set model_cfg (e.g. yolov8n.yaml) to rebuild the head and transfer weights")
        return model

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """SplitC2f for pruning, export-mode head, training hyper-parameters for the ultralytics loss."""
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG

        model = make_prunable(model.float())
        model.model[-1].export = True     # eval forward -> single decoded tensor (no (y, x) tuple)
        model.args = get_cfg(DEFAULT_CFG)  # box/cls/dfl gains used by v8DetectionLoss
        for n, p in model.named_parameters():
            p.requires_grad_(".dfl" not in n)  # DFL projection is a fixed arange
        return model.eval()

    def ignored_layers(self, model) -> list:
        protect = [model.model[i] for i in self.cfg["prune_protect_layers"] if i < len(model.model) - 1]
        return [model.model[-1]] + protect  # never prune the detection head or the thin early blocks

    def bn_calibration_loader(self):
        """Train images at the pretraining resolution (see DEFAULTS['bn_recal_img_size'])."""
        tr = self.load_data()["train"].dataset
        ds = YoloFolder(tr.files[: max(self.cfg["calib_size"], 1)], tr.labels_dir, tr.images_dir,
                        int(self.cfg["bn_recal_img_size"] or self.cfg["img_size"]))
        return DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)

    def example_input(self) -> torch.Tensor:
        return self.load_data()["val"].dataset[0][0].unsqueeze(0)

    # ------------------------------------------------------------------ train / eval
    @staticmethod
    def _loss(model, batch, device):
        imgs, t = batch
        b = {"img": imgs.to(device, non_blocking=True), "batch_idx": t[:, 0].to(device),
             "cls": t[:, 1:2].to(device), "bboxes": t[:, 2:6].to(device)}
        loss, _ = model.loss(b)
        return loss.sum() / imgs.shape[0]

    def _train(self, model, epochs, lr, tag, warmup_frac=0.0):
        # adapter_args.eval_every > 0 evaluates the task metric during training, so a curve shows
        # whether the metric is still moving - falling loss alone does not prove it is.
        every = int(self.cfg.get("eval_every", 0) or 0)
        ev = (lambda m: self.evaluate(m)["score"]) if every else None
        model.criterion = None            # rebuilt for the (possibly pruned) model on the right device
        return train_loop(model, self.load_data()["train"], self._loss, epochs, lr, self.device,
                          weight_decay=self.cfg["weight_decay"], optimizer="adam", grad_clip=10.0,
                          warmup_frac=warmup_frac, history=self.history,
                          eval_fn=ev, eval_every=every, tag=tag)

    def train_baseline(self, model):
        return self._train(model, int(self.cfg["train_epochs"]), self.cfg["lr"], "baseline")

    def finetune(self, model, epochs, lr=None):
        return self._train(model, int(epochs), lr if lr is not None else self.cfg["finetune_lr"],
                           "finetune", warmup_frac=float(self.cfg.get("finetune_warmup_frac", 0.0)))

    def distill_loss(self, student_out, teacher_out, batch):
        raise NotImplementedError("KD for detection is not implemented; use sweep.finetune_mode: plain")

    def postprocess(self, pred: np.ndarray) -> list[np.ndarray]:
        """(B, 4+nc, A) -> per image [k, 6] = x1, y1, x2, y2, score, cls after class-aware NMS."""
        import torchvision

        c, out = self.cfg, []
        p = torch.from_numpy(np.asarray(pred, np.float32)).permute(0, 2, 1)  # B, A, 4+nc
        for pi in p:
            scores, cls = pi[:, 4:].max(1)
            keep = scores > c["conf_thres"]
            box, scores, cls = pi[keep, :4], scores[keep], cls[keep]
            xyxy = torch.cat([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], 1)
            k = torchvision.ops.batched_nms(xyxy, scores, cls, c["iou_thres"])[: c["max_det"]]
            out.append(torch.cat([xyxy[k], scores[k, None], cls[k, None].float()], 1).numpy())
        return out

    def evaluate(self, model) -> dict:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        pred_fn, size = as_predictor(model, self.device), self.cfg["img_size"]
        self.last_map50 = 0.0
        images, anns, dets, img_id = [], [], [], 0
        for imgs, t in self.load_data()["val"]:
            out = pred_fn(imgs)
            out = out[0] if isinstance(out, list) else out
            for bi, det in enumerate(self.postprocess(out)):
                img_id += 1
                images.append({"id": img_id, "width": size, "height": size})
                for row in t[t[:, 0] == bi]:
                    _, cl, cx, cy, w, h = row.tolist()
                    bw, bh = w * size, h * size
                    anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": int(cl), "iscrowd": 0,
                                 "bbox": [cx * size - bw / 2, cy * size - bh / 2, bw, bh], "area": bw * bh})
                for x1, y1, x2, y2, s, cl in det:
                    dets.append({"image_id": img_id, "category_id": int(cl), "score": float(s),
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
        if not dets or not anns:
            return {"metric_name": "mAP50-95", "score": 0.0, "map50": 0.0,
                    "map_small": 0.0, "map_large": 0.0}
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns,
                      "categories": [{"id": i} for i in range(self.cfg["nc"])]}
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            ev = COCOeval(gt, gt.loadRes(dets), "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        self.last_map50 = float(ev.stats[1])
        # mAP50 alongside the ranking metric, as reported detail. mAP50-95 averages IoU thresholds
        # 0.50..0.95, so a box that shifts slightly drops out of the strict ones while still counting
        # at 0.50. Comparing the two separates "the model stopped FINDING objects" from "the model
        # still finds them but localises them less precisely" - which is the difference between a
        # compression step being harmless and being fatal.
        return {"metric_name": "mAP50-95", "score": float(ev.stats[0]),
                "map50": float(ev.stats[1]), "map_small": float(ev.stats[3]),
                "map_large": float(ev.stats[5])}

    def preprocessing(self) -> dict:
        """Mirrors letterbox() + YoloFolder.__getitem__. No mean/std: YOLO consumes [0,1] directly."""
        s = int(self.cfg["img_size"])
        return {"layout": "NCHW", "color": "RGB", "dtype": "float32", "scale": 255.0,
                "mean": None, "std": None,
                "resize": {"to": s, "mode": "letterbox", "interpolation": "bilinear",
                           "pad_value": 114, "keep_aspect_ratio": True,
                           "note": "scale by s/max(w,h), then centre-pad to s x s with 114"},
                "steps": [f"resize keeping aspect ratio so the long side is {s} (bilinear)",
                          f"centre-pad to {s}x{s} with the constant value 114",
                          "convert to float32 and divide by 255", "arrange as NCHW"],
                "output_note": "raw model output is [1, 4+nc, anchors] in letterbox pixel coords; "
                               "undo the pad and scale to map boxes back to the original image"}

    def describe(self) -> dict:
        d = self.load_data()
        return {"task": self.name, "quick": self.quick, "weights": self.cfg["weights"],
                "model_cfg": self.cfg["model_cfg"], "img_size": self.cfg["img_size"],
                "n_train": len(d["train"].dataset), "n_val": len(d["val"].dataset)}
