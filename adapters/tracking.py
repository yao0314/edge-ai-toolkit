"""Multi-object tracking adapter: a compressed detector + an ultralytics tracker, scored on MOT data.

The organisers' own first example workload is "real-time multi-object tracking", and this is the
cheapest honest way to cover it: everything that gets COMPRESSED is the detector, which
DetectionAdapter already handles, so this subclass only changes the two things that actually differ.

  load_data()   frames in order, grouped per sequence (detection shuffles a flat folder)
  evaluate()    run the tracker over each sequence, then score

Ranking stays on mAP, deliberately. MOTA/IDF1 are reported as extra columns instead:
  * mAP measures only what we changed - the detector.
  * MOTA/IDF1 also depend on the TRACKER's hyper-parameters (association threshold, track age),
    which we do not compress, so two compressed models can swap places on tracker tuning noise.
To rank on MOTA instead, set BOTH `score.groups.accuracy.cols: [mota]` AND
`score.groups.accuracy.higher_is_better: true`, and re-state the other three groups: score.groups is
deep-merged, so replacing one group's cols without its flag leaves the default "lower is better"
(which inverts the ranking), and replacing `groups` wholesale drops compression/resource/energy from
the score entirely. See configs/tracking_smoke.yaml for the full block.

Data layout (MOT Challenge):
  <data_root>/<seq>/img1/000001.jpg ...
  <data_root>/<seq>/gt/gt.txt        frame,id,x,y,w,h,conf,cls,visibility   (1-based frame and x,y)
`gt.txt` carries the `id` column that YOLO-format labels do not, and identity is the whole point of
MOTA/IDF1, so a YOLO folder cannot be used for this metric.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from core.runtime import as_predictor

from .detection import DetectionAdapter, collate, letterbox
from .mot_metrics import clear_mot, idf1

log = logging.getLogger(__name__)

DEFAULTS = {
    "data_root": "data/mot",
    "seq_glob": "*",             # which sub-directories are sequences
    "images_sub": "img1",
    "gt_sub": "gt",
    "max_frames": None,          # per sequence, for a quick pass
    "max_sequences": None,
    "tracker": "bytetrack.yaml",  # ultralytics: bytetrack.yaml | botsort.yaml | ocsort.yaml | ...
    "track_conf": 0.25,          # tracking needs a real confidence gate; mAP evaluation does not
    "mot_iou_thres": 0.5,        # IoU at which a prediction counts as the same object
    "gt_classes": None,          # MOT gt class ids to keep (None = all); MOT17 uses 1 = pedestrian
    "gt_min_visibility": 0.0,
    # Which DETECTOR classes may become tracks. A MOT ground truth usually contains one class
    # (pedestrians), so tracking all 80 COCO classes counts every chair and car as a false positive
    # in MOTA/IDF1 while mAP - which matches per class - never sees the problem. None = keep all.
    "track_classes": None,
}


class MotSequence(Dataset):
    """One sequence, frames in file order. Returns (image CHW in [0,1], frame_index)."""

    def __init__(self, files: list[Path], size: int):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        arr, sx, sy, px, py, w0, h0 = letterbox(Image.open(self.files[i]), self.size)
        # .copy(): letterbox returns a read-only view, and torch.from_numpy on a non-writable array
        # gives a tensor whose writes are undefined behaviour
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        # the letterbox transform travels with the frame so boxes can be mapped back to gt pixels
        return x, i, torch.tensor([sx, sy, px, py, w0, h0], dtype=torch.float32)


class MotDetectionView(Dataset):
    """The same frames, but yielding what DetectionAdapter.evaluate() consumes: (image, targets)
    with targets = [n, 5] of cls, cx, cy, w, h normalised to the letterbox.

    This exists so the inherited mAP path runs untouched. MOT gt.txt gives absolute pixels in the
    ORIGINAL frame, so each box goes through the same letterbox scale/pad as the image.
    """

    def __init__(self, files: list[Path], gt: dict[int, list], size: int):
        self.files, self.gt, self.size = files, gt, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        arr, sx, sy, px, py, w0, h0 = letterbox(Image.open(self.files[i]), self.size)
        rows = []
        for _tid, x1, y1, x2, y2 in self.gt.get(i, []):
            cx, cy = (x1 + x2) / 2 * sx + px, (y1 + y2) / 2 * sy + py
            rows.append([0.0, cx / self.size, cy / self.size,
                         (x2 - x1) * sx / self.size, (y2 - y1) * sy / self.size])
        t = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        return x, torch.from_numpy(t)


def read_mot_gt(path: Path, classes=None, min_vis: float = 0.0) -> dict[int, list]:
    """MOT gt.txt -> {frame_index (0-based): [(id, x1, y1, x2, y2), ...]}."""
    out: dict[int, list] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 6:
            continue
        f, tid, x, y, w, h = (float(v) for v in parts[:6])
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        cls = int(float(parts[7])) if len(parts) > 7 else 1
        vis = float(parts[8]) if len(parts) > 8 else 1.0
        if conf == 0 or vis < min_vis or (classes and cls not in classes):
            continue        # conf 0 in MOT gt marks an ignored box
        out[int(f) - 1].append((int(tid), x, y, x + w, y + h))
    return out


class TrackingAdapter(DetectionAdapter):
    name = "tracking"

    def __init__(self, cfg=None, quick=False, device=None):
        super().__init__({**DEFAULTS, **(cfg or {})}, quick, device)
        self._seqs = None
        self.last_mot: dict = {}

    # ------------------------------------------------------------------ data
    def _sequences(self) -> list[tuple[str, list[Path], dict]]:
        if self._seqs is not None:
            return self._seqs
        c = self.cfg
        root = self._path(c["data_root"])
        if not root.is_dir():
            raise FileNotFoundError(
                f"{root} not found. Expected MOT layout <data_root>/<seq>/{c['images_sub']}/*.jpg "
                f"plus <seq>/{c['gt_sub']}/gt.txt - the MOT Challenge layout)")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        seqs = []
        for d in sorted(p for p in root.glob(c["seq_glob"]) if p.is_dir()):
            imgs = sorted(f for f in (d / c["images_sub"]).glob("*") if f.suffix.lower() in exts)
            if not imgs:
                continue
            if c["max_frames"]:
                imgs = imgs[: int(c["max_frames"])]
            classes = c["gt_classes"]
            if classes is not None and not isinstance(classes, (list, tuple, set)):
                classes = [classes]          # a bare int would iterate as nothing / crash
            gt = read_mot_gt(d / c["gt_sub"] / "gt.txt", set(map(int, classes)) if classes else None,
                             float(c["gt_min_visibility"]))
            seqs.append((d.name, imgs, gt))
        if c["max_sequences"]:
            seqs = seqs[: int(c["max_sequences"])]
        if not seqs:
            raise FileNotFoundError(f"no sequences with images under {root}")
        log.info("tracking data: %d sequence(s), %d frames, %d gt boxes (in range)",
                 len(seqs), sum(len(i) for _, i, _ in seqs),
                 sum(len(v) for _, imgs, g in seqs for fi, v in g.items() if 0 <= fi < len(imgs)))
        self._seqs = seqs
        return seqs

    def load_data(self) -> dict:
        """train/calib come from the frames themselves; val is per-sequence and never shuffled."""
        if self._data is not None:
            return self._data
        c = self.cfg
        seqs = self._sequences()
        size = c["img_size"]
        frames = [f for _, imgs, _ in seqs for f in imgs]
        # val must look like a DETECTION set so the inherited mAP path works unchanged; the
        # per-sequence loaders below are the tracking view. Frame indices are per sequence, so the
        # gt dicts are offset as the flat list is concatenated.
        flat_gt, off = {}, 0
        for name, imgs, gt in seqs:
            for fi, boxes in gt.items():
                if 0 <= fi < len(imgs):
                    flat_gt[off + fi] = boxes
                elif fi < 0:
                    # read_mot_gt subtracts 1 for MOT's 1-based frames; a 0-based file lands at -1,
                    # which would index the PREVIOUS sequence's last frame and overwrite its labels
                    log.warning("%s: gt.txt frame index %d is out of range - is the file 0-based? "
                                "MOT gt.txt frames are 1-based", name, fi + 1)
            off += len(imgs)
        # A tracking dataset has no YOLO-format labels, so there is nothing to fine-tune ON here:
        # train/calib are images only (calibration is label-free, which is all quantization needs).
        self._data = {
            "train": DataLoader(MotDetectionView(frames, flat_gt, size), batch_size=c["batch_size"],
                                shuffle=False, collate_fn=collate),
            "val": DataLoader(MotDetectionView(frames, flat_gt, size), batch_size=c["batch_size"],
                              shuffle=False, collate_fn=collate),
            "calib": DataLoader(MotDetectionView(frames[: int(c["calib_size"])], flat_gt, size),
                                batch_size=8, shuffle=False, collate_fn=collate),
            # one loader per sequence: the tracker's state must not cross a sequence boundary
            "sequences": [(name, DataLoader(MotSequence(imgs, size), batch_size=c["batch_size"],
                                            shuffle=False), gt) for name, imgs, gt in seqs],
        }
        return self._data

    def bn_calibration_loader(self):
        """BN re-estimation at the detector's PRETRAINING resolution, like the detection adapter.

        Overriding this without honouring bn_recal_img_size re-estimated BN at img_size (320) for
        weights whose statistics belong to 640, which costs mAP on the unpruned model before any
        compression has happened.
        """
        size = int(self.cfg.get("bn_recal_img_size") or self.cfg["img_size"])
        if size == self.cfg["img_size"]:
            return self.load_data()["calib"]
        seqs = self._sequences()
        frames = [f for _, imgs, _ in seqs for f in imgs][: int(self.cfg["calib_size"])]
        return DataLoader(MotDetectionView(frames, {}, size), batch_size=8, shuffle=False,
                          collate_fn=collate)

    def example_input(self) -> torch.Tensor:
        return self.load_data()["calib"].dataset[0][0].unsqueeze(0)

    def finetune(self, model, epochs, lr=None):
        """No identity-free box labels here, so there is nothing to fine-tune against.

        Quantization and measurement are unaffected (calibration is label-free); pruning simply does
        not get its recovery step, which the pruning study says is where detection pruning fails
        anyway. Point `adapter_args.data_root` at a YOLO-format set and use the detection adapter if
        pruning recovery is wanted.
        """
        if epochs:
            log.warning("tracking adapter has no box-label training set: skipping %d fine-tune epoch(s). "
                        "Prune with the detection adapter if recovery training is needed.", epochs)
        return model

    def train_baseline(self, model):
        return model            # the pretrained detector IS the baseline

    # ------------------------------------------------------------------ evaluate
    def _track_sequence(self, pred_fn, loader, tracker):
        """Detections -> tracker -> [{frame: [(track_id, x1, y1, x2, y2)]}] in ORIGINAL pixels."""
        out: dict[int, list] = defaultdict(list)
        for imgs, idxs, tf in loader:
            raw = pred_fn(imgs)
            raw = raw[0] if isinstance(raw, list) else raw
            for bi, det in enumerate(self.postprocess(raw)):
                fi = int(idxs[bi])
                sx, sy, px, py, w0, h0 = (float(v) for v in tf[bi])
                if len(det):
                    keep = det[:, 4] >= float(self.cfg["track_conf"])
                    cls_keep = self.cfg.get("track_classes")
                    if cls_keep is not None:
                        keep &= np.isin(det[:, 5].astype(int), np.asarray(list(cls_keep), dtype=int))
                    det = det[keep]
                tracks = tracker.update(_as_ultralytics_results(det, self.cfg["img_size"]), None)
                for row in tracks:
                    x1, y1, x2, y2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
                    # undo the letterbox so boxes are comparable with gt.txt pixels
                    out[fi].append((tid, (x1 - px) / max(sx, 1e-9), (y1 - py) / max(sy, 1e-9),
                                    (x2 - px) / max(sx, 1e-9), (y2 - py) / max(sy, 1e-9)))
        return out

    def evaluate(self, model) -> dict:
        """mAP (the ranking metric) plus MOTA / IDF1 / MOTP as reported detail."""
        base = super().evaluate(model)          # mAP over the same frames, reusing the detection path
        try:
            seqs_for_metric = self._run_tracker(model)
        except Exception as e:                  # tracking detail must never fail the whole variant
            log.warning("tracking metrics unavailable: %s: %s", type(e).__name__, e)
            self.last_mot = {}
            return base
        c = clear_mot(seqs_for_metric, float(self.cfg["mot_iou_thres"]))
        f = idf1(seqs_for_metric, float(self.cfg["mot_iou_thres"]))
        self.last_mot = {**{k: c[k] for k in ("mota", "motp", "fn", "fp", "idsw", "n_gt")},
                         **{k: f[k] for k in ("idf1", "idp", "idr")}}
        log.info("tracking: mAP %.4f | MOTA %.4f MOTP %.4f IDF1 %.4f (FN %d FP %d IDSW %d)",
                 base["score"], c["mota"], c["motp"], f["idf1"], c["fn"], c["fp"], c["idsw"])
        return {**base, **self.last_mot}

    def _run_tracker(self, model):
        # Use ultralytics' own TRACKER_MAP rather than a hardcoded pair, so every tracker it ships
        # (bytetrack, botsort, ocsort, deepocsort, fasttrack, tracktrack) is selectable by config
        # and the set cannot go stale when ultralytics adds one.
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        args = IterableSimpleNamespace(**YAML.load(check_yaml(self.cfg["tracker"])))
        kind = TRACKER_MAP.get(getattr(args, "tracker_type", None))
        if kind is None:
            raise ValueError(f"unsupported tracker_type {getattr(args, 'tracker_type', None)!r}; "
                             f"ultralytics offers {sorted(TRACKER_MAP)}")
        pred_fn = as_predictor(model, self.device)
        out = []
        for name, loader, gt in self.load_data()["sequences"]:
            tracks = self._track_sequence(pred_fn, loader, kind(args))
            frames = []
            for fi in range(len(loader.dataset)):
                g, p = gt.get(fi, []), tracks.get(fi, [])
                frames.append({"gt_ids": [t[0] for t in g],
                               "gt_boxes": np.array([t[1:] for t in g], dtype=np.float64).reshape(-1, 4),
                               "pred_ids": [t[0] for t in p],
                               "pred_boxes": np.array([t[1:] for t in p], dtype=np.float64).reshape(-1, 4)})
            out.append(frames)
            log.debug("sequence %s: %d frames tracked", name, len(frames))
        return out

    def describe(self) -> dict:
        seqs = self._sequences()
        # count only gt that survives max_frames, or the log advertises labels no evaluation sees
        n_gt = sum(len(v) for _, imgs, g in seqs for fi, v in g.items() if 0 <= fi < len(imgs))
        return {"task": self.name, "quick": self.quick, "data_root": self.cfg["data_root"],
                "tracker": self.cfg["tracker"], "img_size": self.cfg["img_size"],
                "n_sequences": len(seqs), "n_val": sum(len(i) for _, i, _ in seqs),
                "n_gt_boxes": n_gt,
                "ranking_metric": "mAP50-95 (MOTA/IDF1 reported, not ranked)"}


def _as_ultralytics_results(det: np.ndarray, size: int):
    """Wrap an (N, 6) xyxy+conf+cls array in ultralytics' own Boxes.

    Hand-rolling this wrapper was a mistake: the trackers boolean-index the results object and call
    helpers on it, so the surface they need is larger than it looks and would drift with every
    ultralytics release. Boxes IS what the real pipeline passes them.
    """
    from ultralytics.engine.results import Boxes

    det = np.asarray(det, dtype=np.float32).reshape(-1, 6)
    return Boxes(torch.from_numpy(det), (size, size))
