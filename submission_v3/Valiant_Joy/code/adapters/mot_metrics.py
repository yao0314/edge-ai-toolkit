"""CLEAR MOT metrics (MOTA / MOTP / IDF1) for the tracking adapter.

Implemented here rather than adding `motmetrics` / `TrackEval`, because the requirements file is
pinned and the toolkit has to run offline after the first download. These are *metric definitions*,
not algorithms - the formulas are the published CLEAR MOT (Bernardin & Stiefelhagen 2008) and IDF1
(Ristani et al. 2016) definitions, and `test_mot_metrics.py` checks them against hand-computed
examples. Hungarian matching comes from scipy, which is already a dependency.

Definitions, so the numbers in a report can be read without guessing:
  MOTA = 1 - (FN + FP + IDSW) / GT      (can go negative; 1.0 is perfect)
  MOTP = mean IoU of matched pairs      (localisation quality of the matches only)
  IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN), where identities are matched ONCE per sequence, so a
         track that keeps swapping IDs scores badly even when every frame has a box somewhere.

Why both MOTA and IDF1: MOTA is dominated by detection errors (FN/FP) and counts an ID switch once,
at the frame it happens. IDF1 measures whether an identity is preserved *over its whole life*. A
compressed detector that misses a frame here and there breaks tracks, which barely moves MOTA but
hurts IDF1 - so the pair tells you whether compression cost detections or cost identities.
"""
from __future__ import annotations

import numpy as np

__all__ = ["iou_matrix", "clear_mot", "idf1"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _match(cost_ok: np.ndarray, score: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-score one-to-one matching over allowed pairs only."""
    from scipy.optimize import linear_sum_assignment

    if score.size == 0:
        return []
    # Maximise total SCORE over allowed pairs. The earlier form (big - score, with a large constant
    # for disallowed pairs) makes every allowed pair cheaper than any disallowed one, so the solver
    # maximises the NUMBER of pairs first and only then their quality - which for identity matching
    # can trade one long correct track for several short wrong ones.
    cost = np.where(cost_ok, -score, 0.0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost_ok[r, c]]


def clear_mot(seqs: list[list[dict]], iou_thres: float = 0.5) -> dict:
    """MOTA / MOTP / FN / FP / IDSW over a list of sequences.

    Each sequence is a list of per-frame dicts:
        {"gt_ids": [...], "gt_boxes": Nx4 xyxy, "pred_ids": [...], "pred_boxes": Mx4 xyxy}

    Matching follows CLEAR MOT: a pair from the previous frame is KEPT if it still overlaps above
    the threshold (so a hand-over does not count as a switch), and only the remaining boxes are
    matched freely. An ID switch is counted when a ground-truth identity that was previously
    matched appears matched to a different predicted identity.
    """
    fn = fp = idsw = matches = 0
    iou_sum = 0.0
    n_gt = 0
    for frames in seqs:
        last: dict[int, int] = {}          # gt id -> pred id from the most recent frame it matched
        for fr in frames:
            g_ids = list(fr.get("gt_ids", []))
            p_ids = list(fr.get("pred_ids", []))
            g_box = np.asarray(fr.get("gt_boxes", np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
            p_box = np.asarray(fr.get("pred_boxes", np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
            n_gt += len(g_ids)
            iou = iou_matrix(g_box, p_box)
            ok = iou >= iou_thres

            # 1. keep last frame's pairs that still overlap
            pairs, used_g, used_p = [], set(), set()
            p_index = {pid: i for i, pid in enumerate(p_ids)}
            for gi, gid in enumerate(g_ids):
                pid = last.get(gid)
                if pid is None:
                    continue
                pi = p_index.get(pid)
                # `pi not in used_p`: two ground-truth ids can carry the SAME remembered pred id
                # (one object leaves, another inherits the track, then both reappear). Without this
                # guard both claim the single box, which produced MOTA 1.25 and FP -1 - impossible
                # values, since MOTA is capped at 1 and FP counts unmatched predictions.
                if pi is not None and pi not in used_p and ok[gi, pi]:
                    pairs.append((gi, pi))
                    used_g.add(gi)
                    used_p.add(pi)

            # 2. match whatever is left
            rem_g = [i for i in range(len(g_ids)) if i not in used_g]
            rem_p = [i for i in range(len(p_ids)) if i not in used_p]
            if rem_g and rem_p:
                sub_ok, sub_iou = ok[np.ix_(rem_g, rem_p)], iou[np.ix_(rem_g, rem_p)]
                for r, c in _match(sub_ok, sub_iou):
                    pairs.append((rem_g[r], rem_p[c]))

            for gi, pi in pairs:
                gid, pid = g_ids[gi], p_ids[pi]
                if gid in last and last[gid] != pid:
                    idsw += 1
                last[gid] = pid
                iou_sum += float(iou[gi, pi])
            matches += len(pairs)
            fn += len(g_ids) - len(pairs)
            fp += len(p_ids) - len(pairs)
    # No ground truth means MOTA is undefined, NOT perfect. Returning 1.0 there would make a run
    # whose validation data failed to load look like the best result in the table.
    return {"mota": (1.0 - (fn + fp + idsw) / n_gt) if n_gt else float("nan"),
            "motp": iou_sum / matches if matches else 0.0,
            "fn": fn, "fp": fp, "idsw": idsw, "matches": matches, "n_gt": n_gt}


def idf1(seqs: list[list[dict]], iou_thres: float = 0.5) -> dict:
    """IDF1 / IDP / IDR: identities matched ONCE per sequence, then counted over all frames.

    This is what separates "found a box" from "kept the same object". A tracker that re-IDs the same
    person every few frames can have almost the same MOTA and a much worse IDF1.
    """
    idtp = idfp = idfn = 0
    for frames in seqs:
        gt_ids = sorted({i for fr in frames for i in fr.get("gt_ids", [])})
        pr_ids = sorted({i for fr in frames for i in fr.get("pred_ids", [])})
        gi_of = {g: i for i, g in enumerate(gt_ids)}
        pi_of = {p: i for i, p in enumerate(pr_ids)}
        # overlap[g, p] = how many frames gt identity g and pred identity p are the same object
        overlap = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float64)
        n_g = np.zeros(len(gt_ids))
        n_p = np.zeros(len(pr_ids))
        for fr in frames:
            g_ids, p_ids = list(fr.get("gt_ids", [])), list(fr.get("pred_ids", []))
            # count each identity once per frame: a gt.txt with a duplicated id in one frame (the
            # repo's own smoke data does this) otherwise inflates n_g above the overlap count and
            # drives IDFP negative / IDP above 1
            for g in set(g_ids):
                n_g[gi_of[g]] += 1
            for p in set(p_ids):
                n_p[pi_of[p]] += 1
            iou = iou_matrix(np.asarray(fr.get("gt_boxes", np.zeros((0, 4)))).reshape(-1, 4),
                             np.asarray(fr.get("pred_boxes", np.zeros((0, 4)))).reshape(-1, 4))
            # count each (gt identity, pred identity) at most ONCE per frame, matching how n_g/n_p
            # are counted. Adding one per BOX pair let a duplicated id contribute twice, pushing
            # idtp above the identity's own frame count and idfp/idfn negative.
            hit = {(gi_of[g], pi_of[p])
                   for gi, g in enumerate(g_ids) for pi, p in enumerate(p_ids)
                   if iou[gi, pi] >= iou_thres}
            for gk, pk in hit:
                overlap[gk, pk] += 1
        pairs = _match(overlap > 0, overlap)
        tp = sum(overlap[g, p] for g, p in pairs)
        idtp += tp
        idfp += n_p.sum() - tp
        idfn += n_g.sum() - tp
    denom = 2 * idtp + idfp + idfn
    return {"idf1": (2 * idtp / denom) if denom else float("nan"),      # nothing to score != perfect
            "idp": (idtp / (idtp + idfp)) if idtp + idfp else float("nan"),
            "idr": (idtp / (idtp + idfn)) if idtp + idfn else float("nan"),
            "idtp": int(idtp), "idfp": int(idfp), "idfn": int(idfn)}
