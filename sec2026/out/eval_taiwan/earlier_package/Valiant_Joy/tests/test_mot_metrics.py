"""MOTA / MOTP / IDF1 against hand-computed examples.

These formulas are the reason we do not need `motmetrics` in a pinned, offline-capable
requirements file - so they have to be checked against cases whose answers are worked out by hand,
not against another implementation.
"""
import numpy as np

from adapters.mot_metrics import clear_mot, idf1, iou_matrix


def box(x, y, w=10, h=10):
    return [x, y, x + w, y + h]


def frame(gt, pred):
    """gt / pred: list of (id, x, y) at a fixed 10x10 size."""
    return {"gt_ids": [i for i, *_ in gt], "gt_boxes": np.array([box(x, y) for _, x, y in gt]).reshape(-1, 4),
            "pred_ids": [i for i, *_ in pred], "pred_boxes": np.array([box(x, y) for _, x, y in pred]).reshape(-1, 4)}


def test_iou_matrix_basics():
    assert iou_matrix(np.array([box(0, 0)]), np.array([box(0, 0)]))[0, 0] == 1.0
    assert iou_matrix(np.array([box(0, 0)]), np.array([box(100, 100)]))[0, 0] == 0.0
    # half-overlap along x: intersection 5x10=50, union 200-50=150
    assert abs(iou_matrix(np.array([box(0, 0)]), np.array([box(5, 0)]))[0, 0] - 50 / 150) < 1e-9
    assert iou_matrix(np.zeros((0, 4)), np.array([box(0, 0)])).shape == (0, 1)


def test_perfect_tracking_scores_one():
    seqs = [[frame([(1, 0, 0), (2, 50, 50)], [(7, 0, 0), (8, 50, 50)]) for _ in range(5)]]
    c = clear_mot(seqs)
    assert c["mota"] == 1.0 and c["fn"] == 0 and c["fp"] == 0 and c["idsw"] == 0
    assert c["motp"] == 1.0 and c["n_gt"] == 10
    assert idf1(seqs)["idf1"] == 1.0          # ids differ from gt ids, but the MAPPING is consistent


def test_misses_and_false_positives_are_counted():
    # 3 frames, 1 gt each. Frame 2 has no prediction (1 FN); frame 3 has an extra box (1 FP).
    seqs = [[frame([(1, 0, 0)], [(7, 0, 0)]),
             frame([(1, 0, 0)], []),
             frame([(1, 0, 0)], [(7, 0, 0), (9, 80, 80)])]]
    c = clear_mot(seqs)
    assert (c["fn"], c["fp"], c["idsw"], c["n_gt"]) == (1, 1, 0, 3)
    assert abs(c["mota"] - (1 - 2 / 3)) < 1e-9


def test_id_switch_is_counted_once_and_hurts_idf1_more_than_mota():
    """The same object, detected in every frame, but the tracker renames it half way.
    MOTA loses exactly one ID switch out of 6; IDF1 loses far more, because identity 7 covers only
    half the object's life and identity 8 the other half."""
    seqs = [[frame([(1, 0, 0)], [(7, 0, 0)]) for _ in range(3)]
            + [frame([(1, 0, 0)], [(8, 0, 0)]) for _ in range(3)]]
    c = clear_mot(seqs)
    assert (c["fn"], c["fp"], c["idsw"]) == (0, 0, 1)
    assert abs(c["mota"] - (1 - 1 / 6)) < 1e-9         # 0.833: barely dented
    f = idf1(seqs)
    assert abs(f["idf1"] - 0.5) < 1e-9                 # 0.5: identity was only half preserved
    assert f["idf1"] < c["mota"]                       # the point of reporting both


def test_handover_within_threshold_is_not_a_switch():
    """A pair that survives from the previous frame is kept even when another box is closer, so a
    box passing nearby does not manufacture an ID switch."""
    seqs = [[frame([(1, 0, 0)], [(7, 0, 0)]),
             frame([(1, 0, 0), (2, 6, 0)], [(7, 0, 0), (9, 6, 0)]),
             frame([(1, 0, 0), (2, 6, 0)], [(7, 0, 0), (9, 6, 0)])]]
    assert clear_mot(seqs)["idsw"] == 0


def test_mota_can_go_negative_and_an_empty_set_is_not_perfect():
    import math

    seqs = [[frame([(1, 0, 0)], [(7, 50, 50), (8, 70, 70), (9, 90, 90)])]]   # 1 FN + 3 FP over 1 gt
    assert clear_mot(seqs)["mota"] == 1 - 4 / 1                 # -3.0: MOTA is unbounded below

    # No ground truth means UNDEFINED, not 1.0. Scoring an empty evaluation as perfect would put a
    # run whose data failed to load straight to the top of the table.
    for seqs in ([], [[frame([], [])]]):
        assert math.isnan(clear_mot(seqs)["mota"])
        assert math.isnan(idf1(seqs)["idf1"])
    assert clear_mot([])["fn"] == 0 and clear_mot([])["n_gt"] == 0


def test_low_iou_match_is_rejected():
    """A box in roughly the right place but below the IoU threshold is a miss AND a false positive,
    not a sloppy match - this is what makes MOTA sensitive to localisation collapse."""
    seqs = [[frame([(1, 0, 0)], [(7, 8, 0)])]]      # IoU = 20/180 = 0.11
    c = clear_mot(seqs, iou_thres=0.5)
    assert (c["fn"], c["fp"], c["matches"]) == (1, 1, 0)
    assert clear_mot(seqs, iou_thres=0.1)["matches"] == 1   # threshold is a parameter


def test_one_prediction_cannot_be_claimed_by_two_ground_truths():
    """Two gt ids can carry the SAME remembered pred id - one object leaves, another inherits the
    track, then both reappear. Without a used-check the single box was matched to both, producing
    MOTA 1.25 and FP -1: impossible, since MOTA is capped at 1 and FP counts unmatched predictions."""
    seqs = [[frame([(1, 0, 0)], [(7, 0, 0)]),
             frame([(2, 0, 0)], [(7, 0, 0)]),
             frame([(1, 0, 0), (2, 0, 0)], [(7, 0, 0)])]]
    c = clear_mot(seqs)
    assert c["mota"] <= 1.0
    assert c["fp"] >= 0 and c["fn"] >= 0
    assert c["matches"] <= c["n_gt"]
    assert c["matches"] == 3 and c["fn"] == 1        # frame 3 can only match one of the two


def test_identity_matching_maximises_quality_not_pair_count():
    """The assignment must maximise total overlap, not the NUMBER of matched pairs. One long correct
    identity has to beat several short wrong ones."""
    # id 1 is present for 10 frames and tracked correctly by 7 throughout;
    # id 2 appears once and is also covered by 7 in that frame.
    seqs = [[frame([(1, 0, 0)], [(7, 0, 0)]) for _ in range(10)]
            + [frame([(1, 0, 0), (2, 60, 60)], [(7, 0, 0), (8, 60, 60)])]]
    f = idf1(seqs)
    assert f["idf1"] > 0.9, f"long correct identity should dominate, got {f['idf1']:.3f}"
    assert 0.0 <= f["idp"] <= 1.0 and 0.0 <= f["idr"] <= 1.0


def test_a_duplicated_id_in_one_frame_does_not_break_the_counts():
    """The repo's own generated gt.txt can repeat an id within a frame. Counting it twice pushed
    IDFP negative and IDP above 1."""
    fr = {"gt_ids": [1, 1], "gt_boxes": np.array([box(0, 0), box(0, 0)]),
          "pred_ids": [7], "pred_boxes": np.array([box(0, 0)])}
    f = idf1([[fr, fr]])
    assert f["idfp"] >= 0 and f["idfn"] >= 0
    assert 0.0 <= f["idp"] <= 1.0 and 0.0 <= f["idf1"] <= 1.0
