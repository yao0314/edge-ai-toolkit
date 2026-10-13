"""`sweep.students`: a smaller architecture, distilled from the baseline, swept beside it.

The competition rubric bands compression on the reduction in parameter count AND on-disk size.
Quantization and magnitude pruning leave the parameter count alone, so the only thing that moves
that half is a smaller architecture — and it is only measurable if the teacher and the students
land in ONE table, because the ratios are taken across the sweep.
"""
import pandas as pd
import pytest

from core.score import score_table
from core.sweep import run_sweep_grid, student_tag

FAST = {"threads": [1], "warmup": 1, "runs": 5, "repeats": 1, "min_time_s": 0.05, "spinning": False}
CFG = {
    "task": "dummy", "measure": FAST, "quantize": {"calib_batches": 2},
    "prune": {"round_to": None},
    "sweep": {"prune_ratios": [0.0], "precisions": ["fp32", "int8"], "finetune_epochs": 1,
              "students": [{"width": 4, "epochs": 1}]},
}


def _run(tmp_path, dummy, trained, cfg=None, resume=False):
    return run_sweep_grid(cfg or CFG, lambda c: dummy, lambda ad, c: trained, tmp_path, resume=resume)


def test_student_tag_is_filename_safe():
    assert student_tag({"arch": "mobilenet_v2", "width_mult": 0.5}) == "mobilenet_v2_w050"
    assert student_tag({"arch": "shufflenet_v2_x0_5"}) == "shufflenet_v2_x0_5"
    assert student_tag({}) == "student"


def test_teacher_and_student_share_one_table(tmp_path, dummy, trained):
    df = _run(tmp_path, dummy, trained)
    ok = df[df["error"].isna() & df["skipped"].isna()]
    names = set(ok["variant"])
    assert {"p00_fp32", "p00_int8"} <= names                      # the teacher's own rows
    assert {"student_p00_fp32", "student_p00_int8"} <= names      # and the student's, prefixed
    assert (tmp_path / "sweep.csv").exists()
    assert (tmp_path / "models" / "student_student.pt").exists()  # cached for a resumed sweep


def test_student_actually_has_fewer_parameters(tmp_path, dummy, trained):
    df = _run(tmp_path, dummy, trained)
    teacher = df[df["variant"] == "p00_fp32"].iloc[0]
    student = df[df["variant"] == "student_p00_fp32"].iloc[0]
    assert student["params"] < teacher["params"], "the students dimension changed nothing"
    assert student["size_bytes"] < teacher["size_bytes"]


def test_compression_is_scored_against_the_teacher(tmp_path, dummy, trained):
    """The point of one shared table. Scored alone, every variant of a student sweep normalises to
    itself and compression reads as no gain at all."""
    df = _run(tmp_path, dummy, trained)
    ok = df[df["error"].isna() & df["skipped"].isna()].reset_index(drop=True)
    scored = score_table(ok, {"max_accuracy_drop": None, "max_accuracy_drop_rel": None,
                              "min_accuracy": None})
    teacher = scored[scored["variant"] == "p00_fp32"].iloc[0]
    student = scored[scored["variant"] == "student_p00_int8"].iloc[0]
    assert student["score_compression"] > teacher["score_compression"]

    alone = score_table(ok[ok["variant"].str.startswith("student_")].reset_index(drop=True),
                        {"max_accuracy_drop": None, "max_accuracy_drop_rel": None,
                         "min_accuracy": None})
    # Whatever is best within its own set takes full marks, however little it actually shrank:
    # minmax has nothing but that set to measure against. The exact floor is not the point and was
    # an agg: min artefact - under equal halves a row that is best on one column and worst on the
    # other sits at 0.5 - so assert the property instead: the same student scores differently in
    # the shared table, which is the reason the shared table exists.
    assert alone["score_compression"].max() == 1.0
    alone_student = alone[alone["variant"] == "student_p00_int8"].iloc[0]["score_compression"]
    assert alone_student != student["score_compression"]


def test_compression_scores_file_and_parameters_as_equal_halves(tmp_path, dummy, trained):
    """The organisers were asked how the rubric's two compression quantities combine and answered
    "equal halves", so a row that shrinks the file and not the parameter count earns the file half
    and nothing else. This used to be agg: min, under which such a row scored zero - defensible as a
    reading of "reduction in parameter count AND file size", and not what they meant."""
    df = pd.DataFrame({
        "variant": ["base", "quantized_only", "smaller"],
        "prune_ratio": [0.0, 0.0, 0.0], "precision": ["fp32", "int8", "int8"],
        "accuracy": [0.9, 0.9, 0.9],
        "params": [1000, 1000, 100],          # quantization leaves this untouched
        "size_bytes": [4000, 1000, 100],
    })
    scored = score_table(df, {"max_accuracy_drop": None, "max_accuracy_drop_rel": None,
                              "min_accuracy": None})
    by = dict(zip(scored["variant"], scored["score_compression"]))
    assert by["smaller"] == 1.0, "the row that shrank both should take the whole criterion"
    assert by["base"] == 0.0
    # 4000 -> 1000 of a 4000..100 range is 0.769 of the file half, and none of the parameter half
    size_half = (4000 - 1000) / (4000 - 100)
    assert by["quantized_only"] == pytest.approx(size_half / 2),         "quantization must earn the file half and only the file half"
    assert 0 < by["quantized_only"] < by["smaller"]


def test_sweep_csv_survives_a_crash_between_students(tmp_path, dummy, trained):
    """The file on disk must hold every variant measured so far, not just the current one.

    run_sweep_grid writes sweep.csv from several run_sweep calls and only concatenates at the end,
    so an overwriting flush left a crash during the fourth student with the first three gone — and
    `resume` could not see them either, so it redid them.
    """
    cfg = {**CFG, "sweep": {**CFG["sweep"],
                            "students": [{"width": 4, "epochs": 1}, {"width": 6, "epochs": 1}]}}
    _run(tmp_path, dummy, trained, cfg)
    on_disk = set(pd.read_csv(tmp_path / "sweep.csv")["variant"].astype(str))
    assert {"p00_fp32", "p00_int8"} <= on_disk, "the teacher's rows were overwritten by a student"
    assert any(v.startswith("student_") for v in on_disk)
    assert len([v for v in on_disk if v.endswith("_p00_fp32")]) >= 2


def test_a_fresh_run_does_not_adopt_the_previous_one(tmp_path, dummy, trained):
    """Accumulating across calls must not mean accumulating across runs."""
    (tmp_path / "sweep.csv").write_text("variant,accuracy\nghost_p00_fp32,0.5\n", encoding="utf-8")
    _run(tmp_path, dummy, trained)
    assert "ghost_p00_fp32" not in set(pd.read_csv(tmp_path / "sweep.csv")["variant"].astype(str))


def test_students_skipped_for_a_non_torch_baseline(tmp_path, dummy, caplog):
    """An ONNX baseline cannot teach. The sweep must say so and carry on, not crash."""
    from pathlib import Path

    onnx = Path(__file__).parent / "fixtures" / "tiny_nhwc_b1.onnx"
    df = run_sweep_grid(CFG, lambda c: dummy, lambda ad, c: str(onnx), tmp_path)
    assert not any(str(v).startswith("student_") for v in df["variant"])
    assert "not an nn.Module" in caplog.text
