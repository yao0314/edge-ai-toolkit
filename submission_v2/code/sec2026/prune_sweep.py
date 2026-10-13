"""Sections 6.1 and 6.3 of the task sheet: a structured-pruning sweep with short recovery fine-tuning,
recording accuracy after every stage, then FP32 and static INT8 ONNX per point.

    python sec2026/prune_sweep.py --ratios 0.1 0.2 0.3 --ft-steps 300 --out sec2026/out
    python sec2026/prune_sweep.py --ratios 0.2 --ft-steps 300 --kd 0.7      # distil from the teacher
    python sec2026/prune_sweep.py --ratios 0.1 --ft-steps 20 --val-limit 500 --dry  # plumbing only

For each ratio the official student is pruned once (channels removed, so params, size and MACs
really shrink), BatchNorm statistics are re-estimated on training images, the model is fine-tuned
for --ft-steps optimizer steps (CPU only: the budget is steps, not epochs), and evaluated on the
official val split before and after fine-tuning. Results go to <out>/sweep.csv and
<out>/tradeoff.png; each point keeps pruned.pt (whole module), student_model.onnx and
student_model_int8.onnx under <out>/p<ratio>/.

Pruning reuses the toolkit's core.prune (torch-pruning with the dependency graph for MobileNetV2's
residuals and depthwise convs, and a guard so channel rounding never cuts more than asked).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import export_quant as eq  # noqa: E402


def classifier_layers(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == common.NUM_CLASSES]


def prune(model: nn.Module, ratio: float, round_to: int = 8, global_pruning: bool = False,
          importance: str = "l2", input_shape=common.INPUT_SHAPE) -> tuple[nn.Module, dict]:
    from core.prune import prune_model

    report = {}
    pruned = prune_model(model, torch.randn(*input_shape), ratio, ignored_layers=classifier_layers(model),
                         importance=importance, round_to=round_to, global_pruning=global_pruning,
                         report=report)
    return pruned, report


def finetune(model: nn.Module, train_loader, steps: int, lr: float, teacher: nn.Module | None = None,
             kd_alpha: float = 0.0, kd_t: float = 4.0, log_every: int = 50) -> dict:
    """SGD with cosine decay for a fixed number of steps. With a teacher, the loss is
    kd_alpha * T^2 * KL(teacher_T || student_T) + (1 - kd_alpha) * CE(labels)."""
    if steps <= 0:
        return {"steps": 0, "seconds": 0.0}
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    t0, step, seen, losses = time.time(), 0, 0, []
    while step < steps:
        for images, labels in train_loader:
            out = model(images)
            loss = F.cross_entropy(out, labels)
            if teacher is not None and kd_alpha > 0:
                with torch.no_grad():
                    t_logits = teacher(images)
                kd = F.kl_div(F.log_softmax(out / kd_t, 1), F.softmax(t_logits / kd_t, 1),
                              reduction="batchmean") * kd_t ** 2
                loss = kd_alpha * kd + (1 - kd_alpha) * loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            seen += len(labels)
            losses.append(float(loss))
            if step % log_every == 0 or step == steps:
                el = time.time() - t0
                print(f"    step {step}/{steps}  loss {sum(losses[-log_every:]) / len(losses[-log_every:]):.4f}  "
                      f"{seen / el:.1f} img/s  {el / 60:.1f} min")
            if step >= steps:
                break
    model.eval()
    return {"steps": step, "images": seen, "seconds": round(time.time() - t0, 1),
            "final_loss": round(sum(losses[-20:]) / len(losses[-20:]), 4)}


def recalibrate_bn(model, train_eval_ds, batches: int = 32, batch_size: int = 32):
    from core.prune import recalibrate_bn as _recal

    return _recal(model, common.loader(train_eval_ds, batch_size, shuffle=True), max_batches=batches)


def acc_of(model_or_path, val_loader) -> dict:
    r = common.evaluate(common.predictor(model_or_path), val_loader)
    return {"accuracy": r["accuracy"], "worst_class_accuracy": r["worst_class_accuracy"],
            "worst_class": r["worst_class"], "per_class_recall": r["per_class_recall"]}


FIELDS = ["ratio", "params", "macs", "fp32_size_mb", "int8_size_mb", "max_layer_cut",
          "acc_pruned", "acc_bn", "acc_ft", "worst_ft", "acc_int8", "worst_int8", "int8_sqnr_db",
          "ft_steps", "kd_alpha", "ft_seconds", "total_seconds"]


def run_point(student, ratio, a, train_ds, train_eval_ds, val_loader, teacher, calib, check) -> dict:
    t0 = time.time()
    d = Path(a.out) / f"p{int(round(ratio * 100)):02d}"
    d.mkdir(parents=True, exist_ok=True)
    row = {"ratio": ratio, "kd_alpha": a.kd if teacher is not None else 0.0}
    if ratio > 0:
        model, rep = prune(student, ratio, a.round_to, a.global_pruning, a.importance)
        row["max_layer_cut"] = rep.get("max_layer_cut")
        row["acc_pruned"] = acc_of(model, val_loader)["accuracy"]
        recalibrate_bn(model, train_eval_ds, a.bn_batches, a.batch_size)
        row["acc_bn"] = acc_of(model, val_loader)["accuracy"]
        print(f"  p{ratio:.2f}: pruned {row['acc_pruned']:.4f} -> BN recal {row['acc_bn']:.4f}")
        ft = finetune(model, common.loader(train_ds, a.batch_size, shuffle=True), a.ft_steps, a.lr,
                      teacher, a.kd, a.kd_t)
    else:
        model, ft = student, {"steps": 0, "seconds": 0.0}
    row["ft_steps"], row["ft_seconds"] = ft["steps"], ft["seconds"]
    r = acc_of(model, val_loader)
    row["acc_ft"], row["worst_ft"] = r["accuracy"], r["worst_class_accuracy"]
    torch.save(model, d / "pruned.pt")
    fp32 = eq.export_fp32(model, d / "student_model.onnx")
    row["params"], row["macs"] = eq.onnx_params(fp32), eq.onnx_macs(fp32)
    row["fp32_size_mb"] = round(fp32.stat().st_size / 2**20, 3)
    if not a.no_int8:
        int8 = eq.quantize_static_qdq(fp32, d / "student_model_int8.onnx", calib, a.calib_method)
        row["int8_size_mb"] = round(int8.stat().st_size / 2**20, 3)
        row["int8_sqnr_db"] = eq.health_check(fp32, int8, check)["sqnr_db"]
        r8 = acc_of(int8, val_loader)
        row["acc_int8"], row["worst_int8"] = r8["accuracy"], r8["worst_class_accuracy"]
    row["total_seconds"] = round(time.time() - t0, 1)
    common.save_json({**row, "finetune": ft, "args": vars(a)}, d / "point.json")
    print("  " + json.dumps({k: row.get(k) for k in FIELDS}))
    return row


def plot(rows, base_acc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = sorted(rows, key=lambda r: r["ratio"])
    p0 = next((r["params"] for r in rows if r["ratio"] == 0), rows[0]["params"])
    for key, label, mark in (("acc_ft", "FP32 (after fine-tune)", "o-"), ("acc_int8", "INT8 QDQ", "s--")):
        pts = [(p0 / r["params"], 100 * r[key]) for r in rows if r.get(key) is not None]
        if pts:
            ax.plot(*zip(*pts), mark, label=label)
    for r in rows:
        if r.get("acc_ft") is not None:
            ax.annotate(f"{int(round(100 * r['ratio']))}%", (p0 / r["params"], 100 * r["acc_ft"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    if base_acc:
        ax.axhline(100 * base_acc, color="grey", lw=0.8, ls=":", label="baseline")
    ax.set_xlabel("parameter reduction (x, vs unpruned student)")
    ax.set_ylabel("val top-1 accuracy (%)")
    ax.set_title("Structured pruning ratio vs accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit")
    ap.add_argument("--model", help="default: the kit's student")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--ft-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--kd", type=float, default=0.0, help="KD weight (0 = hard labels only); loads the teacher")
    ap.add_argument("--kd-t", type=float, default=4.0)
    ap.add_argument("--round-to", type=int, default=8)
    ap.add_argument("--global-pruning", action="store_true")
    ap.add_argument("--importance", default="l2", choices=["l1", "l2", "random", "bn_scale"])
    ap.add_argument("--bn-batches", type=int, default=32)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--calib-method", default="minmax", choices=sorted(eq.METHODS))
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--val-limit", type=int, default=None, help="quick plumbing runs only; never report them")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="torch threads for training")
    ap.add_argument("--dry", action="store_true", help="label the output as a plumbing run")
    a = ap.parse_args(argv)

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    student = common.build_student(a.model, a.kit)
    train_ds, val_ds = common.subsets(a.kit, common.train_transform(), a.train_limit, a.val_limit)
    train_eval_ds, _ = common.subsets(a.kit, common.eval_transform(), a.train_limit, a.val_limit)
    val_loader = common.loader(val_ds, a.batch_size)
    teacher = common.build_teacher(kit=a.kit) if a.kd > 0 else None
    arrays = eq.calibration_arrays(train_eval_ds, a.calib + 64, seed=1)
    calib, check = arrays[:a.calib], arrays[a.calib:]
    print(f"train {len(train_ds)} | val {len(val_ds)} | ratios {a.ratios} | ft {a.ft_steps} steps"
          f"{' | KD ' + str(a.kd) if teacher is not None else ''}{' | DRY RUN' if a.dry else ''}")

    csv_path = out / "sweep.csv"
    rows = []
    for ratio in a.ratios:
        rows.append(run_point(student, ratio, a, train_ds, train_eval_ds, val_loader, teacher, calib, check))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:   # rewrite after every point
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    base = next((r["acc_ft"] for r in rows if r["ratio"] == 0), None)
    plot(rows, base, out / "tradeoff.png")
    print(f"-> {csv_path}\n-> {out / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
