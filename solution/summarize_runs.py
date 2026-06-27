"""Summarize manual batch-size runs produced by train_sharded_simclr.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sharded SimCLR run artifacts.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more output directories produced by train_sharded_simclr.py.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(run_dir: Path) -> dict[str, object]:
    metric_rows = read_csv(run_dir / "metrics.csv")
    trace_rows = read_csv(run_dir / "trace_summary.csv")
    if not metric_rows:
        raise RuntimeError(f"empty metrics file in {run_dir}")

    run_name = metric_rows[0]["run_name"]
    local_batch_size = int(metric_rows[0]["local_batch_size"])
    global_batch_size = int(metric_rows[0]["global_batch_size"])
    images_per_second = float(metric_rows[0]["images_per_second"])

    max_stage0_step = max(
        float(row["avg_step_time_s"]) for row in metric_rows if row["role"] == "stage0"
    )
    max_stage1_step = max(
        float(row["avg_step_time_s"]) for row in metric_rows if row["role"] == "stage1"
    )

    phase_totals: dict[str, float] = {}
    for row in trace_rows:
        phase_totals[row["phase"]] = phase_totals.get(row["phase"], 0.0) + float(row["total_s"])

    communication_phases = [
        "send_boundary",
        "recv_boundary",
        "send_boundary_grad",
        "recv_boundary_grad",
        "gather_embeddings",
        "grad_sync_stage0",
        "grad_sync_stage1",
    ]
    compute_phases = [
        "prepare_views",
        "stage0_forward",
        "stage1_forward",
        "loss_calculation",
        "loss_backward",
        "stage0_backward",
        "optimizer_step",
    ]
    communication_s = sum(phase_totals.get(phase, 0.0) for phase in communication_phases)
    compute_s = sum(phase_totals.get(phase, 0.0) for phase in compute_phases)

    if max_stage1_step > max_stage0_step * 1.15:
        diagnosis = "stage1_loss_or_gather_heavy"
    elif max_stage0_step > max_stage1_step * 1.15:
        diagnosis = "stage0_compute_heavy"
    elif communication_s > compute_s * 0.45:
        diagnosis = "communication_or_waiting_visible"
    else:
        diagnosis = "roughly_balanced"

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "images_per_second": images_per_second,
        "max_stage0_step_s": max_stage0_step,
        "max_stage1_step_s": max_stage1_step,
        "communication_s": communication_s,
        "compute_s": compute_s,
        "diagnosis": diagnosis,
    }


def write_diagnosis(output_dir: Path, rows: list[dict[str, object]]) -> None:
    best = max(rows, key=lambda row: float(row["images_per_second"]))
    first = rows[0]
    last = rows[-1]
    change = float(last["images_per_second"]) - float(first["images_per_second"])
    direction = "improved" if change > 0 else "did not improve"

    lines = [
        "# Manual Batch-Size Diagnosis",
        "",
        "The sweep compares throughput, per-stage step time, and phase timing evidence.",
        "",
        f"- Baseline run: `{first['run_name']}` with local batch size {first['local_batch_size']}.",
        f"- Follow-up run: `{last['run_name']}` with local batch size {last['local_batch_size']}.",
        f"- Best observed run: `{best['run_name']}` with {float(best['images_per_second']):.2f} images/s.",
        f"- Result: the follow-up {direction} throughput by {change:.2f} images/s.",
        "",
        "Run details:",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['run_name']}: local_batch_size={row['local_batch_size']}, "
            f"images/s={float(row['images_per_second']):.2f}, "
            f"stage0_step={float(row['max_stage0_step_s']):.4f}s, "
            f"stage1_step={float(row['max_stage1_step_s']):.4f}s, "
            f"diagnosis={row['diagnosis']}"
        )
    lines.extend(
        [
            "",
            "Trace interpretation:",
            "",
            "- `stage0_forward` and `stage0_backward` represent stage-0 local compute.",
            "- `stage1_forward`, `loss_calculation`, and `loss_backward` represent stage-1 and loss-side compute.",
            "- `send_boundary`, `recv_boundary`, `send_boundary_grad`, and `recv_boundary_grad` show point-to-point transfer and waiting.",
            "- `gather_embeddings`, `grad_sync_stage0`, and `grad_sync_stage1` show collective communication.",
            "",
        ]
    )
    (output_dir / "diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(Path(run_dir)) for run_dir in args.run_dirs]
    write_csv(
        output_dir / "manual_batch_size_sweep.csv",
        rows,
        [
            "run_name",
            "run_dir",
            "local_batch_size",
            "global_batch_size",
            "images_per_second",
            "max_stage0_step_s",
            "max_stage1_step_s",
            "communication_s",
            "compute_s",
            "diagnosis",
        ],
    )
    write_diagnosis(output_dir, rows)
    print(f"Wrote {output_dir / 'manual_batch_size_sweep.csv'}")
    print(f"Wrote {output_dir / 'diagnosis_summary.md'}")


if __name__ == "__main__":
    main()
