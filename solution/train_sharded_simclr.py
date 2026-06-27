"""Manual two-stage SimCLR-like distributed training with torch.distributed.

Launch with torchrun, for example:

    torchrun --standalone --nproc_per_node=4 train_sharded_simclr.py --profile

The script intentionally uses low-level torch.distributed primitives instead of
DistributedDataParallel, DistributedSampler, pipeline helpers, or autograd-aware
distributed wrappers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import datasets, models, transforms


IMAGE_SIZE = (3, 224, 224)
NUM_CLASSES = 1000
BOUNDARY_CHANNELS = 128
BOUNDARY_SIZE = 28
DEFAULT_EMBEDDING_DIM = 128
TAG_BOUNDARY_BASE = 1000
TAG_BOUNDARY_GRAD_BASE = 2000


@dataclass
class GroupInfo:
    pair_ranks: list[tuple[int, int]]
    stage0_ranks: list[int]
    stage1_ranks: list[int]
    pair_group: dist.ProcessGroup
    stage0_group: dist.ProcessGroup
    stage1_group: dist.ProcessGroup
    pair_rank: int
    pair_index: int
    stage_index: int


class Stage1WithProjection(nn.Module):
    """ResNet18 tail plus a small projection head for contrastive embeddings."""

    def __init__(self, resnet: models.ResNet, embedding_dim: int) -> None:
        super().__init__()
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.projector = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.projector(x)
        return F.normalize(x, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual sharded SimCLR-like training with low-level torch.distributed."
    )
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=22971)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--metadata-path", type=str, default="")
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--backend",
        choices=("gloo", "nccl"),
        default="gloo",
        help="Use gloo for the CPU course environment; nccl is for CUDA runtimes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset_size < 1:
        raise SystemExit("--dataset-size must be at least 1.")
    if args.local_batch_size < 1:
        raise SystemExit("--local-batch-size must be at least 1.")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive.")
    if args.embedding_dim < 1:
        raise SystemExit("--embedding-dim must be positive.")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive.")


def load_metadata(args: argparse.Namespace) -> None:
    if not args.metadata_path:
        return
    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        raise SystemExit(f"metadata file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args.dataset_size = int(metadata.get("dataset_size", args.dataset_size))
    args.seed = int(metadata.get("seed", args.seed))



def build_group_info(rank: int, world_size: int) -> GroupInfo:
    pair_ranks = [(idx, idx + 1) for idx in range(0, world_size, 2)]
    stage0_ranks = list(range(0, world_size, 2))
    stage1_ranks = list(range(1, world_size, 2))

    pair_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
    for pair in pair_ranks:
        pair_groups[pair] = dist.new_group(ranks=list(pair))

    stage0_group = dist.new_group(ranks=stage0_ranks)
    stage1_group = dist.new_group(ranks=stage1_ranks)

    current_pair = next(pair for pair in pair_ranks if rank in pair)
    pair_group = pair_groups[current_pair]
    pair_index = pair_ranks.index(current_pair)
    pair_rank = current_pair[1] if rank == current_pair[0] else current_pair[0]
    if rank in stage0_ranks:
        stage_index = stage0_ranks.index(rank)
    else:
        stage_index = stage1_ranks.index(rank)

    return GroupInfo(
        pair_ranks=pair_ranks,
        stage0_ranks=stage0_ranks,
        stage1_ranks=stage1_ranks,
        pair_group=pair_group,
        stage0_group=stage0_group,
        stage1_group=stage1_group,
        pair_rank=pair_rank,
        pair_index=pair_index,
        stage_index=stage_index,
    )


def print_communication_structure(rank: int, world_size: int, groups: GroupInfo) -> None:
    if rank == 0:
        print("communication structure", flush=True)
        print(f"  world_group={list(range(world_size))}", flush=True)
        print(f"  pair_groups={groups.pair_ranks}", flush=True)
        print(f"  stage0_group={groups.stage0_ranks}", flush=True)
        print(f"  stage1_group={groups.stage1_ranks}", flush=True)
    dist.barrier()
    print(
        f"rank {rank}: pair_group={groups.pair_ranks[groups.pair_index]} "
        f"paired_rank={groups.pair_rank}",
        flush=True,
    )
    dist.barrier()


def build_stage_model(rank: int, groups: GroupInfo, embedding_dim: int) -> nn.Module:
    resnet = models.resnet18(weights=None)
    if rank in groups.stage0_ranks:
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
    return Stage1WithProjection(resnet=resnet, embedding_dim=embedding_dim)


def broadcast_module_parameters(
    module: nn.Module,
    rank: int,
    group_ranks: list[int],
    group: dist.ProcessGroup,
) -> None:
    if rank not in group_ranks:
        return
    source_rank = group_ranks[0]
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=source_rank, group=group)


def build_fake_dataset(dataset_size: int, seed: int) -> datasets.FakeData:
    return datasets.FakeData(
        size=dataset_size,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        transform=None,
        random_offset=seed,
    )


def build_view_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )


def set_view_seed(seed: int, rank: int, step: int, view_offset: int) -> None:
    value = seed + rank * 100_000 + step * 100 + view_offset
    random.seed(value)
    torch.manual_seed(value)


def materialize_two_views(
    dataset: datasets.FakeData,
    transform: transforms.Compose,
    local_batch_size: int,
    dataset_size: int,
    seed: int,
    rank: int,
    step: int,
    pair_index: int,
) -> torch.Tensor:
    first_views: list[torch.Tensor] = []
    second_views: list[torch.Tensor] = []
    start = (step * local_batch_size + pair_index * local_batch_size) % dataset_size
    for offset in range(local_batch_size):
        sample_index = (start + offset) % dataset_size
        image, _ = dataset[sample_index]
        set_view_seed(seed, rank, step, offset * 2)
        first_views.append(transform(image))
        set_view_seed(seed, rank, step, offset * 2 + 1)
        second_views.append(transform(image))
    return torch.stack(first_views + second_views, dim=0)


@contextmanager
def timed_span(phase_totals: defaultdict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    with record_function(name):
        yield
    phase_totals[name] += time.perf_counter() - start


def expected_boundary_shape(local_batch_size: int) -> tuple[int, int, int, int]:
    return (2 * local_batch_size, BOUNDARY_CHANNELS, BOUNDARY_SIZE, BOUNDARY_SIZE)


def assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], label: str) -> None:
    if tuple(tensor.shape) != expected:
        raise RuntimeError(f"{label} shape mismatch: got {tuple(tensor.shape)}, expected {expected}")


def ensure_finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{label} contains non-finite values.")


def ensure_finite_gradients(module: nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"{label}.{name}.grad contains non-finite values.")


def average_gradients(module: nn.Module, group: dist.ProcessGroup, group_size: int) -> None:
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
        parameter.grad /= group_size


def gather_embeddings_with_live_local(
    local_embeddings: torch.Tensor,
    groups: GroupInfo,
) -> torch.Tensor:
    gathered = [torch.empty_like(local_embeddings.detach()) for _ in groups.stage1_ranks]
    dist.all_gather(gathered, local_embeddings.detach(), group=groups.stage1_group)
    gathered[groups.stage_index] = local_embeddings
    return torch.cat(gathered, dim=0)


def approximate_simclr_loss(
    local_embeddings: torch.Tensor,
    all_embeddings: torch.Tensor,
    local_batch_size: int,
    stage1_group_index: int,
    temperature: float,
) -> torch.Tensor:
    local_views = 2 * local_batch_size
    start_index = stage1_group_index * local_views
    similarities = local_embeddings @ all_embeddings.T
    total_views = all_embeddings.shape[0]
    all_indices = torch.arange(total_views, device=all_embeddings.device)
    losses: list[torch.Tensor] = []

    for local_idx in range(local_views):
        global_idx = start_index + local_idx
        positive_local_idx = local_idx + local_batch_size
        if local_idx >= local_batch_size:
            positive_local_idx = local_idx - local_batch_size
        positive_global_idx = start_index + positive_local_idx

        candidate_mask = all_indices != global_idx
        logits = similarities[local_idx, candidate_mask] / temperature
        target_index = positive_global_idx
        if positive_global_idx > global_idx:
            target_index -= 1
        target = torch.tensor([target_index], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    return torch.stack(losses).mean()


def max_parameter_difference(
    module: nn.Module,
    rank: int,
    group_ranks: list[int],
    group: dist.ProcessGroup,
) -> float:
    if rank not in group_ranks:
        return math.nan
    first_parameter = next(module.parameters()).detach()
    reference = first_parameter.clone()
    dist.broadcast(reference, src=group_ranks[0], group=group)
    local_max = torch.tensor(
        [(first_parameter - reference).abs().max().item()],
        dtype=torch.float64,
    )
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
    return float(local_max.item())


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_diagnosis(output_dir: Path, metrics_rows: list[dict[str, object]]) -> None:
    stage0_rows = [row for row in metrics_rows if row["role"] == "stage0"]
    stage1_rows = [row for row in metrics_rows if row["role"] == "stage1"]
    stage0_step = max(float(row["avg_step_time_s"]) for row in stage0_rows)
    stage1_step = max(float(row["avg_step_time_s"]) for row in stage1_rows)
    images_per_second = float(metrics_rows[0]["images_per_second"])

    if stage1_step > stage0_step * 1.15:
        bottleneck = "stage-1/loss-heavy"
        explanation = (
            "Odd ranks own stage 1, all_gather, and the contrastive loss, so they "
            "were slower than the stage-0 ranks."
        )
    elif stage0_step > stage1_step * 1.15:
        bottleneck = "stage-0 compute-heavy"
        explanation = "Even ranks spent more time in the stage-0 side of the pipeline."
    else:
        bottleneck = "roughly balanced"
        explanation = "Stage-0 and stage-1 ranks had similar average step times."

    text = "\n".join(
        [
            "# Diagnosis Summary",
            "",
            f"- Bottleneck category: `{bottleneck}`",
            f"- Stage-0 max average step time: {stage0_step:.4f}s",
            f"- Stage-1 max average step time: {stage1_step:.4f}s",
            f"- Estimated throughput: {images_per_second:.2f} images/s",
            f"- Explanation: {explanation}",
            "",
            "Use the profiler traces to inspect compute spans, communication spans, "
            "`gather_embeddings`, `loss_calculation`, and waiting around blocking "
            "send/recv or collectives.",
            "",
        ]
    )
    (output_dir / "diagnosis_summary.md").write_text(text, encoding="utf-8")


def run_training(args: argparse.Namespace) -> None:
    load_metadata(args)
    validate_args(args)

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    dist.init_process_group(backend=args.backend)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if world_size < 4 or world_size % 2 != 0:
            raise SystemExit("Launch with an even world size of at least 4 ranks.")

        groups = build_group_info(rank=rank, world_size=world_size)
        role = "stage0" if rank in groups.stage0_ranks else "stage1"
        print_communication_structure(rank=rank, world_size=world_size, groups=groups)

        torch.manual_seed(args.seed)
        random.seed(args.seed + rank)

        model = build_stage_model(rank=rank, groups=groups, embedding_dim=args.embedding_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        if role == "stage0":
            broadcast_module_parameters(model, rank, groups.stage0_ranks, groups.stage0_group)
        else:
            broadcast_module_parameters(model, rank, groups.stage1_ranks, groups.stage1_group)

        dataset = build_fake_dataset(dataset_size=args.dataset_size, seed=args.seed)
        view_transform = build_view_transform()
        boundary_shape = expected_boundary_shape(args.local_batch_size)
        phase_totals: defaultdict[str, float] = defaultdict(float)
        step_times: list[float] = []
        loss_values: list[float] = []

        def training_loop() -> None:
            for step in range(args.steps):
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)

                if role == "stage0":
                    with timed_span(phase_totals, "prepare_views"):
                        views = materialize_two_views(
                            dataset=dataset,
                            transform=view_transform,
                            local_batch_size=args.local_batch_size,
                            dataset_size=args.dataset_size,
                            seed=args.seed,
                            rank=rank,
                            step=step,
                            pair_index=groups.pair_index,
                        )

                    with timed_span(phase_totals, "stage0_forward"):
                        boundary = model(views)
                    assert_shape(boundary, boundary_shape, "boundary_activation")
                    ensure_finite_tensor(boundary, "boundary_activation")

                    with timed_span(phase_totals, "send_boundary"):
                        dist.send(
                            boundary.detach().contiguous(),
                            dst=groups.pair_rank,
                            group=groups.pair_group,
                            tag=TAG_BOUNDARY_BASE + step,
                        )

                    returned_gradient = torch.empty_like(boundary)
                    with timed_span(phase_totals, "recv_boundary_grad"):
                        dist.recv(
                            returned_gradient,
                            src=groups.pair_rank,
                            group=groups.pair_group,
                            tag=TAG_BOUNDARY_GRAD_BASE + step,
                        )
                    ensure_finite_tensor(returned_gradient, "returned_boundary_gradient")

                    with timed_span(phase_totals, "stage0_backward"):
                        boundary.backward(returned_gradient)
                    ensure_finite_gradients(model, "stage0")

                    with timed_span(phase_totals, "grad_sync_stage0"):
                        average_gradients(
                            model,
                            group=groups.stage0_group,
                            group_size=len(groups.stage0_ranks),
                        )

                    with timed_span(phase_totals, "optimizer_step"):
                        optimizer.step()
                else:
                    boundary = torch.empty(boundary_shape, dtype=torch.float32)
                    with timed_span(phase_totals, "recv_boundary"):
                        dist.recv(
                            boundary,
                            src=groups.pair_rank,
                            group=groups.pair_group,
                            tag=TAG_BOUNDARY_BASE + step,
                        )
                    boundary.requires_grad_(True)
                    ensure_finite_tensor(boundary, "received_boundary_activation")

                    with timed_span(phase_totals, "stage1_forward"):
                        local_embeddings = model(boundary)
                    ensure_finite_tensor(local_embeddings, "local_embeddings")

                    with timed_span(phase_totals, "gather_embeddings"):
                        all_embeddings = gather_embeddings_with_live_local(
                            local_embeddings=local_embeddings,
                            groups=groups,
                        )

                    with timed_span(phase_totals, "loss_calculation"):
                        loss = approximate_simclr_loss(
                            local_embeddings=local_embeddings,
                            all_embeddings=all_embeddings,
                            local_batch_size=args.local_batch_size,
                            stage1_group_index=groups.stage_index,
                            temperature=args.temperature,
                        )
                    ensure_finite_tensor(loss, "loss")

                    with timed_span(phase_totals, "loss_backward"):
                        loss.backward()
                    if boundary.grad is None:
                        raise RuntimeError("boundary gradient was not produced on stage 1.")
                    ensure_finite_tensor(boundary.grad, "boundary_gradient")
                    ensure_finite_gradients(model, "stage1")

                    with timed_span(phase_totals, "send_boundary_grad"):
                        dist.send(
                            boundary.grad.detach().contiguous(),
                            dst=groups.pair_rank,
                            group=groups.pair_group,
                            tag=TAG_BOUNDARY_GRAD_BASE + step,
                        )

                    with timed_span(phase_totals, "grad_sync_stage1"):
                        average_gradients(
                            model,
                            group=groups.stage1_group,
                            group_size=len(groups.stage1_ranks),
                        )

                    with timed_span(phase_totals, "optimizer_step"):
                        optimizer.step()
                    loss_values.append(float(loss.item()))

                step_times.append(time.perf_counter() - step_start)

        if args.profile:
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=False,
            ) as prof:
                training_loop()
            trace_path = trace_dir / f"{args.run_name}_rank{rank}.json"
            prof.export_chrome_trace(str(trace_path))
        else:
            training_loop()

        if role == "stage0":
            max_param_diff = max_parameter_difference(
                model,
                rank=rank,
                group_ranks=groups.stage0_ranks,
                group=groups.stage0_group,
            )
        else:
            max_param_diff = max_parameter_difference(
                model,
                rank=rank,
                group_ranks=groups.stage1_ranks,
                group=groups.stage1_group,
            )

        avg_step_time = sum(step_times) / len(step_times)
        avg_loss = sum(loss_values) / len(loss_values) if loss_values else math.nan
        local_summary = {
            "rank": rank,
            "local_rank": local_rank,
            "role": role,
            "pair_rank": groups.pair_rank,
            "pair_index": groups.pair_index,
            "stage_index": groups.stage_index,
            "avg_step_time_s": avg_step_time,
            "loss": avg_loss,
            "max_parameter_difference_after_sync": max_param_diff,
            "phase_totals": dict(phase_totals),
        }
        gathered_summaries = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_summary, object_gather_list=gathered_summaries, dst=0)

        if rank == 0:
            source_images_per_step = args.local_batch_size * len(groups.stage0_ranks)
            global_views_per_step = source_images_per_step * 2
            max_avg_step_time = max(float(row["avg_step_time_s"]) for row in gathered_summaries)
            images_per_second = global_views_per_step / max_avg_step_time

            config = {
                "run_name": args.run_name,
                "backend": args.backend,
                "world_size": world_size,
                "dataset": "torchvision.datasets.FakeData",
                "dataset_size": args.dataset_size,
                "seed": args.seed,
                "image_size": list(IMAGE_SIZE),
                "local_batch_size": args.local_batch_size,
                "global_batch_size_source_images": source_images_per_step,
                "global_views_per_step": global_views_per_step,
                "steps": args.steps,
                "profile": args.profile,
                "temperature": args.temperature,
                "embedding_dim": args.embedding_dim,
                "learning_rate": args.learning_rate,
                "stage0_ranks": groups.stage0_ranks,
                "stage1_ranks": groups.stage1_ranks,
                "pair_groups": [list(pair) for pair in groups.pair_ranks],
                "stage0_split": "conv1/bn1/relu/maxpool/layer1/layer2",
                "stage1_split": "layer3/layer4/avgpool/flatten/projection_head",
            }
            write_json(output_dir / "run_config.json", config)
            write_json(
                output_dir / "communication_groups.json",
                {
                    "world_group": list(range(world_size)),
                    "pair_groups": [list(pair) for pair in groups.pair_ranks],
                    "stage0_group": groups.stage0_ranks,
                    "stage1_group": groups.stage1_ranks,
                },
            )

            metric_rows: list[dict[str, object]] = []
            for row in sorted(gathered_summaries, key=lambda item: int(item["rank"])):
                metric_rows.append(
                    {
                        "run_name": args.run_name,
                        "rank": int(row["rank"]),
                        "role": row["role"],
                        "pair_rank": int(row["pair_rank"]),
                        "local_batch_size": args.local_batch_size,
                        "global_batch_size": source_images_per_step,
                        "global_views_per_step": global_views_per_step,
                        "avg_step_time_s": f"{float(row['avg_step_time_s']):.6f}",
                        "loss": "" if math.isnan(float(row["loss"])) else f"{float(row['loss']):.6f}",
                        "images_per_second": f"{images_per_second:.6f}",
                        "max_parameter_difference_after_sync": f"{float(row['max_parameter_difference_after_sync']):.10f}",
                    }
                )
            write_csv(
                output_dir / "metrics.csv",
                metric_rows,
                [
                    "run_name",
                    "rank",
                    "role",
                    "pair_rank",
                    "local_batch_size",
                    "global_batch_size",
                    "global_views_per_step",
                    "avg_step_time_s",
                    "loss",
                    "images_per_second",
                    "max_parameter_difference_after_sync",
                ],
            )

            phase_rows: list[dict[str, object]] = []
            for row in sorted(gathered_summaries, key=lambda item: int(item["rank"])):
                for phase, total_s in sorted(row["phase_totals"].items()):
                    phase_rows.append(
                        {
                            "run_name": args.run_name,
                            "rank": int(row["rank"]),
                            "role": row["role"],
                            "phase": phase,
                            "total_s": f"{float(total_s):.6f}",
                            "avg_per_step_s": f"{float(total_s) / args.steps:.6f}",
                        }
                    )
            write_csv(
                output_dir / "trace_summary.csv",
                phase_rows,
                ["run_name", "rank", "role", "phase", "total_s", "avg_per_step_s"],
            )
            write_diagnosis(output_dir, metric_rows)

            print("run summary", flush=True)
            print(f"  output_dir={output_dir}", flush=True)
            print(f"  local_batch_size={args.local_batch_size}", flush=True)
            print(f"  global_batch_size={source_images_per_step}", flush=True)
            print(f"  global_views_per_step={global_views_per_step}", flush=True)
            print(f"  estimated_images_per_second={images_per_second:.2f}", flush=True)
            print(f"  metrics={output_dir / 'metrics.csv'}", flush=True)
            if args.profile:
                print(f"  traces={trace_dir}", flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

