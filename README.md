# Distributed DL Capstone Solution

This repository contains a script-based solution for the Torch Distributed capstone. The solution demonstrates distributed training with a manually sharded ResNet18 encoder, launched with `torchrun`.

The main training code is in `solution/train_sharded_simclr.py`. The normal Codespaces flow starts from `solution/run_codespaces_flow.sh`.

## Repository Structure

Upload only the source files and final run artifacts needed to reproduce and discuss the submission:

```text
/workspaces/OU-22971-Distributed_DL/
├── design_doc.md                              # capstone design document
├── README.md                                  # setup, run commands, and analysis
└── solution/
    ├── prepare_fake_data_metadata.py          # writes deterministic FakeData metadata
    ├── train_sharded_simclr.py                # distributed sharded SimCLR training script
    ├── summarize_runs.py                      # compares baseline and follow-up outputs
    ├── run_codespaces_flow.sh                 # full Codespaces flow runner
    ├── prepared/
    │   └── dataset_metadata.json              # final data/workload metadata used by the runs
    └── outputs/
        ├── manual_batch_size_sweep.csv        # baseline vs follow-up comparison table
        ├── diagnosis_summary.md               # combined diagnosis summary
        ├── baseline_b1/
        │   ├── run_config.json                # baseline run configuration
        │   ├── communication_groups.json      # rank pairs and stage groups
        │   ├── metrics.csv                    # per-rank performance metrics
        │   ├── trace_summary.csv              # summarized profiler span timings
        │   ├── diagnosis_summary.md           # baseline bottleneck diagnosis
        │   └── traces/
        │       ├── baseline_b1_rank0.json     # profiler trace for rank 0, stage 0
        │       ├── baseline_b1_rank1.json     # profiler trace for rank 1, stage 1
        │       ├── baseline_b1_rank2.json     # profiler trace for rank 2, stage 0
        │       └── baseline_b1_rank3.json     # profiler trace for rank 3, stage 1
        └── followup_b2/
            ├── run_config.json                # follow-up run configuration
            ├── communication_groups.json      # rank pairs and stage groups
            ├── metrics.csv                    # per-rank performance metrics
            ├── trace_summary.csv              # summarized profiler span timings
            ├── diagnosis_summary.md           # follow-up bottleneck diagnosis
            └── traces/
                ├── followup_b2_rank0.json     # profiler trace for rank 0, stage 0
                ├── followup_b2_rank1.json     # profiler trace for rank 1, stage 1
                ├── followup_b2_rank2.json     # profiler trace for rank 2, stage 0
                └── followup_b2_rank3.json     # profiler trace for rank 3, stage 1
```

## Codespaces Setup

Open the repository in GitHub Codespaces, then run these commands from the repository root:

```bash
cd solution
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
```

Verify the environment:

```bash
python - <<'PY'
import torch
import torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('distributed_available', torch.distributed.is_available())
PY
```

Expected result: `torch.distributed.is_available()` should print `True`.

If you are using the course Conda environment instead of pip, run:

```bash
conda activate 22971-td
cd /workspaces/OU-22971-Distributed_DL/solution
```

## Full Codespaces Flow

Run the full flow from the repository root with:

```bash
cd solution
bash run_codespaces_flow.sh
```

The script runs four steps:

1. prepare deterministic FakeData metadata
2. run the baseline profiled distributed training job with local batch size 1
3. run the follow-up profiled distributed training job with local batch size 2
4. summarize the manual batch-size comparison

The default full-flow settings are:

```text
NPROC_PER_NODE=4
DATASET_SIZE=64
STEPS=3
SEED=22971
```

You can override them if Codespaces is slow:

```bash
STEPS=2 DATASET_SIZE=32 bash run_codespaces_flow.sh
```

Keep `NPROC_PER_NODE=4` for the capstone because the design document requires at least four ranks.

## Manual Step-By-Step Flow

Use this section if you want to run each step separately for the video.

### 1. Data Preparation

The workload uses deterministic `torchvision.datasets.FakeData`, so no real image files are downloaded. This command writes the metadata used by the runs:

```bash
cd solution
python prepare_fake_data_metadata.py --dataset-size 64 --seed 22971 --output-dir prepared
```

Output:

```text
solution/prepared/dataset_metadata.json
```

### 2. Baseline Run

Run a non-profiled baseline smoke run with four ranks and local batch size 1:

```bash
torchrun --standalone --nproc_per_node=4 train_sharded_simclr.py \
  --metadata-path prepared/dataset_metadata.json \
  --local-batch-size 1 \
  --steps 3 \
  --run-name baseline_b1_smoke \
  --output-dir outputs/baseline_b1_smoke
```

This smoke run checks that the distributed job works before collecting profiler traces. It is not required as a final output artifact.

### 3. Baseline Profiled Run

Run the baseline profiled job used for trace analysis:

```bash
torchrun --standalone --nproc_per_node=4 train_sharded_simclr.py \
  --metadata-path prepared/dataset_metadata.json \
  --local-batch-size 1 \
  --steps 3 \
  --profile \
  --run-name baseline_b1 \
  --output-dir outputs/baseline_b1
```

### 4. Follow-Up Profiled Run

The manual tuning decision is to try a larger local batch size and compare `images/s` plus trace evidence:

```bash
torchrun --standalone --nproc_per_node=4 train_sharded_simclr.py \
  --metadata-path prepared/dataset_metadata.json \
  --local-batch-size 2 \
  --steps 3 \
  --profile \
  --run-name followup_b2 \
  --output-dir outputs/followup_b2
```

### 5. Sweep Summary

After both profiled runs finish:

```bash
python summarize_runs.py \
  --run-dirs outputs/baseline_b1 outputs/followup_b2 \
  --output-dir outputs
```

This writes:

```text
solution/outputs/manual_batch_size_sweep.csv
solution/outputs/diagnosis_summary.md
```

`global_batch_size` counts source images. `global_views_per_step` counts the two augmented views per source image. The reported `images/s` uses the augmented views because those are the images that actually pass through the sharded encoder.

## Shard Split And Communication Groups

The model is split manually:

- Stage 0: `conv1`, `bn1`, `relu`, `maxpool`, `layer1`, `layer2`
- Stage 1: `layer3`, `layer4`, `avgpool`, flatten, projection head

Ranks are organized as:

- `pair_group(k) = (2k, 2k+1)`
- even ranks are stage-0 ranks
- odd ranks are stage-1 ranks
- `stage0_group` contains all even ranks
- `stage1_group` contains all odd ranks

Point-to-point communication happens between each even/odd pair:

- even rank sends boundary activations to odd rank
- odd rank sends boundary gradients back to even rank

Collective communication happens inside stage groups:

- stage-1 ranks use `all_gather` to collect embeddings for contrastive loss
- stage-0 ranks use `all_reduce` to average stage-0 gradients
- stage-1 ranks use `all_reduce` to average stage-1 gradients

## Loss Calculation

Each source image is augmented twice, creating a positive pair. Stage 1 produces one embedding per augmented view.

For every local embedding, the loss compares it against all gathered embeddings. Its positive pair is the correct class, and the other embeddings act as negatives. The script removes the embedding itself from the candidate list and applies cross-entropy over cosine-style similarity scores.

## Loss Gradient Approximation

The full SimCLR loss couples all embeddings across all ranks. To keep the project focused on distributed-systems behavior, the script uses the required local approximation:

- each odd rank gathers embeddings from all stage-1 ranks
- remote embeddings are treated as detached constants
- the current rank keeps its own local embeddings attached to autograd
- loss is computed only for local embeddings
- `loss.backward()` produces gradients for the local stage-1 model and for the received boundary activation

The boundary gradient is then sent back to the paired stage-0 rank.

## Bottleneck Categories

The analysis uses the Unit 2 / Unit 3 vocabulary:

- `compute`: long `stage0_forward`, `stage1_forward`, `loss_calculation`, `loss_backward`, or `stage0_backward`
- `communication`: long `send_boundary`, `send_boundary_grad`, `gather_embeddings`, `grad_sync_stage0`, or `grad_sync_stage1`
- `waiting`: one rank arrives early and blocks in `recv`, `send`, `all_gather`, or `all_reduce`
- `memory`: larger batch sizes increase activations, gathered embeddings, and optimizer state pressure

## Output Artifacts

The `prepared/` directory contains the workload configuration:

```text
solution/prepared/dataset_metadata.json
```

Each final run directory contains:

```text
run_config.json
communication_groups.json
metrics.csv
trace_summary.csv
diagnosis_summary.md
traces/<run_name>_rank0.json
traces/<run_name>_rank1.json
traces/<run_name>_rank2.json
traces/<run_name>_rank3.json
```

The combined sweep output contains:

```text
solution/outputs/manual_batch_size_sweep.csv
solution/outputs/diagnosis_summary.md
```

## Output Artifacts Analysis And Discussion

The final comparison is stored in `solution/outputs/manual_batch_size_sweep.csv`.

Observed run results:

```text
baseline_b1: local_batch_size=1, global_batch_size=2, images/s=3.15
followup_b2: local_batch_size=2, global_batch_size=4, images/s=4.16
```

The follow-up improved throughput by about `1.02 images/s`.

The baseline used a smaller local batch size, so each distributed step had less local compute. In that case, communication and waiting are easier to notice in the trace because every step still pays for boundary transfer, embedding gather, gradient return, and gradient synchronization.

The follow-up kept the same distributed structure but increased `local_batch_size` from `1` to `2`. This gave each rank more useful compute per communication round. The traces still show communication or waiting, but the metric evidence shows that the larger batch size improved throughput.

The diagnosis category for both runs was `communication_or_waiting_visible`. This means the system was not perfectly balanced: communication spans such as `recv_boundary`, `send_boundary`, `gather_embeddings`, `send_boundary_grad`, `recv_boundary_grad`, and `grad_sync_*` still matter. However, the follow-up was better because the additional compute improved overall images per second.

For the video trace walkthrough, the most useful files are:

```text
solution/outputs/baseline_b1/traces/baseline_b1_rank1.json
solution/outputs/followup_b2/traces/followup_b2_rank1.json
```

Rank 1 is a stage-1 rank, so it shows:

- `recv_boundary`: stage 1 waits for activation from stage 0
- `stage1_forward`: local compute in the second model shard
- `gather_embeddings`: collective communication across stage-1 ranks
- `loss_calculation`: SimCLR-style contrastive loss
- `loss_backward`: local backward pass on stage 1
- `send_boundary_grad`: returned boundary gradient sent back to stage 0
- `grad_sync_stage1`: collective gradient synchronization

A useful secondary trace pair is:

```text
solution/outputs/baseline_b1/traces/baseline_b1_rank0.json
solution/outputs/followup_b2/traces/followup_b2_rank0.json
```

Rank 0 is a stage-0 rank, so it shows the other side of the pipeline: `stage0_forward`, `send_boundary`, `recv_boundary_grad`, `stage0_backward`, and `grad_sync_stage0`.

The final tuning decision is to prefer the follow-up configuration, `local_batch_size=2`, because it produced higher throughput while preserving the same distributed model split and communication pattern.

## Optional Controller

The optional controller/stretch goal was not implemented, so there is no controller run command for this submission.
