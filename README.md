# Distributed DL Capstone Solution

This solution implements the required SimCLR-like distributed training system with a manually sharded ResNet18 encoder.

The main training logic is in `train_sharded_simclr.py` and is launched with `torchrun`. The submitted flow is script-based and starts from `run_codespaces_flow.sh`.

## Codespaces Setup

Open the repository in GitHub Codespaces, then run these commands from the repository root:

Repository layout used by this README:

```text
/workspaces/OU-22971-Distributed_DL/
├── design_doc.md
├── README.md
└── solution/
    ├── prepare_fake_data_metadata.py
    ├── train_sharded_simclr.py
    ├── summarize_runs.py
    └── run_codespaces_flow.sh
```

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

Run the full flow with one command:

```bash
bash run_codespaces_flow.sh
```

This script runs the following four steps in order:

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

Keep `NPROC_PER_NODE=4` for the capstone because the design doc requires at least four ranks.

## Manual Step-By-Step Flow

Use this section if you want to run each step separately for the video.

### 1. Data Preparation

The workload uses deterministic `torchvision.datasets.FakeData`, so no real image files are downloaded. This command writes the metadata used by the run:

```bash
python prepare_fake_data_metadata.py --dataset-size 64 --seed 22971 --output-dir prepared
```

Output:

```text
prepared/dataset_metadata.json
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

After both runs finish:

```bash
python summarize_runs.py \
  --run-dirs outputs/baseline_b1 outputs/followup_b2 \
  --output-dir outputs
```

This writes:

```text
outputs/manual_batch_size_sweep.csv
outputs/diagnosis_summary.md
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

Each run directory contains:

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
outputs/manual_batch_size_sweep.csv
outputs/diagnosis_summary.md
```

## Analysis And Discussion

The baseline run shows the starting balance between stage-0 compute, stage-1 compute, embedding gathering, loss calculation, gradient synchronization, and waiting.

The follow-up run increases local batch size. If `images/s` improves, the larger batch gave each rank more useful local work per synchronization. If `images/s` does not improve, the trace should show that loss-side work, embedding gathering, waiting, or memory pressure grew faster than useful compute.

The decision is based on `images/s` first, then supported by per-rank step time and profiler span evidence.

## Optional Controller

The optional controller/stretch goal was not implemented, so there is no controller run command for this submission.
