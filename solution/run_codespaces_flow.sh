#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATASET_SIZE="${DATASET_SIZE:-64}"
STEPS="${STEPS:-3}"
SEED="${SEED:-22971}"

echo "Codespaces distributed capstone flow"
echo "------------------------------------"
echo "cwd=${SCRIPT_DIR}"
echo "python=$(command -v python)"
python - <<'PY'
import platform
import torch
import torchvision

print("platform=", platform.platform())
print("torch=", torch.__version__)
print("torchvision=", torchvision.__version__)
print("distributed_available=", torch.distributed.is_available())
PY

echo
echo "Step 1/4: prepare deterministic FakeData metadata"
python prepare_fake_data_metadata.py \
  --dataset-size "${DATASET_SIZE}" \
  --seed "${SEED}" \
  --output-dir prepared

echo
echo "Step 2/4: baseline profiled run, local_batch_size=1"
torchrun \
  --standalone \
  "--nproc_per_node=${NPROC_PER_NODE}" \
  train_sharded_simclr.py \
  --metadata-path prepared/dataset_metadata.json \
  --local-batch-size 1 \
  --steps "${STEPS}" \
  --profile \
  --run-name baseline_b1 \
  --output-dir outputs/baseline_b1

echo
echo "Step 3/4: follow-up profiled run, local_batch_size=2"
torchrun \
  --standalone \
  "--nproc_per_node=${NPROC_PER_NODE}" \
  train_sharded_simclr.py \
  --metadata-path prepared/dataset_metadata.json \
  --local-batch-size 2 \
  --steps "${STEPS}" \
  --profile \
  --run-name followup_b2 \
  --output-dir outputs/followup_b2

echo
echo "Step 4/4: summarize the manual batch-size comparison"
python summarize_runs.py \
  --run-dirs outputs/baseline_b1 outputs/followup_b2 \
  --output-dir outputs

echo
echo "Done. Important artifacts:"
find outputs -maxdepth 3 -type f \
  \( -name "run_config.json" \
  -o -name "communication_groups.json" \
  -o -name "metrics.csv" \
  -o -name "trace_summary.csv" \
  -o -name "diagnosis_summary.md" \
  -o -name "manual_batch_size_sweep.csv" \
  -o -name "*.json" \) \
  | sort
