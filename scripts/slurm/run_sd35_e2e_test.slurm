#!/bin/bash

#SBATCH --output=~/opd-logs/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --array=0
#SBATCH --time=2-0
#SBATCH --mem=50GB
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1

module load python

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

# Prefer project venv if available; otherwise use python from the loaded module.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

JOB_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$REPO_ROOT/outputs/slurm_sd35_e2e/$JOB_TAG"
mkdir -p "$OUT_ROOT/configs"

TRAIN_CFG="$OUT_ROOT/configs/train_sd35_e2e_test.yaml"
RUN_CFG="$OUT_ROOT/configs/run_sd35_e2e_test.yaml"

cat > "$TRAIN_CFG" <<YAML
output_root: "$OUT_ROOT"
run_name: opd-distill-sd35-e2e-test
seed: 42
batch_size: 1
grad_accum_steps: 1
max_train_steps: 2
learning_rate: 0.0001
weight_decay: 0.0001
mixed_precision: true
max_grad_norm: 1.0
log_every: 1
save_every: 1
num_workers: 0
YAML

cat > "$RUN_CFG" <<YAML
experiment_name: opd-sd35-e2e-test
stage: final
seed: 42
output_root: "$OUT_ROOT"
model_config: "$REPO_ROOT/configs/model/sd35.yaml"
train_config: "$TRAIN_CFG"
distill_config: "$REPO_ROOT/configs/distill/sd35_kl.yaml"
eval_config: "$REPO_ROOT/configs/eval/qualtrics.yaml"
prompt_file: "$OUT_ROOT/prompts/prompts.jsonl"
teacher_dataset_manifest: "$OUT_ROOT/datasets/teacher_manifest.json"
YAML

echo "[1/5] Generating prompts"
"$PYTHON_BIN" scripts/generate_prompts.py --config "$RUN_CFG" --num-per-category 1

echo "[2/5] Building teacher dataset"
"$PYTHON_BIN" scripts/build_teacher_dataset.py --config "$RUN_CFG"

echo "[3/5] Running distillation"
"$PYTHON_BIN" scripts/run_distill.py --config "$RUN_CFG"

CKPT_PATH="$(find "$OUT_ROOT/checkpoints" -name final.pt -print -quit || true)"
if [[ -z "$CKPT_PATH" ]]; then
  echo "ERROR: final checkpoint not found under $OUT_ROOT/checkpoints"
  exit 1
fi

echo "[4/5] Running evaluation"
"$PYTHON_BIN" scripts/run_eval.py --config "$RUN_CFG" --ckpt "$CKPT_PATH"

echo "[5/5] Exporting Qualtrics CSV"
"$PYTHON_BIN" scripts/export_qualtrics.py --config "$RUN_CFG"

PROMPT_FILE="$OUT_ROOT/prompts/prompts.jsonl"
MANIFEST_FILE="$OUT_ROOT/datasets/teacher_manifest.json"
EVAL_REPORT="$(find "$OUT_ROOT/metrics" -name eval_report.json -print -quit || true)"
SURVEY_CSV="$(find "$OUT_ROOT/surveys" -name survey.csv -print -quit || true)"
TRAIN_SUMMARY="$(find "$OUT_ROOT/runs" -name train_summary.json -print -quit || true)"

for required_file in "$PROMPT_FILE" "$MANIFEST_FILE" "$CKPT_PATH" "$EVAL_REPORT" "$SURVEY_CSV" "$TRAIN_SUMMARY"; do
  if [[ ! -f "$required_file" ]]; then
    echo "ERROR: Expected file missing: $required_file"
    exit 1
  fi
done

echo "SD3.5 end-to-end test completed successfully."
echo "Output root: $OUT_ROOT"
echo "Checkpoint: $CKPT_PATH"
echo "Eval report: $EVAL_REPORT"
echo "Survey CSV: $SURVEY_CSV"
