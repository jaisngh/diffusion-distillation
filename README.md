# On-Policy Distillation for Diffusion Models (CS234)

This repository contains an end-to-end scaffold for on-policy distillation (OPD) of diffusion models with three hardware-targeted run configs:

- `pilot`: mock teacher/student backend for fast validation and smoke tests.
- `cuda`: SD3.5 Medium -> SD3 Medium (`diffusers`) for a single CUDA GPU.
- `apple`: local-model Apple Silicon quick config using `mps`.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate synthetic prompts:
   ```bash
   python scripts/generate_prompts.py --config configs/pilot_run.yaml --num-per-category 4
   ```
3. Build teacher dataset manifest:
   ```bash
   python scripts/build_teacher_dataset.py --config configs/pilot_run.yaml
   ```
4. Dry-run distillation config:
   ```bash
   python scripts/run_distill.py --config configs/pilot_run.yaml --dry-run
   ```
5. Run smoke distillation:
   ```bash
   python scripts/run_distill.py --config configs/pilot_run.yaml
   ```
6. Evaluate:
   ```bash
   python scripts/run_eval.py --config configs/pilot_run.yaml
   ```
7. Export Qualtrics CSV:
   ```bash
   python scripts/export_qualtrics.py --config configs/pilot_run.yaml
   ```

## Config Profiles

- Lightweight CPU local (default):
  ```bash
  python scripts/run_distill.py --config configs/pilot_run.yaml
  ```
- Single CUDA GPU:
  ```bash
  python scripts/run_distill.py --config configs/sd35m_sd3m_run.yaml
  ```
- Apple Silicon (MPS) quick:
  ```bash
  python scripts/run_distill.py --config configs/sd35m_sd3m_m1_local_quick_run.yaml
  ```

## Stable Diffusion setup

The CUDA config loads:

- Teacher: `stabilityai/stable-diffusion-3.5-medium`
- Student: `stabilityai/stable-diffusion-3-medium-diffusers`

Recommended before first run:

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

If model access is gated, authenticate first:

```bash
huggingface-cli login
```

## Outputs

- Prompts: `outputs/prompts/prompts.jsonl`
- Teacher manifest: `outputs/datasets/teacher_manifest.json`
- Checkpoints: `outputs/checkpoints/<run_name>/`
- Metrics: `outputs/metrics/<run_name>/metrics.jsonl`
- Evaluation report: `outputs/metrics/<run_name>/eval_report.json`
- Survey CSV: `outputs/surveys/<run_name>/survey.csv`

## Notes

- `diffusers` backend is implemented for SD3.5 teacher/student rollouts.
- The pilot path remains the fast local path for smoke tests and CI.
- Device selection is automatic: CUDA is used when available, otherwise Apple Silicon uses `mps`, otherwise CPU.
