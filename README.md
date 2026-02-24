# On-Policy Distillation for Diffusion Models (CS234)

This repository contains an end-to-end scaffold for on-policy distillation (OPD) of diffusion models with a two-stage path:

- `pilot`: mock teacher/student backend for fast validation and smoke tests.
- `final`: SD3.5 Large -> SD3.5 Medium configuration wiring for full-scale runs.

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

## Stage Switch

- Pilot stage (default):
  ```bash
  python scripts/run_distill.py --config configs/pilot_run.yaml --stage pilot
  ```
- Final stage:
  ```bash
  python scripts/run_distill.py --config configs/pilot_run.yaml --stage final
  ```

## Outputs

- Prompts: `outputs/prompts/prompts.jsonl`
- Teacher manifest: `outputs/datasets/teacher_manifest.json`
- Checkpoints: `outputs/checkpoints/<run_name>/`
- Metrics: `outputs/metrics/<run_name>/metrics.jsonl`
- Evaluation report: `outputs/metrics/<run_name>/eval_report.json`
- Survey CSV: `outputs/surveys/<run_name>/survey.csv`

## Notes

- The `diffusers` backend is declared for the SD3.5 stage but not yet implemented in this scaffold.
- The pilot path is fully runnable with local mock models and is used by tests.
