# Colab On-Policy Distillation Guide

This repo includes a single self-contained Colab training script:

- `scripts/run_onpolicy_distill_colab.py`

The script does **on-policy distillation**:
- student rolls out latents,
- teacher supervises on those current student states,
- student is optimized with weighted `KL + epsilon MSE + latent MSE`.

## What the script contains

- Colab bootstrap helpers (optional dependency install + HF model prefetch)
- Prompt generation (built into the same file)
- SD3 teacher/student model loading
- On-policy rollout + distillation loss computation
- Logging (`metrics.jsonl`, `metrics.csv`) and checkpoints

No `src/opd` imports are required.

## Quick start in Colab

1. Open `scripts/run_onpolicy_distill_colab.py`.
2. Edit at least:
   - `PROMPTS_JSONL` (where prompts are read/written)
3. Optional setup flags near the top:
   - `PRINT_COLAB_SETUP`
   - `INSTALL_DEPS`
   - `PREFETCH_MODELS`
   - `COLAB_MODEL_CACHE`
4. Run once with setup enabled (if needed), then disable it:
   - Set flags and/or temporarily uncomment `dep_install()` in `main()`.
   - After first successful setup, comment it back out to avoid reinstall overhead.
5. Run training:
   - `python scripts/run_onpolicy_distill_colab.py`

## Important config blocks

The script is organized with hard-coded sections:

- **Model setup** (teacher/student IDs, dtype, image size, freeze knobs)
- **Train setup** (batch size, grad accumulation, LR, mixed precision)
- **Distill setup** (weights for KL/MSE losses, KL warmup)
- **Prompt setup** (auto-generate prompts + categories/templates)
- **Colab setup options** (install/prefetch switches)

## Outputs

By default, outputs are written under `outputs_colab/`:

- `outputs_colab/metrics/<run_name>/metrics.jsonl`
- `outputs_colab/metrics/<run_name>/metrics.csv`
- `outputs_colab/checkpoints/<run_name>/step_*.pt`
- `outputs_colab/checkpoints/<run_name>/final.pt`
- `outputs_colab/runs/<run_name>/train_summary.json`

## Notes

- Requires a CUDA-enabled Colab runtime.
- Prompt generation can be disabled by setting `GENERATE_PROMPTS = False`.
- When prompt generation is disabled, `PROMPTS_JSONL` must already exist.
