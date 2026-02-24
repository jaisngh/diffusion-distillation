# PLAN: On-Policy Distillation for Diffusion Models (CS234)

## Summary
- Build an end-to-end, single-GPU-first pipeline for on-policy distillation of diffusion models, targeting digital-ad prompt generation.
- Use a two-stage execution path:
  - Stage A (pilot): validate pipeline on a smaller teacher/student pair.
  - Stage B (final): run the same pipeline on SD3.5 Large (teacher) -> SD3.5 Medium (student).
- First execution step includes repository initialization, directory scaffolding, and writing this plan to `PLAN.md`.

## Repository Bootstrap (first implementation task)
1. Initialize git:
   - `git init`
2. Create project structure:
   - `mkdir -p configs/{model,train,distill,eval} src/opd/{data,models,rollout,reward,train,eval,utils} scripts tests/{unit,integration} docs experiments/{runs,artifacts} outputs/{checkpoints,metrics,images,surveys}`
3. Create baseline files:
   - `PLAN.md`, `README.md`, `.gitignore`, `pyproject.toml`, `requirements.txt`
4. Add `.gitignore` entries for model caches, checkpoints, generated images, run artifacts, and survey exports.

## Proposed Directory Layout
```text
diffusion-distillation/
  PLAN.md
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  configs/
    model/{pilot.yaml,sd35.yaml}
    train/{single_gpu.yaml}
    distill/{kl_per_timestep.yaml}
    eval/{quant.yaml,qualtrics.yaml}
  src/opd/
    data/{prompt_gen.py,dataset.py}
    models/{teacher.py,student.py,schedulers.py,pipeline_factory.py}
    rollout/{trajectory.py,sampling.py}
    reward/{kl.py,mse.py,losses.py}
    train/{loop.py,optimizer.py,checkpoint.py}
    eval/{quant_metrics.py,qual_metrics.py,survey_export.py}
    utils/{seed.py,logging.py,device.py,io.py}
  scripts/
    generate_prompts.py
    build_teacher_dataset.py
    run_distill.py
    run_eval.py
    export_qualtrics.py
  tests/
    unit/{test_kl.py,test_rollout_alignment.py,test_config_load.py}
    integration/{test_smoke_distill.py,test_eval_pipeline.py}
  experiments/{runs,artifacts}
  outputs/{checkpoints,metrics,images,surveys}
  docs/{design.md,experiment_log.md}
```

## Task Partition (decision-complete)
### Task 1: Project scaffolding and config contracts
- Implement config loader and typed config schema for model, training, reward, and eval.
- Define run naming, output directory conventions, and reproducibility keys (seed, commit hash, config hash).
- Deliverable: `scripts/run_distill.py --config ...` can parse and print resolved config without training.

### Task 2: Synthetic prompt generation and data build
- Create prompt generator constrained to digital ad categories (fashion, food, travel, tech, etc.).
- Generate `prompts.jsonl` with fields: `prompt_id`, `prompt`, `category`, `seed`.
- Build teacher outputs per prompt (image + optional latent trajectory snapshot policy).
- Deliverable: reproducible synthetic dataset with metadata manifest.

### Task 3: Model wrappers and trajectory alignment
- Build wrappers for teacher and student inference with shared scheduler/timestep alignment.
- Implement rollout API that generates student denoising trajectories and teacher reference trajectory at identical timesteps.
- Deliverable: one prompt yields aligned `(z_t_student, z_t_teacher)` pairs for all timesteps.

### Task 4: Reward/loss implementation
- Implement per-timestep KL divergence objective between teacher and student predicted distributions.
- Implement auxiliary losses:
  - epsilon-prediction MSE (stability),
  - final latent/image MSE (monitoring and ablations).
- Define weighted aggregate objective and logging for each component.
- Deliverable: loss values are finite and decrease in short smoke runs.

### Task 5: On-policy distillation training loop
- Implement training loop with:
  - rollout generation from current student policy,
  - teacher grading on same prompts/timesteps,
  - backprop with gradient accumulation, mixed precision, checkpointing.
- Add memory-safe defaults for single GPU.
- Deliverable: resumable training producing checkpoints and metrics JSONL/CSV.

### Task 6: Quantitative evaluation pipeline
- Implement pre/post distillation comparison on held-out prompts:
  - teacher-student image MSE,
  - optional CLIP-based similarity and diversity metrics.
- Output summary tables and plots in `outputs/metrics`.
- Deliverable: `scripts/run_eval.py` generates before/after report.

### Task 7: Qualitative evaluation (Qualtrics-ready)
- Create side-by-side image pair exporter (distilled vs non-distilled student).
- Generate randomized survey CSV with blinded labels and prompt text.
- Deliverable: `scripts/export_qualtrics.py` outputs importable Qualtrics package artifacts.

### Task 8: Stage migration from pilot to SD3.5
- Keep interfaces unchanged; switch only model config and compute knobs.
- Run pilot acceptance first, then SD3.5 final experiments.
- Deliverable: identical commands with different config files.

### Task 9: Documentation and reproducibility hardening
- Document setup, runbook, experiment registry format, failure recovery.
- Add “minimal reproducible run” and “full SD3.5 run” instructions.
- Deliverable: `README.md` and `docs/design.md` sufficient for handoff.

## Public APIs / Interfaces / Types
- CLI entrypoints:
  - `scripts/generate_prompts.py --config <path>`
  - `scripts/build_teacher_dataset.py --config <path>`
  - `scripts/run_distill.py --config <path> [--resume <ckpt>] [--stage pilot|final]`
  - `scripts/run_eval.py --config <path> --ckpt <path>`
  - `scripts/export_qualtrics.py --config <path> --run-id <id>`
- Core Python interfaces:
  - `RolloutEngine.rollout(prompt_batch) -> TrajectoryBatch`
  - `RewardComputer.compute(trajectory_batch) -> LossBreakdown`
  - `DistillTrainer.step(batch) -> TrainMetrics`
- Data schemas:
  - `prompts.jsonl`: `prompt_id:str, prompt:str, category:str, seed:int`
  - `metrics.jsonl`: `step:int, total_loss:float, kl_loss:float, mse_loss:float, lr:float, grad_norm:float`
  - `survey.csv`: `question_id, prompt_id, prompt_text, image_a_path, image_b_path, randomized_label`

## Test Cases and Scenarios
- Unit tests:
  - KL loss correctness and numerical stability.
  - Timestep/shape alignment between teacher and student rollouts.
  - Config validation and default resolution.
- Integration tests:
  - Smoke distillation run on tiny subset (2-4 prompts, few timesteps, few steps).
  - End-to-end eval generation from checkpoint.
  - Qualtrics CSV export integrity and randomization correctness.
- Acceptance criteria:
  - No NaNs/Infs across smoke training.
  - Post-distillation student improves vs pre-distillation on held-out prompt MSE.
  - Re-running with same seed/config reproduces metrics within tolerance.
  - All scripts run via documented commands on single-GPU setup.

## Failure Modes and Mitigations
- OOM on single GPU: lower batch size, gradient accumulation, mixed precision, checkpointing.
- Scheduler mismatch: enforce shared timesteps and assert identical schedule length.
- Reward collapse/noisy training: clip gradients, warm-start with MSE auxiliary loss, lower KL weight early.
- Prompt overfitting: enforce held-out categories and report split-wise metrics.

## Assumptions and Defaults
- Chosen defaults:
  - Two-stage path (`pilot -> SD3.5`) to de-risk implementation.
  - Single-GPU-first optimization.
  - End-to-end baseline delivery before optimization-heavy work.
- Assumed toolchain:
  - Python + PyTorch + Diffusers + Accelerate.
- Scope boundaries:
  - Survey generation is in scope; survey response analysis automation is out of initial scope.
  - Full-scale hyperparameter sweeps are out of initial scope; include one ablation pass only.
