# Design Notes

## Architecture

The project is organized around a typed config contract and five executable stages:

1. Prompt generation.
2. Teacher dataset build.
3. Distillation training.
4. Quantitative evaluation.
5. Qualitative survey export.

Core package modules:

- `opd.data`: prompt synthesis + dataset utilities.
- `opd.models`: teacher/student wrappers and scheduler.
- `opd.rollout`: on-policy rollout engine producing aligned trajectories.
- `opd.reward`: KL + auxiliary losses and weighted objective.
- `opd.train`: optimizer, checkpointing, trainer loop.
- `opd.eval`: quant metrics and survey export.
- `opd.utils`: config loading, logging, reproducibility helpers.

## Reproducibility

Every run tracks:

- `seed`
- `config_hash`
- `git_commit`
- canonical output directory keyed by `run_name`

## Stage behavior

- `pilot`: uses `backend=mock` to validate training/eval logic quickly.
- `final`: points at SD3.5 model IDs with identical interfaces.

## Failure guards

- Non-finite loss check in trainer loop.
- Gradient clipping.
- Scheduler/timestep alignment through one scheduler shared by teacher/student rollout.
