# Diffusion Distillation Notebook

This repository has been reduced to a single Google Colab notebook, [opd.ipynb](/Users/jai18/Desktop/CS234/diffusion-distillation/opd.ipynb), plus this README.

The notebook is a cleaned, self-contained version of the original training workflow for on-policy distillation of Stable Diffusion 3 models. It:

- installs the Colab dependencies,
- generates a small prompt set or reuses an existing JSONL file,
- trains a student `stable-diffusion-3-medium-diffusers` model against a `stable-diffusion-3.5-medium` teacher,
- saves checkpoints, streamed metrics, and a training summary,
- renders teacher, original student, and distilled student evaluation images,
- zips the evaluation directory and attempts a Colab download at the end.

## Running It

1. Open [opd.ipynb](/Users/jai18/Desktop/CS234/diffusion-distillation/opd.ipynb) in Google Colab.
2. Switch the runtime to a CUDA GPU.
3. If your Hugging Face access is gated, uncomment `notebook_login()` in the config cell.
4. Run the notebook from top to bottom.

## Default Outputs

The notebook writes everything under `/content/opd_outputs`:

- `prompts/prompts.jsonl`
- `checkpoints/<run_name>/final.pt`
- `metrics/<run_name>/metrics.jsonl`
- `runs/<run_name>/train_summary.json`
- `eval_images/<run_name>/...`
- `eval_images.zip`

## Notes

- The default config is a Colab-sized smoke test, not a full training run.
- Increase `MAX_TRAIN_STEPS`, prompt count, and image size only after the base run succeeds.
- The repository no longer depends on the old `src/`, `scripts/`, `configs/`, or `tests/` tree.
