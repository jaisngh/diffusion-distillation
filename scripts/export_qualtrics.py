#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opd.data.dataset import PromptDataset
from opd.eval.survey_export import build_survey_rows, save_survey_csv
from opd.utils.io import ensure_output_layout, load_app_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qualtrics-ready survey CSV.")
    parser.add_argument("--config", required=True, help="Path to top-level run config YAML.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier for metadata.")
    parser.add_argument("--stage", default=None, help="Optional stage override: pilot|final.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config, stage_override=args.stage)
    paths = ensure_output_layout(config)
    dataset = PromptDataset.from_jsonl(config.prompt_file)
    selected = dataset.records[: config.eval.export_num_pairs]
    selected_rows = [
        {"prompt_id": row.prompt_id, "prompt": row.prompt, "category": row.category, "seed": row.seed}
        for row in selected
    ]

    baseline_map = {
        row["prompt_id"]: str(paths["images"] / "baseline" / f'{row["prompt_id"]}.png')
        for row in selected_rows
    }
    distilled_map = {
        row["prompt_id"]: str(paths["images"] / "distilled" / f'{row["prompt_id"]}.png')
        for row in selected_rows
    }

    rows = build_survey_rows(
        prompt_rows=selected_rows,
        distilled_image_paths=distilled_map,
        baseline_image_paths=baseline_map,
        seed=config.seed,
    )
    target = save_survey_csv(paths["surveys"] / "survey.csv", rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id or config.run_id,
                "survey_csv": str(target),
                "num_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
