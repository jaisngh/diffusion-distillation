#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opd.data.prompt_gen import CATEGORY_TEMPLATES, generate_prompt_records, save_prompts_jsonl
from opd.utils.io import ensure_output_layout, load_app_config, save_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic prompts for OPD.")
    parser.add_argument("--config", required=True, help="Path to top-level run config YAML.")
    parser.add_argument("--stage", default=None, help="Optional stage override: pilot|final.")
    parser.add_argument(
        "--num-per-category",
        type=int,
        default=8,
        help="Prompts to generate per category.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(CATEGORY_TEMPLATES.keys()),
        help="Comma-separated categories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config, stage_override=args.stage)
    paths = ensure_output_layout(config)

    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    records = generate_prompt_records(
        num_per_category=args.num_per_category,
        seed=config.seed,
        categories=categories,
    )
    save_prompts_jsonl(config.prompt_file, records)
    save_json(
        paths["prompts"] / "prompts_manifest.json",
        {
            "num_records": len(records),
            "categories": categories,
            "prompt_file": config.prompt_file,
            "seed": config.seed,
        },
    )
    print(json.dumps({"prompt_file": config.prompt_file, "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
