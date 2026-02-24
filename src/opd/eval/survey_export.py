from __future__ import annotations

import csv
import random
from pathlib import Path


def build_survey_rows(
    prompt_rows: list[dict[str, object]],
    distilled_image_paths: dict[str, str],
    baseline_image_paths: dict[str, str],
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(prompt_rows, start=1):
        prompt_id = str(row["prompt_id"])
        prompt_text = str(row["prompt"])
        distilled = distilled_image_paths.get(prompt_id, "")
        baseline = baseline_image_paths.get(prompt_id, "")
        if rng.random() < 0.5:
            image_a_path = distilled
            image_b_path = baseline
            randomized_label = "A=distilled"
        else:
            image_a_path = baseline
            image_b_path = distilled
            randomized_label = "B=distilled"
        rows.append(
            {
                "question_id": f"q_{idx:04d}",
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "image_a_path": image_a_path,
                "image_b_path": image_b_path,
                "randomized_label": randomized_label,
            }
        )
    return rows


def save_survey_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "prompt_id",
        "prompt_text",
        "image_a_path",
        "image_b_path",
        "randomized_label",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target
