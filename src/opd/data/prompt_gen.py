from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "fashion": [
        "A premium {adjective} fashion ad for {audience}, featuring {style} styling and studio lighting",
        "A social media banner advertising {adjective} apparel for {audience} with {style} aesthetics",
    ],
    "food": [
        "A mouthwatering ad for {adjective} {product} aimed at {audience}, cinematic close-up",
        "A digital campaign visual promoting {product} with {style} composition and bold typography",
    ],
    "travel": [
        "An aspirational ad for {adjective} travel packages to {destination}, {style} visual style",
        "A conversion-focused ad creative for {destination} vacations targeting {audience}",
    ],
    "tech": [
        "A modern ad for a {adjective} {product} targeting {audience}, rendered in {style} style",
        "A launch campaign visual for {product} with {style} gradients and product hero shot",
    ],
    "fitness": [
        "A high-energy ad promoting {adjective} fitness programs for {audience}, {style} visual language",
        "A performance marketing image for {product} with dynamic motion and {style} palette",
    ],
}

ADJECTIVES = ["minimal", "bold", "luxury", "vibrant", "futuristic", "clean", "playful"]
AUDIENCES = ["young professionals", "college students", "new parents", "small business owners"]
PRODUCTS = ["smartwatch", "meal kit", "running shoes", "wireless earbuds", "protein shake"]
STYLES = ["high-contrast", "editorial", "retro-modern", "photoreal", "flat-design"]
DESTINATIONS = ["Kyoto", "Barcelona", "Iceland", "Bali", "New York City"]


def _prompt_id(category: str, prompt: str, seed: int) -> str:
    digest = hashlib.sha1(f"{category}|{prompt}|{seed}".encode("utf-8")).hexdigest()
    return digest[:16]


def generate_prompt_records(
    num_per_category: int,
    seed: int,
    categories: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    selected_categories = list(categories) if categories is not None else list(CATEGORY_TEMPLATES)
    records: list[dict[str, object]] = []
    for category in selected_categories:
        templates = CATEGORY_TEMPLATES.get(category)
        if not templates:
            raise ValueError(f"Unknown category '{category}'.")
        for index in range(num_per_category):
            template = rng.choice(templates)
            prompt = template.format(
                adjective=rng.choice(ADJECTIVES),
                audience=rng.choice(AUDIENCES),
                product=rng.choice(PRODUCTS),
                style=rng.choice(STYLES),
                destination=rng.choice(DESTINATIONS),
            )
            prompt_seed = seed + index
            records.append(
                {
                    "prompt_id": _prompt_id(category, prompt, prompt_seed),
                    "prompt": prompt,
                    "category": category,
                    "seed": prompt_seed,
                }
            )
    return records


def save_prompts_jsonl(path: str | Path, records: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
