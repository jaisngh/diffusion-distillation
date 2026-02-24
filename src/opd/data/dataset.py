from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class PromptRecord:
    prompt_id: str
    prompt: str
    category: str
    seed: int


class PromptDataset:
    def __init__(self, records: list[PromptRecord]) -> None:
        self.records = records

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PromptDataset":
        target = Path(path)
        records: list[PromptRecord] = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append(
                    PromptRecord(
                        prompt_id=row["prompt_id"],
                        prompt=row["prompt"],
                        category=row["category"],
                        seed=int(row["seed"]),
                    )
                )
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def prompts(self) -> list[str]:
        return [record.prompt for record in self.records]


def iter_prompt_batches(
    dataset: PromptDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[list[PromptRecord]]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [dataset.records[idx] for idx in batch_indices]
