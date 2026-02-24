from __future__ import annotations

from typing import Iterable


def preference_rate(votes: Iterable[str], positive_label: str = "distilled") -> float:
    votes = list(votes)
    if not votes:
        return 0.0
    positive = sum(1 for vote in votes if vote == positive_label)
    return positive / len(votes)
