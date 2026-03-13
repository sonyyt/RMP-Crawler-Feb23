"""
Phase 1 — University Selection
Randomly samples NUM_UNIVERSITIES institutions from the R1/R2 list.
"""

from __future__ import annotations
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


def load_universities(path: Path = config.UNIVERSITIES_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        universities = json.load(f)
    log.info("Loaded %d universities from %s", len(universities), path)
    return universities


def deduplicate(universities: list[dict]) -> list[dict]:
    """Remove duplicate entries by (name, state)."""
    seen: set[tuple] = set()
    unique = []
    for u in universities:
        key = (u["name"].lower(), u["state"])
        if key not in seen:
            seen.add(key)
            unique.append(u)
    return unique


def select_universities(
    seed: int = config.RANDOM_SEED,
    n: int = config.NUM_UNIVERSITIES,
    path: Path = config.UNIVERSITIES_FILE,
) -> list[dict]:
    universities = deduplicate(load_universities(path))
    log.info("Unique universities after dedup: %d", len(universities))

    if n > len(universities):
        log.warning(
            "Requested %d universities but only %d available; using all.", n, len(universities)
        )
        n = len(universities)

    rng = random.Random(seed)
    selected = rng.sample(universities, n)
    log.info("Selected %d universities (seed=%d)", len(selected), seed)
    return selected


def save_selected(selected: list[dict], path: Path = config.SELECTED_UNIS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    log.info("Saved selected universities to %s", path)


def run() -> list[dict]:
    selected = select_universities()
    save_selected(selected)

    r1_count = sum(1 for u in selected if u["carnegie"] == "R1")
    r2_count = sum(1 for u in selected if u["carnegie"] == "R2")
    log.info("Breakdown: %d R1, %d R2", r1_count, r2_count)

    return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    unis = run()
    for u in unis:
        print(f"[{u['carnegie']}] {u['name']} ({u['state']})")
