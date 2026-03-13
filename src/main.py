"""
Main pipeline orchestrator.

Usage:
  python src/main.py                  # run all phases
  python src/main.py --phase 1        # university selection only
  python src/main.py --phase 2        # RMP crawl only
  python src/main.py --phase 3        # tenure estimation (v2: Google + LLM)
  python src/main.py --phase 3 --validate  # validate on first 100 professors
  python src/main.py --resume         # skip already-processed items
  python src/main.py --phase 2 --resume
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Ensure log and data directories exist
config.LOG_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(level: str = config.LOG_LEVEL) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=numeric, format=fmt, handlers=handlers)


# ── Phase runners ──────────────────────────────────────────────────────────────

def phase1() -> list[dict]:
    from src.university_selector import run as select_universities
    log = logging.getLogger("phase1")
    log.info("═══ Phase 1: University selection ═══")
    universities = select_universities()
    log.info("Done. %d universities saved to %s", len(universities), config.SELECTED_UNIS_FILE)
    return universities


def phase2(resume: bool = False) -> list[dict]:
    from src.rmp_crawler import crawl_universities
    log = logging.getLogger("phase2")
    log.info("═══ Phase 2: RMP crawling ═══")

    if not config.SELECTED_UNIS_FILE.exists():
        log.error("Selected universities file not found. Run Phase 1 first.")
        sys.exit(1)

    with open(config.SELECTED_UNIS_FILE, "r", encoding="utf-8") as f:
        selected = json.load(f)

    professors = crawl_universities(selected, resume=resume)
    log.info(
        "Done. %d qualifying professors saved to %s", len(professors), config.PROFESSORS_RMP_FILE
    )
    return professors


def phase3(resume: bool = False, validate: bool = False) -> list[dict]:
    from src.tenure_estimator_v2 import estimate_tenure_for_all
    log = logging.getLogger("phase3")
    log.info("═══ Phase 3 v2: Tenure estimation (Google + LLM) ═══")

    # Prefer the filtered subset (>=100 ratings) if it exists; fall back to full set
    phase3_input = config.DATA_DIR / "professors_phase3_subset.json"
    if not phase3_input.exists():
        phase3_input = config.PROFESSORS_RMP_FILE
    if not phase3_input.exists():
        log.error("RMP professors file not found. Run Phase 2 first.")
        sys.exit(1)

    log.info("Phase 3 reading from %s", phase3_input)
    with open(phase3_input, "r", encoding="utf-8") as f:
        professors = json.load(f)

    log.info("Loaded %d professors", len(professors))
    results = estimate_tenure_for_all(
        professors, resume=resume, validation_mode=validate,
    )
    log.info("Done. %d results saved to %s", len(results), config.PROFESSORS_FINAL_V2_FILE)
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RMP Professor Tenure Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only this phase (1, 2, or 3). Omit to run all phases.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed universities/professors (checkpoint resume).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Phase 3 on validation subset only (first 100 professors).",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("main")

    log.info("RMP Tenure Crawler starting (seed=%d, min_ratings=%d, min_confidence=%.2f)",
             config.RANDOM_SEED, config.MIN_RATINGS, config.MIN_CONFIDENCE)

    if args.phase is None or args.phase == 1:
        phase1()
    if args.phase is None or args.phase == 2:
        phase2(resume=args.resume)
    if args.phase is None or args.phase == 3:
        phase3(resume=args.resume, validate=args.validate)

    if args.phase is None:
        log.info("All phases complete.")
        _print_summary()


def _print_summary() -> None:
    log = logging.getLogger("summary")
    try:
        with open(config.PROFESSORS_FINAL_V2_FILE, "r", encoding="utf-8") as f:
            final = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    total = len(final)
    high = sum(1 for p in final if p.get("confidence_label") == "high")
    medium = sum(1 for p in final if p.get("confidence_label") == "medium")
    low = sum(1 for p in final if p.get("confidence_label") == "low")

    log.info("─── Final Output Summary ───")
    log.info("Total professors with tenure estimates : %d", total)
    log.info("  High confidence (≥0.85)              : %d", high)
    log.info("  Medium confidence (0.60–0.84)        : %d", medium)
    log.info("  Low confidence (0.35–0.59)           : %d", low)

    # Top disciplines
    from collections import Counter
    disciplines = Counter(p.get("discipline", "Unknown") for p in final)
    log.info("Top disciplines:")
    for disc, count in disciplines.most_common(10):
        log.info("  %-40s %d", disc, count)

    log.info("Output file : %s", config.PROFESSORS_FINAL_V2_FILE)
    log.info("CSV file    : %s", config.PROFESSORS_FINAL_V2_CSV)


if __name__ == "__main__":
    main()
