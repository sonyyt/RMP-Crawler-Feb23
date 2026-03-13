"""
Phase 5 — Build Analysis Dataset

Reads data/raw_rmp_reviews.csv and produces data/processed_reviews.csv
with additional computed features:
  - sentiment score (VADER compound, -1 to 1)
  - review_length (word count)
  - Restricts to professors with known tenure_year
  - Clips relative_year to [-10, +10] window
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

RELATIVE_YEAR_WINDOW = (-10, 10)


def build_analysis_dataset(
    raw_path: Path = config.DATA_DIR / "raw_rmp_reviews.csv",
    out_path: Path = config.DATA_DIR / "processed_reviews.csv",
) -> pd.DataFrame:

    df = pd.read_csv(raw_path)
    log.info("Raw reviews loaded: %d rows", len(df))

    # ── Keep only professors with a known tenure_year ─────────────────────────
    df = df[df["tenure_year"].notna() & df["review_year"].notna()].copy()
    df["tenure_year"] = df["tenure_year"].astype(int)
    df["review_year"] = df["review_year"].astype(int)
    df["relative_year"] = df["review_year"] - df["tenure_year"]
    log.info("After filtering to professors with tenure_year: %d rows", len(df))

    # ── Clip to analysis window ───────────────────────────────────────────────
    lo, hi = RELATIVE_YEAR_WINDOW
    df = df[(df["relative_year"] >= lo) & (df["relative_year"] <= hi)].copy()
    log.info("After clipping to relative_year [%d, %d]: %d rows", lo, hi, len(df))

    # ── Sentiment ─────────────────────────────────────────────────────────────
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text: str) -> float:
        if not isinstance(text, str) or text.strip() == "":
            return float("nan")
        return analyzer.polarity_scores(text)["compound"]

    log.info("Computing sentiment scores…")
    df["sentiment"] = df["review_text"].apply(get_sentiment)

    # ── Review length ─────────────────────────────────────────────────────────
    df["review_length"] = df["review_text"].apply(
        lambda t: len(str(t).split()) if isinstance(t, str) else 0
    )

    # ── Binary would_take_again ───────────────────────────────────────────────
    # RMP stores as 0/1 or True/False
    df["would_take_again_bin"] = pd.to_numeric(df["would_take_again"], errors="coerce")

    # ── Pre / post indicator ──────────────────────────────────────────────────
    df["post_tenure"] = (df["relative_year"] >= 0).astype(int)

    # ── Select and order output columns ──────────────────────────────────────
    out_cols = [
        "professor_id", "professor_name", "university", "carnegie",
        "tenure_year", "review_id", "review_date", "review_year",
        "relative_year", "post_tenure",
        "overall_rating", "clarity_rating", "difficulty",
        "would_take_again_bin", "sentiment", "review_length",
        "review_text", "course", "is_online", "grade", "rating_tags",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols]

    df.to_csv(out_path, index=False)
    log.info("Saved processed dataset → %s  (%d rows, %d professors)",
             out_path, len(df), df["professor_id"].nunique())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_analysis_dataset()
