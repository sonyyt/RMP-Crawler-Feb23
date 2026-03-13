"""
Phase 4 — Individual Review Crawler

For every professor in professors_filtered.csv, fetches ALL individual
reviews from RateMyProfessors via the unofficial GraphQL API, then splits
them into pre-tenure and post-tenure CSV files.

Output layout
─────────────
data/reviews/<slug>_pre.csv   — reviews dated before tenure_year
data/reviews/<slug>_post.csv  — reviews dated >= tenure_year
data/reviews/all_reviews.csv  — all reviews combined with professor metadata
data/reviews/crawl_summary.csv — per-professor counts and crawl status

Usage
─────
    python src/review_crawler.py               # crawl all, skip already-done
    python src/review_crawler.py --no-resume   # re-crawl everything
"""

from __future__ import annotations

import argparse
import base64
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

# ── HTTP session (mirrors rmp_crawler.py setup) ────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": config.RMP_AUTH_HEADER,
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.ratemyprofessors.com/",
        "Origin": "https://www.ratemyprofessors.com",
    }
)

# ── GraphQL query for individual reviews ──────────────────────────────────────

RATINGS_QUERY = """
query RatingsListQuery($count: Int!, $id: ID!, $cursor: String) {
  node(id: $id) {
    ... on Teacher {
      id
      firstName
      lastName
      numRatings
      ratings(first: $count, after: $cursor) {
        edges {
          node {
            id
            comment
            date
            class
            helpfulRating
            clarityRating
            difficultyRating
            wouldTakeAgain
            attendanceMandatory
            textbookUse
            isForOnlineClass
            isForCredit
            tags
            thumbsUpTotal
            thumbsDownTotal
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────


def _graphql(query: str, variables: dict, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = SESSION.post(
                config.RMP_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt + random.uniform(0, 1)
            log.warning(
                "RMP request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    log.error("All %d RMP attempts exhausted.", retries)
    return None


def _rmp_delay() -> None:
    lo, hi = config.RMP_DELAY_RANGE
    time.sleep(random.uniform(lo, hi))


def extract_numeric_id(rmp_url: str) -> Optional[str]:
    """Extract numeric professor ID from an RMP profile URL."""
    m = re.search(r"/professor/(\d+)", rmp_url)
    return m.group(1) if m else None


def encode_professor_id(numeric_id: str) -> str:
    """
    Encode a numeric RMP professor ID to the base64 node ID expected by GraphQL.
    e.g. 1981360  →  base64("Teacher-1981360")
    """
    return base64.b64encode(f"Teacher-{numeric_id}".encode()).decode()


def parse_review_year(date_str: str) -> Optional[int]:
    """Return the 4-digit year from an RMP date string (ISO or plain date)."""
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).year
        return datetime.strptime(date_str[:10], "%Y-%m-%d").year
    except (ValueError, AttributeError):
        m = re.search(r"\b(20\d{2})\b", date_str)
        return int(m.group(1)) if m else None


def make_slug(name: str) -> str:
    """Filesystem-safe slug from a professor name."""
    slug = name.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", slug)


# ── Core review fetcher ────────────────────────────────────────────────────────


def fetch_professor_reviews(encoded_id: str) -> list[dict]:
    """
    Paginate through all reviews for a professor.
    Returns a list of raw review dicts (before professor metadata is added).
    """
    reviews: list[dict] = []
    cursor: Optional[str] = None
    page = 0

    while True:
        page += 1
        variables = {"count": 20, "id": encoded_id, "cursor": cursor}
        data = _graphql(RATINGS_QUERY, variables)
        if data is None:
            log.error("Failed to fetch reviews on page %d", page)
            break

        node = (data.get("data") or {}).get("node")
        if node is None:
            log.warning("GraphQL returned no node for id=%s", encoded_id)
            break

        ratings_data = node.get("ratings") or {}
        edges = ratings_data.get("edges") or []
        page_info = ratings_data.get("pageInfo") or {}

        for edge in edges:
            r = edge.get("node") or {}

            # Normalise tags to a pipe-separated string
            raw_tags = r.get("tags") or []
            tags_str = "|".join(raw_tags) if isinstance(raw_tags, list) else str(raw_tags)

            date_str = r.get("date") or ""
            review_year = parse_review_year(date_str)

            # quality ≈ average of helpfulRating and clarityRating
            helpful = r.get("helpfulRating")
            clarity = r.get("clarityRating")
            if helpful is not None and clarity is not None:
                quality = round((helpful + clarity) / 2, 2)
            elif helpful is not None:
                quality = helpful
            elif clarity is not None:
                quality = clarity
            else:
                quality = None

            reviews.append(
                {
                    "review_id": r.get("id", ""),
                    "date": date_str,
                    "review_year": review_year,
                    "comment": (r.get("comment") or "").strip(),
                    "class_name": r.get("class") or "",
                    "quality": quality,
                    "helpful_rating": helpful,
                    "clarity_rating": clarity,
                    "difficulty": r.get("difficultyRating"),
                    "would_take_again": r.get("wouldTakeAgain"),
                    "attendance_mandatory": r.get("attendanceMandatory") or "",
                    "textbook_use": r.get("textbookUse"),
                    "is_online": r.get("isForOnlineClass"),
                    "is_for_credit": r.get("isForCredit"),
                    "tags": tags_str,
                    "thumbs_up": r.get("thumbsUpTotal") or 0,
                    "thumbs_down": r.get("thumbsDownTotal") or 0,
                }
            )

        log.debug("Page %d: %d total reviews so far", page, len(reviews))

        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        _rmp_delay()

    return reviews


# ── Main pipeline ──────────────────────────────────────────────────────────────


def crawl_professor_reviews(
    professors_csv: Path = config.PROFESSORS_FILTERED_CSV,
    output_dir: Path = config.REVIEWS_DIR,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Crawl reviews for every professor in professors_csv.

    For each professor the function writes:
      <output_dir>/<slug>_pre.csv   — reviews before tenure_year
      <output_dir>/<slug>_post.csv  — reviews from tenure_year onward

    It also writes:
      <output_dir>/all_reviews.csv      — all reviews combined
      <output_dir>/crawl_summary.csv    — per-professor counts

    Returns the combined DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_profs = pd.read_csv(professors_csv)
    log.info("Loaded %d rows from %s", len(df_profs), professors_csv)

    # Deduplicate: same RMP URL = same professor profile
    df_profs = df_profs.drop_duplicates(subset=["rmp_url"]).reset_index(drop=True)
    log.info("Unique professor profiles: %d", len(df_profs))

    all_reviews: list[dict] = []
    summary_rows: list[dict] = []

    for idx, row in df_profs.iterrows():
        name: str = str(row["name"])
        slug: str = make_slug(name)
        university: str = str(row.get("university", ""))
        rmp_url: str = str(row.get("rmp_url", ""))
        discipline: str = str(row.get("discipline", ""))
        carnegie: str = str(row.get("carnegie", ""))
        confidence: float = float(row.get("confidence", 0))
        confidence_label: str = str(row.get("confidence_label", ""))

        raw_ty = row.get("tenure_year")
        tenure_year: Optional[int] = int(raw_ty) if pd.notna(raw_ty) else None

        log.info(
            "[%d/%d] %s | %s | tenure=%s",
            idx + 1,
            len(df_profs),
            name,
            university,
            tenure_year,
        )

        pre_path = output_dir / f"{slug}_pre.csv"
        post_path = output_dir / f"{slug}_post.csv"

        # Resume: load already-crawled data
        if resume and pre_path.exists() and post_path.exists():
            log.info("  → Already done, loading cached CSVs")
            pre_df = pd.read_csv(pre_path)
            post_df = pd.read_csv(post_path)
            all_reviews.extend(pre_df.to_dict("records"))
            all_reviews.extend(post_df.to_dict("records"))
            summary_rows.append(
                {
                    "professor_name": name,
                    "professor_slug": slug,
                    "university": university,
                    "tenure_year": tenure_year,
                    "total_reviews": len(pre_df) + len(post_df),
                    "pre_tenure_reviews": len(pre_df),
                    "post_tenure_reviews": len(post_df),
                    "has_sufficient_pre": len(pre_df) >= config.MIN_REVIEWS_PER_PERIOD,
                    "has_sufficient_post": len(post_df) >= config.MIN_REVIEWS_PER_PERIOD,
                    "status": "cached",
                }
            )
            continue

        # Extract and encode professor ID
        numeric_id = extract_numeric_id(rmp_url)
        if numeric_id is None:
            log.warning("  → Cannot extract ID from URL: %s — skipping", rmp_url)
            summary_rows.append(
                {
                    "professor_name": name,
                    "professor_slug": slug,
                    "university": university,
                    "tenure_year": tenure_year,
                    "total_reviews": 0,
                    "pre_tenure_reviews": 0,
                    "post_tenure_reviews": 0,
                    "has_sufficient_pre": False,
                    "has_sufficient_post": False,
                    "status": "no_id",
                }
            )
            continue

        encoded_id = encode_professor_id(numeric_id)

        # Fetch all reviews from RMP
        reviews = fetch_professor_reviews(encoded_id)
        log.info("  → Fetched %d reviews", len(reviews))

        if not reviews:
            log.warning("  → No reviews returned")
            pd.DataFrame().to_csv(pre_path, index=False)
            pd.DataFrame().to_csv(post_path, index=False)
            summary_rows.append(
                {
                    "professor_name": name,
                    "professor_slug": slug,
                    "university": university,
                    "tenure_year": tenure_year,
                    "total_reviews": 0,
                    "pre_tenure_reviews": 0,
                    "post_tenure_reviews": 0,
                    "has_sufficient_pre": False,
                    "has_sufficient_post": False,
                    "status": "no_reviews",
                }
            )
            _rmp_delay()
            continue

        # Attach professor metadata and classify period
        for r in reviews:
            r["professor_name"] = name
            r["professor_slug"] = slug
            r["university"] = university
            r["discipline"] = discipline
            r["carnegie"] = carnegie
            r["tenure_year"] = tenure_year
            r["confidence"] = confidence
            r["confidence_label"] = confidence_label

            ry = r.get("review_year")
            if tenure_year is not None and ry is not None:
                r["years_from_tenure"] = ry - tenure_year
                r["period"] = "post_tenure" if ry >= tenure_year else "pre_tenure"
            else:
                r["years_from_tenure"] = None
                r["period"] = "unknown"

        pre_reviews = [r for r in reviews if r.get("period") == "pre_tenure"]
        post_reviews = [r for r in reviews if r.get("period") == "post_tenure"]

        pre_df = pd.DataFrame(pre_reviews)
        post_df = pd.DataFrame(post_reviews)
        pre_df.to_csv(pre_path, index=False)
        post_df.to_csv(post_path, index=False)
        log.info(
            "  → Pre: %d  Post: %d  Unknown: %d",
            len(pre_reviews),
            len(post_reviews),
            len(reviews) - len(pre_reviews) - len(post_reviews),
        )

        all_reviews.extend(reviews)
        summary_rows.append(
            {
                "professor_name": name,
                "professor_slug": slug,
                "university": university,
                "tenure_year": tenure_year,
                "total_reviews": len(reviews),
                "pre_tenure_reviews": len(pre_reviews),
                "post_tenure_reviews": len(post_reviews),
                "has_sufficient_pre": len(pre_reviews) >= config.MIN_REVIEWS_PER_PERIOD,
                "has_sufficient_post": len(post_reviews) >= config.MIN_REVIEWS_PER_PERIOD,
                "status": "crawled",
            }
        )

        _rmp_delay()

    # Save combined outputs
    all_df = pd.DataFrame(all_reviews)
    all_df.to_csv(config.REVIEWS_ALL_FILE, index=False)
    log.info("All reviews → %s  (%d rows)", config.REVIEWS_ALL_FILE, len(all_df))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.REVIEWS_CRAWL_SUMMARY, index=False)
    log.info("Crawl summary → %s", config.REVIEWS_CRAWL_SUMMARY)

    # Quick stats
    suf = summary_df[summary_df["has_sufficient_pre"] & summary_df["has_sufficient_post"]]
    log.info(
        "Professors with ≥%d reviews in BOTH periods: %d / %d",
        config.MIN_REVIEWS_PER_PERIOD,
        len(suf),
        len(summary_df),
    )

    return all_df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl RMP reviews per professor")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-crawl all professors even if CSVs already exist",
    )
    parser.add_argument(
        "--professors-csv",
        type=Path,
        default=config.PROFESSORS_FILTERED_CSV,
        help="Path to professors_filtered.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    crawl_professor_reviews(
        professors_csv=args.professors_csv,
        output_dir=config.REVIEWS_DIR,
        resume=not args.no_resume,
    )
