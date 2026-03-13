"""
Phase 4 — RMP Individual Review Scraper

Fetches every review for each professor in professors_final_v2.csv using
the RMP unofficial GraphQL API.  Saves:
  - data/cache/{professor_id}.json   (per-professor raw cache)
  - data/raw_rmp_reviews.csv         (all reviews, one row per review)
  - data/professors.csv              (professor metadata used for analysis)
"""

from __future__ import annotations
import base64
import json
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

# ── Session (reuse auth from rmp_crawler) ──────────────────────────────────────

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

# ── GraphQL query for individual ratings ───────────────────────────────────────

RATINGS_QUERY = """
query RatingsListQuery($count: Int!, $id: ID!, $cursor: String) {
  node(id: $id) {
    ... on Teacher {
      id
      firstName
      lastName
      numRatings
      avgRating
      ratings(first: $count, after: $cursor) {
        edges {
          node {
            id
            comment
            class
            date
            helpfulRating
            clarityRating
            difficultyRating
            wouldTakeAgain
            ratingTags
            isForOnlineClass
            isForCredit
            grade
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

PAGE_SIZE = 20


def _encode_teacher_id(numeric_id: str | int) -> str:
    """Encode numeric professor ID to RMP base64 node ID: Teacher-{id}."""
    raw = f"Teacher-{numeric_id}"
    return base64.b64encode(raw.encode()).decode()


def _extract_professor_id(rmp_url: str) -> str | None:
    """Extract numeric ID from https://www.ratemyprofessors.com/professor/123456."""
    try:
        return rmp_url.rstrip("/").split("/")[-1]
    except Exception:
        return None


def _graphql(query: str, variables: dict, retries: int = 3) -> dict | None:
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
            log.warning("Request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return None


def fetch_all_reviews(numeric_id: str) -> list[dict]:
    """Fetch every review for a professor by numeric RMP ID."""
    encoded_id = _encode_teacher_id(numeric_id)
    reviews = []
    cursor = None

    while True:
        variables = {
            "count": PAGE_SIZE,
            "id": encoded_id,
            "cursor": cursor,
        }
        data = _graphql(RATINGS_QUERY, variables)
        if data is None:
            log.error("GraphQL failed for professor %s", numeric_id)
            break

        teacher = data.get("data", {}).get("node")
        if not teacher:
            log.warning("No teacher node for professor %s", numeric_id)
            break

        ratings = teacher.get("ratings", {})
        edges = ratings.get("edges", [])
        page_info = ratings.get("pageInfo", {})

        for edge in edges:
            node = edge.get("node", {})
            reviews.append(node)

        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        time.sleep(random.uniform(*config.RMP_DELAY_RANGE))

    return reviews


def scrape_all_professors(
    professors_csv: Path = config.DATA_DIR / "professors_final_v2.csv",
    cache_dir: Path = config.DATA_DIR / "cache",
    raw_out: Path = config.DATA_DIR / "raw_rmp_reviews.csv",
    prof_out: Path = config.DATA_DIR / "professors.csv",
    resume: bool = True,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(professors_csv)
    log.info("Loaded %d professors", len(df))

    # Save canonical professors.csv (metadata only)
    meta_cols = ["name", "university", "university_state", "carnegie",
                 "discipline", "rmp_url", "tenure_year", "confidence",
                 "confidence_label", "rank", "num_ratings"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    prof_meta = df[meta_cols].copy()
    prof_meta["professor_id"] = prof_meta["rmp_url"].apply(
        lambda u: _extract_professor_id(str(u)) if pd.notna(u) else None
    )
    prof_meta.to_csv(prof_out, index=False)
    log.info("Saved professor metadata → %s", prof_out)

    # Determine which professors are already cached
    already_done: set[str] = set()
    if resume:
        for f in cache_dir.glob("*.json"):
            already_done.add(f.stem)

    all_reviews: list[dict] = []
    total = len(df)

    for i, row in df.iterrows():
        rmp_url = str(row.get("rmp_url", ""))
        professor_id = _extract_professor_id(rmp_url)
        if not professor_id:
            log.warning("[%d/%d] No RMP ID for %s — skipping", i + 1, total, row["name"])
            continue

        # Load from cache if available
        cache_file = cache_dir / f"{professor_id}.json"
        if professor_id in already_done and cache_file.exists():
            log.info("[%d/%d] Cache hit: %s (%s)", i + 1, total, row["name"], professor_id)
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            reviews = cached.get("reviews", [])
        else:
            log.info("[%d/%d] Scraping: %s (%s)", i + 1, total, row["name"], professor_id)
            reviews = fetch_all_reviews(professor_id)
            cached = {
                "professor_id": professor_id,
                "name": row["name"],
                "university": row["university"],
                "tenure_year": row.get("tenure_year"),
                "reviews": reviews,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2, ensure_ascii=False)
            log.info("  → %d reviews fetched, cached to %s", len(reviews), cache_file.name)
            time.sleep(random.uniform(*config.RMP_DELAY_RANGE))

        # Flatten reviews with professor metadata
        tenure_year = row.get("tenure_year")
        for rev in reviews:
            date_str = rev.get("date", "")
            try:
                review_year = int(date_str[:4]) if date_str else None
            except (ValueError, TypeError):
                review_year = None

            relative_year = None
            if review_year and pd.notna(tenure_year):
                relative_year = review_year - int(tenure_year)

            def _clean(s):
                """Strip embedded newlines/carriage returns from text fields."""
                if not isinstance(s, str):
                    return s
                return s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()

            all_reviews.append({
                "professor_id": professor_id,
                "professor_name": row["name"],
                "university": row["university"],
                "carnegie": row.get("carnegie"),
                "tenure_year": tenure_year,
                "review_id": rev.get("id"),
                "review_date": date_str,
                "review_year": review_year,
                "relative_year": relative_year,
                "overall_rating": rev.get("helpfulRating"),  # RMP "helpful" = overall quality
                "clarity_rating": rev.get("clarityRating"),
                "difficulty": rev.get("difficultyRating"),
                "would_take_again": rev.get("wouldTakeAgain"),
                "review_text": _clean(rev.get("comment", "")),
                "course": _clean(rev.get("class", "")),
                "is_online": rev.get("isForOnlineClass"),
                "is_for_credit": rev.get("isForCredit"),
                "grade": rev.get("grade"),
                "rating_tags": _clean(rev.get("ratingTags", "")),
            })

    raw_df = pd.DataFrame(all_reviews)
    raw_df.to_csv(raw_out, index=False)
    log.info("Saved %d total reviews → %s", len(raw_df), raw_out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_DIR / "review_scraper.log"),
        ],
    )
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    scrape_all_professors()
