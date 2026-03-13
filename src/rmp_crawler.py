"""
Phase 2 — RMP Professor Crawler
Uses Rate My Professor's unofficial GraphQL API to collect professor data.
No individual professor pages are visited (listing only).
"""

from __future__ import annotations
import base64
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

# ── HTTP session ───────────────────────────────────────────────────────────────

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

# ── GraphQL queries ────────────────────────────────────────────────────────────

SCHOOL_SEARCH_QUERY = """
query NewSearchSchoolsQuery($query: SchoolSearchQuery!) {
  newSearch {
    schools(query: $query) {
      edges {
        node {
          id
          name
          city
          state
          numRatings
        }
      }
    }
  }
}
"""

TEACHER_SEARCH_QUERY = """
query TeacherSearchPaginatedQuery(
  $count: Int!
  $cursor: String
  $query: TeacherSearchQuery!
) {
  search: newSearch {
    teachers(query: $query, first: $count, after: $cursor) {
      edges {
        node {
          id
          firstName
          lastName
          avgRating
          numRatings
          wouldTakeAgainPercent
          avgDifficulty
          department
          school {
            id
            name
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_rmp_id(encoded_id: str) -> str:
    """
    RMP GraphQL returns base64-encoded IDs like 'VGVhY2hlci0xMjM0NTY='.
    Decoded: 'Teacher-123456'. We extract the numeric part.
    """
    try:
        decoded = base64.b64decode(encoded_id + "==").decode("utf-8")
        return decoded.split("-")[-1]
    except Exception:
        return encoded_id


def _rmp_url(encoded_id: str) -> str:
    numeric = _decode_rmp_id(encoded_id)
    return f"https://www.ratemyprofessors.com/professor/{numeric}"


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
            log.warning("RMP request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    log.error("All %d RMP attempts exhausted.", retries)
    return None


def _rmp_delay() -> None:
    lo, hi = config.RMP_DELAY_RANGE
    time.sleep(random.uniform(lo, hi))

# ── School lookup ──────────────────────────────────────────────────────────────

def find_school_id(university: dict) -> Optional[str]:
    """
    Look up RMP school ID for a university dict with keys 'name' and 'state'.
    Falls back to SCHOOL_NAME_OVERRIDES if canonical name yields no result.
    Returns the RMP base64-encoded school ID, or None.
    """
    name = config.SCHOOL_NAME_OVERRIDES.get(university["name"], university["name"])
    state = university["state"]

    data = _graphql(SCHOOL_SEARCH_QUERY, {"query": {"text": name}})
    if data is None:
        return None

    edges = (
        data.get("data", {})
        .get("newSearch", {})
        .get("schools", {})
        .get("edges", [])
    )

    if not edges:
        log.warning("No RMP school found for '%s'", name)
        return None

    # Prefer a school in the right state with matching name
    name_lower = name.lower()
    state_upper = state.upper()

    for edge in edges:
        node = edge["node"]
        if (
            node.get("state", "").upper() == state_upper
            and name_lower in node["name"].lower()
        ):
            log.debug("Matched school: %s (id=%s)", node["name"], node["id"])
            return node["id"]

    # Fallback: closest name match regardless of state
    for edge in edges:
        node = edge["node"]
        if name_lower in node["name"].lower():
            log.debug(
                "Fallback match for '%s': %s (%s)", name, node["name"], node.get("state", "?")
            )
            return node["id"]

    # Last resort: first result
    first = edges[0]["node"]
    log.debug("Using first result for '%s': %s", name, first["name"])
    return first["id"]

# ── Professor listing ──────────────────────────────────────────────────────────

def fetch_professors(school_id: str, min_ratings: int = config.MIN_RATINGS) -> list[dict]:
    """
    Paginate through all professors at a school. Returns only those with
    numRatings >= min_ratings. Does NOT visit individual professor pages.
    """
    professors: list[dict] = []
    cursor: Optional[str] = None
    page = 0

    while True:
        page += 1
        variables = {
            "count": config.RMP_PAGE_SIZE,
            "cursor": cursor,
            "query": {"schoolID": school_id, "text": ""},
        }
        data = _graphql(TEACHER_SEARCH_QUERY, variables)
        if data is None:
            log.error("Failed to fetch page %d for school %s", page, school_id)
            break

        search_data = (
            data.get("data", {})
            .get("search", {})
            .get("teachers", {})
        )
        edges = search_data.get("edges", [])
        page_info = search_data.get("pageInfo", {})

        for edge in edges:
            node = edge.get("node", {})
            num_ratings = node.get("numRatings", 0) or 0
            if num_ratings < min_ratings:
                continue

            would_take_again = node.get("wouldTakeAgainPercent")
            # RMP uses -1 to indicate "not enough data"
            if would_take_again is not None and would_take_again < 0:
                would_take_again = None

            encoded_id = node.get("id", "")
            professors.append(
                {
                    "rmp_id": encoded_id,
                    "name": f"{node.get('firstName', '')} {node.get('lastName', '')}".strip(),
                    "first_name": node.get("firstName", ""),
                    "last_name": node.get("lastName", ""),
                    "quality": node.get("avgRating"),
                    "num_ratings": num_ratings,
                    "discipline": node.get("department", ""),
                    "would_take_again_pct": would_take_again,
                    "difficulty": node.get("avgDifficulty"),
                    "school_name": node.get("school", {}).get("name", ""),
                    "rmp_url": _rmp_url(encoded_id),
                }
            )

        log.debug(
            "Page %d: %d edges, running total qualifying: %d", page, len(edges), len(professors)
        )

        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        _rmp_delay()

    return professors

# ── Main Phase 2 entry point ───────────────────────────────────────────────────

def crawl_universities(
    selected_unis: list[dict],
    output_path: Path = config.PROFESSORS_RMP_FILE,
    failures_path: Path = config.RMP_FAILURES_FILE,
    resume: bool = False,
) -> list[dict]:
    """
    For each university, find its RMP school ID and collect qualifying professors.
    Writes results incrementally (checkpoint after each university).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress if resuming
    processed_names: set[str] = set()
    all_professors: list[dict] = []
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_professors = json.load(f)
        processed_names = {p.get("_university_name", "") for p in all_professors}
        # Remove internal key before use
        for p in all_professors:
            p.pop("_university_name", None)
        log.info("Resuming: %d universities already processed", len(processed_names))

    failures: list[dict] = []

    for i, uni in enumerate(selected_unis, 1):
        uni_name = uni["name"]

        if uni_name in processed_names:
            log.info("[%d/%d] Skipping (already done): %s", i, len(selected_unis), uni_name)
            continue

        log.info("[%d/%d] Processing: %s [%s]", i, len(selected_unis), uni_name, uni["carnegie"])

        school_id = find_school_id(uni)
        if school_id is None:
            log.warning("Could not find RMP school for '%s' — skipping", uni_name)
            failures.append({"university": uni_name, "reason": "school not found"})
            _rmp_delay()
            continue

        professors = fetch_professors(school_id)
        log.info(
            "  → Found %d qualifying professors (≥%d ratings)", len(professors), config.MIN_RATINGS
        )

        # Attach university metadata
        for p in professors:
            p["university"] = uni_name
            p["university_state"] = uni["state"]
            p["carnegie"] = uni["carnegie"]

        all_professors.extend(professors)

        # Checkpoint: save with internal tracking key
        checkpoint_data = []
        for p in all_professors:
            entry = dict(p)
            entry["_university_name"] = entry["university"]
            checkpoint_data.append(entry)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        log.info("  → Checkpoint saved (%d total professors so far)", len(all_professors))

        _rmp_delay()

    # Save failures
    if failures:
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        log.warning("%d universities failed — see %s", len(failures), failures_path)

    # Clean up checkpoint key from final file
    for p in all_professors:
        p.pop("_university_name", None)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_professors, f, indent=2, ensure_ascii=False)

    log.info(
        "Phase 2 complete: %d professors across %d universities saved to %s",
        len(all_professors),
        len(selected_unis) - len(failures),
        output_path,
    )
    return all_professors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected_path = config.SELECTED_UNIS_FILE
    if not selected_path.exists():
        log.error("Run Phase 1 first: python src/university_selector.py")
        sys.exit(1)
    with open(selected_path, "r") as f:
        unis = json.load(f)
    crawl_universities(unis)
