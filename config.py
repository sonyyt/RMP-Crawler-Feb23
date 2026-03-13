"""
Project-wide configuration.
Edit values here or override with environment variables.
"""

from __future__ import annotations
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED: int = int(os.getenv("RMP_SEED", "42"))
NUM_UNIVERSITIES: int = int(os.getenv("RMP_NUM_UNIS", "100"))

# ── RMP scraping ───────────────────────────────────────────────────────────────
RMP_GRAPHQL_URL: str = "https://www.ratemyprofessors.com/graphql"
RMP_AUTH_HEADER: str = "Basic dGVzdDp0ZXN0"  # base64("test:test") – public
RMP_PAGE_SIZE: int = 20
MIN_RATINGS: int = int(os.getenv("RMP_MIN_RATINGS", "5"))

# Delay between RMP API calls (seconds, chosen uniformly at random)
RMP_DELAY_RANGE: tuple[float, float] = (1.0, 3.0)

# ── Google search ──────────────────────────────────────────────────────────────
# Delay between Google requests (seconds)
GOOGLE_DELAY_RANGE: tuple[float, float] = (2.0, 4.0)
GOOGLE_MAX_RESULTS: int = 5

# ── Tenure estimation ──────────────────────────────────────────────────────────
# Minimum confidence score to include professor in final output
MIN_CONFIDENCE: float = float(os.getenv("RMP_MIN_CONFIDENCE", "0.35"))

# Assumed years from Asst. Prof start to tenure
TENURE_LAG_LOW: int = 6
TENURE_LAG_HIGH: int = 7

# ── Data paths ─────────────────────────────────────────────────────────────────
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"

UNIVERSITIES_FILE = DATA_DIR / "universities_r1_r2.json"
SELECTED_UNIS_FILE = DATA_DIR / "selected_universities.json"
PROFESSORS_RMP_FILE = DATA_DIR / "professors_rmp.json"
PROFESSORS_FINAL_FILE = DATA_DIR / "professors_final.json"
PROFESSORS_FINAL_CSV = DATA_DIR / "professors_final.csv"
RMP_FAILURES_FILE = DATA_DIR / "rmp_failures.json"

# ── OpenAI API (Phase 3 v2) ───────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Google Custom Search API ─────────────────────────────────────────────────
GOOGLE_CSE_API_KEY: str = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_CX: str = os.getenv("GOOGLE_CSE_CX", "")

# ── Tenure estimation v2 ─────────────────────────────────────────────────────
PHD_YEAR_CUTOFF: int = int(os.getenv("RMP_PHD_CUTOFF", "2010"))
MAX_PAGES_PER_PROFESSOR: int = 5
MAX_PAGE_CHARS: int = 15000
VALIDATION_SUBSET_SIZE: int = int(os.getenv("RMP_VALIDATION_SIZE", "100"))

# ── Data paths (v2) ──────────────────────────────────────────────────────────
PROFESSORS_FINAL_V2_FILE = DATA_DIR / "professors_final_v2.json"
PROFESSORS_FINAL_V2_CSV = DATA_DIR / "professors_final_v2.csv"
PROFESSORS_EXCLUDED_FILE = DATA_DIR / "professors_excluded.json"

# ── Phase 4: Review crawling ──────────────────────────────────────────────────
PROFESSORS_FILTERED_CSV = DATA_DIR / "professors_filtered.csv"
REVIEWS_DIR = DATA_DIR / "reviews"
REVIEWS_ALL_FILE = REVIEWS_DIR / "all_reviews.csv"
REVIEWS_CRAWL_SUMMARY = REVIEWS_DIR / "crawl_summary.csv"

# Minimum reviews per period to include a professor in analysis
MIN_REVIEWS_PER_PERIOD: int = int(os.getenv("RMP_MIN_REVIEWS_PERIOD", "3"))

# ── Phase 5: Analysis ─────────────────────────────────────────────────────────
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_PLOTS_DIR = ANALYSIS_DIR / "plots"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_LEVEL: str = os.getenv("RMP_LOG_LEVEL", "INFO")

# ── School name overrides ──────────────────────────────────────────────────────
# If RMP can't find a university by its canonical name, specify the RMP name here.
# Format: {"canonical name": "rmp search name"}
SCHOOL_NAME_OVERRIDES: dict[str, str] = {
    "Indiana University-Purdue University Indianapolis": "IUPUI",
    "University of North Carolina at Chapel Hill": "UNC Chapel Hill",
    "Pennsylvania State University": "Penn State",
    "Missouri University of Science and Technology": "Missouri S&T",
    "CUNY Graduate Center": "City University of New York",
}
