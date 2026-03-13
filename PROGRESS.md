# RMP Professor Tenure Crawler — Progress Log

> **For the next agent**: This file tracks every step taken, current results, and remaining TODOs.
> Read PLAN.md first for full context and architecture. Then resume from the "Current Status" section.

---

## Current Status

**Date**: 2026-02-24
**Phase**: Phase 3 RUNNING (PID 1049820, started 13:21 UTC)
**Blocker**: DuckDuckGo consistently returns 202 rate-limit; pipeline falls back to Semantic Scholar (conf=0.55) + direct faculty page fetch. Results are accumulating.

---

## Steps Taken

### 2026-02-23 — Initial Setup & Verification

- [x] Created project directory structure under `/home/sz/RMP_Crawler_Feb23/`
- [x] Wrote `PLAN.md` with full project architecture and method
- [x] Wrote `PROGRESS.md` (this file)
- [x] Created `data/universities_r1_r2.json` with 284 universities (279 after dedup); 128 R1, 156 R2
- [x] Created `requirements.txt` with all dependencies
- [x] Created `config.py` with tunable parameters
- [x] Created `src/university_selector.py` — random selection of 100 universities
- [x] Created `src/rmp_crawler.py` — GraphQL-based RMP scraper
- [x] Created `src/tenure_estimator.py` — multi-source tenure year estimator
- [x] Created `src/main.py` — full pipeline orchestrator
- [x] Added `from __future__ import annotations` to all source files (Python 3.8 compatibility)
- [x] **Phase 1 verified and run** — 51 R1 + 49 R2 universities selected (seed=42)
- [x] **RMP API smoke-tested** — confirmed live: MIT → school ID `U2Nob29sLTU4MA==`,
      fetched 7 qualifying professors (e.g. Denis Auroux, Mathematics, rating 4.8, 60 ratings)

---

## Current Results

### Phase 1 — University Selection
- Status: **COMPLETE**
- Output: `data/selected_universities.json` (100 universities: 51 R1, 49 R2)
- Random seed: 42

### Phase 2 — RMP Crawling
- Status: **COMPLETE**
- Output: `data/professors_rmp.json` — **16,899 professors** from 89/100 universities
- 11 universities returned 0 qualifying professors (some are ID mismatches, some legitimately small)
- Notable counts: Arizona State (1193), DePaul (645), South Carolina (624), Cal State Sacramento (548)
- Zero-result schools (likely ID mismatches): Boston University, Indiana University, UCF, UMass Amherst, UMD College Park, UIC
- Filtered subset created: `data/professors_phase3_subset.json` — 3,000 professors with ≥100 ratings

### Phase 3 — Tenure Estimation
- Status: **RUNNING** (PID 1049820 as of 2026-02-24 13:21 UTC)
- Input: `data/professors_phase3_subset.json` (3,000 professors, sorted by ratings DESC)
- Output: `data/professors_final.json` (live checkpoint after each professor)
- ~5–8 seconds per professor; estimated 4–7 hours total
- Current results: accumulating; mostly 0.55 confidence from Semantic Scholar
- Sources: (1) direct faculty page URL guessing, (2) Semantic Scholar API, (3) DDG fallback (always 429, fails fast at 5s interval)
- To check progress: `tail -f logs/phase3.log` and `python3 -c "import json; d=json.load(open('data/professors_final.json')); print(len(d))"`

---

## How to Run

### Prerequisites
```bash
cd /home/sz/RMP_Crawler_Feb23
pip install -r requirements.txt
```

### Full pipeline (all phases)
```bash
python src/main.py
```

### Individual phases
```bash
# Phase 1: Select universities
python src/main.py --phase 1

# Phase 2: Crawl RMP
python src/main.py --phase 2

# Phase 3: Estimate tenure
python src/main.py --phase 3

# Resume interrupted run (skips already-processed universities)
python src/main.py --resume
```

### Configuration
Edit `config.py` to change:
- `RANDOM_SEED` — change for different university sample
- `MIN_RATINGS` — minimum RMP ratings threshold (default: 40)
- `MIN_CONFIDENCE` — minimum confidence to include in final output (default: 0.35)
- `GOOGLE_DELAY_RANGE` — (min, max) seconds between Google searches
- `RMP_DELAY_RANGE` — (min, max) seconds between RMP API calls

---

## Known Issues & Observations

- **DDG always rate-limited (202)**: DuckDuckGo blocks all search requests immediately. The pipeline falls through to Semantic Scholar (conf=0.55) for professors with papers indexed there. DDG is kept as last resort but set to 5s min interval.
- **Direct faculty URL guessing**: Rarely finds pages (URL patterns vary widely by university). Works best for major R1s.
- **S2 affiliation matching is approximate**: Uses first publication year as proxy for hire year; not always accurate for lateral hires or late-career hires.
- **School ID mismatches** (Phase 2): Boston University, Indiana University, UCF, UMass Amherst, University of Maryland College Park, UIC returned 0 or 1 qualifying professors. These can be re-crawled after adding entries to `SCHOOL_NAME_OVERRIDES` in `config.py`.
- **Zero R1 schools for Yale, Dartmouth, Rice**: These are elite schools with few professors reaching 40+ ratings. Legitimate.
- **Most results at conf=0.55** (Semantic Scholar career estimate): Acceptable for statistical analysis but not for individual-level precision.

---

## TODOs for Next Agent

### Immediate
- [x] Phase 1 complete
- [x] Phase 2 complete (16,899 professors)
- [ ] **Phase 3 currently running** — wait for completion or monitor progress
  - `tail -f logs/phase3.log`
  - If process dies, restart with `python3 src/main.py --phase 3 --resume`

### Quality checks after Phase 2
- [ ] Check what percentage of universities were successfully matched in RMP
- [ ] Check distribution of professors by discipline
- [ ] Verify `would_take_again_pct` is being recorded (some profs have -1 = not enough data)

### Quality checks after Phase 3
- [ ] Check confidence score distribution
- [ ] Manually spot-check 10–20 professors with high confidence scores
- [ ] Check for obvious errors (tenure year in future, or before 1970)

### Potential improvements
- [ ] Add Bing search as fallback if Google blocks
- [ ] Add university faculty page scraping as a structured source
  (many universities list start year on department pages)
- [ ] Add semantic search using an LLM to parse ambiguous CV text
- [ ] Handle edge cases: joint appointments, industry tenure, international systems

---

## File Inventory

| File | Status | Description |
|------|--------|-------------|
| `PLAN.md` | Done | Full project plan |
| `PROGRESS.md` | Done | This file |
| `requirements.txt` | Done | Python deps |
| `config.py` | Done | Configuration |
| `data/universities_r1_r2.json` | Done | 281 R1/R2 universities |
| `data/selected_universities.json` | **Missing** | Run Phase 1 |
| `data/professors_rmp.json` | **Missing** | Run Phase 2 |
| `data/professors_final.json` | **Missing** | Run Phase 3 |
| `data/professors_final.csv` | **Missing** | Run Phase 3 |
| `src/university_selector.py` | Done | Phase 1 script |
| `src/rmp_crawler.py` | Done | Phase 2 script |
| `src/tenure_estimator.py` | Done | Phase 3 script |
| `src/main.py` | Done | Orchestrator |

---

## Logs Location

Runtime logs are written to `logs/pipeline.log` (created on first run).

---

## Notes on Data Quality

- **RMP discipline** is a free-text field entered by students, not standardized.
  Some normalization is applied in `rmp_crawler.py` but expect variation.
- **Tenure estimation** is inherently approximate. Treat all estimates as having
  ±1–2 year uncertainty even at high confidence.
- **LinkedIn snippets** from Google search (without LinkedIn API) are often
  truncated and may miss exact years.
- **R1 vs R2 label** is included per university for downstream filtering.
