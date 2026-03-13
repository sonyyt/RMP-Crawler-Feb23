# RMP Professor Tenure Crawler

An automated data pipeline that scrapes [Rate My Professor (RMP)](https://www.ratemyprofessors.com/) to build a confidence-scored dataset of tenured professors across US research universities.

---

## What It Does

The pipeline runs in three phases:

### Phase 1 — University Selection
- Draws from the Carnegie Classification of Institutions of Higher Education (R1 + R2 doctoral universities, ~281 total)
- Randomly selects 100 universities using a seeded draw (reproducible via `config.py`)
- Output: `data/selected_universities.json`

### Phase 2 — RMP Professor Scraping
- Queries RMP's unofficial GraphQL API (no Selenium required)
- For each selected university: looks up the RMP school ID, then paginates all professors
- Filters to professors with **≥ 40 ratings**
- Collects: name, department, avg rating, difficulty, would-take-again %, RMP URL/ID
- Checkpoints after every university so the run can be resumed on failure
- Output: `data/professors_rmp.json`

### Phase 3 — Tenure Year Estimation
- Targets likely tenured professors (Associate Professor / Full Professor rank)
- Searches multiple public sources per professor:
  1. University faculty page (`site:[university].edu`)
  2. Public CV (Google `filetype:pdf`)
  3. LinkedIn snippet
  4. General Google search
- Extracts tenure year using pattern matching with a confidence score (0–1):

  | Signal | Example | Confidence |
  |--------|---------|------------|
  | Explicit tenure/promotion mention | "tenured 2015" | 0.90 |
  | "Associate Professor since YYYY" | | 0.85 |
  | Asst. Prof start year + 6–7 yrs | "Asst Prof 2008" → ~2014 | 0.50 |
  | PhD year + 6–10 yrs (last resort) | "PhD 2005" → ~2012 | 0.35 |

- Only records with confidence ≥ 0.35 are kept; ≥ 0.70 are flagged high-confidence
- Output: `data/professors_final_v2.json` / `data/professors_final_v2.csv`

---

## Output Fields

| Field | Description |
|-------|-------------|
| `name` | Full professor name |
| `university` | Institution name |
| `discipline` | Academic department (from RMP) |
| `rmp_url` | Direct RMP profile URL |
| `tenure_year` | Estimated or confirmed tenure year |
| `tenure_year_range` | `[low, high]` range if uncertain |
| `confidence` | Confidence score (0–1) |
| `confidence_label` | `"high"` / `"medium"` / `"low"` |
| `source` | Source type where year was found |
| `source_url` | URL of the source |
| `rank` | `"Associate Professor"` or `"Full Professor"` |

---

## Project Structure

```
RMP_Crawler_Feb23/
├── config.py                        # Seeds, thresholds, API keys
├── requirements.txt
├── src/
│   ├── main.py                      # Pipeline entry point
│   ├── university_selector.py       # Phase 1: university sampling
│   ├── rmp_crawler.py               # Phase 2: RMP GraphQL scraping
│   └── tenure_estimator_v2.py       # Phase 3: tenure year extraction
└── data/
    ├── universities_r1_r2.json      # Full R1/R2 university list
    ├── selected_universities.json   # 100 sampled universities
    ├── professors_rmp.json          # Raw RMP data (Phase 2 output)
    ├── professors_filtered.json     # Filtered subset for Phase 3
    └── professors_final_v2.json     # Final tenure-scored dataset
```

---

## Setup

```bash
pip install -r requirements.txt
```

Configure `config.py`:
- `RANDOM_SEED` — for reproducible university sampling (default: `42`)
- `GOOGLE_API_KEY` / `GOOGLE_CSE_ID` — for Google Custom Search in Phase 3
- `MIN_RATINGS` — minimum RMP ratings threshold (default: `40`)

---

## Running the Pipeline

```bash
# Run all phases
python src/main.py

# Or run phases individually (see src/main.py for entry points)
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| HTTP requests | `requests` |
| HTML parsing | `beautifulsoup4` + `lxml` |
| Data manipulation | `pandas` |
| Search integration | `google-api-python-client` |
| Rate limiting | `time.sleep` with random jitter |
| Checkpointing | JSON append-on-success |

---

## Ethical Considerations

- Random 1–3 s delays between RMP API calls; 3–8 s between Google searches
- Only the RMP professor listing API is used — no individual review pages are scraped
- All data is collected for academic research purposes only
- Respects `robots.txt` where applicable
