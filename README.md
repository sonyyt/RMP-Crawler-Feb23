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
- Uses the **OpenAI Responses API** with the `web_search_preview` tool — no Google CSE required
- Two-stage design to minimize API costs (~300 web searches vs ~4,700 in a naive approach):
  1. **Stage 1** — one web search per university: fetches the full CS/related faculty list and identifies which professors are tenured (Associate or Full Professor)
  2. **Stage 2** — one web search per confirmed tenured professor: finds their specific tenure year
- Results are structured JSON extracted by the model with a confidence score (0–1):

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

Create a `.env` file in the project root (never commit this):

```bash
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini   # optional, defaults to gpt-4o-mini
```

Other settings in `config.py`:
- `RANDOM_SEED` — for reproducible university sampling (default: `42`)
- `MIN_RATINGS` — minimum RMP ratings threshold (default: `40`)
- `MIN_CONFIDENCE` — minimum confidence to include in final output (default: `0.35`)

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
| Web search + extraction | OpenAI Responses API (`web_search_preview`) |
| Rate limiting | `time.sleep` with random jitter |
| Checkpointing | JSON append-on-success |

---

## Ethical Considerations

- Random 1–3 s delays between RMP API calls; 3–8 s between Google searches
- Only the RMP professor listing API is used — no individual review pages are scraped
- All data is collected for academic research purposes only
- Respects `robots.txt` where applicable
