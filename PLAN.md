# RMP Professor Tenure Crawler — Project Plan

## Objective

Build an automated pipeline that:
1. Randomly selects 100 US universities from Carnegie R1 and R2 classifications
2. For each university, scrapes Rate My Professor (RMP) for professors with ≥40 ratings
3. Filters for tenured associate or full professors
4. Estimates their tenure year from LinkedIn, public CVs, and Google search
5. Outputs a confidence-scored dataset ready for analysis

---

## Phase 1 — University Selection

### Source
- Carnegie Classification of Institutions of Higher Education (2021)
  - R1: Doctoral Universities – Very High Research Activity (~146 institutions)
  - R2: Doctoral Universities – High Research Activity (~135 institutions)
- Pre-compiled list stored in `data/universities_r1_r2.json`

### Method
- Randomly sample 100 universities using a seeded random draw (for reproducibility)
- Seed stored in `config.py` (default seed = 42, can be overridden)
- Output: `data/selected_universities.json`

---

## Phase 2 — RMP Professor Scraping

### API
- Rate My Professor exposes an unofficial GraphQL API
  - Endpoint: `https://www.ratemyprofessors.com/graphql`
  - Auth: HTTP Basic `Authorization: Basic dGVzdDp0ZXN0` (publicly known)
- No login or Selenium required for professor listing

### Queries
1. **School lookup**: search by university name → get RMP school ID
2. **Professor listing**: paginate all professors for a given school ID
   - Fields collected: `id`, `firstName`, `lastName`, `avgRating`, `numRatings`,
     `wouldTakeAgainPercent`, `avgDifficulty`, `department`
   - No detail-page visits (no individual professor pages)
3. **Filter**: keep only professors with `numRatings >= 40`

### Output fields per professor
| Field | Description |
|-------|-------------|
| name | Full name |
| university | University name |
| discipline | RMP department string |
| quality | avgRating (0–5) |
| num_ratings | Number of ratings |
| would_take_again_pct | Percentage who'd take again |
| difficulty | avgDifficulty (0–5) |
| rmp_url | Direct RMP professor URL |
| rmp_id | Internal RMP numeric ID |

- Output: `data/professors_rmp.json`
- Checkpointing: saved after every university to allow resume on failure

---

## Phase 3 — Tenure Estimation

### Target subset
- Only professors likely to be tenured: "Associate Professor" or "Professor"
  (identified via RMP department field keyword matching where possible; also
   handled via search result parsing in the tenure step)

### Sources queried per professor (in order of reliability)
1. **University faculty page** — Google: `"[Name]" site:[university].edu`
2. **Public CV** — Google: `"[Name]" "[University]" filetype:pdf CV`
3. **LinkedIn snippet** — Google: `site:linkedin.com "[Name]" "[University]"`
4. **General Google search** — `"[Name]" "[University]" tenure year`

### Year extraction logic
| Pattern | Example | Confidence |
|---------|---------|------------|
| Explicit tenure/promotion mention | "tenured 2015", "promoted to Associate Professor in 2014" | 0.90 |
| "Associate Professor since YYYY" | | 0.85 |
| Start year as Asst. Prof + 6–7 yrs | "Assistant Professor 2008" → est. 2014–2015 | 0.50 |
| PhD year + 6–10 yrs (last resort) | "PhD 2005" → est. 2011–2015 | 0.35 |

### Confidence threshold for final output
- **High confidence (≥0.70)**: included in primary output (`professors_final.json`)
- **Low confidence (0.35–0.69)**: included with flag, in secondary output
- **Below 0.35**: excluded

### Output fields (final)
| Field | Description |
|-------|-------------|
| name | Full name |
| university | University name |
| discipline | Academic discipline |
| rmp_url | RMP URL |
| tenure_year | Estimated or confirmed year |
| tenure_year_range | [low, high] range if uncertain |
| confidence | Score 0–1 |
| confidence_label | "high" / "medium" / "low" |
| source | Where the year was found |
| source_url | URL of source |
| rank | "Associate Professor" or "Full Professor" |

- Primary output: `data/professors_final.json`
- Also exported to `data/professors_final.csv` for easy inspection

---

## Technical Stack

| Component | Library |
|-----------|---------|
| HTTP requests | `requests` |
| HTML parsing | `beautifulsoup4` |
| Data manipulation | `pandas` |
| Logging | `logging` (stdlib) |
| Rate limiting | `time.sleep` with jitter |
| Checkpointing | JSON append-on-success |
| Config | `config.py` + env vars |

---

## Project File Structure

```
RMP_Crawler_Feb23/
├── PLAN.md                         # This file
├── PROGRESS.md                     # Live progress log
├── requirements.txt
├── config.py
├── data/
│   ├── universities_r1_r2.json     # Full R1/R2 list (~281 universities)
│   ├── selected_universities.json  # 100 randomly selected
│   ├── professors_rmp.json         # All professors from Phase 2
│   └── professors_final.json       # Tenure-estimated output
├── src/
│   ├── __init__.py
│   ├── university_selector.py
│   ├── rmp_crawler.py
│   ├── tenure_estimator.py
│   └── main.py
```

---

## Ethical & Rate-Limiting Considerations

- Add random 1–3 second delays between RMP API calls
- Add random 3–8 second delays between Google searches
- Use descriptive User-Agent strings
- Never scrape individual RMP professor review pages (only listing)
- Respect robots.txt for non-Google sources where possible
- All data is used for academic research purposes only

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| RMP API changes | Abstract API calls; easy to swap GraphQL queries |
| Google CAPTCHA blocks | Exponential backoff; rotate search queries; low request rate |
| Missing RMP school match | Fuzzy name matching with fallback to manual mapping |
| Professor rank not in RMP data | Use Google search to confirm rank before tenure estimation |
| Incomplete tenure data | Report confidence score; include in separate "low-confidence" file |

---

## Deliverables

1. `data/professors_final.json` — primary structured output
2. `data/professors_final.csv` — same data in CSV
3. `PROGRESS.md` — updated log of steps, results, and remaining TODOs
4. All intermediate data files for reproducibility
