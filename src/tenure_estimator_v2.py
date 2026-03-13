"""
Phase 3 v2 — Tenure Estimator (Batched by University)

Two-stage approach to minimize OpenAI web search API costs:

  Stage 1: ONE web search per university.
           Ask: "Among these professors at X university, who is tenured?"
           → Eliminates ~95% of professors cheaply (batch by university).

  Stage 2: ONE web search per tenured professor.
           Ask: "When did this professor get tenure?"
           → Only pays for the ~5% that are actually tenured.

Cost reduction: ~92 Stage 1 + ~200 Stage 2 = ~300 web searches
                vs ~4,700 individual searches in the old approach.

Uses OpenAI Responses API with web_search_preview tool.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

CURRENT_YEAR = 2026
MAX_NAMES_PER_BATCH = 50  # Max professors per extraction call

# ── Non-tenure-track keyword filter (applied in Python) ──────────────────────

NON_TENURE_TRACK_KEYWORDS = [
    "adjunct", "instructor", "lecturer", "clinical",
    "visiting", "research professor", "research associate",
    "of instruction", "of practice", "of the practice",
    "of teaching", "teaching professor", "teaching associate",
    "teaching assistant professor",
    "instructional", "postdoc", "emeritus",
]

# ── Stage 1 Prompts (University-level batch identification) ──────────────────

STAGE1_SEARCH_PROMPT = """\
I have a list of professors from {university}'s Computer Science department. \
I need to determine which of them are currently tenured. \
Please search for {university}'s Computer Science (or related) department \
faculty directory page to find their current titles and ranks.

Here are the professors I need to check:
{professor_list}

For each professor, find their current title on the university website and \
determine if they are tenured (Associate Professor or Full Professor on \
the tenure track). Report your findings for each professor.\
"""

STAGE1_EXTRACT_PROMPT = """\
Based on the search results about {university}'s Computer Science faculty, \
determine the tenure status of each professor below.

Professors to check:
{professor_list}

Return ONLY a JSON object:
{{
  "professors": [
    {{
      "name": "Exact Name As Given",
      "found": true or false,
      "title": "exact title from source or null",
      "is_tenured": true or false or null
    }}
  ]
}}

You MUST include ALL {count} professors from the list (one entry each).

Rules for "is_tenured":
- true ONLY for: tenured Associate Professor or tenured Full Professor
- false for: Assistant Professor, Adjunct, Instructor, Lecturer, \
Clinical Professor, Visiting Professor, Research Professor, \
Research Associate Professor, Professor of Instruction, \
Professor of Practice, Professor of Teaching, Teaching Professor, \
Teaching Associate Professor, Teaching Assistant Professor, \
Emeritus, Postdoc, or any non-tenure-track title
- null ONLY if the professor was not found in search results\
"""

# ── Stage 2 Prompts (Individual tenure year search) ──────────────────────────

STAGE2_SEARCH_PROMPT = """\
Search for detailed career information about {name}, a tenured {discipline} \
professor at {university}. I need to determine:

1. When they became a tenured Associate Professor (their tenure year)
2. When they started as Assistant Professor (to estimate tenure year as start + 6)
3. Their PhD year

Look for their CV, university bio page, personal website, or any source \
with career timeline information.\
"""

STAGE2_EXTRACT_PROMPT = """\
Based on the search results about {name} at {university} ({discipline}), \
extract tenure timing information.

Return ONLY a JSON object:
{{
  "current_title": "exact title from source",
  "rank": "Full Professor" or "Associate Professor",
  "phd_year": YYYY or null,
  "tenure_year": YYYY or null,
  "tenure_year_source": "explicit" or "promotion_year" or "assoc_prof_start" or "asst_prof_plus_6" or "phd_plus_6" or null,
  "confidence_notes": "brief explanation"
}}

Rules:
- tenure_year = the year they became a tenured Associate Professor
- If promoted to Associate Professor in YYYY → source = "promotion_year"
- If "Associate Professor since YYYY" → source = "assoc_prof_start"
- If only Asst Prof start year YYYY → tenure_year = YYYY + 6, source = "asst_prof_plus_6"
- If only PhD year → tenure_year = phd_year + 6, source = "phd_plus_6"
- If explicit tenure mention ("tenured in YYYY") → source = "explicit"\
"""


# ── TenureAnalyzer ────────────────────────────────────────────────────────────

class TenureAnalyzer:
    """Two-stage tenure analysis using OpenAI web search."""

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._search_calls = 0
        self._extract_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        log.info("OpenAI client initialized (model=%s)", config.OPENAI_MODEL)

    def stage1_search(self, university: str, names: list[str]) -> str:
        """
        Stage 1: ONE web search per university (or chunk of names).
        Includes professor names in the prompt so the LLM checks each one.
        """
        prof_list = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))
        prompt = STAGE1_SEARCH_PROMPT.format(
            university=university,
            professor_list=prof_list,
        )
        try:
            resp = self.client.responses.create(
                model=config.OPENAI_MODEL,
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
            self._search_calls += 1
            if resp.usage:
                self._input_tokens += resp.usage.input_tokens
                self._output_tokens += resp.usage.output_tokens
            text = resp.output_text or ""
            log.debug("  Stage 1 search: %d chars response", len(text))
            return text
        except Exception as e:
            log.error("  Stage 1 search failed for %s: %s", university, e)
            return ""

    def stage1_extract(self, search_text: str, university: str,
                       names: list[str]) -> list[dict]:
        """Extract structured tenure status from Stage 1 search results."""
        prof_list = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))
        prompt = STAGE1_EXTRACT_PROMPT.format(
            university=university,
            professor_list=prof_list,
            count=len(names),
        )
        try:
            resp = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                temperature=0.0,
                max_tokens=max(1024, len(names) * 80),
                messages=[
                    {"role": "system",
                     "content": "Extract structured data. Return only valid JSON."},
                    {"role": "user",
                     "content": f"{search_text}\n\n---\n\n{prompt}"},
                ],
                response_format={"type": "json_object"},
            )
            self._extract_calls += 1
            if resp.usage:
                self._input_tokens += resp.usage.prompt_tokens
                self._output_tokens += resp.usage.completion_tokens
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            return data.get("professors", [])
        except Exception as e:
            log.error("  Stage 1 extraction failed for %s: %s", university, e)
            return []

    def stage2_analyze(self, professor: dict) -> dict:
        """Stage 2: Individual web search + extraction for tenure year."""
        name = professor["name"]
        university = professor["university"]
        discipline = professor.get("discipline", "Computer Science")

        # Web search
        prompt = STAGE2_SEARCH_PROMPT.format(
            name=name, university=university, discipline=discipline,
        )
        try:
            resp = self.client.responses.create(
                model=config.OPENAI_MODEL,
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
            search_text = resp.output_text or ""
            self._search_calls += 1
            if resp.usage:
                self._input_tokens += resp.usage.input_tokens
                self._output_tokens += resp.usage.output_tokens
        except Exception as e:
            log.error("    Stage 2 search failed for %s: %s", name, e)
            return _empty_result(f"Search error: {e}")

        if len(search_text) < 20:
            return _empty_result("No useful search results")

        # Extract structured data
        time.sleep(0.3)
        extract_prompt = STAGE2_EXTRACT_PROMPT.format(
            name=name, university=university, discipline=discipline,
        )
        try:
            resp = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                temperature=0.0,
                max_tokens=512,
                messages=[
                    {"role": "system",
                     "content": "Extract structured data. Return only valid JSON."},
                    {"role": "user",
                     "content": f"{search_text}\n\n---\n\n{extract_prompt}"},
                ],
                response_format={"type": "json_object"},
            )
            self._extract_calls += 1
            if resp.usage:
                self._input_tokens += resp.usage.prompt_tokens
                self._output_tokens += resp.usage.completion_tokens
            raw = resp.choices[0].message.content
            return _parse_json(raw)
        except Exception as e:
            log.error("    Stage 2 extraction failed for %s: %s", name, e)
            return _empty_result(f"Extraction error: {e}")

    @property
    def stats(self) -> dict:
        search_fee = self._search_calls * 0.025
        input_cost = self._input_tokens * 0.15 / 1_000_000
        output_cost = self._output_tokens * 0.60 / 1_000_000
        return {
            "search_calls": self._search_calls,
            "extract_calls": self._extract_calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "search_fee_usd": round(search_fee, 2),
            "token_cost_usd": round(input_cost + output_cost, 4),
            "total_est_cost_usd": round(search_fee + input_cost + output_cost, 2),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_result(reason: str = "No data") -> dict:
    return {
        "current_title": None,
        "rank": None,
        "phd_year": None,
        "tenure_year": None,
        "tenure_year_source": None,
        "confidence_notes": reason,
    }


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        for key in ("current_title", "rank", "phd_year", "tenure_year",
                     "tenure_year_source"):
            result.setdefault(key, None)
        result.setdefault("confidence_notes", "")
        return result
    except json.JSONDecodeError:
        return _empty_result("JSON parse error")


def _fuzzy_match_name(name: str, results: list[dict]) -> Optional[dict]:
    """Match a professor name against Stage 1 extraction results."""
    parts = name.lower().strip().split()
    last = parts[-1] if parts else ""
    first = parts[0] if parts else ""

    for r in results:
        r_parts = r.get("name", "").lower().strip().split()
        r_last = r_parts[-1] if r_parts else ""
        r_first = r_parts[0] if r_parts else ""

        # Last name + first name/initial match
        if r_last == last and r_first and first:
            if r_first[0] == first[0]:
                return r
    return None


# ── Confidence scoring ────────────────────────────────────────────────────────

CONFIDENCE_MAP = {
    "explicit": 0.95,
    "promotion_year": 0.90,
    "assoc_prof_start": 0.85,
    "asst_prof_plus_6": 0.70,
    "phd_plus_6": 0.55,
}


def compute_confidence(llm_result: dict) -> tuple[float, str]:
    source = llm_result.get("tenure_year_source")
    has_year = llm_result.get("tenure_year") is not None

    if has_year and source in CONFIDENCE_MAP:
        conf = CONFIDENCE_MAP[source]
        return conf, "high" if conf >= 0.85 else "medium"

    if not has_year:
        return 0.40, "low"

    return 0.0, "none"


# ── Entry builders ────────────────────────────────────────────────────────────

def _build_entry(prof: dict, llm_result: dict, confidence: float,
                 conf_label: str) -> dict:
    return {
        "name": prof["name"],
        "university": prof.get("university", ""),
        "university_state": prof.get("university_state", ""),
        "carnegie": prof.get("carnegie", ""),
        "discipline": prof.get("discipline", ""),
        "quality": prof.get("quality"),
        "num_ratings": prof.get("num_ratings"),
        "would_take_again_pct": prof.get("would_take_again_pct"),
        "difficulty": prof.get("difficulty"),
        "rmp_url": prof.get("rmp_url", ""),
        "current_title": llm_result.get("current_title"),
        "rank": llm_result.get("rank"),
        "phd_year": llm_result.get("phd_year"),
        "tenure_year": llm_result.get("tenure_year"),
        "tenure_year_source": llm_result.get("tenure_year_source"),
        "confidence": confidence,
        "confidence_label": conf_label,
        "llm_notes": llm_result.get("confidence_notes", ""),
    }


def _build_excluded_entry(prof: dict, reason: str,
                          title: str | None = None) -> dict:
    return {
        "name": prof["name"],
        "university": prof.get("university", ""),
        "university_state": prof.get("university_state", ""),
        "discipline": prof.get("discipline", ""),
        "num_ratings": prof.get("num_ratings"),
        "rmp_url": prof.get("rmp_url", ""),
        "excluded": True,
        "exclusion_reason": reason,
        "current_title": title,
    }


def _save_json(path: Path, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def estimate_tenure_for_all(
    professors: list[dict],
    output_path: Path | None = None,
    csv_path: Path | None = None,
    resume: bool = False,
    validation_mode: bool = False,
) -> list[dict]:
    """
    Two-stage batched tenure estimation.

    Stage 1: One web search per university → identify tenured professors.
    Stage 2: One web search per tenured professor → find tenure year.
    """
    output_path = output_path or config.PROFESSORS_FINAL_V2_FILE
    csv_path = csv_path or config.PROFESSORS_FINAL_V2_CSV
    excluded_path = config.PROFESSORS_EXCLUDED_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analyzer = TenureAnalyzer()

    # ── Resume ────────────────────────────────────────────────────────────
    processed_keys: set[str] = set()
    results: list[dict] = []
    excluded: list[dict] = []

    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_keys = {r["name"] + r.get("university", "") for r in results}

    if resume and excluded_path.exists():
        with open(excluded_path, "r", encoding="utf-8") as f:
            excluded = json.load(f)
        processed_keys.update(
            r["name"] + r.get("university", "") for r in excluded
        )

    log.info("Resume: %d professors already processed", len(processed_keys))

    # ── Validation mode ───────────────────────────────────────────────────
    if validation_mode:
        professors = professors[:config.VALIDATION_SUBSET_SIZE]
        log.info("VALIDATION MODE: %d professors", len(professors))

    # ── Group by university (unprocessed, deduplicated) ───────────────────
    uni_groups: dict[str, list[dict]] = defaultdict(list)
    seen_keys: set[str] = set()
    for prof in professors:
        key = prof["name"] + prof.get("university", "")
        if key not in processed_keys and key not in seen_keys:
            uni_groups[prof.get("university", "Unknown")].append(prof)
            seen_keys.add(key)

    total_remaining = sum(len(v) for v in uni_groups.values())
    log.info(
        "To process: %d professors across %d universities",
        total_remaining, len(uni_groups),
    )

    if total_remaining == 0:
        log.info("Nothing to process.")
        return results

    uni_list = sorted(uni_groups.keys())
    stage2_total = 0

    for ui, uni_name in enumerate(uni_list, 1):
        uni_profs = uni_groups[uni_name]
        log.info(
            "━━ [%d/%d] %s (%d professors) ━━",
            ui, len(uni_list), uni_name, len(uni_profs),
        )

        # ── Stage 1: Search faculty page + identify tenured ──────────────
        # Process in chunks of MAX_NAMES_PER_BATCH
        tenured_candidates: list[tuple[dict, dict]] = []  # (prof, stage1_match)

        for chunk_start in range(0, len(uni_profs), MAX_NAMES_PER_BATCH):
            chunk = uni_profs[chunk_start:chunk_start + MAX_NAMES_PER_BATCH]
            chunk_names = [p["name"] for p in chunk]

            log.info("  Stage 1 chunk: professors %d–%d of %d",
                     chunk_start + 1,
                     min(chunk_start + MAX_NAMES_PER_BATCH, len(uni_profs)),
                     len(uni_profs))

            # Web search (includes professor names in prompt)
            search_text = analyzer.stage1_search(uni_name, chunk_names)

            if not search_text:
                log.warning("  Stage 1 search returned nothing for chunk")
                for prof in chunk:
                    excluded.append(
                        _build_excluded_entry(prof, "stage1_search_failed")
                    )
                continue

            # Extract structured results
            time.sleep(0.3)
            batch_results = analyzer.stage1_extract(
                search_text, uni_name, chunk_names,
            )

            # Build lookup by lowercase name
            result_map = {}
            for r in batch_results:
                result_map[r.get("name", "").lower().strip()] = r

            # Match each professor to extraction results
            for prof in chunk:
                name = prof["name"]
                match = result_map.get(name.lower().strip())
                if match is None:
                    match = _fuzzy_match_name(name, batch_results)

                if match and match.get("is_tenured") is True:
                    # Check keyword filter on Stage 1 title
                    title_lower = (match.get("title") or "").lower()
                    if any(kw in title_lower for kw in NON_TENURE_TRACK_KEYWORDS):
                        excluded.append(_build_excluded_entry(
                            prof, "non_tenure_track", match.get("title"),
                        ))
                        log.info("  EXCLUDED (filter): %s — %s",
                                 name, match.get("title"))
                    else:
                        tenured_candidates.append((prof, match))
                        log.info("  TENURED: %s — %s", name, match.get("title"))
                elif match and match.get("is_tenured") is False:
                    excluded.append(_build_excluded_entry(
                        prof, "not_tenured", match.get("title"),
                    ))
                else:
                    # Not found or null
                    excluded.append(_build_excluded_entry(
                        prof, "not_found_in_directory",
                    ))

            time.sleep(0.5)

        # Save Stage 1 results before entering Stage 2
        _save_json(excluded_path, excluded)

        log.info(
            "  Stage 1 done: %d tenured candidates, %d excluded",
            len(tenured_candidates), len(uni_profs) - len(tenured_candidates),
        )
        stage2_total += len(tenured_candidates)

        # ── Stage 2: Tenure year details for each candidate ──────────────
        for ti, (prof, s1_match) in enumerate(tenured_candidates, 1):
            log.info("  Stage 2 [%d/%d]: %s",
                     ti, len(tenured_candidates), prof["name"])

            llm_result = analyzer.stage2_analyze(prof)

            confidence, conf_label = compute_confidence(llm_result)

            # PhD filter
            phd_year = llm_result.get("phd_year")
            if phd_year and phd_year < config.PHD_YEAR_CUTOFF:
                entry = _build_entry(prof, llm_result, confidence, conf_label)
                entry["excluded"] = True
                entry["exclusion_reason"] = f"phd_before_{config.PHD_YEAR_CUTOFF}"
                excluded.append(entry)
                log.info("    EXCLUDED: PhD %d < cutoff %d",
                         phd_year, config.PHD_YEAR_CUTOFF)
                _save_json(output_path, results)
                _save_json(excluded_path, excluded)
                time.sleep(0.3)
                continue

            # Verify Stage 2 confirms tenure (Stage 1 can be wrong)
            rank = (llm_result.get("rank") or "").lower()
            title = (llm_result.get("current_title") or "").lower()
            combined_rt = f"{rank} {title}"
            if "assistant professor" in combined_rt and "associate" not in combined_rt:
                entry = _build_entry(prof, llm_result, confidence, conf_label)
                entry["excluded"] = True
                entry["exclusion_reason"] = "not_tenured_assistant_prof"
                excluded.append(entry)
                log.info("    EXCLUDED: Assistant Professor (not yet tenured)")
                _save_json(output_path, results)
                _save_json(excluded_path, excluded)
                time.sleep(0.3)
                continue

            # Title keyword filter on Stage 2 result
            combined = f"{rank} {title}"
            if any(kw in combined for kw in NON_TENURE_TRACK_KEYWORDS):
                entry = _build_entry(prof, llm_result, confidence, conf_label)
                entry["excluded"] = True
                entry["exclusion_reason"] = "non_tenure_track"
                excluded.append(entry)
                log.info("    EXCLUDED: non-TT title '%s'",
                         llm_result.get("current_title"))
                _save_json(output_path, results)
                _save_json(excluded_path, excluded)
                time.sleep(0.3)
                continue

            # Tenure year sanity check
            tenure_year = llm_result.get("tenure_year")
            if tenure_year is not None:
                if tenure_year < 1980 or tenure_year > CURRENT_YEAR:
                    entry = _build_entry(prof, llm_result, confidence, conf_label)
                    entry["excluded"] = True
                    entry["exclusion_reason"] = f"tenure_year_out_of_range_{tenure_year}"
                    excluded.append(entry)
                    _save_json(output_path, results)
                    _save_json(excluded_path, excluded)
                    time.sleep(0.3)
                    continue

            # ── INCLUDE ──
            entry = _build_entry(prof, llm_result, confidence, conf_label)
            entry["stage1_title"] = s1_match.get("title")
            results.append(entry)
            log.info(
                "    INCLUDED: title=%s, tenure=%s (src=%s), phd=%s, conf=%.2f",
                entry["current_title"], entry["tenure_year"],
                entry["tenure_year_source"], entry["phd_year"], confidence,
            )

            # Checkpoint
            _save_json(output_path, results)
            _save_json(excluded_path, excluded)
            time.sleep(0.3)

        # University complete
        log.info(
            "  University done. Running totals: %d included, %d excluded | %s",
            len(results), len(excluded), analyzer.stats,
        )
        time.sleep(1.0)

    # ── Final CSV export ──────────────────────────────────────────────────
    try:
        import pandas as pd
        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            log.info("CSV saved to %s", csv_path)
    except ImportError:
        log.warning("pandas not installed; skipping CSV export")

    log.info("Phase 3 v2 complete:")
    log.info("  Included : %d professors", len(results))
    log.info("  Excluded : %d professors", len(excluded))
    log.info("  Stage 2 searches: %d (tenured candidates)", stage2_total)
    log.info("  API stats: %s", analyzer.stats)

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Tenure Estimator v2 (Batched)")
    parser.add_argument("--validate", action="store_true",
                        help="Process only first 100 professors")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed professors")
    args = parser.parse_args()

    input_path = config.DATA_DIR / "professors_phase3_subset.json"
    if not input_path.exists():
        input_path = config.PROFESSORS_RMP_FILE
    if not input_path.exists():
        log.error("No input file found. Run Phase 2 first.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        profs = json.load(f)

    log.info("Loaded %d professors from %s", len(profs), input_path)
    estimate_tenure_for_all(
        profs, resume=args.resume, validation_mode=args.validate,
    )
