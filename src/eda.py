"""
Phase 6 — Exploratory Data Analysis + Hypothesis Testing

Reads data/processed_reviews.csv and produces:
  - output/plots/*.png
  - output/findings_summary.txt

Steps:
  1. Distribution of reviews across relative years
  2. Average rating / difficulty / sentiment vs relative year (with CI)
  3. Pre-vs-post tenure comparisons (t-tests + effect sizes)
  4. Regression with professor fixed effects
  5. Heterogeneity: R1 vs R2, review count groups
"""

from __future__ import annotations
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

PLOT_DIR = config.ROOT_DIR / "output" / "plots"
FINDINGS_FILE = config.ROOT_DIR / "output" / "findings_summary.txt"


# ── Plotting helpers ───────────────────────────────────────────────────────────

STYLE = {
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}

def _save(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", path)


def _yearly_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean ± SEM per relative_year for a given column."""
    g = df.groupby("relative_year")[col]
    return pd.DataFrame({
        "mean": g.mean(),
        "sem": g.sem(),
        "n": g.count(),
    }).reset_index()


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _ttest_summary(pre: pd.Series, post: pd.Series, label: str) -> dict:
    pre, post = pre.dropna(), post.dropna()
    if len(pre) < 5 or len(post) < 5:
        return {"variable": label, "n_pre": len(pre), "n_post": len(post),
                "mean_pre": np.nan, "mean_post": np.nan, "delta": np.nan,
                "t": np.nan, "p": np.nan, "cohens_d": np.nan}
    t, p = stats.ttest_ind(pre, post, equal_var=False)
    return {
        "variable": label,
        "n_pre": len(pre),
        "n_post": len(post),
        "mean_pre": round(pre.mean(), 4),
        "mean_post": round(post.mean(), 4),
        "delta": round(post.mean() - pre.mean(), 4),
        "t": round(t, 3),
        "p": round(p, 4),
        "cohens_d": round(_cohens_d(pre, post), 4),
    }


# ── Plot 1: Review count distribution ─────────────────────────────────────────

def plot_review_distribution(df: pd.DataFrame) -> None:
    counts = df.groupby("relative_year").size().reset_index(name="count")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.bar(counts["relative_year"], counts["count"], color="#4C72B0", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Tenure year")
        ax.set_xlabel("Relative year (0 = tenure year)")
        ax.set_ylabel("Number of reviews")
        ax.set_title("Distribution of RMP Reviews Relative to Tenure Year")
        ax.legend()
        _save(fig, "01_review_distribution.png")


# ── Plot 2–4: Time-series of key metrics ──────────────────────────────────────

def _plot_metric_over_time(df: pd.DataFrame, col: str, ylabel: str,
                            title: str, fname: str, ylim=None) -> None:
    stats_df = _yearly_stats(df, col)
    stats_df = stats_df[stats_df["n"] >= 3]  # require ≥3 reviews per year

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(stats_df["relative_year"], stats_df["mean"],
                color="#4C72B0", linewidth=2, marker="o", markersize=4)
        ax.fill_between(
            stats_df["relative_year"],
            stats_df["mean"] - 1.96 * stats_df["sem"],
            stats_df["mean"] + 1.96 * stats_df["sem"],
            alpha=0.2, color="#4C72B0",
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Tenure year")
        ax.set_xlabel("Relative year (0 = tenure year)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()
        _save(fig, fname)


def plot_metrics_over_time(df: pd.DataFrame) -> None:
    _plot_metric_over_time(df, "overall_rating", "Mean overall rating (1–5)",
                           "Overall Rating vs. Relative Tenure Year",
                           "02_rating_over_time.png", ylim=(1, 5))
    _plot_metric_over_time(df, "difficulty", "Mean difficulty rating (1–5)",
                           "Difficulty vs. Relative Tenure Year",
                           "03_difficulty_over_time.png", ylim=(1, 5))
    _plot_metric_over_time(df, "sentiment", "Mean VADER sentiment (−1 to 1)",
                           "Review Sentiment vs. Relative Tenure Year",
                           "04_sentiment_over_time.png", ylim=(-0.5, 1.0))
    if "would_take_again_bin" in df.columns:
        _plot_metric_over_time(df, "would_take_again_bin", "Fraction who would take again",
                               "Would Take Again vs. Relative Tenure Year",
                               "05_would_take_again_over_time.png", ylim=(0, 1))


# ── Plot 5: Pre vs post box plots ─────────────────────────────────────────────

def plot_pre_post_boxplots(df: pd.DataFrame) -> None:
    metrics = [
        ("overall_rating", "Overall Rating"),
        ("difficulty", "Difficulty"),
        ("sentiment", "Sentiment"),
    ]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=False)
        for ax, (col, label) in zip(axes, metrics):
            pre = df[df["post_tenure"] == 0][col].dropna()
            post = df[df["post_tenure"] == 1][col].dropna()
            ax.boxplot([pre, post], labels=["Pre-tenure", "Post-tenure"],
                       patch_artist=True,
                       boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                       medianprops=dict(color="red", linewidth=2))
            ax.set_title(label)
            ax.set_ylabel(label)
        fig.suptitle("Pre- vs Post-Tenure Distributions", fontsize=13, y=1.01)
        _save(fig, "06_pre_post_boxplots.png")


# ── Plot 6: Heterogeneity — R1 vs R2 ──────────────────────────────────────────

def plot_r1_vs_r2(df: pd.DataFrame) -> None:
    if "carnegie" not in df.columns:
        return
    for col, ylabel, fname in [
        ("overall_rating", "Mean overall rating", "07_r1r2_rating.png"),
        ("difficulty",     "Mean difficulty",     "08_r1r2_difficulty.png"),
    ]:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots()
            for carnegie, color in [("R1", "#4C72B0"), ("R2", "#DD8452")]:
                sub = df[df["carnegie"] == carnegie]
                s = _yearly_stats(sub, col)
                s = s[s["n"] >= 3]
                ax.plot(s["relative_year"], s["mean"],
                        color=color, linewidth=2, marker="o",
                        markersize=4, label=carnegie)
                ax.fill_between(
                    s["relative_year"],
                    s["mean"] - 1.96 * s["sem"],
                    s["mean"] + 1.96 * s["sem"],
                    alpha=0.15, color=color,
                )
            ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
            ax.set_xlabel("Relative year")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}: R1 vs R2 Universities")
            ax.legend()
            _save(fig, fname)


# ── Plot 7: Within-professor variability over time ────────────────────────────

def plot_rating_variability(df: pd.DataFrame) -> None:
    std_df = df.groupby("relative_year")["overall_rating"].std().reset_index()
    std_df.columns = ["relative_year", "std"]
    count_df = df.groupby("relative_year")["overall_rating"].count().reset_index()
    std_df = std_df.merge(count_df, on="relative_year")
    std_df = std_df[std_df["overall_rating"] >= 5]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(std_df["relative_year"], std_df["std"],
                color="#55A868", linewidth=2, marker="o", markersize=4)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Tenure year")
        ax.set_xlabel("Relative year")
        ax.set_ylabel("Std. dev. of overall rating")
        ax.set_title("Rating Variability Over Time (polarization signal)")
        ax.legend()
        _save(fig, "09_rating_variability.png")


# ── Statistical tests ─────────────────────────────────────────────────────────

def run_hypothesis_tests(df: pd.DataFrame) -> list[dict]:
    pre = df[df["post_tenure"] == 0]
    post = df[df["post_tenure"] == 1]

    results = [
        _ttest_summary(pre["overall_rating"], post["overall_rating"], "Overall Rating"),
        _ttest_summary(pre["difficulty"],     post["difficulty"],     "Difficulty"),
        _ttest_summary(pre["sentiment"],      post["sentiment"],      "Sentiment"),
    ]
    if "would_take_again_bin" in df.columns:
        results.append(
            _ttest_summary(pre["would_take_again_bin"], post["would_take_again_bin"],
                           "Would Take Again")
        )

    results_df = pd.DataFrame(results)
    log.info("\n%s", results_df.to_string(index=False))
    return results


# ── Regression with professor fixed effects ───────────────────────────────────

def run_regression(df: pd.DataFrame) -> dict[str, object]:
    """
    OLS: metric ~ relative_year + C(professor_id)
    Tests whether relative_year has a significant within-professor trend.
    """
    out = {}
    for col in ["overall_rating", "difficulty", "sentiment"]:
        sub = df[["relative_year", col, "professor_id"]].dropna()
        if len(sub) < 50:
            continue
        try:
            model = smf.ols(f"{col} ~ relative_year + C(professor_id)", data=sub).fit()
            coef = model.params.get("relative_year", float("nan"))
            pval = model.pvalues.get("relative_year", float("nan"))
            out[col] = {"coef": round(coef, 5), "p": round(pval, 4), "n": len(sub)}
            log.info("Regression %s: coef=%.5f p=%.4f (n=%d)", col, coef, pval, len(sub))
        except Exception as e:
            log.warning("Regression failed for %s: %s", col, e)
    return out


# ── Findings summary ──────────────────────────────────────────────────────────

def write_findings(df: pd.DataFrame, tests: list[dict], regressions: dict) -> None:
    n_profs = df["professor_id"].nunique()
    n_reviews = len(df)
    n_pre = (df["post_tenure"] == 0).sum()
    n_post = (df["post_tenure"] == 1).sum()

    lines = [
        "=" * 70,
        "RMP PROFESSOR TENURE ANALYSIS — FINDINGS SUMMARY",
        "=" * 70,
        "",
        f"Dataset: {n_reviews:,} reviews across {n_profs} professors",
        f"Pre-tenure reviews:  {n_pre:,}",
        f"Post-tenure reviews: {n_post:,}",
        "",
        "── Pre vs. Post Tenure (Welch t-test) ──────────────────────────────",
    ]

    for r in tests:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
        d_label = ""
        if not np.isnan(r["cohens_d"]):
            d_label = f"  d={r['cohens_d']:.3f}"
        lines.append(
            f"  {r['variable']:<22} pre={r['mean_pre']:.3f}  post={r['mean_post']:.3f}"
            f"  Δ={r['delta']:+.3f}  p={r['p']:.4f} {sig}{d_label}"
        )

    lines += [
        "",
        "── Within-professor regression (metric ~ relative_year + FE) ───────",
    ]
    for col, info in regressions.items():
        sig = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 else "*" if info["p"] < 0.05 else "ns"
        lines.append(
            f"  {col:<22} coef={info['coef']:+.5f}  p={info['p']:.4f} {sig}  (n={info['n']})"
        )

    lines += [
        "",
        "── Hypotheses suggested by data ─────────────────────────────────────",
        "",
    ]

    # Auto-generate hypothesis language from results
    for r in tests:
        if np.isnan(r["p"]):
            continue
        var = r["variable"]
        delta = r["delta"]
        sig = r["p"] < 0.05
        if sig and abs(r["cohens_d"]) >= 0.1:
            direction = "increases" if delta > 0 else "decreases"
            lines.append(
                f"  H: {var} {direction} after tenure "
                f"(Δ={delta:+.3f}, d={r['cohens_d']:.3f}, p={r['p']:.4f})"
            )
        else:
            lines.append(
                f"  H: {var} shows no significant change after tenure "
                f"(p={r['p']:.4f}, insufficient effect size)"
            )

    lines += ["", "=" * 70]

    summary = "\n".join(lines)
    print(summary)
    FINDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_FILE.write_text(summary, encoding="utf-8")
    log.info("Findings written → %s", FINDINGS_FILE)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_eda(
    processed_path: Path = config.DATA_DIR / "processed_reviews.csv",
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    log.info("Loaded processed reviews: %d rows, %d professors",
             len(df), df["professor_id"].nunique())

    if len(df) == 0:
        log.error("No data to analyze. Run review_scraper.py and analysis.py first.")
        return

    plot_review_distribution(df)
    plot_metrics_over_time(df)
    plot_pre_post_boxplots(df)
    plot_r1_vs_r2(df)
    plot_rating_variability(df)

    tests = run_hypothesis_tests(df)
    regressions = run_regression(df)
    write_findings(df, tests, regressions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_eda()
