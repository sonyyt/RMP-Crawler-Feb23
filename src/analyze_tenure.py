"""
Phase 5 — Tenure Impact Analysis

Answers four research questions using the review CSVs produced by
review_crawler.py:

  Q1. Do quality / difficulty / would-take-again change before vs. after tenure?
  Q2. Do review *sentiments* and *keywords* change before vs. after tenure?
  Q3. Does *review frequency* (reviews per year) change before vs. after tenure?
  Q4. Does the number of years before/after tenure correlate with those changes?

Outputs (all under data/analysis/):
  q1_numeric_changes.csv      — per-professor numeric metric deltas
  q1_summary.csv              — group-level stats + statistical test results
  q2_sentiment_changes.csv    — per-professor sentiment deltas
  q2_keywords_pre.csv         — top TF-IDF keywords for pre-tenure corpus
  q2_keywords_post.csv        — top TF-IDF keywords for post-tenure corpus
  q2_tag_frequencies.csv      — tag counts pre / post
  q3_frequency_changes.csv    — per-professor review-rate deltas
  q3_summary.csv              — group-level frequency stats
  q4_correlations.csv         — Spearman correlations (years × metric change)
  plots/                      — PNG figures for all four questions

Usage
─────
    python src/analyze_tenure.py              # uses default data paths
    python src/analyze_tenure.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

# ── Optional heavy imports (graceful degradation) ────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for scripts
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
    sns.set_theme(style="whitegrid", palette="muted")
except ImportError:
    PLOT_AVAILABLE = False
    log.warning("matplotlib/seaborn not available — plots will be skipped")

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    log.warning("scipy not available — statistical tests will be skipped")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # Download required corpora once
    for _corpus in ("vader_lexicon", "stopwords", "punkt"):
        try:
            nltk.data.find(f"sentiment/{_corpus}" if _corpus == "vader_lexicon" else _corpus)
        except LookupError:
            nltk.download(_corpus, quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    log.warning("nltk not available — semantic analysis will be limited")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available — TF-IDF keyword extraction will be skipped")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(
    reviews_path: Path = config.REVIEWS_ALL_FILE,
    professors_path: Path = config.PROFESSORS_FILTERED_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and lightly clean the review + professor DataFrames."""
    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Reviews file not found: {reviews_path}\n"
            "Run `python src/review_crawler.py` first."
        )

    df_rev = pd.read_csv(reviews_path, low_memory=False)
    df_prof = pd.read_csv(professors_path)

    # Type coercions
    df_rev["review_year"] = pd.to_numeric(df_rev["review_year"], errors="coerce")
    df_rev["tenure_year"] = pd.to_numeric(df_rev["tenure_year"], errors="coerce")
    df_rev["quality"] = pd.to_numeric(df_rev["quality"], errors="coerce")
    df_rev["difficulty"] = pd.to_numeric(df_rev["difficulty"], errors="coerce")
    df_rev["years_from_tenure"] = pd.to_numeric(df_rev["years_from_tenure"], errors="coerce")

    # Normalise would_take_again to 0/1/NaN
    def _wta(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, bool):
            return float(v)
        s = str(v).lower().strip()
        if s in ("true", "1", "yes"):
            return 1.0
        if s in ("false", "0", "no"):
            return 0.0
        return np.nan

    df_rev["would_take_again"] = df_rev["would_take_again"].apply(_wta)

    # Only keep known periods
    df_rev = df_rev[df_rev["period"].isin(["pre_tenure", "post_tenure"])].copy()
    log.info(
        "Reviews loaded: %d total  (%d pre, %d post)",
        len(df_rev),
        (df_rev["period"] == "pre_tenure").sum(),
        (df_rev["period"] == "post_tenure").sum(),
    )
    return df_rev, df_prof


# ── Shared utilities ──────────────────────────────────────────────────────────


def _professors_with_both_periods(
    df: pd.DataFrame,
    min_reviews: int = config.MIN_REVIEWS_PER_PERIOD,
) -> pd.DataFrame:
    """Keep only professors that have ≥ min_reviews in each period."""
    counts = (
        df.groupby(["professor_slug", "period"])
        .size()
        .unstack(fill_value=0)
    )
    both = counts[
        (counts.get("pre_tenure", 0) >= min_reviews)
        & (counts.get("post_tenure", 0) >= min_reviews)
    ].index
    filtered = df[df["professor_slug"].isin(both)].copy()
    log.info(
        "Professors with ≥%d reviews in both periods: %d",
        min_reviews,
        len(both),
    )
    return filtered


def _wilcoxon(pre_vals: np.ndarray, post_vals: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired (per-professor) means."""
    if not SCIPY_AVAILABLE or len(pre_vals) < 5:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, p = scipy_stats.wilcoxon(pre_vals, post_vals, alternative="two-sided")
            return float(stat), float(p)
        except Exception:
            return float("nan"), float("nan")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two arrays."""
    diff = a - b
    if diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def _save_fig(fig, name: str, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → Plot saved: %s", path)


# ── Q1: Numeric metrics ───────────────────────────────────────────────────────


def analyze_q1_numeric(
    df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """
    Q1: Do quality / difficulty / would-take-again change after tenure?

    Strategy
    ────────
    For each professor compute the mean of each metric in the pre and post
    period.  Then run a Wilcoxon signed-rank test across all professors on
    the paired means (non-parametric; robust to small / skewed samples).
    """
    log.info("=== Q1: Numeric metrics ===")
    df_both = _professors_with_both_periods(df)

    metrics = {
        "quality": "Quality (avg helpful+clarity, 1–5)",
        "difficulty": "Difficulty (1–5, higher = harder)",
        "would_take_again": "Would Take Again (0=No, 1=Yes)",
    }

    # Per-professor means per period
    agg = (
        df_both.groupby(["professor_slug", "professor_name", "university", "period"])[
            list(metrics.keys())
        ]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot_table(
        index=["professor_slug", "professor_name", "university"],
        columns="period",
        values=list(metrics.keys()),
    )
    pivot.columns = [f"{m}_{p}" for m, p in pivot.columns]
    pivot = pivot.reset_index()

    # Deltas (post − pre: positive = higher after tenure)
    for m in metrics:
        pre_col = f"{m}_pre_tenure"
        post_col = f"{m}_post_tenure"
        if pre_col in pivot.columns and post_col in pivot.columns:
            pivot[f"{m}_delta"] = pivot[post_col] - pivot[pre_col]

    pivot.to_csv(out_dir / "q1_numeric_changes.csv", index=False)
    log.info("  Per-professor deltas → q1_numeric_changes.csv (%d rows)", len(pivot))

    # Group-level summary
    rows = []
    for m, label in metrics.items():
        pre_col = f"{m}_pre_tenure"
        post_col = f"{m}_post_tenure"
        delta_col = f"{m}_delta"
        if pre_col not in pivot.columns or post_col not in pivot.columns:
            continue
        valid = pivot[[pre_col, post_col, delta_col]].dropna()
        pre_arr = valid[pre_col].values
        post_arr = valid[post_col].values
        stat, p = _wilcoxon(pre_arr, post_arr)
        d = _cohens_d(post_arr, pre_arr)
        direction = "↑ increases" if valid[delta_col].mean() > 0 else "↓ decreases"
        rows.append(
            {
                "metric": m,
                "label": label,
                "n_professors": len(valid),
                "mean_pre": round(float(pre_arr.mean()), 3),
                "mean_post": round(float(post_arr.mean()), 3),
                "mean_delta": round(float(valid[delta_col].mean()), 3),
                "median_delta": round(float(valid[delta_col].median()), 3),
                "cohens_d": round(d, 3),
                "wilcoxon_stat": stat,
                "p_value": round(p, 4) if not np.isnan(p) else "n/a",
                "significant_p05": bool(p < 0.05) if not np.isnan(p) else False,
                "direction": direction,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "q1_summary.csv", index=False)
    log.info("  Summary → q1_summary.csv")
    log.info("\n%s", summary_df[["metric", "mean_pre", "mean_post", "mean_delta", "p_value", "direction"]].to_string(index=False))

    # Plots
    if PLOT_AVAILABLE:
        _q1_plots(pivot, metrics, plots_dir)

    return summary_df


def _q1_plots(pivot: pd.DataFrame, metrics: dict, plots_dir: Path) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (m, label) in zip(axes, metrics.items()):
        pre_col = f"{m}_pre_tenure"
        post_col = f"{m}_post_tenure"
        if pre_col not in pivot or post_col not in pivot:
            continue
        valid = pivot[[pre_col, post_col]].dropna()
        data = pd.DataFrame(
            {
                "Pre-tenure": valid[pre_col].values,
                "Post-tenure": valid[post_col].values,
            }
        )
        data_long = data.melt(var_name="Period", value_name="Value")
        sns.violinplot(data=data_long, x="Period", y="Value", ax=ax, inner="box", cut=0)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Mean (per professor)")
        # draw connecting lines
        for _, row in valid.iterrows():
            ax.plot([0, 1], [row[pre_col], row[post_col]], color="grey", alpha=0.3, lw=0.8)

    fig.suptitle("Q1: Numeric Metric Changes Before vs. After Tenure", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "q1_metric_violins.png", plots_dir)

    # Box plot of deltas
    delta_cols = [f"{m}_delta" for m in metrics if f"{m}_delta" in pivot.columns]
    if delta_cols:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        delta_data = pivot[delta_cols].dropna(how="all")
        delta_data.columns = [c.replace("_delta", "") for c in delta_data.columns]
        delta_long = delta_data.melt(var_name="Metric", value_name="Delta (post − pre)")
        sns.boxplot(data=delta_long, x="Metric", y="Delta (post − pre)", ax=ax2)
        ax2.axhline(0, color="red", linestyle="--", lw=1, label="No change")
        ax2.set_title("Q1: Distribution of Metric Deltas (Post − Pre Tenure)")
        ax2.legend()
        fig2.tight_layout()
        _save_fig(fig2, "q1_delta_boxplot.png", plots_dir)


# ── Q2: Semantic analysis ─────────────────────────────────────────────────────


def analyze_q2_semantic(
    df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """
    Q2: Do review sentiments and keywords change after tenure?

    Sub-analyses
    ────────────
    (a) VADER compound sentiment score per review → compare pre vs. post
    (b) TF-IDF top-50 keywords for pre- vs. post-tenure corpora
    (c) Tag frequency pre vs. post tenure
    """
    log.info("=== Q2: Semantic analysis ===")
    df_both = _professors_with_both_periods(df)
    df_text = df_both[df_both["comment"].notna() & (df_both["comment"] != "")].copy()

    # ── (a) Sentiment ──────────────────────────────────────────────────────────
    sentiment_df = pd.DataFrame()
    if NLTK_AVAILABLE:
        sia = SentimentIntensityAnalyzer()
        df_text["sentiment_compound"] = df_text["comment"].apply(
            lambda t: sia.polarity_scores(str(t))["compound"]
        )
        df_text["sentiment_positive"] = df_text["comment"].apply(
            lambda t: sia.polarity_scores(str(t))["pos"]
        )
        df_text["sentiment_negative"] = df_text["comment"].apply(
            lambda t: sia.polarity_scores(str(t))["neg"]
        )

        sent_agg = (
            df_text.groupby(["professor_slug", "professor_name", "period"])[
                ["sentiment_compound", "sentiment_positive", "sentiment_negative"]
            ]
            .mean()
            .reset_index()
        )
        sent_pivot = sent_agg.pivot_table(
            index=["professor_slug", "professor_name"],
            columns="period",
            values=["sentiment_compound", "sentiment_positive", "sentiment_negative"],
        )
        sent_pivot.columns = [f"{m}_{p}" for m, p in sent_pivot.columns]
        sent_pivot = sent_pivot.reset_index()
        for s in ["sentiment_compound", "sentiment_positive", "sentiment_negative"]:
            pre_c = f"{s}_pre_tenure"
            post_c = f"{s}_post_tenure"
            if pre_c in sent_pivot.columns and post_c in sent_pivot.columns:
                sent_pivot[f"{s}_delta"] = sent_pivot[post_c] - sent_pivot[pre_c]

        sentiment_df = sent_pivot
        sentiment_df.to_csv(out_dir / "q2_sentiment_changes.csv", index=False)

        valid_sent = sent_pivot[
            ["sentiment_compound_pre_tenure", "sentiment_compound_post_tenure"]
        ].dropna()
        if len(valid_sent) >= 5:
            stat, p = _wilcoxon(
                valid_sent["sentiment_compound_pre_tenure"].values,
                valid_sent["sentiment_compound_post_tenure"].values,
            )
            log.info(
                "  Sentiment compound  pre=%.3f  post=%.3f  Wilcoxon p=%.4f",
                valid_sent["sentiment_compound_pre_tenure"].mean(),
                valid_sent["sentiment_compound_post_tenure"].mean(),
                p if not np.isnan(p) else -1,
            )

        if PLOT_AVAILABLE:
            _q2_sentiment_plot(df_text, plots_dir)
    else:
        log.info("  NLTK not available — skipping sentiment")

    # ── (b) TF-IDF keywords ────────────────────────────────────────────────────
    if SKLEARN_AVAILABLE:
        _q2_tfidf(df_text, out_dir, plots_dir)
    else:
        log.info("  scikit-learn not available — skipping TF-IDF")

    # ── (c) Tag frequencies ────────────────────────────────────────────────────
    _q2_tags(df_both, out_dir, plots_dir)

    return sentiment_df


def _q2_sentiment_plot(df_text: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution of compound scores
    for period, color, label in [
        ("pre_tenure", "steelblue", "Pre-tenure"),
        ("post_tenure", "coral", "Post-tenure"),
    ]:
        sub = df_text[df_text["period"] == period]["sentiment_compound"].dropna()
        axes[0].hist(sub, bins=30, alpha=0.6, color=color, label=label)
    axes[0].set_title("Q2: Sentiment Score Distribution")
    axes[0].set_xlabel("VADER Compound Score (−1 = most negative, +1 = most positive)")
    axes[0].legend()

    # Box plot per period
    sent_data = df_text[["period", "sentiment_compound"]].dropna()
    sent_data["period"] = sent_data["period"].map(
        {"pre_tenure": "Pre-tenure", "post_tenure": "Post-tenure"}
    )
    sns.boxplot(data=sent_data, x="period", y="sentiment_compound", ax=axes[1])
    axes[1].set_title("Q2: Sentiment by Period")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("VADER Compound Score")

    fig.tight_layout()
    _save_fig(fig, "q2_sentiment.png", plots_dir)


def _q2_tfidf(df_text: pd.DataFrame, out_dir: Path, plots_dir: Path) -> None:
    stop = set()
    if NLTK_AVAILABLE:
        try:
            stop = set(stopwords.words("english"))
        except Exception:
            pass
    stop.update(["professor", "class", "course", "really", "just", "also"])

    def _corpus(period: str) -> list[str]:
        return df_text[df_text["period"] == period]["comment"].dropna().tolist()

    pre_corpus = _corpus("pre_tenure")
    post_corpus = _corpus("post_tenure")

    if not pre_corpus or not post_corpus:
        log.warning("  Not enough text for TF-IDF")
        return

    vec = TfidfVectorizer(
        stop_words=list(stop),
        max_features=200,
        ngram_range=(1, 2),
        min_df=3,
    )
    vec.fit(pre_corpus + post_corpus)
    vocab = vec.get_feature_names_out()

    def _top_keywords(corpus: list[str], n: int = 50) -> pd.DataFrame:
        mat = vec.transform(corpus)
        scores = np.asarray(mat.mean(axis=0)).flatten()
        idx = np.argsort(scores)[::-1][:n]
        return pd.DataFrame({"keyword": vocab[idx], "tfidf_score": scores[idx]})

    kw_pre = _top_keywords(pre_corpus)
    kw_post = _top_keywords(post_corpus)
    kw_pre.to_csv(out_dir / "q2_keywords_pre.csv", index=False)
    kw_post.to_csv(out_dir / "q2_keywords_post.csv", index=False)
    log.info("  TF-IDF keywords → q2_keywords_pre.csv / q2_keywords_post.csv")

    if PLOT_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, kw_df, title in [
            (axes[0], kw_pre.head(20), "Pre-tenure Top-20 Keywords"),
            (axes[1], kw_post.head(20), "Post-tenure Top-20 Keywords"),
        ]:
            sns.barh(kw_df["keyword"][::-1], kw_df["tfidf_score"][::-1], ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Mean TF-IDF Score")
        fig.suptitle("Q2: Most Distinctive Keywords Before vs. After Tenure", fontsize=13)
        fig.tight_layout()
        _save_fig(fig, "q2_keywords.png", plots_dir)

    # Word clouds
    if WORDCLOUD_AVAILABLE and PLOT_AVAILABLE:
        for corpus, fname, title in [
            (pre_corpus, "q2_wordcloud_pre.png", "Pre-tenure Reviews"),
            (post_corpus, "q2_wordcloud_post.png", "Post-tenure Reviews"),
        ]:
            text = " ".join(corpus)
            wc = WordCloud(
                width=800, height=400, stopwords=stop, background_color="white"
            ).generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            ax_wc.set_title(title, fontsize=14)
            _save_fig(fig_wc, fname, plots_dir)


def _q2_tags(df_both: pd.DataFrame, out_dir: Path, plots_dir: Path) -> None:
    rows = []
    for period in ["pre_tenure", "post_tenure"]:
        sub = df_both[df_both["period"] == period]["tags"].dropna()
        tag_counts: dict[str, int] = {}
        for cell in sub:
            for tag in str(cell).split("|"):
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for tag, count in tag_counts.items():
            rows.append({"period": period, "tag": tag, "count": count})

    if not rows:
        return
    tag_df = pd.DataFrame(rows)
    tag_pivot = tag_df.pivot_table(
        index="tag", columns="period", values="count", fill_value=0
    ).reset_index()
    # Normalise by total reviews in each period
    n_pre = (df_both["period"] == "pre_tenure").sum()
    n_post = (df_both["period"] == "post_tenure").sum()
    if "pre_tenure" in tag_pivot.columns:
        tag_pivot["pre_pct"] = tag_pivot["pre_tenure"] / max(n_pre, 1) * 100
    if "post_tenure" in tag_pivot.columns:
        tag_pivot["post_pct"] = tag_pivot["post_tenure"] / max(n_post, 1) * 100
    tag_pivot = tag_pivot.sort_values(
        by=tag_pivot.columns[tag_pivot.columns.str.endswith("pct")].tolist(),
        ascending=False,
    )
    tag_pivot.to_csv(out_dir / "q2_tag_frequencies.csv", index=False)
    log.info("  Tag frequencies → q2_tag_frequencies.csv")

    if PLOT_AVAILABLE and "pre_pct" in tag_pivot.columns and "post_pct" in tag_pivot.columns:
        top_tags = tag_pivot.nlargest(20, "pre_pct")
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(top_tags))
        w = 0.35
        ax.barh(x - w / 2, top_tags["pre_pct"], w, label="Pre-tenure", color="steelblue")
        ax.barh(x + w / 2, top_tags["post_pct"], w, label="Post-tenure", color="coral")
        ax.set_yticks(x)
        ax.set_yticklabels(top_tags["tag"])
        ax.set_xlabel("% of reviews with this tag")
        ax.set_title("Q2: Top-20 Review Tags Pre vs. Post Tenure")
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, "q2_tags.png", plots_dir)


# ── Q3: Review frequency ──────────────────────────────────────────────────────


def analyze_q3_frequency(
    df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """
    Q3: Does review frequency (reviews per year) change after tenure?

    For each professor:
      rate_pre  = pre-tenure reviews  / years of pre-tenure coverage
      rate_post = post-tenure reviews / years of post-tenure coverage
    """
    log.info("=== Q3: Review frequency ===")
    df_both = _professors_with_both_periods(df)

    rows = []
    for slug, grp in df_both.groupby("professor_slug"):
        name = grp["professor_name"].iloc[0]
        university = grp["university"].iloc[0]
        tenure_year = grp["tenure_year"].iloc[0]

        pre = grp[grp["period"] == "pre_tenure"]
        post = grp[grp["period"] == "post_tenure"]

        def _rate(sub: pd.DataFrame) -> Optional[float]:
            years = sub["review_year"].dropna()
            if len(years) < 2:
                return float(len(sub))  # can't measure rate; return raw count
            span = years.max() - years.min()
            return float(len(sub) / span) if span > 0 else float(len(sub))

        rate_pre = _rate(pre)
        rate_post = _rate(post)
        pre_span = (
            pre["review_year"].max() - pre["review_year"].min()
            if pre["review_year"].notna().sum() >= 2
            else None
        )
        post_span = (
            post["review_year"].max() - post["review_year"].min()
            if post["review_year"].notna().sum() >= 2
            else None
        )

        rows.append(
            {
                "professor_slug": slug,
                "professor_name": name,
                "university": university,
                "tenure_year": tenure_year,
                "n_pre": len(pre),
                "n_post": len(post),
                "pre_span_years": pre_span,
                "post_span_years": post_span,
                "rate_pre": round(rate_pre, 3) if rate_pre is not None else None,
                "rate_post": round(rate_post, 3) if rate_post is not None else None,
                "rate_delta": (
                    round(rate_post - rate_pre, 3)
                    if rate_pre is not None and rate_post is not None
                    else None
                ),
            }
        )

    freq_df = pd.DataFrame(rows)
    freq_df.to_csv(out_dir / "q3_frequency_changes.csv", index=False)
    log.info("  Frequency table → q3_frequency_changes.csv (%d rows)", len(freq_df))

    valid = freq_df[["rate_pre", "rate_post", "rate_delta"]].dropna()
    stat, p = _wilcoxon(valid["rate_pre"].values, valid["rate_post"].values)
    summary = {
        "n_professors": len(valid),
        "mean_rate_pre": round(float(valid["rate_pre"].mean()), 3),
        "mean_rate_post": round(float(valid["rate_post"].mean()), 3),
        "mean_delta": round(float(valid["rate_delta"].mean()), 3),
        "median_delta": round(float(valid["rate_delta"].median()), 3),
        "wilcoxon_stat": stat,
        "p_value": round(p, 4) if not np.isnan(p) else "n/a",
        "significant_p05": bool(p < 0.05) if not np.isnan(p) else False,
        "direction": "↑ increases" if valid["rate_delta"].mean() > 0 else "↓ decreases",
    }
    pd.DataFrame([summary]).to_csv(out_dir / "q3_summary.csv", index=False)
    log.info(
        "  Frequency: pre=%.2f  post=%.2f  delta=%.2f  p=%.4s",
        summary["mean_rate_pre"],
        summary["mean_rate_post"],
        summary["mean_delta"],
        summary["p_value"],
    )

    if PLOT_AVAILABLE:
        _q3_plots(freq_df, valid, plots_dir)

    return freq_df


def _q3_plots(freq_df: pd.DataFrame, valid: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Paired violin
    rate_long = pd.DataFrame(
        {
            "Pre-tenure": valid["rate_pre"].values,
            "Post-tenure": valid["rate_post"].values,
        }
    ).melt(var_name="Period", value_name="Reviews per year")
    sns.violinplot(
        data=rate_long, x="Period", y="Reviews per year", ax=axes[0], inner="box", cut=0
    )
    for _, row in valid.iterrows():
        axes[0].plot([0, 1], [row["rate_pre"], row["rate_post"]], color="grey", alpha=0.3, lw=0.8)
    axes[0].set_title("Q3: Review Rate Before vs. After Tenure")

    # Delta distribution
    axes[1].hist(valid["rate_delta"], bins=20, color="teal", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", lw=1)
    axes[1].set_title("Q3: Change in Review Rate (Post − Pre)")
    axes[1].set_xlabel("Δ Reviews per year")

    fig.tight_layout()
    _save_fig(fig, "q3_frequency.png", plots_dir)

    # Year-by-year review counts heatmap (professors × year)
    if "review_year" in freq_df.columns:
        pass  # skip heatmap if not enough data


# ── Q4: Years before/after tenure as moderator ────────────────────────────────


def analyze_q4_years_impact(
    df: pd.DataFrame,
    q1_df: pd.DataFrame,
    q3_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """
    Q4: Does the number of years of pre/post-tenure data moderate the changes?

    Approach
    ────────
    For each professor compute:
      years_pre  = tenure_year − min(pre review year)
      years_post = max(post review year) − tenure_year

    Then compute Spearman correlations between (years_pre, years_post) and
    each metric delta from Q1/Q3.
    """
    log.info("=== Q4: Years-before/after-tenure impact ===")
    df_both = _professors_with_both_periods(df)

    # Build per-professor years info
    years_rows = []
    for slug, grp in df_both.groupby("professor_slug"):
        tenure_year = grp["tenure_year"].iloc[0]
        pre = grp[grp["period"] == "pre_tenure"]["review_year"].dropna()
        post = grp[grp["period"] == "post_tenure"]["review_year"].dropna()
        if tenure_year is None or tenure_year != tenure_year:
            continue
        yrs_pre = float(tenure_year) - float(pre.min()) if len(pre) > 0 else None
        yrs_post = float(post.max()) - float(tenure_year) if len(post) > 0 else None
        years_rows.append(
            {
                "professor_slug": slug,
                "tenure_year": tenure_year,
                "years_pre": yrs_pre,
                "years_post": yrs_post,
            }
        )

    years_df = pd.DataFrame(years_rows)

    # Merge with Q1 deltas
    q1_delta_cols = [c for c in q1_df.columns if c.endswith("_delta")]
    merged = years_df.merge(
        q1_df[["professor_slug"] + q1_delta_cols], on="professor_slug", how="left"
    )

    # Merge with Q3 rate_delta
    if "professor_slug" in q3_df.columns and "rate_delta" in q3_df.columns:
        merged = merged.merge(
            q3_df[["professor_slug", "rate_delta"]].rename(
                columns={"rate_delta": "review_rate_delta"}
            ),
            on="professor_slug",
            how="left",
        )

    merged.to_csv(out_dir / "q4_correlations_raw.csv", index=False)

    # Spearman correlations
    corr_rows = []
    outcome_cols = q1_delta_cols + (
        ["review_rate_delta"] if "review_rate_delta" in merged.columns else []
    )
    for predictor in ["years_pre", "years_post"]:
        for outcome in outcome_cols:
            pair = merged[[predictor, outcome]].dropna()
            if len(pair) < 5 or not SCIPY_AVAILABLE:
                corr, p = float("nan"), float("nan")
            else:
                corr, p = scipy_stats.spearmanr(pair[predictor], pair[outcome])
            corr_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": len(pair),
                    "spearman_r": round(float(corr), 3),
                    "p_value": round(float(p), 4) if not np.isnan(p) else "n/a",
                    "significant_p05": bool(p < 0.05) if not np.isnan(p) else False,
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "q4_correlations.csv", index=False)
    log.info("  Spearman correlations → q4_correlations.csv")
    log.info("\n%s", corr_df.to_string(index=False))

    if PLOT_AVAILABLE:
        _q4_plots(merged, outcome_cols, plots_dir)

    return corr_df


def _q4_plots(merged: pd.DataFrame, outcome_cols: list[str], plots_dir: Path) -> None:
    predictors = [c for c in ["years_pre", "years_post"] if c in merged.columns]
    outcomes = [c for c in outcome_cols if c in merged.columns]
    if not predictors or not outcomes:
        return

    n_rows = len(predictors)
    n_cols = len(outcomes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for r, pred in enumerate(predictors):
        for c, out in enumerate(outcomes):
            ax = axes[r][c]
            pair = merged[[pred, out]].dropna()
            ax.scatter(pair[pred], pair[out], alpha=0.6, s=50)
            if len(pair) >= 3 and SCIPY_AVAILABLE:
                slope, intercept, *_ = scipy_stats.linregress(pair[pred], pair[out])
                x_range = np.linspace(pair[pred].min(), pair[pred].max(), 100)
                ax.plot(x_range, slope * x_range + intercept, color="red", lw=1.5)
            ax.axhline(0, color="grey", linestyle="--", lw=0.8)
            ax.set_xlabel(pred.replace("_", " "))
            ax.set_ylabel(out.replace("_", " "))
            ax.set_title(f"{pred} → {out}")

    fig.suptitle("Q4: Years Before/After Tenure vs. Metric Changes", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "q4_scatter.png", plots_dir)


# ── Main ──────────────────────────────────────────────────────────────────────


def run_all(
    reviews_path: Path = config.REVIEWS_ALL_FILE,
    professors_path: Path = config.PROFESSORS_FILTERED_CSV,
    out_dir: Path = config.ANALYSIS_DIR,
    plots_dir: Path = config.ANALYSIS_PLOTS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_rev, df_prof = load_data(reviews_path, professors_path)

    q1_df = analyze_q1_numeric(df_rev, out_dir, plots_dir)
    q2_df = analyze_q2_semantic(df_rev, out_dir, plots_dir)
    q3_df = analyze_q3_frequency(df_rev, out_dir, plots_dir)
    q4_df = analyze_q4_years_impact(df_rev, q1_df, q3_df, out_dir, plots_dir)

    log.info("\n══════════════════════════════════════════")
    log.info("Analysis complete.  Results in: %s", out_dir)
    log.info("  Plots in:  %s", plots_dir)
    log.info("══════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tenure impact analysis (Q1–Q4)")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=config.REVIEWS_ALL_FILE,
        help="Path to all_reviews.csv produced by review_crawler.py",
    )
    parser.add_argument(
        "--professors",
        type=Path,
        default=config.PROFESSORS_FILTERED_CSV,
        help="Path to professors_filtered.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.ANALYSIS_DIR,
        help="Output directory for analysis CSVs and plots",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    run_all(
        reviews_path=args.reviews,
        professors_path=args.professors,
        out_dir=args.out_dir,
        plots_dir=args.out_dir / "plots",
    )
