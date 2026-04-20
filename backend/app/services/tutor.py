from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.models import GraphRecommendation

STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "and", "or", "in", "on", "with", "is", "are", "be", "from",
    "make", "graph", "chart", "plot", "show", "about", "dataset", "data", "should", "how", "what", "i",
    "we", "our", "me", "my", "do", "does", "this", "that", "can", "could", "would", "best", "good",
}


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if math.isfinite(value) and abs(value) >= 1000:
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    return f"{value:.{digits}g}"


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if token not in STOPWORDS]


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _match_columns(question: str, columns: Sequence[str]) -> List[str]:
    q = question.lower().strip()
    q_tokens = _token_set(q)
    scored: List[Tuple[float, str]] = []

    for col in columns:
        col_lower = col.lower()
        col_tokens = _token_set(col_lower)
        score = 0.0
        if col_lower in q:
            score += 4.0
        overlap = len(q_tokens.intersection(col_tokens))
        if overlap:
            score += overlap * 1.5
        ratio = SequenceMatcher(None, q, col_lower).ratio()
        if ratio >= 0.55:
            score += ratio
        if col_tokens and col_tokens.issubset(q_tokens):
            score += 2.0
        if score > 0:
            scored.append((score, col))

    scored.sort(key=lambda item: item[0], reverse=True)
    matched: List[str] = []
    for _, col in scored:
        if col not in matched:
            matched.append(col)
    return matched[:6]


def _question_intent(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ["heatmap", "correlation matrix", "all relationships", "all variables related"]):
        return "matrix"
    if any(word in q for word in ["trend", "over time", "change over time", "timeline", "time series", "increase", "decrease"]):
        return "trend"
    if any(word in q for word in ["correlation", "relationship", "associate", "associated", "related to", "vs", "against"]):
        return "relationship"
    if any(word in q for word in ["compare", "difference", "between groups", "across groups", "higher than", "lower than"]):
        return "comparison"
    if any(word in q for word in ["distribution", "spread", "range", "outlier", "skew", "normal"]):
        return "distribution"
    if any(word in q for word in ["count", "frequency", "how many", "proportion", "share", "composition", "percentage"]):
        return "composition"
    if any(word in q for word in ["top", "highest", "lowest", "rank", "ranking"]):
        return "ranking"
    if any(word in q for word in ["missing", "na", "null", "empty"]):
        return "missingness"
    return "overview"


def _choose_first(preferred: Iterable[str], available: Sequence[str], exclude: Optional[Sequence[str]] = None) -> Optional[str]:
    exclude_set = set(exclude or [])
    for col in preferred:
        if col in available and col not in exclude_set:
            return col
    for col in available:
        if col not in exclude_set:
            return col
    return None


def _skew_label(skew: float) -> str:
    if skew >= 1:
        return "strongly right-skewed"
    if skew >= 0.35:
        return "mildly right-skewed"
    if skew <= -1:
        return "strongly left-skewed"
    if skew <= -0.35:
        return "mildly left-skewed"
    return "roughly symmetric"


def _distribution_stats(df: pd.DataFrame, col: str) -> Dict:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return {}
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    outlier_count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    mean = float(series.mean())
    median = float(series.median())
    skew = float(series.skew()) if len(series) >= 3 else 0.0
    return {
        "n": int(len(series)),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": mean,
        "median": median,
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "outliers": outlier_count,
        "skew": skew,
        "skew_label": _skew_label(skew),
    }


def _distribution_observations(df: pd.DataFrame, col: str) -> Tuple[str, List[str], List[str]]:
    stats = _distribution_stats(df, col)
    if not stats:
        return (f"{col} does not have enough numeric values to analyze.", [], [])

    answer = (
        f"In this dataset, {col} runs from {_fmt(stats['min'])} to {_fmt(stats['max'])}. "
        f"The mean is {_fmt(stats['mean'])} and the median is {_fmt(stats['median'])}, so the distribution looks {stats['skew_label']}."
    )
    if stats["outliers"]:
        answer += f" There are about {stats['outliers']} likely outliers by the 1.5×IQR rule."

    obs = [
        f"{col} spans {_fmt(stats['min'])} to {_fmt(stats['max'])} across {stats['n']} non-missing rows.",
        f"Center: mean {_fmt(stats['mean'])}, median {_fmt(stats['median'])}, standard deviation {_fmt(stats['std'])}.",
        f"Middle 50% of values fall between {_fmt(stats['q1'])} and {_fmt(stats['q3'])}.",
    ]
    if stats["outliers"]:
        obs.append(f"Potential outliers detected: {stats['outliers']}.")

    interp = [
        "Use a histogram to see the overall shape and whether values cluster into one or more peaks.",
        "Add a box plot when you want outliers and spread to stand out immediately.",
        "If the mean is far from the median, the distribution is skewed and the median is often a safer summary.",
    ]
    return answer, obs, interp


def _relationship_observations(df: pd.DataFrame, x: str, y: str) -> Tuple[str, List[str], List[str]]:
    subset = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 3:
        return (f"There are not enough paired values in {x} and {y} to estimate a relationship.", [], [])

    corr = float(subset[x].corr(subset[y]))
    strength = "weak"
    if abs(corr) >= 0.75:
        strength = "strong"
    elif abs(corr) >= 0.4:
        strength = "moderate"
    direction = "positive" if corr >= 0 else "negative"
    slope = np.polyfit(subset[x], subset[y], 1)[0] if len(subset) >= 2 else float("nan")

    answer = (
        f"{x} and {y} show a {strength} {direction} relationship in this dataset (Pearson r = {_fmt(corr, 2)}). "
        f"That means {y} tends to {'increase' if corr >= 0 else 'decrease'} as {x} gets larger."
    )

    obs = [
        f"Pearson correlation: {_fmt(corr, 2)} using {len(subset)} paired rows.",
        f"Approximate fitted slope: {_fmt(slope)} change in {y} for each 1-unit increase in {x}.",
        f"{x} ranges from {_fmt(subset[x].min())} to {_fmt(subset[x].max())}; {y} ranges from {_fmt(subset[y].min())} to {_fmt(subset[y].max())}.",
    ]

    interp = [
        "A scatterplot shows the raw point cloud; that is the main graph to start with.",
        "If thousands of points overlap, switch to a hexbin plot so dense regions stop hiding each other.",
        "Correlation shows association, not causation, so use the pattern as evidence of linkage rather than proof of mechanism.",
    ]
    return answer, obs, interp


def _comparison_frame(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    subset = df[[group_col, value_col]].copy()
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna()
    return subset


def _comparison_observations(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[str, List[str], List[str]]:
    subset = _comparison_frame(df, group_col, value_col)
    if subset.empty:
        return (f"There are not enough non-missing values to compare {value_col} across {group_col}.", [], [])

    grouped = subset.groupby(group_col)[value_col].agg(["mean", "median", "count", "std"]).sort_values("mean", ascending=False)
    top_group = grouped.index[0]
    bottom_group = grouped.index[-1]
    answer = (
        f"The average {value_col} is highest in {top_group} ({_fmt(grouped.iloc[0]['mean'])}) and lowest in {bottom_group} ({_fmt(grouped.iloc[-1]['mean'])}). "
        f"This suggests the groups differ in central tendency, but you should also check spread and sample size."
    )

    obs = [
        f"Highest mean {value_col}: {idx} = {_fmt(row['mean'])} (median {_fmt(row['median'])}, n={int(row['count'])})."
        for idx, row in grouped.head(4).iterrows()
    ]
    interp = [
        "Use a box plot when you need to compare both the center and spread across groups.",
        "A violin plot is helpful when shape matters and each group has enough rows.",
        "A bar chart of means is fine for a quick ranking, but it hides within-group variability.",
    ]
    return answer, obs, interp


def _trend_frame(df: pd.DataFrame, time_col: str, value_col: str, group_col: Optional[str] = None) -> pd.DataFrame:
    keep_cols = [time_col, value_col] + ([group_col] if group_col else [])
    subset = df[keep_cols].copy()
    subset[time_col] = pd.to_datetime(subset[time_col], errors="coerce")
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=[time_col, value_col]).sort_values(time_col)
    return subset


def _trend_observations(df: pd.DataFrame, time_col: str, value_col: str, group_col: Optional[str] = None) -> Tuple[str, List[str], List[str]]:
    subset = _trend_frame(df, time_col, value_col, group_col)
    if len(subset) < 3:
        return (f"There are not enough time-ordered values in {time_col} and {value_col} to estimate a trend.", [], [])

    if group_col and group_col in subset.columns:
        grouped = subset.groupby(group_col)[value_col].agg(["mean", "count"]).sort_values("mean", ascending=False)
        lead = grouped.index[0]
        answer = (
            f"Across {time_col}, {value_col} changes over time, and {lead} has the highest average level ({_fmt(grouped.iloc[0]['mean'])}). "
            f"Use separate lines by {group_col} if you need to compare trend shapes across groups."
        )
        obs = [
            f"Time range: {subset[time_col].min().date()} to {subset[time_col].max().date()}.",
            f"Overall {value_col} range: {_fmt(subset[value_col].min())} to {_fmt(subset[value_col].max())}.",
            f"Highest average by {group_col}: {lead} ({_fmt(grouped.iloc[0]['mean'])}, n={int(grouped.iloc[0]['count'])}).",
        ]
    else:
        first_val = float(subset.iloc[0][value_col])
        last_val = float(subset.iloc[-1][value_col])
        change = last_val - first_val
        pct = (change / first_val * 100.0) if first_val not in (0, 0.0) else None
        direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed about the same"
        answer = (
            f"Across the recorded time range, {value_col} {direction} from {_fmt(first_val)} to {_fmt(last_val)}"
            + (f" ({_fmt(pct, 2)}%)." if pct is not None else ".")
        )
        obs = [
            f"Time range: {subset[time_col].min().date()} to {subset[time_col].max().date()}.",
            f"Lowest {value_col}: {_fmt(subset[value_col].min())}; highest: {_fmt(subset[value_col].max())}.",
            f"Net change across the observed period: {_fmt(change)}.",
        ]

    interp = [
        "Use a line chart as the default trend graph because it preserves order and makes slope changes easy to see.",
        "Add a rolling mean when the line is noisy and you care about the broader trend instead of point-to-point fluctuation.",
        "If the metric is cumulative, an area chart can work as a second graph, but start with a line chart first.",
    ]
    return answer, obs, interp


def _composition_observations(df: pd.DataFrame, col: str, second_col: Optional[str] = None) -> Tuple[str, List[str], List[str]]:
    counts = df[col].astype(str).fillna("Missing").value_counts(dropna=False)
    if counts.empty:
        return (f"{col} does not have enough usable values to summarize.", [], [])

    total = max(1, int(counts.sum()))
    top_label = str(counts.index[0])
    top_count = int(counts.iloc[0])
    top_share = top_count / total * 100
    answer = f"{top_label} is the largest {col} category with {top_count} rows ({_fmt(top_share, 3)}% of the dataset)."

    obs = [f"{label}: {int(count)} rows ({_fmt(count / total * 100, 3)}%)." for label, count in counts.head(5).items()]
    if second_col:
        obs.append(f"Because you also have {second_col}, a stacked bar chart can show how {col} composition changes inside each {second_col} group.")
    interp = [
        "Use a sorted bar chart when you mainly care about which categories dominate.",
        "Use a normalized stacked bar chart when you need to compare composition across a second grouping variable.",
        "Be careful with very small categories because they can look dramatic while representing only a few rows.",
    ]
    return answer, obs, interp


def _ranking_observations(df: pd.DataFrame, group_col: str, value_col: str) -> Tuple[str, List[str], List[str]]:
    subset = _comparison_frame(df, group_col, value_col)
    if subset.empty:
        return (f"There is not enough non-missing data to rank {group_col} by {value_col}.", [], [])

    grouped = subset.groupby(group_col)[value_col].mean().sort_values(ascending=False)
    answer = f"The top {group_col} by mean {value_col} is {grouped.index[0]} with an average of {_fmt(grouped.iloc[0])}."
    obs = [f"{idx}: mean {value_col} = {_fmt(val)}." for idx, val in grouped.head(5).items()]
    interp = [
        "A sorted horizontal bar chart is the clearest way to show ranking.",
        "A lollipop chart is a lighter alternative when you want the same ranking story with less ink.",
        "Always show the metric used for ranking in the axis label so the ordering is easy to defend.",
    ]
    return answer, obs, interp


def _matrix_observations(record: Dict) -> Tuple[str, List[str], List[str]]:
    top_corrs = record.get("top_correlations", [])
    if not top_corrs:
        return ("This dataset does not have enough numeric columns for a useful correlation heatmap.", [], [])
    top = top_corrs[0]
    answer = (
        f"The strongest numeric relationship in this dataset is {top['x']} vs {top['y']} with correlation {_fmt(top['corr'], 2)}. "
        f"A correlation heatmap is the fastest way to scan all numeric variables at once."
    )
    obs = [f"{pair['x']} vs {pair['y']}: r = {_fmt(pair['corr'], 2)}." for pair in top_corrs[:5]]
    interp = [
        "Use a heatmap when you want a whole-dataset overview of numeric relationships instead of a single pair.",
        "Strong positive cells mean two variables rise together; strong negative cells mean they move in opposite directions.",
        "Follow a heatmap with a scatterplot for the most interesting pair so you can see whether the relationship is linear or driven by outliers.",
    ]
    return answer, obs, interp


def _missingness_observations(record: Dict) -> Tuple[str, List[str], List[str]]:
    columns = record.get("columns", [])
    ranked = sorted(columns, key=lambda c: c.get("missing_count", 0), reverse=True)
    top = ranked[0] if ranked else None
    if not top or top.get("missing_count", 0) == 0:
        return ("The uploaded dataset has no detected missing values in its columns.", ["All columns appear complete in this file."], ["You can focus on the substantive analysis without a missing-data graph."])
    answer = f"The most missing values are in {top['name']} ({top['missing_count']} rows). A missingness bar chart is the best first check."
    obs = [f"{col['name']}: {col['missing_count']} missing values." for col in ranked[:6] if col.get("missing_count", 0) > 0]
    interp = [
        "Use a missingness bar chart to spot which columns need cleaning first.",
        "If missing values cluster by time or group, that pattern can bias the result more than the raw count suggests.",
        "Handle missingness before interpreting trends or relationships, especially when one key variable is heavily incomplete.",
    ]
    return answer, obs, interp


def _fallback_answer(record: Dict) -> Tuple[str, List[str], List[str]]:
    description = record["description"]
    numeric_cols = record["numeric_columns"]
    categorical_cols = record["categorical_columns"]
    top_corrs = record.get("top_correlations", [])

    answer = f"This dataset is about: {description}. "
    if numeric_cols:
        answer += f"Its main quantitative columns include {', '.join(numeric_cols[:4])}. "
    if categorical_cols:
        answer += f"Its grouping columns include {', '.join(categorical_cols[:4])}. "
    if top_corrs:
        top = top_corrs[0]
        answer += f"The strongest numeric pattern so far is {top['x']} vs {top['y']} (r={_fmt(top['corr'], 2)})."

    obs = record.get("overview_bullets", [])[:4]
    interp = [
        "Ask about a trend, comparison, ranking, relationship, distribution, or composition to get a more specific answer.",
        "The best graph depends on which columns matter and whether you care about change, spread, or group differences.",
    ]
    return answer, obs, interp


def _code_histogram(col: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"df['{col}'].dropna().plot(kind='hist', bins=20)\n"
        f"plt.xlabel('{col}')\n"
        "plt.ylabel('Count')\n"
        f"plt.title('Distribution of {col}')\n"
        "plt.show()"
    )


def _code_density(col: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"df['{col}'].dropna().plot(kind='density')\n"
        f"plt.xlabel('{col}')\n"
        f"plt.title('Density of {col}')\n"
        "plt.show()"
    )


def _code_box(col: str, group_col: Optional[str] = None) -> str:
    if group_col:
        return (
            "import matplotlib.pyplot as plt\n"
            f"df.boxplot(column='{col}', by='{group_col}', rot=45)\n"
            f"plt.title('{col} by {group_col}')\n"
            "plt.suptitle('')\n"
            f"plt.ylabel('{col}')\n"
            "plt.show()"
        )
    return (
        "import matplotlib.pyplot as plt\n"
        f"df[['{col}']].plot(kind='box')\n"
        f"plt.title('Box plot of {col}')\n"
        "plt.show()"
    )


def _code_violin(col: str, group_col: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        "plot_df = df[[\"%s\", \"%s\"]].dropna()\n" % (group_col, col) +
        "groups = [plot_df.loc[plot_df['%s'] == g, '%s'].values for g in plot_df['%s'].unique()]\n" % (group_col, col, group_col) +
        "labels = plot_df['%s'].unique()\n" % group_col +
        "plt.violinplot(groups, showmeans=True, showextrema=True)\n"
        "plt.xticks(range(1, len(labels) + 1), labels, rotation=45)\n"
        f"plt.ylabel('{col}')\n"
        f"plt.title('{col} by {group_col}')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_scatter(x: str, y: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"df.plot(kind='scatter', x='{x}', y='{y}', alpha=0.7)\n"
        f"plt.title('{y} vs {x}')\n"
        "plt.show()"
    )


def _code_hexbin(x: str, y: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"df.plot.hexbin(x='{x}', y='{y}', gridsize=25, mincnt=1)\n"
        f"plt.title('Hexbin of {y} vs {x}')\n"
        "plt.show()"
    )


def _code_line(x: str, y: str, group_col: Optional[str] = None) -> str:
    if group_col:
        return (
            "import matplotlib.pyplot as plt\n"
            f"plot_df = df[['{x}', '{y}', '{group_col}']].dropna().copy()\n"
            f"plot_df['{x}'] = pd.to_datetime(plot_df['{x}'])\n"
            f"plot_df = plot_df.sort_values('{x}')\n"
            f"for name, sub in plot_df.groupby('{group_col}'):\n"
            f"    plt.plot(sub['{x}'], sub['{y}'], label=name)\n"
            f"plt.xlabel('{x}')\n"
            f"plt.ylabel('{y}')\n"
            f"plt.title('{y} over time by {group_col}')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    return (
        "import matplotlib.pyplot as plt\n"
        f"plot_df = df[['{x}', '{y}']].dropna().copy()\n"
        f"plot_df['{x}'] = pd.to_datetime(plot_df['{x}'])\n"
        f"plot_df = plot_df.sort_values('{x}')\n"
        f"plt.plot(plot_df['{x}'], plot_df['{y}'])\n"
        f"plt.xlabel('{x}')\n"
        f"plt.ylabel('{y}')\n"
        f"plt.title('{y} over time')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_rolling_line(x: str, y: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"plot_df = df[['{x}', '{y}']].dropna().copy()\n"
        f"plot_df['{x}'] = pd.to_datetime(plot_df['{x}'])\n"
        f"plot_df = plot_df.sort_values('{x}')\n"
        f"plot_df['rolling_mean'] = plot_df['{y}'].rolling(window=5, min_periods=1).mean()\n"
        f"plt.plot(plot_df['{x}'], plot_df['{y}'], alpha=0.35, label='Raw')\n"
        f"plt.plot(plot_df['{x}'], plot_df['rolling_mean'], linewidth=2, label='Rolling mean')\n"
        "plt.legend()\n"
        f"plt.title('{y} over time with smoothing')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_area(x: str, y: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"plot_df = df[['{x}', '{y}']].dropna().copy()\n"
        f"plot_df['{x}'] = pd.to_datetime(plot_df['{x}'])\n"
        f"plot_df = plot_df.sort_values('{x}')\n"
        f"plt.fill_between(plot_df['{x}'], plot_df['{y}'], alpha=0.4)\n"
        f"plt.plot(plot_df['{x}'], plot_df['{y}'])\n"
        f"plt.xlabel('{x}')\n"
        f"plt.ylabel('{y}')\n"
        f"plt.title('{y} over time (area view)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_bar(group_col: str, value_col: Optional[str] = None) -> str:
    if value_col:
        return (
            "import matplotlib.pyplot as plt\n"
            f"plot_df = df.groupby('{group_col}')['{value_col}'].mean().sort_values(ascending=False)\n"
            "plot_df.plot(kind='bar')\n"
            f"plt.ylabel('Mean {value_col}')\n"
            f"plt.title('Mean {value_col} by {group_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    return (
        "import matplotlib.pyplot as plt\n"
        f"df['{group_col}'].value_counts().sort_values(ascending=False).plot(kind='bar')\n"
        "plt.ylabel('Count')\n"
        f"plt.title('Count by {group_col}')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_stacked_bar(col: str, second_col: str) -> str:
    return (
        "import matplotlib.pyplot as plt\n"
        f"plot_df = pd.crosstab(df['{second_col}'], df['{col}'], normalize='index')\n"
        "plot_df.plot(kind='bar', stacked=True)\n"
        f"plt.ylabel('Proportion within {second_col}')\n"
        f"plt.title('{col} composition within {second_col}')\n"
        "plt.legend(title='%s', bbox_to_anchor=(1.05, 1), loc='upper left')\n" % col +
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_heatmap(numeric_columns: Sequence[str]) -> str:
    cols = list(numeric_columns[:8])
    cols_repr = ", ".join([f"'{c}'" for c in cols])
    return (
        "import matplotlib.pyplot as plt\n"
        f"corr = df[[{cols_repr}]].corr(numeric_only=True)\n"
        "plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)\n"
        "plt.colorbar(label='Correlation')\n"
        "plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')\n"
        "plt.yticks(range(len(corr.columns)), corr.columns)\n"
        "plt.title('Correlation heatmap')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _code_missingness(columns: Sequence[str]) -> str:
    cols_repr = ", ".join([f"'{c}'" for c in columns[:12]])
    return (
        "import matplotlib.pyplot as plt\n"
        f"missing = df[[{cols_repr}]].isna().sum().sort_values(ascending=False)\n"
        "missing.plot(kind='bar')\n"
        "plt.ylabel('Missing values')\n"
        "plt.title('Missing values by column')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _recommend_distribution(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> List[GraphRecommendation]:
    _, _, interp = _distribution_observations(df, col)
    charts = [
        GraphRecommendation(
            chart_type="Histogram",
            title=f"Distribution of {col}",
            why_this_chart=f"Start with a histogram when you want the clearest overall view of how {col} is distributed.",
            x_column=col,
            steps=[
                f"Put {col} on the x-axis.",
                "Use count on the y-axis.",
                "Start around 20 bins and adjust if the shape looks too jagged or too flat.",
                "Check for skew, long tails, multiple peaks, and gaps.",
            ],
            python_code=_code_histogram(col),
            interpretation_notes=interp,
            good_when="You want the main distribution story fast.",
        ),
        GraphRecommendation(
            chart_type="Density plot",
            title=f"Density of {col}",
            why_this_chart="A density plot gives a smoother view of shape when the histogram bins feel arbitrary.",
            x_column=col,
            steps=[
                f"Use the same numeric column {col}.",
                "Plot a density curve instead of bars.",
                "Compare it with the histogram if the curve looks too smooth.",
            ],
            python_code=_code_density(col),
            interpretation_notes=["Peaks in the curve suggest clusters of common values.", "A long tail still signals skew even without bars."],
            good_when="You care more about shape than raw counts.",
        ),
        GraphRecommendation(
            chart_type="Box plot" if not group_col else "Grouped box plot",
            title=f"{col} spread" if not group_col else f"{col} by {group_col}",
            why_this_chart="A box plot is the quickest way to spot spread, median, and outliers.",
            x_column=group_col if group_col else col,
            y_column=col if group_col else None,
            group_column=group_col,
            steps=[
                "Plot the median line, box, whiskers, and outliers.",
                "Compare medians and interquartile ranges across groups if you grouped it.",
            ],
            python_code=_code_box(col, group_col),
            interpretation_notes=["Points beyond the whiskers are potential outliers.", "Wide boxes mean greater variability in the middle 50% of the data."],
            good_when="You need spread and outliers to be obvious.",
        ),
    ]
    if group_col:
        charts.append(
            GraphRecommendation(
                chart_type="Violin plot",
                title=f"{col} by {group_col}",
                why_this_chart="A violin plot is useful when you care about shape within each group, not just summary lines.",
                x_column=group_col,
                y_column=col,
                group_column=group_col,
                steps=[
                    f"Split {col} by {group_col}.",
                    "Show the density shape for each group side by side.",
                    "Compare both median level and shape differences.",
                ],
                python_code=_code_violin(col, group_col),
                interpretation_notes=["Bulges in the violin show where values concentrate.", "Different shapes across groups can matter even when means look similar."],
                good_when="Each group has enough rows and shape matters.",
            )
        )
    return charts[:4]


def _recommend_relationship(df: pd.DataFrame, x: str, y: str) -> List[GraphRecommendation]:
    return [
        GraphRecommendation(
            chart_type="Scatterplot",
            title=f"{y} vs {x}",
            why_this_chart=f"A scatterplot is the main graph for checking whether {x} and {y} move together.",
            x_column=x,
            y_column=y,
            steps=[
                f"Place {x} on the x-axis and {y} on the y-axis.",
                "Plot one point per row after dropping missing values.",
                "Look for a rising, falling, curved, or clustered point cloud.",
                "Add a trend line if you want a quick directional summary.",
            ],
            python_code=_code_scatter(x, y),
            interpretation_notes=[
                "An upward cloud suggests a positive relationship; downward suggests negative.",
                "A few far-away points can create a misleading correlation, so inspect outliers directly.",
            ],
            good_when="You want the raw relationship, not just the summary statistic.",
        ),
        GraphRecommendation(
            chart_type="Hexbin plot",
            title=f"Dense regions of {y} vs {x}",
            why_this_chart="A hexbin plot works better than a scatterplot when many points overlap.",
            x_column=x,
            y_column=y,
            steps=[
                "Use the same two numeric columns as a scatterplot.",
                "Bin the plane into hexagons so dense regions become visible.",
                "Check where the highest-density bins concentrate.",
            ],
            python_code=_code_hexbin(x, y),
            interpretation_notes=["Darker or hotter bins show where the dataset is most concentrated.", "Use this when the scatterplot turns into a solid blob."],
            good_when="You have hundreds or thousands of points.",
        ),
    ]


def _recommend_comparison(df: pd.DataFrame, group_col: str, value_col: str) -> List[GraphRecommendation]:
    return [
        GraphRecommendation(
            chart_type="Box plot",
            title=f"{value_col} by {group_col}",
            why_this_chart="This is usually the best first comparison graph because it shows center, spread, and outliers together.",
            x_column=group_col,
            y_column=value_col,
            group_column=group_col,
            steps=[
                f"Put {group_col} on the x-axis and {value_col} on the y-axis.",
                "Compare median lines and box heights between groups.",
                "Check whether groups overlap strongly or separate clearly.",
            ],
            python_code=_code_box(value_col, group_col),
            interpretation_notes=[
                "Large median gaps suggest meaningful differences in typical values.",
                "Overlap still matters; one high mean does not guarantee clean separation.",
            ],
            good_when="You want a defensible group comparison, not just averages.",
        ),
        GraphRecommendation(
            chart_type="Bar chart of means",
            title=f"Mean {value_col} by {group_col}",
            why_this_chart="Use this when you need a quick average ranking across groups.",
            x_column=group_col,
            y_column=value_col,
            group_column=group_col,
            steps=[
                f"Group rows by {group_col}.",
                f"Compute the mean of {value_col} in each group.",
                "Sort bars from highest to lowest to make the ranking clear.",
            ],
            python_code=_code_bar(group_col, value_col),
            interpretation_notes=["This hides spread, so pair it with a box plot if variability matters.", "Bars alone can overstate certainty when some groups are tiny."],
            good_when="You only need a fast ranking of group means.",
        ),
        GraphRecommendation(
            chart_type="Violin plot",
            title=f"Shape of {value_col} by {group_col}",
            why_this_chart="A violin plot adds distribution shape to the group comparison.",
            x_column=group_col,
            y_column=value_col,
            group_column=group_col,
            steps=[
                "Plot each group's distribution shape side by side.",
                "Compare where values concentrate, not just the median.",
            ],
            python_code=_code_violin(value_col, group_col),
            interpretation_notes=["Wide sections of the violin show common value ranges.", "Different shapes can matter even when means are similar."],
            good_when="Each group has enough rows and shape matters.",
        ),
    ]


def _recommend_trend(df: pd.DataFrame, time_col: str, value_col: str, group_col: Optional[str] = None) -> List[GraphRecommendation]:
    charts = [
        GraphRecommendation(
            chart_type="Line chart" if not group_col else "Grouped line chart",
            title=f"{value_col} over time" if not group_col else f"{value_col} over time by {group_col}",
            why_this_chart="A line chart is the default trend graph because it preserves ordering and makes rate changes easy to read.",
            x_column=time_col,
            y_column=value_col,
            group_column=group_col,
            steps=[
                f"Convert {time_col} to datetime.",
                f"Sort rows by {time_col}.",
                f"Plot {time_col} on the x-axis and {value_col} on the y-axis.",
                "If you have groups, draw one line per group and add a legend.",
            ],
            python_code=_code_line(time_col, value_col, group_col),
            interpretation_notes=[
                "Watch for sustained rises or drops instead of overreacting to one noisy point.",
                "If lines cross between groups, the group ordering changes over time and a single average may be misleading.",
            ],
            good_when="Time order matters and you care about shape of change.",
        ),
        GraphRecommendation(
            chart_type="Smoothed line chart",
            title=f"Smoothed {value_col} over time",
            why_this_chart="Use a rolling mean when the raw line is noisy and you want the broad direction to stand out.",
            x_column=time_col,
            y_column=value_col,
            steps=[
                "Keep the raw line faint in the background.",
                "Overlay a rolling mean or moving average.",
                "Use the smoothed line to discuss direction, not exact point values.",
            ],
            python_code=_code_rolling_line(time_col, value_col),
            interpretation_notes=["Smoothing clarifies direction but can hide short spikes.", "Use the raw line too if sudden events matter."],
            good_when="Point-to-point noise hides the larger trend.",
        ),
        GraphRecommendation(
            chart_type="Area chart",
            title=f"Cumulative-looking trend of {value_col}",
            why_this_chart="An area chart is a secondary option when the value is cumulative or you want to emphasize total magnitude over time.",
            x_column=time_col,
            y_column=value_col,
            steps=[
                "Start with the line chart first.",
                "Switch to area only if the filled shape helps the story instead of cluttering it.",
            ],
            python_code=_code_line(time_col, value_col, group_col).replace("plt.plot", "plt.fill_between").replace(", label=name", ""),
            interpretation_notes=["Area charts can exaggerate size changes, so use them carefully.", "They work best for totals or cumulative measures."],
            good_when="The variable behaves like a running total or cumulative burden.",
        ),
    ]
    return charts


def _recommend_composition(df: pd.DataFrame, col: str, second_col: Optional[str] = None) -> List[GraphRecommendation]:
    charts = [
        GraphRecommendation(
            chart_type="Sorted bar chart",
            title=f"Count by {col}",
            why_this_chart="A sorted bar chart is the cleanest way to show which categories dominate.",
            x_column=col,
            steps=[
                f"Count rows in each {col} category.",
                "Sort the bars from highest to lowest.",
                "Use percentages too if totals differ between datasets or groups.",
            ],
            python_code=_code_bar(col),
            interpretation_notes=["The largest bars show the dominant categories immediately.", "Very small bars may still matter if they represent rare but important cases."],
            good_when="You want an overall category count story.",
        )
    ]
    if second_col:
        charts.append(
            GraphRecommendation(
                chart_type="Normalized stacked bar chart",
                title=f"{col} composition within {second_col}",
                why_this_chart="Use this when you want to compare proportions across a second grouping variable.",
                x_column=second_col,
                group_column=col,
                steps=[
                    f"Build a cross-tab of {second_col} by {col}.",
                    "Normalize each bar to 100%.",
                    "Compare how the category mix changes from group to group.",
                ],
                python_code=_code_stacked_bar(col, second_col),
                interpretation_notes=["This graph answers composition questions better than a plain count bar chart.", "Equal-height bars make proportion changes easier to compare."],
                good_when="You care about shares within groups, not just raw counts.",
            )
        )
    return charts


def _recommend_ranking(df: pd.DataFrame, group_col: str, value_col: str) -> List[GraphRecommendation]:
    return [
        GraphRecommendation(
            chart_type="Horizontal bar chart",
            title=f"Ranking of {group_col} by mean {value_col}",
            why_this_chart="Sorted bars are the clearest ranking graph.",
            x_column=value_col,
            y_column=group_col,
            group_column=group_col,
            steps=[
                f"Aggregate {value_col} by {group_col}.",
                "Sort descending.",
                "Plot as horizontal bars so long labels stay readable.",
            ],
            python_code=_code_bar(group_col, value_col),
            interpretation_notes=["Ranking helps identify leaders and laggards fast.", "Include sample sizes when a high rank may come from a tiny group."],
            good_when="You want ordered performance, not distribution details.",
        )
    ]


def _recommend_matrix(record: Dict) -> List[GraphRecommendation]:
    numeric_cols = record.get("numeric_columns", [])
    return [
        GraphRecommendation(
            chart_type="Correlation heatmap",
            title="Correlation overview",
            why_this_chart="A heatmap is the quickest scan of all numeric relationships at once.",
            steps=[
                "Select the main numeric columns.",
                "Compute a correlation matrix.",
                "Inspect the strongest positive and negative cells, then follow up with scatterplots.",
            ],
            python_code=_code_heatmap(numeric_cols),
            interpretation_notes=["Warm cells indicate positive relationships; cool cells indicate negative ones.", "Use it for screening, then validate with pairwise plots."],
            good_when="You want a whole-dataset relationship overview.",
        )
    ]


def _recommend_missingness(record: Dict) -> List[GraphRecommendation]:
    cols = [col["name"] for col in record.get("columns", []) if col.get("missing_count", 0) > 0]
    return [
        GraphRecommendation(
            chart_type="Missingness bar chart",
            title="Missing values by column",
            why_this_chart="This graph helps you see data-quality issues before you interpret results.",
            steps=[
                "Count missing values in each column.",
                "Sort descending so the worst columns stand out.",
                "Decide whether to clean, impute, or exclude those columns before analysis.",
            ],
            python_code=_code_missingness(cols or [c["name"] for c in record.get("columns", [])[:10]]),
            interpretation_notes=["Heavy missingness can distort every downstream graph.", "If one key column is sparse, treat any pattern from it cautiously."],
            good_when="You need a fast data-quality check.",
        )
    ]


def build_analysis(record: Dict, question: str) -> Dict:
    df: pd.DataFrame = record["df"]
    numeric_cols = record["numeric_columns"]
    categorical_cols = record["categorical_columns"]
    datetime_cols = record["datetime_columns"]
    all_cols = list(df.columns)

    matched_columns = _match_columns(question, all_cols)
    intent = _question_intent(question)

    recommended_graphs: List[GraphRecommendation] = []
    observations: List[str] = []
    interpretation_help: List[str] = []
    follow_ups: List[str] = []
    confidence_note = ""

    direct_answer = ""

    if intent == "trend" and datetime_cols and numeric_cols:
        time_col = _choose_first(matched_columns, datetime_cols) or datetime_cols[0]
        value_col = _choose_first(matched_columns, numeric_cols, exclude=[time_col]) or numeric_cols[0]
        group_col = _choose_first(matched_columns, categorical_cols)
        direct_answer, observations, interpretation_help = _trend_observations(df, time_col, value_col, group_col if group_col and group_col not in {time_col, value_col} else None)
        recommended_graphs = _recommend_trend(df, time_col, value_col, group_col if group_col and group_col not in {time_col, value_col} else None)
        follow_ups = [
            f"Do you want to compare separate {group_col} groups over time?" if group_col else "Do you want a smoothed line to reduce noise?",
            f"Should we test whether the rate of change in {value_col} is consistent over time?",
        ]
        confidence_note = "High confidence because the question maps cleanly to a time column and a numeric measure."

    elif intent == "relationship" and len(numeric_cols) >= 2:
        chosen = [c for c in matched_columns if c in numeric_cols][:2]
        if len(chosen) < 2:
            chosen = numeric_cols[:2]
        x_col, y_col = chosen[0], chosen[1]
        direct_answer, observations, interpretation_help = _relationship_observations(df, x_col, y_col)
        recommended_graphs = _recommend_relationship(df, x_col, y_col)
        follow_ups = [
            f"Do you want to control for another variable that might affect both {x_col} and {y_col}?",
            f"Should we check whether the relationship stays the same inside each major group?",
        ]
        confidence_note = "High confidence because two numeric columns were available for a direct relationship check."

    elif intent == "comparison" and categorical_cols and numeric_cols:
        group_col = _choose_first(matched_columns, categorical_cols) or categorical_cols[0]
        value_col = _choose_first(matched_columns, numeric_cols) or numeric_cols[0]
        direct_answer, observations, interpretation_help = _comparison_observations(df, group_col, value_col)
        recommended_graphs = _recommend_comparison(df, group_col, value_col)
        follow_ups = [
            f"Do you want to compare medians as well as means for {value_col}?",
            f"Should we look for outliers inside each {group_col} group?",
        ]
        confidence_note = "High confidence because the dataset has both a grouping column and a numeric measure to compare."

    elif intent == "distribution" and numeric_cols:
        col = _choose_first(matched_columns, numeric_cols) or numeric_cols[0]
        group_col = _choose_first(matched_columns, categorical_cols)
        direct_answer, observations, interpretation_help = _distribution_observations(df, col)
        recommended_graphs = _recommend_distribution(df, col, group_col if group_col else None)
        follow_ups = [
            f"Do you want to compare the distribution of {col} across groups?" if categorical_cols else f"Do you want to check whether {col} has outliers?",
            f"Should we transform {col} if the skew is too strong?",
        ]
        confidence_note = "High confidence because the question maps to a numeric variable's spread."

    elif intent == "composition" and categorical_cols:
        col = _choose_first(matched_columns, categorical_cols) or categorical_cols[0]
        second_col = None
        for match in matched_columns:
            if match in categorical_cols and match != col:
                second_col = match
                break
        direct_answer, observations, interpretation_help = _composition_observations(df, col, second_col)
        recommended_graphs = _recommend_composition(df, col, second_col)
        follow_ups = [
            f"Do you want raw counts or within-group percentages for {col}?",
            f"Should we compare {col} composition across another grouping variable?",
        ]
        confidence_note = "Good confidence because the question maps to category counts or proportions."

    elif intent == "ranking" and categorical_cols and numeric_cols:
        group_col = _choose_first(matched_columns, categorical_cols) or categorical_cols[0]
        value_col = _choose_first(matched_columns, numeric_cols) or numeric_cols[0]
        direct_answer, observations, interpretation_help = _ranking_observations(df, group_col, value_col)
        recommended_graphs = _recommend_ranking(df, group_col, value_col)
        follow_ups = [
            f"Do you want to rank groups by median {value_col} instead of mean?",
            f"Should we add the group sample sizes before trusting the ranking?",
        ]
        confidence_note = "Good confidence because the ranking can be computed directly from the observed rows."

    elif intent == "matrix" and len(numeric_cols) >= 2:
        direct_answer, observations, interpretation_help = _matrix_observations(record)
        recommended_graphs = _recommend_matrix(record)
        follow_ups = [
            "Which pair from the heatmap do you want to inspect next with a scatterplot?",
            "Do you want to remove highly collinear variables before modeling?",
        ]
        confidence_note = "Good confidence because the answer comes from whole-dataset numeric relationships."

    elif intent == "missingness":
        direct_answer, observations, interpretation_help = _missingness_observations(record)
        recommended_graphs = _recommend_missingness(record)
        follow_ups = [
            "Do you want to inspect whether missingness clusters by time or group?",
            "Should we decide whether to impute, filter, or keep the missing rows?",
        ]
        confidence_note = "Good confidence because missingness is directly measurable from the dataset."

    else:
        direct_answer, observations, interpretation_help = _fallback_answer(record)
        if len(numeric_cols) >= 2:
            recommended_graphs = _recommend_matrix(record) + _recommend_relationship(df, numeric_cols[0], numeric_cols[1])[:1]
        elif numeric_cols:
            recommended_graphs = _recommend_distribution(df, numeric_cols[0])[:2]
        elif categorical_cols:
            recommended_graphs = _recommend_composition(df, categorical_cols[0])[:1]
        follow_ups = [
            "Are you trying to compare groups, show a trend, inspect a distribution, or test a relationship?",
            "Which one or two columns matter most to your question?",
        ]
        confidence_note = "Broader question, so the answer is an overview built from dataset structure and strongest patterns."

    # cap output size
    recommended_graphs = recommended_graphs[:4]
    observations = observations[:6]
    interpretation_help = interpretation_help[:6]
    follow_ups = [item for item in follow_ups if item][:4]

    return {
        "direct_answer": direct_answer,
        "question_intent": intent,
        "recommended_graphs": [graph.model_dump() for graph in recommended_graphs],
        "observations": observations,
        "interpretation_help": interpretation_help,
        "follow_up_questions": follow_ups,
        "matched_columns": matched_columns[:6],
        "confidence_note": confidence_note,
    }
