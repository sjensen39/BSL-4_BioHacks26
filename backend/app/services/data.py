from __future__ import annotations

import io
import uuid
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app.config import settings
from app.models import ColumnProfile


_DATETIME_PARSE_THRESHOLD = 0.7


def _try_read_csv(data: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read the CSV file. {last_error}")


def _detect_datetime(series: pd.Series) -> bool:
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    non_null = series.dropna().astype(str)
    if len(non_null) < 3:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", utc=False, format="mixed")
    success_rate = parsed.notna().mean() if len(parsed) else 0.0
    return bool(success_rate >= _DATETIME_PARSE_THRESHOLD)


def _normalize_missing(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else as_float
    return str(value) if isinstance(value, pd.Timestamp) else value


def _safe_unique_ratio(series: pd.Series) -> float:
    denom = max(1, int(series.notna().sum()))
    return float(series.nunique(dropna=True)) / denom


def _column_profile(df: pd.DataFrame, col: str) -> Tuple[ColumnProfile, str]:
    series = df[col]
    if _detect_datetime(series):
        detected = "datetime"
    elif is_bool_dtype(series):
        detected = "boolean"
    elif is_numeric_dtype(series):
        detected = "numeric"
    else:
        unique_ratio = _safe_unique_ratio(series)
        detected = "categorical" if unique_ratio <= 0.3 or series.nunique(dropna=True) <= 20 else "text"

    samples: List[str] = []
    for value in series.dropna().astype(str).head(4).tolist():
        if value not in samples:
            samples.append(value)

    profile = ColumnProfile(
        name=col,
        detected_type=detected,
        missing_count=int(series.isna().sum()),
        unique_count=int(series.nunique(dropna=True)),
        sample_values=samples,
    )
    return profile, detected


def _numeric_stats(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    for col in numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "outliers": outliers,
            "n": int(len(series)),
        }
    return stats


def _categorical_stats(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    for col in categorical_columns:
        counts = df[col].astype(str).fillna("Missing").value_counts(dropna=False).head(settings.top_categories)
        total = max(1, int(len(df)))
        stats[col] = {
            "top_counts": [{"label": str(label), "count": int(count), "share": float(count) / total} for label, count in counts.items()]
        }
    return stats


def _datetime_stats(df: pd.DataFrame, datetime_columns: List[str]) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    for col in datetime_columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parsed = parsed.dropna()
        if parsed.empty:
            continue
        stats[col] = {
            "min": parsed.min().isoformat(),
            "max": parsed.max().isoformat(),
            "n": int(len(parsed)),
        }
    return stats


def _top_correlations(df: pd.DataFrame, numeric_columns: List[str]) -> List[Dict]:
    if len(numeric_columns) < 2:
        return []
    numeric_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(numeric_only=True)
    pairs: List[Dict] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            pairs.append({"x": cols[i], "y": cols[j], "corr": float(value)})
    pairs.sort(key=lambda item: abs(item["corr"]), reverse=True)
    return pairs[:6]


def build_dataset_record(filename: str, description: str, data: bytes) -> Dict:
    if len(data) > settings.max_upload_bytes:
        raise ValueError("CSV file is too large for this prototype.")

    df = _try_read_csv(data)
    if df.empty:
        raise ValueError("The CSV file is empty.")

    df.columns = [str(c).strip() for c in df.columns]

    columns: List[ColumnProfile] = []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    datetime_columns: List[str] = []

    for col in df.columns:
        profile, detected = _column_profile(df, col)
        columns.append(profile)
        if detected == "numeric":
            numeric_columns.append(col)
        elif detected == "categorical":
            categorical_columns.append(col)
        elif detected == "datetime":
            datetime_columns.append(col)

    preview = [
        {k: _normalize_missing(v) for k, v in row.items()}
        for row in df.head(settings.preview_rows).to_dict(orient="records")
    ]

    numeric_stats = _numeric_stats(df, numeric_columns)
    categorical_stats = _categorical_stats(df, categorical_columns)
    datetime_stats = _datetime_stats(df, datetime_columns)
    top_correlations = _top_correlations(df, numeric_columns)

    overview = [
        f"{len(df)} rows across {len(df.columns)} columns.",
        f"Numeric columns: {', '.join(numeric_columns[:6]) or 'none detected'}.",
        f"Categorical columns: {', '.join(categorical_columns[:6]) or 'none detected'}.",
    ]
    if datetime_columns:
        overview.append(f"Time-based columns: {', '.join(datetime_columns[:4])}.")
    if top_correlations:
        top = top_correlations[0]
        overview.append(f"Strongest numeric relationship detected: {top['x']} vs {top['y']} (r={top['corr']:.2f}).")

    return {
        "data_id": str(uuid.uuid4()),
        "filename": filename,
        "description": description.strip(),
        "df": df,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [c.model_dump() for c in columns],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "preview_rows": preview,
        "overview_bullets": overview,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "datetime_stats": datetime_stats,
        "top_correlations": top_correlations,
    }
