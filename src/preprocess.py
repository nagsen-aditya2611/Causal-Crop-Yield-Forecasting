#preprocess.py

# well the functions in this file were made from the code
# from the second notebook named 02_panel_cleaning.ipynb

"""
contains cleaning rules, handles missingvals, 
sorts panel, and balancing checks
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ID_COLS = ["cropname", "countrycode", "admid", "harvestyear"]

NUMERIC_COLS = [
    "yield",
    "production",
    "harvestarea",
    "sos",
    "eos",
    "awc",
    "bulkdensity",
    "drainageclass",
    "latitude",
    "longitude",
    "regionarea",
    "croparea",
    "cropareapercentage",
    "avgtmin",
    "avgtmax",
    "avgtavg",
    "avgrad",
    "avget0",
    "avgvpd",
    "avgcwb",
    "avgssm",
    "avgrsm",
    "avgndvi",
    "avgfpar",
]

CORE_REQUIRED_COLS = [
    "cropname",
    "countrycode",
    "admid",
    "harvestyear",
    "yield",
    "avgssm",
    "avgndvi",
    "avgfpar",
]

RANGE_RULES = {
    "yield": (0, None),
    "production": (0, None),
    "harvestarea": (0, None),
    "awc": (0, None),
    "bulkdensity": (0, None),
    "drainageclass": (0, None),
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "cropareapercentage": (0, 100),
    "avgndvi": (-1, 1),
    "avgfpar": (0, 100),
    "avgssm": (0, None),
    "avgrsm": (0, None),
}


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return out


def assert_required_columns(df: pd.DataFrame, required: Iterable[str] = CORE_REQUIRED_COLS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["cropname", "countrycode", "admid"]:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    if "harvestyear" in out.columns:
        out["harvestyear"] = pd.to_numeric(out["harvestyear"], errors="coerce").astype("Int64")

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").copy()


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["cropname", "countrycode", "admid", "harvestyear"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def validate_panel_key(df: pd.DataFrame, key_cols: Iterable[str] = ID_COLS) -> pd.DataFrame:
    key_cols = [c for c in key_cols if c in df.columns]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    return df.loc[dup_mask, key_cols].sort_values(key_cols)


def deduplicate_panel(df: pd.DataFrame, key_cols: Iterable[str] = ID_COLS, keep: str = "first") -> pd.DataFrame:
    key_cols = [c for c in key_cols if c in df.columns]
    return df.drop_duplicates(subset=key_cols, keep=keep).copy()


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame({
        "column": df.columns,
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean().values * 100).round(2),
        "dtype": df.dtypes.astype(str).values,
    })
    return rep.sort_values(["pct_missing", "column"], ascending=[False, True]).reset_index(drop=True)


def range_violations(df: pd.DataFrame, rules: dict = RANGE_RULES) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        s = df[col]
        bad = pd.Series(False, index=s.index)
        if lo is not None:
            bad |= s < lo
        if hi is not None:
            bad |= s > hi
        rows.append({
            "column": col,
            "n_violations": int(bad.sum()),
            "pct_violations": round(float(bad.mean() * 100), 4),
        })
    return pd.DataFrame(rows).sort_values(["n_violations", "column"], ascending=[False, True]).reset_index(drop=True)


def filter_invalid_core_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "harvestyear" in out.columns:
        out = out[out["harvestyear"].notna()]
    if "yield" in out.columns:
        out = out[out["yield"].notna()]
        out = out[out["yield"] >= 0]

    for col in ["cropname", "countrycode", "admid"]:
        if col in out.columns:
            out = out[out[col].notna()]
            out = out[out[col].astype(str).str.len() > 0]

    return out.reset_index(drop=True)


def add_basic_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"production", "harvestarea"} <= set(out.columns):
        out["yield_recomputed"] = out["production"] / out["harvestarea"]
        out.loc[out["harvestarea"] == 0, "yield_recomputed"] = pd.NA

    if {"avgtmin", "avgtavg", "avgtmax"} <= set(out.columns):
        out["temp_order_flag"] = (
            (out["avgtmin"] <= out["avgtavg"]) &
            (out["avgtavg"] <= out["avgtmax"])
        )

    return out


def unit_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = [c for c in ["cropname", "countrycode", "admid"] if c in df.columns]
    return (
        df.groupby(grp_cols, dropna=False)
        .agg(
            n_years=("harvestyear", "nunique"),
            min_year=("harvestyear", "min"),
            max_year=("harvestyear", "max"),
            mean_yield=("yield", "mean"),
        )
        .reset_index()
    )


def build_clean_panel(path_in: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data(path_in)
    df = normalize_columns(df)
    assert_required_columns(df)
    df = drop_empty_rows(df)
    df = coerce_types(df)

    dup_df = validate_panel_key(df)
    df = deduplicate_panel(df)

    miss_df = missing_report(df)
    range_df = range_violations(df)

    df = filter_invalid_core_rows(df)
    df = add_basic_flags(df)
    df = sort_panel(df)

    unit_df = unit_summary(df)
    return df, dup_df, miss_df, range_df, unit_df


def save_outputs(
    df: pd.DataFrame,
    dup_df: pd.DataFrame,
    miss_df: pd.DataFrame,
    range_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    out_dir: str | Path,
    stem: str = "crop_panel",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{stem}_clean.csv", index=False)
    dup_df.to_csv(out_dir / f"{stem}_duplicates.csv", index=False)
    miss_df.to_csv(out_dir / f"{stem}_missing_report.csv", index=False)
    range_df.to_csv(out_dir / f"{stem}_range_violations.csv", index=False)
    unit_df.to_csv(out_dir / f"{stem}_unit_summary.csv", index=False)


if __name__ == "__main__":
    RAW_PATH = "data/raw/final_crop_dataset.csv"
    OUT_DIR = "data/processed"

    clean_df, dup_df, miss_df, range_df, unit_df = build_clean_panel(RAW_PATH)
    save_outputs(clean_df, dup_df, miss_df, range_df, unit_df, OUT_DIR)
