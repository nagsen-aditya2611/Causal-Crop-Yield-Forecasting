#features.py

"""
this file is for scaling, z-score creation, 
feature selection, pair-table constructions
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


ID_COLS = ["crop_name", "country_code", "adm_id", "harvest_year"]
DEFAULT_SCALE_COLS = ["avg_ssm", "avg_ndvi", "avg_fpar", "avg_cwb", "avg_tavg", "yield"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower().str.replace(" ", "_")
    return out


def add_scaled_features(
    df: pd.DataFrame,
    scale_cols: list[str] = DEFAULT_SCALE_COLS,
) -> pd.DataFrame:
    out = df.copy()
    for c in scale_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(out[scale_cols]),
        columns=[f"{c}_z" for c in scale_cols],
        index=out.index,
    )

    return pd.concat([out, scaled], axis=1)


def make_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("adm_id", as_index=False)
          .agg(
              n_years=("harvest_year", "nunique"),
              min_year=("harvest_year", "min"),
              max_year=("harvest_year", "max"),
              n_rows=("harvest_year", "size"),
          )
    )


def make_pair_frame(
    panel_z: pd.DataFrame,
    x_col: str,
    y_col: str,
    pair_name: str,
) -> pd.DataFrame:
    keep_cols = [
        "crop_name", "country_code", "adm_id", "harvest_year",
        "awc", "bulk_density", "drainage_class", "latitude", "longitude",
        x_col, y_col,
    ]
    out = panel_z[keep_cols].copy()
    out = out.rename(columns={x_col: "x", y_col: "y"})
    out["pair"] = pair_name
    return out
