# ccm_utils.py

"""
for embedding search, CCM excecution, 
convergence summaries, and result formatting
"""

from __future__ import annotations

import pandas as pd
import pyEDM


def run_ccm(
    df: pd.DataFrame,
    columns: str,
    target: str,
    lib_sizes: str = "10 80 10",
    E: int = 3,
    tau: int = -1,
    sample: int = 100,
    Tp: int = 0,
    embedded: bool = False,
    seed: int = 0,
):
    return pyEDM.CCM(
        dataFrame=df,
        columns=columns,
        target=target,
        libSizes=lib_sizes,
        sample=sample,
        E=E,
        Tp=Tp,
        tau=tau,
        embedded=embedded,
        seed=seed,
    )


def summarize_ccm(ccm_result: pd.DataFrame) -> pd.DataFrame:
    out = ccm_result.copy()
    return out


def best_embedding_by_rho(
    df: pd.DataFrame,
    columns: str,
    target: str,
    E_values=range(1, 8),
    lib_sizes: str = "10 80 10",
    tau: int = -1,
    sample: int = 100,
):
    rows = []
    for E in E_values:
        res = pyEDM.CCM(
            dataFrame=df,
            columns=columns,
            target=target,
            libSizes=lib_sizes,
            sample=sample,
            E=E,
            tau=tau,
        )
        rows.append(
            {
                "E": E,
                "rho_mean": res["rho"].mean() if "rho" in res.columns else None,
                "rho_max": res["rho"].max() if "rho" in res.columns else None,
            }
        )
    return pd.DataFrame(rows)
