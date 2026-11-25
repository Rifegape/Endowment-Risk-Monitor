"""Modelo de factores: regresión OLS de cada clase de activo sobre los factores."""

import numpy as np
import pandas as pd


def factor_regression(assets: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Devuelve betas, alpha anualizado y R² por activo."""
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    rows = {}
    for code in assets.columns:
        y = assets[code].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1 - resid.var() / y.var()
        rows[code] = {"alpha_anual": coef[0] * 12, **dict(zip(factors.columns, coef[1:])), "r2": r2}
    return pd.DataFrame(rows).T


def portfolio_factor_exposure(betas: pd.DataFrame, weights: pd.Series, factor_cols: list[str]) -> pd.Series:
    w = weights.reindex(betas.index).fillna(0) / weights.sum()
    return betas[factor_cols].T @ w
