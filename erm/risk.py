"""Métricas de riesgo de portafolio."""

import numpy as np
import pandas as pd

PERIODS = 12  # datos mensuales


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(returns.columns).fillna(0) / weights.sum()
    return returns @ w


def ann_return(r: pd.Series) -> float:
    return float((1 + r).prod() ** (PERIODS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(PERIODS))


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def var_cvar_hist(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """VaR y CVaR históricos a 1 mes, expresados como pérdida positiva."""
    q = np.quantile(r, 1 - level)
    tail = r[r <= q]
    return float(-q), float(-tail.mean())


def var_param(r: pd.Series, level: float = 0.95) -> float:
    """VaR paramétrico normal a 1 mes (pérdida positiva)."""
    from scipy.stats import norm
    return float(-(r.mean() + norm.ppf(1 - level) * r.std(ddof=1)))


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Contribución de cada activo a la volatilidad anual del portafolio.

    RC_i = w_i * (Σw)_i / σ_p. Las contribuciones suman σ_p.
    """
    w = (weights.reindex(returns.columns).fillna(0) / weights.sum()).to_numpy()
    cov = returns.cov().to_numpy() * PERIODS
    sigma = float(np.sqrt(w @ cov @ w))
    marginal = cov @ w / sigma
    rc = w * marginal
    return pd.DataFrame(
        {"peso": w, "vol_marginal": marginal, "contribucion": rc, "pct_riesgo": rc / sigma},
        index=returns.columns,
    )


def tracking_error(returns: pd.DataFrame, actual: pd.Series, policy: pd.Series) -> float:
    active = portfolio_returns(returns, actual) - portfolio_returns(returns, policy)
    return ann_vol(active)


def rolling_vol(r: pd.Series, window: int = 36) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(PERIODS)


def summary(r: pd.Series, rf: pd.Series | None = None) -> dict:
    var95, cvar95 = var_cvar_hist(r, 0.95)
    var99, cvar99 = var_cvar_hist(r, 0.99)
    out = {
        "ret_anual": ann_return(r),
        "vol_anual": ann_vol(r),
        "max_dd": max_drawdown(r),
        "var95": var95, "cvar95": cvar95,
        "var99": var99, "cvar99": cvar99,
        "var95_param": var_param(r, 0.95),
    }
    if rf is not None:
        excess = r - rf.reindex(r.index)
        out["sharpe"] = float(excess.mean() / excess.std(ddof=1) * np.sqrt(PERIODS))
    return out
