"""Des-suavizamiento de retornos de activos privados (Geltner, 1993).

Una tasación incorpora el valor de mercado con rezago, por lo que el retorno
reportado se comporta como r_rep(t) = a * r_rep(t-1) + (1 - a) * r_true(t).
Invirtiendo esa relación:

    r_true(t) = (r_rep(t) - a * r_rep(t-1)) / (1 - a)

donde a se estima como la autocorrelación de primer orden del retorno
reportado. El resultado conserva la media y revela la volatilidad y las
correlaciones económicas que la tasación esconde.
"""

import numpy as np
import pandas as pd


def ar1(series: pd.Series) -> float:
    """Autocorrelación de primer orden (estimador OLS sin constante sobre la serie centrada)."""
    x = series.dropna().to_numpy()
    x = x - x.mean()
    return float(np.dot(x[1:], x[:-1]) / np.dot(x[:-1], x[:-1]))


def geltner(series: pd.Series, rho: float | None = None, max_rho: float = 0.95) -> pd.Series:
    """Des-suaviza una serie de retornos. Si rho no se entrega, se estima por AR(1)."""
    rho = ar1(series) if rho is None else rho
    rho = float(np.clip(rho, 0.0, max_rho))
    out = (series - rho * series.shift(1)) / (1 - rho)
    out.iloc[0] = series.iloc[0]
    return out


def desmooth_frame(returns: pd.DataFrame, private_codes: list[str]) -> tuple[pd.DataFrame, dict]:
    """Des-suaviza las columnas privadas y devuelve también el rho estimado por activo."""
    out = returns.copy()
    rhos = {}
    for code in private_codes:
        rhos[code] = ar1(returns[code])
        out[code] = geltner(returns[code], rhos[code])
    return out, rhos
