"""Datos de retornos mensuales.

Por defecto el monitor usa datos SINTÉTICOS generados con una semilla fija,
para que el proyecto funcione sin licencias de datos. Para usar datos reales,
reemplaza data/factor_returns.csv y data/asset_returns.csv con el mismo
formato (fecha en la primera columna, un retorno mensual decimal por columna).

Los activos privados se simulan en dos versiones: el retorno "económico"
(no observable) y el retorno "reportado", suavizado como una tasación que
incorpora el valor de mercado con rezago. El monitor solo usa el reportado,
como ocurre en la práctica, y lo des-suaviza con el método de Geltner.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ASSET_CODES, FACTORS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FACTOR_FILE = DATA_DIR / "factor_returns.csv"
ASSET_FILE = DATA_DIR / "asset_returns.csv"

# Parámetros mensuales de los factores: media y volatilidad
_FACTOR_MU = np.array([0.0070, 0.0060, 0.0015, 0.0030, 0.0010])
_FACTOR_VOL = np.array([0.045, 0.050, 0.030, 0.012, 0.008])
_FACTOR_CORR = np.array([
    #  RVG    RVC    FX     TAS    CRED
    [1.00, 0.60, -0.45, -0.10, 0.45],
    [0.60, 1.00, -0.40, 0.00, 0.35],
    [-0.45, -0.40, 1.00, 0.10, -0.35],
    [-0.10, 0.00, 0.10, 1.00, -0.10],
    [0.45, 0.35, -0.35, -0.10, 1.00],
])

# Modelo de cada activo: carry/alpha mensual, betas a factores, vol idiosincrática
_ASSET_MODEL = {
    #           carry    RVG   RVC   FX    TAS   CRED   idio
    "RV_CL":   (0.0000, 0.00, 1.00, 0.00, 0.00, 0.00, 0.010),
    "RV_GL":   (0.0000, 1.00, 0.00, 1.00, 0.00, 0.00, 0.006),
    "RV_EM":   (-0.0010, 1.15, 0.10, 1.00, 0.00, 0.00, 0.022),
    "RF_GOB":  (0.0000, 0.00, 0.00, 0.00, 1.00, 0.00, 0.003),
    "RF_CORP": (0.0005, 0.00, 0.00, 0.00, 0.85, 1.00, 0.004),
    "RF_GL":   (0.0005, 0.00, 0.00, 1.00, 0.40, 0.60, 0.005),
    "PE":      (0.0020, 1.30, 0.00, 1.00, 0.00, 0.00, 0.030),
    "PD":      (0.0045, 0.30, 0.00, 0.60, 0.00, 1.60, 0.010),
    "INMO":    (0.0035, 0.00, 0.40, 0.00, 0.60, 0.00, 0.015),
    "CAJA":    (0.0035, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0008),
}

# Coeficiente de suavizamiento de la tasación (AR(1) del retorno reportado)
SMOOTHING = {"PE": 0.70, "PD": 0.55, "INMO": 0.80}

# Shocks deterministas para que la historia sintética tenga episodios reconocibles
_EPISODES = {
    # fecha: (RVG, RVC, FX, TAS, CRED)
    "2008-09": (-0.09, -0.10, 0.08, 0.01, -0.03),
    "2008-10": (-0.17, -0.15, 0.10, 0.01, -0.05),
    "2008-11": (-0.07, -0.05, 0.03, 0.01, -0.03),
    "2009-02": (-0.10, -0.04, 0.03, 0.00, -0.01),
    "2011-08": (-0.07, -0.06, 0.03, 0.00, -0.01),
    "2019-10": (0.00, -0.07, 0.04, -0.01, -0.01),
    "2019-11": (0.00, -0.09, 0.06, -0.02, -0.01),
    "2020-03": (-0.13, -0.15, 0.06, -0.01, -0.03),
    "2022-06": (-0.08, 0.00, 0.02, -0.02, -0.01),
    "2022-09": (-0.09, -0.03, 0.03, -0.02, -0.01),
}


def generate_synthetic(start: str = "2005-01", end: str = "2026-10", seed: int = 7):
    """Genera factores, retornos económicos y retornos reportados (mensuales)."""
    rng = np.random.default_rng(seed)
    dates = pd.period_range(start, end, freq="M").to_timestamp("M")
    n = len(dates)

    # Innovaciones t-Student (colas gruesas) correlacionadas
    df = 6
    chol = np.linalg.cholesky(_FACTOR_CORR)
    z = rng.standard_t(df, size=(n, len(FACTORS))) / np.sqrt(df / (df - 2))
    f = _FACTOR_MU + (z @ chol.T) * _FACTOR_VOL
    factors = pd.DataFrame(f, index=dates, columns=FACTORS)
    for ym, shock in _EPISODES.items():
        idx = factors.index.strftime("%Y-%m") == ym
        factors.loc[idx] = np.array(shock)

    true = pd.DataFrame(index=dates, columns=ASSET_CODES, dtype=float)
    for code in ASSET_CODES:
        carry, *betas, idio = _ASSET_MODEL[code]
        true[code] = carry + factors.values @ np.array(betas) + rng.normal(0, idio, n)

    reported = true.copy()
    for code, a in SMOOTHING.items():
        r = np.zeros(n)
        r[0] = true[code].iloc[0]
        for t in range(1, n):
            r[t] = a * r[t - 1] + (1 - a) * true[code].iloc[t]
        reported[code] = r

    return factors, true, reported


def write_sample_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    factors, _, reported = generate_synthetic()
    factors.round(6).to_csv(FACTOR_FILE, index_label="fecha")
    reported.round(6).to_csv(ASSET_FILE, index_label="fecha")


def load_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga factores y retornos de activos (reportados). Crea la muestra si falta."""
    if not (FACTOR_FILE.exists() and ASSET_FILE.exists()):
        write_sample_data()
    factors = pd.read_csv(FACTOR_FILE, index_col=0, parse_dates=True)
    assets = pd.read_csv(ASSET_FILE, index_col=0, parse_dates=True)
    missing = set(ASSET_CODES) - set(assets.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {ASSET_FILE.name}: {sorted(missing)}")
    common = factors.index.intersection(assets.index)
    return factors.loc[common, FACTORS], assets.loc[common, ASSET_CODES]


if __name__ == "__main__":
    write_sample_data()
    print(f"Datos de ejemplo escritos en {DATA_DIR}")
