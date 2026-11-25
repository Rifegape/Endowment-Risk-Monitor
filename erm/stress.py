"""Pruebas de estrés por factores y efecto denominador sobre los rangos de la política."""

import pandas as pd

from .config import ASSETS


def apply_shock(
    betas: pd.DataFrame, shocks: dict, weights: pd.Series, aum: float, rhos: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Impacto instantáneo de un shock de factores.

    retorno (económico): Σ beta_k * shock_k, con betas estimadas sobre retornos
    des-suavizados, que reflejan el valor económico de los activos privados.

    retorno_reportado: lo que muestra la contabilidad al cierre del mes del
    shock. Los activos líquidos lo reflejan completo; los privados solo una
    fracción (1 - rho), porque la tasación incorpora el shock con rezago.
    Los pesos posteriores se calculan con valores reportados, que es como se
    controla el cumplimiento de la política: ahí aparece el efecto denominador.
    """
    rhos = rhos or {}
    shock = pd.Series(shocks).reindex(betas.columns.intersection(list(shocks))).fillna(0)
    asset_ret = betas[shock.index] @ shock
    lag = pd.Series({c: 1 - rhos.get(c, 0.0) for c in asset_ret.index})
    reported_ret = asset_ret * lag
    w = weights / weights.sum()
    value_before = w * aum
    value_after = value_before * (1 + asset_ret)
    total_after = value_after.sum()
    reported_after = value_before * (1 + reported_ret)
    w_after = reported_after / reported_after.sum() * 100

    table = pd.DataFrame({
        "retorno": asset_ret,
        "retorno_reportado": reported_ret,
        "impacto_mm": value_after - value_before,
        "peso_antes": w * 100,
        "peso_despues": w_after,
    })
    lim = {a.code: (a.lo, a.hi) for a in ASSETS}
    table["min"] = [lim[c][0] for c in table.index]
    table["max"] = [lim[c][1] for c in table.index]
    table["fuera_de_rango"] = (table["peso_despues"] < table["min"]) | (table["peso_despues"] > table["max"])

    total = {
        "retorno": float(total_after / aum - 1),
        "retorno_reportado": float(reported_after.sum() / aum - 1),
        "impacto_mm": float(total_after - aum),
        "patrimonio_despues": float(total_after),
        "n_fuera_de_rango": int(table["fuera_de_rango"].sum()),
    }
    return table, total
