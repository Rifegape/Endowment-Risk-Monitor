"""Proyección de liquidez del endowment.

Modelo de compromisos tipo Takahashi-Alexander (Yale), en versión agregada
con tasas constantes por programa:

    llamados_t      = tasa_llamado * no_llamado_{t-1}
    distribuciones_t = tasa_distribucion * NAV_{t-1} * (1 + g)
    NAV_t           = NAV_{t-1} * (1 + g) + llamados_t - distribuciones_t
    no_llamado_t    = no_llamado_{t-1} - llamados_t + compromisos_nuevos_t

La cartera líquida financia llamados y gasto, y recibe distribuciones y
donaciones. El gasto sigue una regla de % sobre el patrimonio promedio de
los últimos 3 años. El escenario de estrés aplica una caída a la cartera
líquida el año 1, recorta distribuciones y acelera llamados por 2 años.
"""

from dataclasses import dataclass

import pandas as pd

from .config import ASSET_BY_CODE, AUM_MM, POLICY, PRIVATE_PROGRAMS


@dataclass
class StressInputs:
    liquid_shock: float = 0.0       # caída de la cartera líquida en el año 1 (ej. -0.25)
    private_shock: float = 0.0      # caída del NAV privado en el año 1 (rezagada, menor)
    dist_cut: float = 0.0           # recorte de distribuciones en años 1-2 (ej. 0.5)
    call_boost: float = 0.0         # aceleración de llamados en años 1-2 (ej. 0.25)


def project(
    spending_rate: float = POLICY.spending_rate,
    commit_scale: float = 1.0,
    stress: StressInputs | None = None,
    years: int = POLICY.horizon_years,
    aum: float = AUM_MM,
) -> pd.DataFrame:
    stress = stress or StressInputs()
    progs = {p.code: p for p in PRIVATE_PROGRAMS}
    nav = {c: aum * ASSET_BY_CODE[c].actual / 100 for c in progs}
    unfunded = {c: p.unfunded_mm for c, p in progs.items()}
    liquid = aum - sum(nav.values())
    history = [aum] * 3

    rows = []
    for t in range(1, years + 1):
        early = t <= 2
        calls = dists = 0.0
        for c, p in progs.items():
            g = p.growth + (stress.private_shock if t == 1 else 0.0)
            cr = min(1.0, p.call_rate * (1 + stress.call_boost if early else 1))
            dr = p.dist_rate * (1 - stress.dist_cut if early else 1)
            call = cr * unfunded[c]
            dist = dr * nav[c] * (1 + g)
            nav[c] = nav[c] * (1 + g) + call - dist
            unfunded[c] = unfunded[c] - call + p.new_commit_mm * commit_scale
            calls += call
            dists += dist

        spending = spending_rate * sum(history[-3:]) / 3
        r_liq = POLICY.liquid_return + (stress.liquid_shock if t == 1 else 0.0)
        liquid = liquid * (1 + r_liq) - calls + dists - spending + POLICY.donations_mm
        total = liquid + sum(nav.values())
        history.append(total)
        rows.append({
            "anio": t,
                        "patrimonio": total,
            "liquido": liquid,
            "nav_privado": sum(nav.values()),
            "no_llamado": sum(unfunded.values()),
            "llamados": calls,
            "distribuciones": dists,
            "gasto": spending,
            "pct_privado": sum(nav.values()) / total * 100,
        })

    df = pd.DataFrame(rows).set_index("anio")
    # Exposición ilíquida potencial: lo invertido en privados más lo ya comprometido y no llamado
    df["iliquido_potencial"] = (df["nav_privado"] + df["no_llamado"]) / df["patrimonio"] * 100
    # Compromisos no llamados como % de la cartera líquida
    df["no_llamado_sobre_liquido"] = df["no_llamado"] / df["liquido"] * 100
    return df


def private_max_range() -> float:
    from .config import BUCKET_LIMITS
    return BUCKET_LIMITS["Alternativos"][1]


def private_policy_weight() -> float:
    return sum(ASSET_BY_CODE[p.code].policy for p in PRIVATE_PROGRAMS)
