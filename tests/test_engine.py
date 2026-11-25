import numpy as np
import pandas as pd
import pytest

from erm import factors as fm
from erm import liquidity as liq
from erm import risk, stress
from erm.config import ASSETS, AUM_MM, PRIVATE_CODES, SCENARIOS
from erm.data import generate_synthetic, load_returns
from erm.desmoothing import ar1, desmooth_frame, geltner

W = pd.Series({a.code: a.actual for a in ASSETS})


@pytest.fixture(scope="module")
def data():
    f, a = load_returns()
    d, rhos = desmooth_frame(a, PRIVATE_CODES)
    return f, a, d, rhos


def test_weights_sum_to_100():
    assert sum(a.actual for a in ASSETS) == pytest.approx(100)
    assert sum(a.policy for a in ASSETS) == pytest.approx(100)


def test_geltner_recovers_smoothing_coefficient():
    rng = np.random.default_rng(0)
    true = pd.Series(rng.normal(0.01, 0.05, 5000))
    a = 0.7
    rep = true.copy()
    for t in range(1, len(rep)):
        rep.iloc[t] = a * rep.iloc[t - 1] + (1 - a) * true.iloc[t]
    assert ar1(rep) == pytest.approx(a, abs=0.03)
    des = geltner(rep, a)
    assert np.corrcoef(des[1:], true[1:])[0, 1] > 0.99
    assert des.std() == pytest.approx(true.std(), rel=0.02)


def test_desmoothing_raises_private_volatility(data):
    _, a, d, _ = data
    for c in PRIVATE_CODES:
        assert d[c].std() > 1.5 * a[c].std()


def test_risk_contributions_add_up_to_volatility(data):
    _, _, d, _ = data
    rc = risk.risk_contributions(d, W)
    port_vol = risk.ann_vol(risk.portfolio_returns(d, W))
    assert rc["contribucion"].sum() == pytest.approx(port_vol, rel=1e-6)
    assert rc["pct_riesgo"].sum() == pytest.approx(1.0)


def test_cvar_exceeds_var(data):
    _, _, d, _ = data
    r = risk.portfolio_returns(d, W)
    var, cvar = risk.var_cvar_hist(r, 0.95)
    assert cvar >= var > 0


def test_factor_betas_recover_model(data):
    f, _, d, _ = data
    b = fm.factor_regression(d, f)
    assert b.loc["RV_GL", "RV_GLOBAL"] == pytest.approx(1.0, abs=0.1)
    assert b.loc["RV_GL", "USDCLP"] == pytest.approx(1.0, abs=0.1)
    assert b.loc["RF_GOB", "TASAS"] == pytest.approx(1.0, abs=0.1)


def test_stress_conserves_value(data):
    f, _, d, rhos = data
    b = fm.factor_regression(d, f)
    table, tot = stress.apply_shock(b, SCENARIOS[0].shocks, W, AUM_MM, rhos)
    assert table["impacto_mm"].sum() == pytest.approx(tot["impacto_mm"])
    assert table["peso_despues"].sum() == pytest.approx(100)
    assert tot["retorno"] < 0
    # los privados reconocen con rezago: pérdida reportada menor que la económica
    assert abs(tot["retorno_reportado"]) < abs(tot["retorno"])


def test_liquidity_projection_balances():
    df = liq.project()
    assert np.allclose(df["patrimonio"], df["liquido"] + df["nav_privado"])
    stressed = liq.project(stress=liq.StressInputs(-0.25, -0.10, 0.5, 0.25))
    assert stressed["pct_privado"].max() > df["pct_privado"].max()


def test_synthetic_data_is_reproducible():
    f1, _, r1 = generate_synthetic(seed=7)
    f2, _, r2 = generate_synthetic(seed=7)
    pd.testing.assert_frame_equal(r1, r2)
