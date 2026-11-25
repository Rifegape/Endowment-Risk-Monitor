"""Endowment Risk Monitor — dashboard en Dash.

Ejecutar localmente:  python app.py   (http://127.0.0.1:8050)
Producción (Render):  gunicorn app:server
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from erm import factors as fm
from erm import liquidity as liq
from erm import risk
from erm import stress as st
from erm.config import (
    ASSET_BY_CODE, ASSETS, AUM_MM, BUCKET_LIMITS, FACTOR_LABELS, FACTORS, POLICY,
    PRIVATE_CODES, PRIVATE_PROGRAMS, SCENARIOS,
)
from erm.data import load_returns
from erm.desmoothing import desmooth_frame

# ---------------------------------------------------------------- datos y cálculos base
FACTOR_RET, REPORTED = load_returns()
DESMOOTHED, RHOS = desmooth_frame(REPORTED, PRIVATE_CODES)
W_ACTUAL = pd.Series({a.code: a.actual for a in ASSETS})
W_POLICY = pd.Series({a.code: a.policy for a in ASSETS})
NAMES = {a.code: a.name for a in ASSETS}
BETAS_DES = fm.factor_regression(DESMOOTHED, FACTOR_RET)
BETAS_REP = fm.factor_regression(REPORTED, FACTOR_RET)
LAST_DATE = REPORTED.index[-1]
FACTOR_SHORT = {"RV_GLOBAL": "RV global", "RV_CHILE": "RV Chile", "USDCLP": "USD/CLP", "TASAS": "Tasas", "CREDITO": "Crédito"}

# ---------------------------------------------------------------- estilo de gráficos
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"          # categóricos 1 y 2
NEG, POS, MID = "#d03b3b", "#2a78d6", "#f0efec"
CRITICAL = "#d03b3b"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def base_layout(fig: go.Figure, height: int = 340, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK2),
        hoverlabel=dict(bgcolor="#fcfcfb", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
        **{"bargap": 0.35, **kw},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:,.{d}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kpi(label: str, value: str, note: str = "") -> html.Div:
    return html.Div([html.Div(label, className="label"), html.Div(value, className="value"),
                     html.Div(note, className="note")], className="kpi")


def card(title: str, sub: str, *children) -> html.Div:
    return html.Div([html.H3(title), html.P(sub, className="sub"), *children], className="card")


TABLE_STYLE = dict(
    style_as_list_view=True,
    style_table={"overflowX": "auto"},
    style_header={"backgroundColor": "transparent", "color": INK2, "fontWeight": 600,
                  "borderBottom": f"1px solid {AXIS}", "fontFamily": FONT, "fontSize": 12},
    style_cell={"backgroundColor": "transparent", "color": INK, "fontFamily": FONT, "fontSize": 13,
                "padding": "6px 10px", "border": "none", "borderBottom": f"1px solid {GRID}",
                "textAlign": "right", "fontVariantNumeric": "tabular-nums"},
    style_cell_conditional=[{"if": {"column_id": "Clase"}, "textAlign": "left"}],
)


def window_slice(df: pd.DataFrame, years: str) -> pd.DataFrame:
    return df if years == "all" else df.iloc[-int(years) * 12:]


# ---------------------------------------------------------------- pestaña Resumen
def tab_resumen() -> html.Div:
    r_des = risk.portfolio_returns(DESMOOTHED, W_ACTUAL)
    s = risk.summary(r_des.iloc[-120:], DESMOOTHED["CAJA"])
    te = risk.tracking_error(DESMOOTHED.iloc[-120:], W_ACTUAL, W_POLICY)
    alt = sum(ASSET_BY_CODE[c].actual for c in PRIVATE_CODES)
    unfunded = sum(p.unfunded_mm for p in PRIVATE_PROGRAMS)
    illiq = (alt / 100 * AUM_MM + unfunded) / AUM_MM

    kpis = html.Div([
        kpi("Patrimonio", f"$ {num(AUM_MM)} MM", f"CLP · cierre {LAST_DATE:%m-%Y}"),
        kpi("Retorno anual 10 años", pct(s["ret_anual"]), "nominal en CLP"),
        kpi("Volatilidad 10 años", pct(s["vol_anual"]), "con privados des-suavizados"),
        kpi("VaR 95% a 1 mes", f"$ {num(s['var95'] * AUM_MM)} MM", f"{pct(s['var95'])} del patrimonio"),
        kpi("Tracking error vs política", pct(te, 2), "anual, 10 años"),
        kpi("Exposición ilíquida potencial", pct(illiq), f"NAV privado + no llamado · tope {num(POLICY.max_illiquid_potential)}%"),
    ], className="grid g6")

    order = [a.code for a in ASSETS][::-1]
    labels = [NAMES[c] for c in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[ASSET_BY_CODE[c].hi - ASSET_BY_CODE[c].lo for c in order],
        base=[ASSET_BY_CODE[c].lo for c in order], orientation="h", name="Rango de la política",
        marker=dict(color=GRID, line=dict(width=0)), width=0.6,
        hovertemplate="%{y}<br>Rango: %{base:.0f}% a %{customdata:.0f}%<extra></extra>",
        customdata=[ASSET_BY_CODE[c].hi for c in order],
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[ASSET_BY_CODE[c].actual for c in order], orientation="h", name="Peso efectivo",
        marker=dict(color=S1, cornerradius=4), width=0.28,
        hovertemplate="%{y}<br>Efectivo: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=[ASSET_BY_CODE[c].policy for c in order], mode="markers", name="Objetivo",
        marker=dict(symbol="line-ns", size=18, line=dict(width=2.5, color=INK)),
        hovertemplate="%{y}<br>Objetivo: %{x:.1f}%<extra></extra>",
    ))
    base_layout(fig, 380, barmode="overlay", xaxis_title=None)
    fig.update_xaxes(ticksuffix="%", range=[0, 32])

    rows = []
    for a in ASSETS:
        ok = a.lo <= a.actual <= a.hi
        rows.append({"Clase": a.name, "Efectivo": f"{num(a.actual, 1)}%", "Objetivo": f"{num(a.policy, 1)}%",
                     "Desvío": f"{num(a.actual - a.policy, 1)} pp", "Rango": f"{num(a.lo)}–{num(a.hi)}%",
                     "Estado": "✓ Dentro" if ok else "✕ Fuera"})
    for b, (lo, hi) in BUCKET_LIMITS.items():
        v = sum(a.actual for a in ASSETS if a.bucket == b)
        pol = sum(a.policy for a in ASSETS if a.bucket == b)
        rows.append({"Clase": f"Total {b.lower()}", "Efectivo": f"{num(v, 1)}%", "Objetivo": f"{num(pol, 1)}%",
                     "Desvío": f"{num(v - pol, 1)} pp", "Rango": f"{num(lo)}–{num(hi)}%",
                     "Estado": "✓ Dentro" if lo <= v <= hi else "✕ Fuera"})
    table = dash_table.DataTable(
        data=rows, columns=[{"name": c, "id": c} for c in rows[0]], **TABLE_STYLE,
        style_data_conditional=[
            {"if": {"filter_query": '{Estado} contains "Dentro"', "column_id": "Estado"}, "color": "#006300"},
            {"if": {"filter_query": '{Estado} contains "Fuera"', "column_id": "Estado"}, "color": CRITICAL, "fontWeight": 600},
            {"if": {"filter_query": '{Clase} contains "Total"'}, "fontWeight": 600},
        ],
    )
    return html.Div([
        kpis, html.Div(style={"height": 16}),
        html.Div([
            card("Asignación efectiva vs política", "Barra: peso efectivo · trazo negro: objetivo · banda gris: rango permitido",
                 dcc.Graph(figure=fig, config={"displayModeBar": False})),
            card("Cumplimiento de la política de inversión", "Control de rangos por clase de activo y por grupo", table),
        ], className="grid g2"),
    ])


# ---------------------------------------------------------------- pestaña Riesgo
def tab_riesgo() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([html.Label("Retornos de activos privados"), dcc.RadioItems(
                id="r-basis", value="des", inline=True, inputStyle={"marginRight": 6, "marginLeft": 12},
                options=[{"label": "Des-suavizados (económicos)", "value": "des"},
                         {"label": "Reportados (tasación)", "value": "rep"}])], className="control wide"),
            html.Div([html.Label("Ventana"), dcc.Dropdown(
                id="r-window", value="10", clearable=False,
                options=[{"label": "3 años", "value": "3"}, {"label": "5 años", "value": "5"},
                         {"label": "10 años", "value": "10"}, {"label": "Toda la historia", "value": "all"}])],
                className="control"),
        ], className="controls"),
        html.Div(id="r-kpis", className="grid g6"),
        html.Div(style={"height": 16}),
        html.Div([
            card("Contribución al riesgo", "Peso en el patrimonio vs participación en la volatilidad total",
                 dcc.Graph(id="r-contrib", config={"displayModeBar": False})),
            card("Volatilidad móvil 36 meses", "El suavizamiento de la tasación esconde parte del riesgo",
                 dcc.Graph(id="r-rolling", config={"displayModeBar": False})),
        ], className="grid g2"),
        card("Drawdown del portafolio", "Caída desde el máximo previo, con la base de retornos seleccionada",
             dcc.Graph(id="r-dd", config={"displayModeBar": False})),
    ])


# ---------------------------------------------------------------- pestaña Alternativos
def tab_alternativos() -> html.Div:
    rows = []
    for c in PRIVATE_CODES:
        rep, des = REPORTED[c], DESMOOTHED[c]
        rows.append({
            "Clase": NAMES[c], "Rho (AR1)": num(RHOS[c], 2),
            "Vol. reportada": pct(risk.ann_vol(rep)), "Vol. des-suavizada": pct(risk.ann_vol(des)),
            "Corr. RV global (rep.)": num(rep.corr(FACTOR_RET["RV_GLOBAL"]), 2),
            "Corr. RV global (des.)": num(des.corr(FACTOR_RET["RV_GLOBAL"]), 2),
            "Máx. caída (rep.)": pct(risk.max_drawdown(rep)), "Máx. caída (des.)": pct(risk.max_drawdown(des)),
        })
    table = dash_table.DataTable(data=rows, columns=[{"name": k, "id": k} for k in rows[0]], **TABLE_STYLE)
    return html.Div([
        card("Efecto del suavizamiento en activos privados",
             "Método de Geltner: r_true(t) = (r_rep(t) − ρ·r_rep(t−1)) / (1 − ρ), con ρ = autocorrelación de primer orden",
             table),
        html.Div([
            html.Div([html.Label("Clase de activo"), dcc.Dropdown(
                id="a-asset", value="PE", clearable=False,
                options=[{"label": NAMES[c], "value": c} for c in PRIVATE_CODES])], className="control"),
        ], className="controls"),
        html.Div([
            card("Valor acumulado", "Índice base 100: la serie reportada se ve más estable que la económica",
                 dcc.Graph(id="a-index", config={"displayModeBar": False})),
            card("Retornos mensuales, últimos 5 años", "La serie des-suavizada recupera la amplitud de los movimientos",
                 dcc.Graph(id="a-monthly", config={"displayModeBar": False})),
        ], className="grid g2"),
        html.Div([
            html.P([html.B("Por qué importa: "),
                    "si el comité mide riesgo con valores de tasación, subestima la volatilidad del portafolio, "
                    "sobrestima la diversificación de los alternativos y asigna más a ellos de lo que justificaría "
                    "su riesgo económico. Des-suavizar antes de optimizar o reportar corrige ese sesgo."],
                   className="note-box"),
        ], className="card"),
    ])


# ---------------------------------------------------------------- pestaña Factores
def tab_factores() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([html.Label("Retornos de activos privados"), dcc.RadioItems(
                id="f-basis", value="des", inline=True, inputStyle={"marginRight": 6, "marginLeft": 12},
                options=[{"label": "Des-suavizados", "value": "des"}, {"label": "Reportados", "value": "rep"}])],
                className="control wide"),
        ], className="controls"),
        html.Div([
            card("Betas por clase de activo", "Regresión OLS mensual sobre 5 factores · azul = exposición positiva, rojo = negativa",
                 dcc.Graph(id="f-heat", config={"displayModeBar": False})),
            card("Exposición del portafolio a factores", "Suma de betas ponderada por el peso efectivo",
                 dcc.Graph(id="f-port", config={"displayModeBar": False})),
        ], className="grid g2"),
    ])


# ---------------------------------------------------------------- pestaña Estrés
def tab_estres() -> html.Div:
    sliders = []
    for f in FACTORS:
        sliders.append(html.Div([
            html.Label(FACTOR_LABELS[f]),
            html.Div(dcc.Slider(id=f"s-{f}", min=-50, max=50, step=1, value=0, marks={-50: "−50%", 0: "0", 50: "+50%"},
                                tooltip={"placement": "top", "always_visible": True, "template": "{value}%"}),
                     className="slider-pad"),
        ]))
    return html.Div([
        html.Div([
            html.Div([html.Label("Escenario"), dcc.Dropdown(
                id="s-scenario", value=0, clearable=False,
                options=[{"label": s.name, "value": i} for i, s in enumerate(SCENARIOS)]
                + [{"label": "Personalizado (mover los controles)", "value": -1}])], className="control wide"),
        ], className="controls"),
        html.Div(id="s-desc", className="note-box", style={"marginBottom": 12}),
        card("Shocks por factor", "Al elegir un escenario se cargan sus valores; puedes ajustarlos libremente",
             html.Div(sliders, className="sliders")),
        html.Div(id="s-kpis", className="grid g3"),
        html.Div(style={"height": 16}),
        html.Div([
            card("Impacto por clase de activo", "Pérdida o ganancia económica en MM CLP",
                 dcc.Graph(id="s-bar", config={"displayModeBar": False})),
            card("Pesos después del shock y rangos de la política",
                 "Con valores reportados al cierre del mes: los privados aún no reflejan toda la caída",
                 html.Div(id="s-table")),
        ], className="grid g23"),
    ])


# ---------------------------------------------------------------- pestaña Liquidez
def tab_liquidez() -> html.Div:
    def sl(id_, label, mn, mx, step, val, marks):
        return html.Div([html.Label(label), html.Div(
            dcc.Slider(id=id_, min=mn, max=mx, step=step, value=val, marks=marks,
                       tooltip={"placement": "top", "always_visible": True}), className="slider-pad")])
    return html.Div([
        card("Supuestos", "Programa de compromisos, regla de gasto y escenario de estrés",
             html.Div([
                 sl("l-spend", "Tasa de gasto (% del patrimonio prom. 3 años)", 3, 6, 0.25, POLICY.spending_rate * 100,
                    {3: "3%", 4.5: "4,5%", 6: "6%"}),
                 sl("l-commit", "Ritmo de compromisos nuevos (× plan)", 0, 2, 0.1, 1.0, {0: "0×", 1: "1×", 2: "2×"}),
                 sl("l-shock", "Caída cartera líquida año 1 (%)", -50, 0, 1, -25, {-50: "−50%", -25: "−25%", 0: "0"}),
                 sl("l-dist", "Recorte de distribuciones años 1–2 (%)", 0, 100, 5, 50, {0: "0", 50: "50%", 100: "100%"}),
                 sl("l-calls", "Aceleración de llamados años 1–2 (%)", 0, 100, 5, 25, {0: "0", 50: "50%", 100: "100%"}),
             ], className="sliders")),
        html.Div(id="l-kpis", className="grid g3"),
        html.Div(style={"height": 16}),
        html.Div([
            card("Alternativos como % del patrimonio", "Proyección a 10 años · línea punteada: tope de la política",
                 dcc.Graph(id="l-pct", config={"displayModeBar": False})),
            card("Exposición ilíquida potencial", "(NAV privado + compromisos no llamados) / patrimonio",
                 dcc.Graph(id="l-illiq", config={"displayModeBar": False})),
        ], className="grid g2"),
        card("Flujos proyectados en escenario de estrés", "MM CLP por año", html.Div(id="l-table")),
    ])


# ---------------------------------------------------------------- app
app = Dash(__name__, title="Endowment Risk Monitor", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Header([
        html.Div([
            html.H1("Endowment Risk Monitor"),
            html.P("Riesgo, alternativos, estrés y liquidez de un portafolio multiactivo tipo endowment"),
        ]),
        html.Span("Datos sintéticos de ejemplo · CLP", className="badge"),
    ], className="top"),
    dcc.Tabs(id="tabs", value="resumen", className="tabs", parent_className="tabs-container", children=[
        dcc.Tab(label="Resumen", value="resumen", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Riesgo", value="riesgo", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Alternativos", value="alt", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Factores", value="factores", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Estrés", value="estres", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Liquidez", value="liquidez", className="tab", selected_className="tab--selected"),
    ]),
    html.Div(id="tab-content"),
    html.Footer([
        "Proyecto demostrativo de Ricardo García Pereira, CFA. Los datos son sintéticos y los escenarios ilustrativos; "
        "no constituyen recomendación de inversión. Reemplaza data/*.csv y erm/config.py para usarlo con un portafolio real.",
    ]),
], className="wrap")

TABS = {"resumen": tab_resumen, "riesgo": tab_riesgo, "alt": tab_alternativos,
        "factores": tab_factores, "estres": tab_estres, "liquidez": tab_liquidez}


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return TABS[tab]()


@app.callback(
    Output("r-kpis", "children"), Output("r-contrib", "figure"), Output("r-rolling", "figure"), Output("r-dd", "figure"),
    Input("r-basis", "value"), Input("r-window", "value"),
)
def update_risk(basis, window):
    R = window_slice(DESMOOTHED if basis == "des" else REPORTED, window)
    port = risk.portfolio_returns(R, W_ACTUAL)
    s = risk.summary(port, R["CAJA"])
    kpis = [
        kpi("Volatilidad anual", pct(s["vol_anual"])),
        kpi("VaR 95% 1 mes (histórico)", pct(s["var95"]), f"$ {num(s['var95'] * AUM_MM)} MM"),
        kpi("CVaR 95% 1 mes", pct(s["cvar95"]), f"$ {num(s['cvar95'] * AUM_MM)} MM"),
        kpi("VaR 99% 1 mes", pct(s["var99"]), f"paramétrico 95%: {pct(s['var95_param'])}"),
        kpi("Máxima caída", pct(s["max_dd"])),
        kpi("Sharpe (vs caja)", num(s["sharpe"], 2), f"retorno anual {pct(s['ret_anual'])}"),
    ]

    rc = risk.risk_contributions(R, W_ACTUAL).sort_values("pct_riesgo")
    labels = [NAMES[c] for c in rc.index]
    fig_c = go.Figure([
        go.Bar(y=labels, x=rc["peso"] * 100, orientation="h", name="Peso", marker=dict(color=S1, cornerradius=4),
               hovertemplate="%{y}<br>Peso: %{x:.1f}%<extra></extra>"),
        go.Bar(y=labels, x=rc["pct_riesgo"] * 100, orientation="h", name="% del riesgo",
               marker=dict(color=S2, cornerradius=4), hovertemplate="%{y}<br>Riesgo: %{x:.1f}%<extra></extra>"),
    ])
    base_layout(fig_c, 380, barmode="group", bargroupgap=0.12)
    fig_c.update_xaxes(ticksuffix="%")

    full_rep = risk.portfolio_returns(REPORTED, W_ACTUAL)
    full_des = risk.portfolio_returns(DESMOOTHED, W_ACTUAL)
    rv_rep, rv_des = risk.rolling_vol(full_rep).dropna(), risk.rolling_vol(full_des).dropna()
    fig_r = go.Figure([
        go.Scatter(x=rv_des.index, y=rv_des * 100, name="Des-suavizado", line=dict(color=S1, width=2),
                   hovertemplate="%{x|%m-%Y}: %{y:.1f}%<extra>Des-suavizado</extra>"),
        go.Scatter(x=rv_rep.index, y=rv_rep * 100, name="Reportado", line=dict(color=S2, width=2),
                   hovertemplate="%{x|%m-%Y}: %{y:.1f}%<extra>Reportado</extra>"),
    ])
    base_layout(fig_r, 380, hovermode="x unified")
    fig_r.update_yaxes(ticksuffix="%", rangemode="tozero")

    dd = risk.drawdown_series(port)
    fig_d = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=S1, width=2),
                                 fillcolor="rgba(42,120,214,0.12)", name="Drawdown",
                                 hovertemplate="%{x|%m-%Y}: %{y:.1f}%<extra></extra>"))
    base_layout(fig_d, 260, showlegend=False, hovermode="x")
    fig_d.update_yaxes(ticksuffix="%")
    return kpis, fig_c, fig_r, fig_d


@app.callback(Output("a-index", "figure"), Output("a-monthly", "figure"), Input("a-asset", "value"))
def update_alt(code):
    rep, des = REPORTED[code], DESMOOTHED[code]
    fig_i = go.Figure([
        go.Scatter(x=des.index, y=100 * (1 + des).cumprod(), name="Des-suavizado", line=dict(color=S1, width=2),
                   hovertemplate="%{x|%m-%Y}: %{y:.0f}<extra>Des-suavizado</extra>"),
        go.Scatter(x=rep.index, y=100 * (1 + rep).cumprod(), name="Reportado", line=dict(color=S2, width=2),
                   hovertemplate="%{x|%m-%Y}: %{y:.0f}<extra>Reportado</extra>"),
    ])
    base_layout(fig_i, 340, hovermode="x unified")
    rep5, des5 = rep.iloc[-60:], des.iloc[-60:]
    fig_m = go.Figure([
        go.Bar(x=des5.index, y=des5 * 100, name="Des-suavizado", marker=dict(color=S1),
               hovertemplate="%{x|%m-%Y}: %{y:.1f}%<extra>Des-suavizado</extra>"),
        go.Bar(x=rep5.index, y=rep5 * 100, name="Reportado", marker=dict(color=S2),
               hovertemplate="%{x|%m-%Y}: %{y:.1f}%<extra>Reportado</extra>"),
    ])
    base_layout(fig_m, 340, barmode="group", bargap=0.25, bargroupgap=0.05)
    fig_m.update_yaxes(ticksuffix="%")
    return fig_i, fig_m


@app.callback(Output("f-heat", "figure"), Output("f-port", "figure"), Input("f-basis", "value"))
def update_factors(basis):
    b = BETAS_DES if basis == "des" else BETAS_REP
    z = b[FACTORS].to_numpy()
    lim = float(np.ceil(np.abs(z).max() * 10) / 10)
    xlabels = [FACTOR_SHORT[f] for f in FACTORS]
    ylabels = [f"{NAMES[c]}  ·  R² {num(r2, 2)}" for c, r2 in zip(b.index, b["r2"])]
    text = [[num(v, 2) for v in row] for row in z]
    fig_h = go.Figure(go.Heatmap(
        z=z, x=xlabels, y=ylabels, zmin=-lim, zmax=lim, text=text, texttemplate="%{text}",
        colorscale=[[0, NEG], [0.5, MID], [1, POS]], showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>", textfont=dict(size=12),
    ))
    base_layout(fig_h, 420)
    fig_h.update_yaxes(autorange="reversed", showgrid=False)
    fig_h.update_xaxes(side="top", showgrid=False, tickfont=dict(color=INK2))

    exp = fm.portfolio_factor_exposure(b, W_ACTUAL, FACTORS)
    colors = [POS if v >= 0 else NEG for v in exp]
    fig_p = go.Figure(go.Bar(x=[FACTOR_SHORT[f] for f in exp.index], y=exp.values, marker=dict(color=colors, cornerradius=4),
                             text=[num(v, 2) for v in exp.values], textposition="outside", textfont=dict(color=INK2),
                             hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    base_layout(fig_p, 420, showlegend=False)
    return fig_h, fig_p


@app.callback([Output(f"s-{f}", "value") for f in FACTORS], Input("s-scenario", "value"))
def load_scenario(i):
    if i is None or i < 0:
        return [0] * len(FACTORS)
    return [round(SCENARIOS[i].shocks.get(f, 0) * 100) for f in FACTORS]


@app.callback(
    Output("s-desc", "children"), Output("s-kpis", "children"), Output("s-bar", "figure"), Output("s-table", "children"),
    Input("s-scenario", "value"), *[Input(f"s-{f}", "value") for f in FACTORS],
)
def update_stress(i, *vals):
    shocks = {f: (v or 0) / 100 for f, v in zip(FACTORS, vals)}
    desc = SCENARIOS[i].description if i is not None and i >= 0 else "Escenario definido con los controles."
    table, tot = st.apply_shock(BETAS_DES, shocks, W_ACTUAL, AUM_MM, RHOS)
    kpis = [
        kpi("Impacto económico", f"$ {num(tot['impacto_mm'])} MM", f"{pct(tot['retorno'])} del patrimonio"),
        kpi("Impacto reportado al cierre del mes", pct(tot["retorno_reportado"]),
            "los privados reconocen la caída con rezago"),
        kpi("Clases fuera de rango", str(tot["n_fuera_de_rango"]), "con pesos a valor reportado"),
    ]
    t = table.sort_values("impacto_mm")
    colors = [NEG if v < 0 else POS for v in t["impacto_mm"]]
    fig = go.Figure(go.Bar(y=[NAMES[c] for c in t.index], x=t["impacto_mm"], orientation="h",
                           marker=dict(color=colors, cornerradius=4), customdata=t["retorno"] * 100,
                           hovertemplate="%{y}<br>$ %{x:,.0f} MM · %{customdata:.1f}%<extra></extra>"))
    base_layout(fig, 380, showlegend=False)

    rows = [{"Clase": NAMES[c], "Ret. econ.": pct(r.retorno), "Ret. rep.": pct(r.retorno_reportado),
             "Antes": f"{num(r.peso_antes, 1)}%", "Después": f"{num(r.peso_despues, 1)}%",
             "Rango": f"{num(r['min'])}–{num(r['max'])}%", "Estado": "✕ Fuera" if r.fuera_de_rango else "✓ Dentro"}
            for c, r in table.iterrows()]
    alt_after = table.loc[PRIVATE_CODES, "peso_despues"].sum()
    lo, hi = BUCKET_LIMITS["Alternativos"]
    rows.append({"Clase": "Total alternativos", "Ret. econ.": "", "Ret. rep.": "",
                 "Antes": f"{num(table.loc[PRIVATE_CODES, 'peso_antes'].sum(), 1)}%",
                 "Después": f"{num(alt_after, 1)}%", "Rango": f"{lo}–{hi}%",
                 "Estado": "✓ Dentro" if lo <= alt_after <= hi else "✕ Fuera"})
    dt = dash_table.DataTable(
        data=rows, columns=[{"name": k, "id": k} for k in rows[0]], **TABLE_STYLE,
        style_data_conditional=[
            {"if": {"filter_query": '{Estado} contains "Fuera"', "column_id": "Estado"}, "color": CRITICAL, "fontWeight": 600},
            {"if": {"filter_query": '{Estado} contains "Dentro"', "column_id": "Estado"}, "color": "#006300"},
            {"if": {"filter_query": '{Clase} contains "Total"'}, "fontWeight": 600},
        ],
    )
    return desc, kpis, fig, dt


@app.callback(
    Output("l-kpis", "children"), Output("l-pct", "figure"), Output("l-illiq", "figure"), Output("l-table", "children"),
    Input("l-spend", "value"), Input("l-commit", "value"), Input("l-shock", "value"),
    Input("l-dist", "value"), Input("l-calls", "value"),
)
def update_liquidity(spend, commit, shock, dist, calls):
    base = liq.project(spending_rate=spend / 100, commit_scale=commit)
    stress_in = liq.StressInputs(liquid_shock=shock / 100, private_shock=shock / 100 * 0.4,
                                 dist_cut=dist / 100, call_boost=calls / 100)
    stressed = liq.project(spending_rate=spend / 100, commit_scale=commit, stress=stress_in)
    cap = liq.private_max_range()
    years_over = int((stressed["pct_privado"] > cap).sum())

    kpis = [
        kpi("Máx. % alternativos (base)", f"{num(base['pct_privado'].max(), 1)}%", f"tope de la política {cap}%"),
        kpi("Máx. % alternativos (estrés)", f"{num(stressed['pct_privado'].max(), 1)}%",
            f"{years_over} de {len(stressed)} años sobre el tope"),
        kpi("Gasto año 1", f"$ {num(base['gasto'].iloc[0])} MM", f"{num(spend, 2)}% del patrimonio promedio"),
    ]

    x = [f"Año {t}" for t in base.index]

    def two_lines(col, limit, ytitle):
        fig = go.Figure([
            go.Scatter(x=x, y=base[col], name="Base", line=dict(color=S1, width=2), mode="lines+markers",
                       marker=dict(size=8, line=dict(width=2, color="#fcfcfb")),
                       hovertemplate="%{x}: %{y:.1f}%<extra>Base</extra>"),
            go.Scatter(x=x, y=stressed[col], name="Estrés", line=dict(color=S2, width=2), mode="lines+markers",
                       marker=dict(size=8, line=dict(width=2, color="#fcfcfb")),
                       hovertemplate="%{x}: %{y:.1f}%<extra>Estrés</extra>"),
        ])
        fig.add_hline(y=limit, line=dict(color=INK2, width=1.5, dash="dot"),
                      annotation_text=f"Tope {num(limit)}%", annotation_position="top left",
                      annotation_font=dict(color=INK2, size=11))
        base_layout(fig, 340, hovermode="x unified")
        fig.update_yaxes(ticksuffix="%", title=None)
        return fig

    fig_pct = two_lines("pct_privado", cap, "%")
    fig_ill = two_lines("iliquido_potencial", POLICY.max_illiquid_potential, "%")

    t = stressed
    rows = [{"Clase": f"Año {i}", "Llamados": num(r.llamados), "Distribuciones": num(r.distribuciones),
             "Gasto": num(r.gasto), "Cartera líquida": num(r.liquido), "NAV privado": num(r.nav_privado),
             "No llamado": num(r.no_llamado), "% alternativos": f"{num(r.pct_privado, 1)}%"}
            for i, r in t.iterrows()]
    dt = dash_table.DataTable(data=rows, columns=[{"name": ("Año" if k == "Clase" else k), "id": k} for k in rows[0]],
                              **TABLE_STYLE)
    return kpis, fig_pct, fig_ill, dt


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
