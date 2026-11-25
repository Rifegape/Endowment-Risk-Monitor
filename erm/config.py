"""Configuración del portafolio ilustrativo.

Todo lo que un usuario querría cambiar para adaptar el monitor a su propio
endowment está aquí: clases de activos, pesos efectivos, política de inversión
(pesos objetivo y rangos), compromisos no llamados y supuestos de liquidez.
Moneda base: CLP. Montos en millones de pesos (MM CLP).
"""

from dataclasses import dataclass, field

BASE_CURRENCY = "CLP"
AUM_MM = 65_000.0  # Patrimonio total del endowment, MM CLP

FACTORS = ["RV_GLOBAL", "RV_CHILE", "USDCLP", "TASAS", "CREDITO"]
FACTOR_LABELS = {
    "RV_GLOBAL": "Renta variable global (USD)",
    "RV_CHILE": "Renta variable Chile",
    "USDCLP": "Dólar / peso",
    "TASAS": "Tasas locales (retorno bono UF)",
    "CREDITO": "Spread de crédito (exceso de retorno)",
}


@dataclass(frozen=True)
class AssetClass:
    code: str
    name: str
    bucket: str            # Renta variable / Renta fija / Alternativos / Caja
    actual: float          # peso efectivo, % del patrimonio
    policy: float          # peso objetivo de la política, %
    lo: float              # rango mínimo, %
    hi: float              # rango máximo, %
    private: bool = False  # True = valorización por tasación (retornos suavizados)
    liquid: bool = True    # disponible para cubrir llamados y gasto en < 1 mes


ASSETS: list[AssetClass] = [
    AssetClass("RV_CL", "Renta variable Chile", "Renta variable", 7.5, 8, 5, 12),
    AssetClass("RV_GL", "Renta variable global", "Renta variable", 24.0, 22, 15, 30),
    AssetClass("RV_EM", "Renta variable emergente", "Renta variable", 5.5, 5, 0, 10),
    AssetClass("RF_GOB", "Renta fija Chile gobierno UF", "Renta fija", 13.0, 15, 10, 25),
    AssetClass("RF_CORP", "Renta fija Chile corporativa", "Renta fija", 11.0, 12, 5, 20),
    AssetClass("RF_GL", "Renta fija global IG", "Renta fija", 7.0, 8, 0, 15),
    AssetClass("PE", "Private equity", "Alternativos", 12.5, 10, 5, 15, private=True, liquid=False),
    AssetClass("PD", "Deuda privada", "Alternativos", 6.5, 7, 0, 10, private=True, liquid=False),
    AssetClass("INMO", "Inmobiliario", "Alternativos", 10.5, 10, 5, 15, private=True, liquid=False),
    AssetClass("CAJA", "Caja y money market", "Caja", 2.5, 3, 0, 10),
]

ASSET_CODES = [a.code for a in ASSETS]
ASSET_BY_CODE = {a.code: a for a in ASSETS}
PRIVATE_CODES = [a.code for a in ASSETS if a.private]


@dataclass(frozen=True)
class PrivateProgram:
    """Supuestos del modelo de compromisos (Takahashi-Alexander simplificado)."""
    code: str
    unfunded_mm: float      # compromisos no llamados hoy, MM CLP
    call_rate: float        # fracción del no llamado que se llama cada año
    dist_rate: float        # fracción del NAV que se distribuye cada año
    growth: float           # crecimiento anual esperado del NAV
    new_commit_mm: float    # compromisos nuevos por año (programa de vintage)


PRIVATE_PROGRAMS: list[PrivateProgram] = [
    PrivateProgram("PE", unfunded_mm=4_500, call_rate=0.35, dist_rate=0.18, growth=0.10, new_commit_mm=2_000),
    PrivateProgram("PD", unfunded_mm=2_500, call_rate=0.45, dist_rate=0.25, growth=0.08, new_commit_mm=1_200),
    PrivateProgram("INMO", unfunded_mm=1_500, call_rate=0.40, dist_rate=0.09, growth=0.07, new_commit_mm=600),
]


@dataclass(frozen=True)
class EndowmentPolicy:
    spending_rate: float = 0.045        # regla de gasto: % del patrimonio promedio de 3 años
    donations_mm: float = 1_200.0       # donaciones nuevas al año, MM CLP
    liquid_return: float = 0.065        # retorno nominal esperado de la cartera líquida
    max_illiquid_potential: float = 50.0  # tope para (NAV privado + no llamado) / patrimonio, %
    horizon_years: int = 10


POLICY = EndowmentPolicy()

# Límites agregados por grupo de activos (% del patrimonio)
BUCKET_LIMITS = {
    "Renta variable": (25, 45),
    "Renta fija": (20, 50),
    "Alternativos": (15, 35),
    "Caja": (0, 10),
}


@dataclass(frozen=True)
class StressScenario:
    name: str
    description: str
    shocks: dict = field(default_factory=dict)  # shock instantáneo por factor, en decimal


# Escenarios ilustrativos inspirados en episodios históricos. Las magnitudes son
# aproximadas y deben calibrarse con datos propios antes de usarse en un comité.
SCENARIOS: list[StressScenario] = [
    StressScenario(
        "Crisis financiera global (tipo 2008)",
        "Caída fuerte de bolsas, salida de capitales y alza del dólar; ampliación de spreads.",
        {"RV_GLOBAL": -0.40, "RV_CHILE": -0.35, "USDCLP": 0.25, "TASAS": 0.04, "CREDITO": -0.08},
    ),
    StressScenario(
        "Pandemia (tipo marzo 2020)",
        "Shock breve y profundo de liquidez; refugio en dólar y bonos soberanos.",
        {"RV_GLOBAL": -0.20, "RV_CHILE": -0.25, "USDCLP": 0.12, "TASAS": 0.01, "CREDITO": -0.04},
    ),
    StressScenario(
        "Alza global de tasas (tipo 2022)",
        "Inflación persistente, bancos centrales restrictivos; caen acciones y bonos a la vez.",
        {"RV_GLOBAL": -0.18, "RV_CHILE": 0.00, "USDCLP": 0.05, "TASAS": -0.08, "CREDITO": -0.03},
    ),
    StressScenario(
        "Shock local Chile (tipo octubre 2019)",
        "Incertidumbre doméstica: cae la bolsa local, se deprecia el peso, suben tasas largas.",
        {"RV_GLOBAL": 0.00, "RV_CHILE": -0.20, "USDCLP": 0.12, "TASAS": -0.03, "CREDITO": -0.02},
    ),
]
