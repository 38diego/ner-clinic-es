"""Tests de la tabla de Quan (2005): CIE-10 → categoría de Charlson (#129)."""

import ast
import inspect
import re
import string
import textwrap

import pytest

from ner_clinic.charlson import quan
from ner_clinic.charlson.reglas import REGLAS
from ner_clinic.tipos import CategoriaCharlson as C


@pytest.mark.parametrize(
    ("codigo", "categoria"),
    [
        ("E11.9", C.DIABETES),
        ("E11.2", C.DIABETES_COMPLICADA),
        ("J44", C.PULMONAR_CRONICA),
        ("I63", C.CEREBROVASCULAR),
        ("C81", C.TUMOR_MALIGNO),
        # Formas de escribir el código y subcategorías de la CIE-10-ES (CodiEsp)
        ("e11.9", C.DIABETES),
        ("E119", C.DIABETES),
        (" i63.9 ", C.CEREBROVASCULAR),
        ("E11 .9", C.DIABETES),
        ("E11.21", C.DIABETES_COMPLICADA),
        ("J44.1", C.PULMONAR_CRONICA),
        ("I25.2", C.INFARTO_MIOCARDIO),
        ("K70.3", C.HEPATICA_LEVE),
        ("K70.4", C.HEPATICA_SEVERA),
        ("C78.7", C.TUMOR_METASTASICO),
        ("C80", C.TUMOR_METASTASICO),
        ("B20", C.VIH_SIDA),
        ("N18.3", C.RENAL_CRONICA),
        ("G30.9", C.DEMENCIA),
    ],
)
def test_categoria_de(codigo, categoria):
    assert quan.categoria_de(codigo) is categoria


@pytest.mark.parametrize(
    "codigo",
    [
        "I10",  # hipertensión: no es de Charlson
        "E11",  # repartido entre diabetes con y sin complicación
        "I25",  # solo I25.2 y I25.5
        "B23",  # el rango del VIH salta B23
        "C44",  # piel no melanoma: fuera de "cualquier tumor maligno"
        "D05",  # carcinoma in situ
        "E08.9",  # diabetes por otra afección: solo existe en la CIE-10-CM
        "",
    ],
)
def test_sin_categoria(codigo):
    assert quan.categoria_de(codigo) is None


def test_cada_categoria_tiene_codigos():
    assert set(quan.TABLA_QUAN) == set(C)
    for categoria in C:
        assert quan.codigos_de(categoria)


@pytest.mark.parametrize(
    ("entrada", "prefijos"),
    [
        ("I21", ["I21"]),
        ("I25.2", ["I252"]),
        ("K70.0-K70.3", ["K700", "K701", "K702", "K703"]),
        ("C30-C34", ["C30", "C31", "C32", "C33", "C34"]),
    ],
)
def test_expandir(entrada, prefijos):
    assert quan._expandir(entrada) == prefijos


@pytest.mark.parametrize("entrada", ["C30-K34", "K70.0-K71.3", "C34-C30", "I2", "I21.10"])
def test_expandir_rechaza_entradas_mal_escritas(entrada):
    with pytest.raises(ValueError):
        quan._expandir(entrada)


# --- Cruce con el paquete `coder` de rOpenSci -------------------------------------------
# Columna `regex_icd10` de la hoja `charlson` de `data-raw/classcodes.xlsx`, commit
# 0826da00250cce270b71a5076d481b728192c205 (2025-05-05) de github.com/ropensci/coder. Es
# la versión de Quan 2005 para CIE-10. `coder` busca cada regex al inicio del código sin
# punto.
CODER_ICD10 = {
    C.INFARTO_MIOCARDIO: r"I2([12]|52)",
    C.INSUFICIENCIA_CARDIACA: r"I(099|1(10|3[02])|255|4(2[05-9]|3)|50)|P290",
    C.VASCULAR_PERIFERICA: r"I7([01]|3[189]|71|9[02])|K55[189]|Z95[89]",
    C.CEREBROVASCULAR: r"G4[56]|H340|I6",
    C.DEMENCIA: r"F0([0-3]|51)|G3(0|11)",
    C.PULMONAR_CRONICA: r"(I27[89])|J(4[0-7]|6([0-7]|84)|70[13])",
    C.REUMATOLOGICA: r"M(0[56]|3(15|[2-4]|5[13]|60))",
    C.ULCERA_PEPTICA: r"K2[5-8]",
    C.HEPATICA_LEVE: r"B18|K7(0[0-39]|1[3457]|[34]|6[023489])|Z944",
    C.DIABETES: r"E1[0-4][01689]",
    C.HEMIPLEJIA: r"G(041|114|8(0[12]|[12]|3[0-49]))",
    C.RENAL_CRONICA: r"I1(20|31)|N(0([35][2-7])|1[89]|250)|Z(49[012]|9(40|92))",
    C.DIABETES_COMPLICADA: r"E1[0-4][2-57]",
    C.TUMOR_MALIGNO: r"C([01]|2[0-6]|3[0-47-9]|4[0135-9]|5[0-8]|6|7[0-6]|8[1-58]|9[0-7])",
    C.HEPATICA_SEVERA: r"I(8(5[09]|64)|982)|K7(04|[12]1|29|6[5-7])",
    C.TUMOR_METASTASICO: r"C(7[7-9]|80)",
    C.VIH_SIDA: r"B2[0124]",
}


def _categoria_coder(codigo: str) -> C | None:
    encontradas = [c for c, regex in CODER_ICD10.items() if re.match(regex, codigo)]
    assert len(encontradas) <= 1, (codigo, encontradas)
    return encontradas[0] if encontradas else None


def test_coincide_con_coder_en_todos_los_codigos():
    """Todos los códigos de 3 y 4 caracteres posibles, de A00 a Z99.9."""
    codigos = [
        f"{letra}{n:02d}{sub}"
        for letra in string.ascii_uppercase
        for n in range(100)
        for sub in ["", *string.digits]
    ]
    distintos = [
        (codigo, quan.categoria_de(codigo), _categoria_coder(codigo))
        for codigo in codigos
        if quan.categoria_de(codigo) is not _categoria_coder(codigo)
    ]
    assert distintos == []


def test_un_codigo_en_dos_categorias_falla(monkeypatch):
    tabla = {C.DIABETES: ("E11.9",), C.DIABETES_COMPLICADA: ("E11.8-E11.9",)}
    monkeypatch.setattr(quan, "TABLA_QUAN", tabla)
    with pytest.raises(ValueError, match="E119 está en dos categorías"):
        quan._prefijos()


# --- Diferencias con las reglas actuales (docs/datos.md) --------------------------------

_CODIGO = re.compile(r"^[A-Z]\d{2}(\.\d)?$")


def _codigos_de_la_regla(regla) -> set[str]:
    """Los códigos que puede dar `regla.subclasificar`, sacados de su código fuente.

    Son las constantes con forma de código. Si la función arma el código con una
    f-string (`f"{prefijo}.7"`), se combinan los prefijos que asigna a una variable
    ("E10", "E11") con los sufijos de sus f-strings (".7").
    """
    funcion = regla.subclasificar
    if funcion.__name__ == "<lambda>":
        return {funcion("")}
    arbol = ast.parse(textwrap.dedent(inspect.getsource(funcion)))
    constantes = [
        n for n in ast.walk(arbol) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    sufijos = {
        parte.value
        for f in ast.walk(arbol)
        if isinstance(f, ast.JoinedStr)
        for parte in f.values
        if isinstance(parte, ast.Constant)
    }
    asignados = {
        n.value.value
        for n in ast.walk(arbol)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    }
    prefijos = (
        {p for p in asignados if isinstance(p, str) and _CODIGO.match(p)} if sufijos else set()
    )
    codigos = {c.value for c in constantes if _CODIGO.match(c.value)} - prefijos
    return codigos | {p + s for p in prefijos for s in sufijos}


def test_diferencias_con_las_reglas_documentadas():
    distintos = {
        (regla.categoria, codigo, quan.categoria_de(codigo))
        for regla in REGLAS
        for codigo in _codigos_de_la_regla(regla)
        if quan.categoria_de(codigo) is not regla.categoria
    }
    assert distintos == {
        (C.DIABETES_COMPLICADA, "E10.8", C.DIABETES),
        (C.DIABETES_COMPLICADA, "E11.8", C.DIABETES),
        (C.RENAL_CRONICA, "N03.9", None),
        (C.TUMOR_MALIGNO, "C80", C.TUMOR_METASTASICO),
        # #63: la fibrosis pulmonar es pulmonar crónica en el Charlson original y en los
        # criterios del set sintético, pero Quan no incluye J84 (solo J70.1 y J70.3)
        (C.PULMONAR_CRONICA, "J84", None),
    }


def test_la_diabetes_complicada_sale_de_las_f_strings():
    regla = next(r for r in REGLAS if r.categoria is C.DIABETES_COMPLICADA)
    assert _codigos_de_la_regla(regla) == {
        f"{tipo}.{sub}" for tipo in ("E10", "E11") for sub in "234578"
    }


def test_cada_regla_da_al_menos_un_codigo():
    for regla in REGLAS:
        assert _codigos_de_la_regla(regla), regla.categoria
