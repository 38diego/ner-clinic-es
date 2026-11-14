"""Formato y cobertura del set sintético reservado (issue #44, PRD de evaluación FR8).

Solo valida las anotaciones del JSONL: no corre `analizar()` ni el modelo sobre estas
notas. El set es reservado y se mide una sola vez.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from ner_clinic.charlson.indice import calcular_desde_categorias
from ner_clinic.tipos import CategoriaCharlson

RAIZ = Path(__file__).resolve().parents[1]
DIR_SET = RAIZ / "evaluacion" / "sintetico"
RUTA_NOTAS = DIR_SET / "notas.jsonl"
RUTAS_EXPERIMENTOS = [
    RAIZ / "experimentos" / "errores_tesis" / "notas.jsonl",
    RAIZ / "experimentos" / "siglas" / "notas.jsonl",
]

CAMPOS_NOTA = {"id", "texto", "entidades", "indice", "registrados"}
CAMPOS_NOTA_OBLIGATORIOS = {"id", "texto", "entidades", "indice"}
CAMPOS_ENTIDAD = {"texto", "estado", "categoria", "registrado"}
ESTADOS = {"confirmada", "negada", "familiar"}


def _cargar_modulo_cobertura():
    """Carga `cobertura.py` por ruta, sin importar el paquete `evaluacion`."""
    spec = importlib.util.spec_from_file_location("cobertura_sintetico", DIR_SET / "cobertura.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


cobertura = _cargar_modulo_cobertura()


def _leer_jsonl(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


NOTAS = _leer_jsonl(RUTA_NOTAS)
CONTEOS = cobertura.contar(NOTAS)


# --- Formato ----------------------------------------------------------------


def test_hay_al_menos_100_notas():
    assert len(NOTAS) >= 100


def test_ids_unicos():
    ids = [n["id"] for n in NOTAS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("nota", NOTAS, ids=lambda n: n["id"])
def test_formato_de_la_nota(nota):
    assert CAMPOS_NOTA_OBLIGATORIOS <= set(nota) <= CAMPOS_NOTA
    assert isinstance(nota["id"], str) and nota["id"]
    assert isinstance(nota["texto"], str) and nota["texto"].strip()
    assert isinstance(nota["entidades"], list) and nota["entidades"]
    assert isinstance(nota["indice"], int) and nota["indice"] >= 0
    if "registrados" in nota:
        assert nota["registrados"]
        assert all(isinstance(r, str) and r.strip() for r in nota["registrados"])


@pytest.mark.parametrize("nota", NOTAS, ids=lambda n: n["id"])
def test_formato_de_las_entidades(nota):
    categorias = {c.value for c in CategoriaCharlson}
    vistas = set()
    for e in nota["entidades"]:
        assert {"texto", "estado"} <= set(e) <= CAMPOS_ENTIDAD, e
        assert e["estado"] in ESTADOS, e
        assert e["texto"] in nota["texto"], f"{e['texto']!r} no aparece literal en la nota"
        assert e["texto"].lower() not in vistas, f"{e['texto']!r} anotada dos veces"
        vistas.add(e["texto"].lower())
        if "categoria" in e:
            assert e["categoria"] in categorias, e
        if "registrado" in e:
            assert isinstance(e["registrado"], bool), e
            assert "registrados" in nota, "`registrado` sin `registrados` en la nota"


@pytest.mark.parametrize("nota", [n for n in NOTAS if "registrados" in n], ids=lambda n: n["id"])
def test_confirmadas_con_registrado_cuando_hay_registrados(nota):
    for e in nota["entidades"]:
        if e["estado"] == "confirmada":
            assert "registrado" in e, e


@pytest.mark.parametrize("nota", NOTAS, ids=lambda n: n["id"])
def test_indice_coherente_con_las_categorias_anotadas(nota):
    """El índice anotado es el de Charlson 1987 sobre las categorías de las confirmadas."""
    categorias = [
        CategoriaCharlson(e["categoria"])
        for e in nota["entidades"]
        if e["estado"] == "confirmada" and "categoria" in e
    ]
    assert nota["indice"] == calcular_desde_categorias(categorias)


def test_no_repite_notas_de_experimentos():
    previas = [n["texto"] for ruta in RUTAS_EXPERIMENTOS for n in _leer_jsonl(ruta)]
    assert len(previas) == 55
    for nota in NOTAS:
        for previa in previas:
            assert previa not in nota["texto"] and nota["texto"] not in previa, nota["id"]


def test_sin_numeros_tipo_cedula():
    for nota in NOTAS:
        assert not re.search(r"\d{7,}", nota["texto"]), nota["id"]


# --- Cobertura mínima (issue #44) -------------------------------------------


def test_cobertura_de_estados():
    assert CONTEOS["estados"]["negada"] >= 60
    assert CONTEOS["estados"]["familiar"] >= 40


def test_las_17_categorias_al_menos_2_veces_confirmadas():
    assert len(CONTEOS["categorias"]) == 17
    assert set(CONTEOS["categorias"]) == {c.value for c in CategoriaCharlson}
    faltan = {c: n for c, n in CONTEOS["categorias"].items() if n < 2}
    assert not faltan


def test_jerarquias_leve_severa():
    assert CONTEOS["jerarquias"]["diabetes"] >= 1
    assert CONTEOS["jerarquias"]["hepática"] >= 1


def test_siglas_registrados_y_longitudes():
    assert CONTEOS["siglas"] >= 30
    assert CONTEOS["notas_con_registrados"] >= 15
    assert CONTEOS["una_linea"] >= 1
    assert CONTEOS["largas"] >= 1


def test_formas_de_negacion_y_parentesco():
    for forma in ("niega", "sin", "no presenta", "se descarta", "negativo para"):
        assert CONTEOS["negaciones"].get(forma, 0) >= 1, forma
    assert "otra" not in CONTEOS["negaciones"]
    assert "otro" not in CONTEOS["parentescos"]
    assert len(CONTEOS["parentescos"]) >= 5


def test_readme_trae_la_tabla_generada():
    """La tabla del README es la salida de `cobertura.py`, no un conteo a mano."""
    readme = (DIR_SET / "README.md").read_text(encoding="utf-8")
    assert cobertura.tabla(CONTEOS) in readme


def test_es_sigla():
    assert cobertura.es_sigla("DM2")
    assert cobertura.es_sigla("EPOC")
    assert not cobertura.es_sigla("asma")
    assert not cobertura.es_sigla("CA DE COLON")


# --- Doble anotación ciega -------------------------------------------------------------


def _cargar_modulo_acuerdo():
    """Carga `acuerdo.py` por ruta, como `cobertura.py`."""
    spec = importlib.util.spec_from_file_location("acuerdo_sintetico", DIR_SET / "acuerdo.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


acuerdo = _cargar_modulo_acuerdo()
ACUERDO = acuerdo.comparar(
    acuerdo.cargar(acuerdo.CARPETA / "a.jsonl"), acuerdo.cargar(acuerdo.CARPETA / "b.jsonl")
)


def test_los_dos_anotadores_anotaron_las_mismas_notas():
    ids = {n["id"] for n in NOTAS}
    assert set(acuerdo.cargar(acuerdo.CARPETA / "a.jsonl")) == ids
    assert set(acuerdo.cargar(acuerdo.CARPETA / "b.jsonl")) == ids


def test_tabla_de_acuerdo_del_readme_es_la_salida_del_script():
    readme = (DIR_SET / "README.md").read_text(encoding="utf-8")
    assert acuerdo.tabla(ACUERDO) in readme


def test_acuerdo_suficiente_para_usar_el_set():
    assert ACUERDO.f1_entidades >= 0.9
    assert ACUERDO.kappa_estado >= 0.8


def test_coinciden_por_contencion_sin_mayusculas():
    assert acuerdo.coinciden("DM2", "dm2 con nefropatía")
    assert not acuerdo.coinciden("EPOC", "asma")


def test_discrepancias_lista_cada_tipo():
    a = {
        "x": {
            "entidades": [
                {"texto": "HTA", "estado": "negada"},
                {"texto": "ICC", "estado": "confirmada", "categoria": "insuficiencia_cardiaca"},
                {"texto": "asma", "estado": "confirmada"},
            ],
            "indice": 1,
        }
    }
    b = {
        "x": {
            "entidades": [
                {"texto": "HTA", "estado": "confirmada"},
                {"texto": "ICC", "estado": "confirmada"},
                {"texto": "EPOC", "estado": "confirmada"},
            ],
            "indice": 2,
        }
    }
    lineas = acuerdo.discrepancias(acuerdo.comparar(a, b))
    assert any("estado 'HTA'" in x for x in lineas)
    assert any("categoría 'ICC'" in x for x in lineas)
    assert any("solo A 'asma'" in x for x in lineas)
    assert any("solo B 'EPOC'" in x for x in lineas)
    assert any("índice 1 / 2" in x for x in lineas)
