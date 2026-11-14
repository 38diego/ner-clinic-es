"""Tests de la corrección de ortografía antes de las reglas de Charlson (#115).

Todas las frases son sintéticas.
"""

import time

import pytest

from ner_clinic.charlson import correccion
from ner_clinic.charlson.base import normalizar
from ner_clinic.charlson.reglas import mapear as mapear_exacto
from ner_clinic.tipos import CategoriaCharlson as C


@pytest.mark.parametrize(("palabra", "frase"), sorted(correccion.FRASES_DEL_VOCABULARIO.items()))
def test_cada_palabra_del_vocabulario_dispara_su_regla_en_su_frase(palabra, frase):
    assert palabra in normalizar(frase).split()
    assert mapear_exacto(frase) is not None


def test_vocabulario_normalizado_y_sin_siglas_cortas():
    for palabra in correccion.VOCABULARIO:
        assert palabra == normalizar(palabra)
        assert palabra.isalpha()
    assert correccion.VOCABULARIO.isdisjoint(correccion.PALABRAS_COMUNES)


# --- US2: lo que sí se corrige -----------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "categoria", "correccion_esperada"),
    [
        ("diabetis mellitus tipo 2", C.DIABETES, ("diabetis", "diabetes")),
        ("eppoc", C.PULMONAR_CRONICA, ("eppoc", "epoc")),
        ("EPPOC reagudizado", C.PULMONAR_CRONICA, ("eppoc", "epoc")),
        ("cirrocis", C.HEPATICA_LEVE, ("cirrocis", "cirrosis")),
        ("insuficiencia cardiacca", C.INSUFICIENCIA_CARDIACA, ("cardiacca", "cardiaca")),
        ("insufciencia renal crónica", C.RENAL_CRONICA, ("insufciencia", "insuficiencia")),
        ("carcinomma de mama", C.TUMOR_MALIGNO, ("carcinomma", "carcinoma")),
        ("artritis reumatiode", C.REUMATOLOGICA, ("reumatiode", "reumatoide")),
        ("infatro agudo de miocardio", C.INFARTO_MIOCARDIO, ("infatro", "infarto")),
        ("Hemipleja derecha", C.HEMIPLEJIA, ("hemipleja", "hemiplejia")),
    ],
)
def test_corrige_una_letra(texto, categoria, correccion_esperada):
    mapeo = correccion.mapear(texto)

    assert mapeo is not None
    assert mapeo.categoria is categoria
    assert mapeo.correcciones == (correccion_esperada,)
    assert mapeo.por_correccion


def test_sin_errores_no_hay_correcciones():
    mapeo = correccion.mapear("diabetes mellitus tipo 2")

    assert mapeo == correccion.Mapeo(C.DIABETES, "E11.9")
    assert not mapeo.por_correccion


def test_corrige_varias_palabras_de_la_misma_frase():
    mapeo = correccion.mapear("enfermedad pulmonr obtructiva cronica")

    assert mapeo is not None and mapeo.categoria is C.PULMONAR_CRONICA
    assert mapeo.correcciones == (("pulmonr", "pulmonar"), ("obtructiva", "obstructiva"))


# --- Lo que no se corrige ------------------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    [
        "época de lluvias",  # palabra común a distancia 1 de "epoc"
        "penal",  # a distancia 1 de "renal"
        "dolor torácico",
        "cefalea tensional",
        "neumonía",
        "hta",  # menos de 5 letras
        "diabeteeess",  # a distancia 3
        "crónico",  # cambio de género, no error
        "úlcera séptica",  # "septica" no es "peptica"
        "sidra",  # a distancia 1 de "sida", que solo acepta una letra repetida
        "saida",
        "arteritis temporal",
        "lesión en la periferia",
    ],
)
def test_no_inventa_enfermedades(texto):
    assert correccion.mapear(texto) is None
    assert correccion.corregir(texto)[1] == ()


def test_palabras_cortas_no_se_corrigen():
    assert correccion.corregir("lupu")[1] == ()  # 4 letras


def test_ambigua_entre_dos_palabras_no_se_corrige(monkeypatch):
    monkeypatch.setattr(correccion, "VOCABULARIO", frozenset({"cancer", "cancel"}))
    assert correccion.corregir("cancen")[1] == ()


def test_misma_palabra_en_otro_genero_no_es_ambigua():
    # "cardiacs" está a distancia 1 de "cardiaca" y de "cardiaco": es la misma palabra.
    texto, correcciones = correccion.corregir("insuficiencia cardiacs")
    assert correcciones == (("cardiacs", "cardiaca"),)
    assert texto == "insuficiencia cardiaca"


def test_la_exclusion_sigue_aplicando_despues_de_corregir():
    # "diabetis gestacional": se corrige a "diabetes gestacional", que la regla excluye.
    assert correccion.mapear("diabetis gestacional") is None


def test_la_correccion_es_rapida():
    textos = ["insufciencia cardiacca congestiva descompensada en paciente anciano"] * 200
    inicio = time.perf_counter()
    for texto in textos:
        correccion.mapear(texto)
    assert time.perf_counter() - inicio < 1.0


@pytest.mark.parametrize("palabra", sorted(correccion.PALABRAS_COMUNES))
def test_cada_palabra_comun_estaria_a_distancia_1_de_una_clave(palabra, monkeypatch):
    """La lista protege algo: sin ella, la palabra se corregiría."""
    assert correccion._corregir_palabra(palabra) is None
    monkeypatch.setattr(correccion, "PALABRAS_COMUNES", frozenset())
    assert correccion._corregir_palabra(palabra) is not None or len(palabra) < 5


@pytest.mark.parametrize(
    ("palabra", "esperada"),
    [("eppoc", "epoc"), ("epocc", "epoc"), ("siida", "sida"), ("sidda", "sida")],
)
def test_claves_de_4_letras_solo_con_letra_repetida(palabra, esperada):
    assert correccion._corregir_palabra(palabra) == esperada


@pytest.mark.parametrize("palabra", ["sidra", "saida", "silda", "epcoc", "epoca"])
def test_claves_de_4_letras_no_aceptan_otros_errores(palabra):
    assert correccion._corregir_palabra(palabra) is None


def test_terminos_clinicos_bien_escritos_no_cambian_la_categoria():
    # "neuropatia" no se corrige a "nefropatia": da la complicación neurológica (E11.4).
    mapeo = correccion.mapear("diabetis con neuropatia")
    assert mapeo is not None
    assert (mapeo.categoria, mapeo.cie10) == (C.DIABETES_COMPLICADA, "E11.4")
    assert mapeo.correcciones == (("diabetis", "diabetes"),)


def test_solo_se_reportan_las_correcciones_necesarias():
    # "malignno" se corrige a "maligno", pero la categoría (diabetes) sale igual sin eso.
    assert correccion.corregir("diabetis mellitus, tumor malignno?")[1] == (
        ("diabetis", "diabetes"),
        ("malignno", "maligno"),
    )
    mapeo = correccion.mapear("diabetis mellitus, tumor malignno?")
    assert mapeo is not None and mapeo.categoria is C.DIABETES
    assert mapeo.correcciones == (("diabetis", "diabetes"),)


def test_las_dos_correcciones_necesarias_se_reportan():
    mapeo = correccion.mapear("insufciencia cardiacca")
    assert mapeo is not None and mapeo.categoria is C.INSUFICIENCIA_CARDIACA
    assert mapeo.correcciones == (("insufciencia", "insuficiencia"), ("cardiacca", "cardiaca"))


def test_cada_palabra_corregida_se_reporta_una_vez():
    texto, correcciones = correccion.corregir("diabetis tipo 2, diabetis descompensada")
    assert correcciones == (("diabetis", "diabetes"),)
    assert texto == "diabetes tipo 2, diabetes descompensada"


@pytest.mark.parametrize(
    ("texto", "categoria"),
    [
        ("enfermedad bascular periferica", C.VASCULAR_PERIFERICA),  # b/v
        ("tumor malino", C.TUMOR_MALIGNO),
        ("insuficiencia reinal cronica", C.RENAL_CRONICA),
    ],
)
def test_palabras_raras_del_diccionario_se_corrigen(texto, categoria):
    mapeo = correccion.mapear(texto)
    assert mapeo is not None and mapeo.categoria is categoria
