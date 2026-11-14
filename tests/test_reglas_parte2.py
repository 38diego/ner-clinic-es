"""Reglas de Charlson, parte 2: las 8 categorías portadas de la tesis.

Todos los textos son sintéticos.
"""

import pytest

from ner_clinic.charlson.reglas import REGLAS_PARTE2, mapear, normalizar
from ner_clinic.charlson.reglas_parte2 import (
    subclasificar_cie10_hemiplejia,
    subclasificar_cie10_renal,
)
from ner_clinic.tipos import CategoriaCharlson as C

# --- normalizar ---


def test_normalizar_pasa_a_minusculas_y_quita_tildes():
    assert normalizar("Diabetes Mellitus TIPO 2 Insulinorrequiriente") == (
        "diabetes mellitus tipo 2 insulinorrequiriente"
    )
    assert normalizar("Nefropatía diabética") == "nefropatia diabetica"


def test_normalizar_quita_caracteres_de_control_y_colapsa_espacios():
    assert normalizar("  vih\x07   en\ttratamiento \n ") == "vih en tratamiento"


# --- estructura ---


def test_reglas_en_el_orden_de_la_tesis():
    assert [r.categoria for r in REGLAS_PARTE2] == [
        C.DIABETES,
        C.DIABETES_COMPLICADA,
        C.HEMIPLEJIA,
        C.RENAL_CRONICA,
        C.TUMOR_MALIGNO,
        C.HEPATICA_SEVERA,
        C.TUMOR_METASTASICO,
        C.VIH_SIDA,
    ]


def test_texto_sin_comorbilidad_de_la_parte2_no_mapea():
    assert mapear("hipertension arterial") is None
    assert mapear("") is None


def _casos(categoria, pares):
    return [pytest.param(texto, categoria, cie10, id=texto) for texto, cie10 in pares]


# --- diabetes sin complicación (subclasificar_cie10_diabetes) ---

DIABETES = _casos(
    C.DIABETES,
    [
        # tipo 1
        ("diabetes mellitus tipo 1 en coma", "E10.0"),
        ("dm1 con cetoacidosis", "E10.1"),
        ("diabetes juvenil descompensada", "E10.6"),
        ("dm tipo 1", "E10.9"),
        # tipo 2
        ("diabetes tipo 2 con coma", "E11.0"),
        ("dm2 con cetosis", "E11.1"),
        ("dmt2 con hiperglicemia", "E11.6"),
        ("diabetes mellitus tipo 2 insulinorrequiriente", "E11.9"),
        # sin tipo (la tesis daba E11.0 y E11.1; corregido en #24)
        ("diabetes en coma", "E14.0"),
        ("cetoacidosis diabetica", "E14.1"),
        ("diabetes mal control", "E14.6"),
        ("diabetes mellitus", "E14.9"),
    ],
)

# --- diabetes con daño de órgano blanco (subclasificar_cie10_diabetes_cronica) ---

DIABETES_COMPLICADA = _casos(
    C.DIABETES_COMPLICADA,
    [
        ("nefropatia diabetica", "E11.2"),
        ("retinopatia diabetica", "E11.3"),
        ("neuropatia diabetica", "E11.4"),
        ("pie diabetico", "E11.5"),
        ("dm1 con enfermedad renal", "E10.2"),
        ("diabetes con complicaciones microvasculares", "E11.7"),
        ("dm2 con nefropatia y retinopatia diabetica", "E11.7"),
        ("diabetes con compromiso de organo blanco", "E11.8"),
    ],
)

# --- hemiplejía / paraplejía (subclasificar_cie10_hemiplejia) ---

HEMIPLEJIA = _casos(
    C.HEMIPLEJIA,
    [
        ("paraplejia secuelar", "G82"),
        ("Tetraparesia espástica", "G82"),
        ("hemiparesia derecha", "G81"),
        ("mielitis transversa", "G04.1"),
        ("paralisis cerebral espastica", "G80.1"),
        ("paralisis cerebral hemiparetica", "G80.2"),
        ("sindrome de cauda equina", "G83.4"),
        ("monoparesia de brazo derecho", "G83.2"),
        ("monoplejia de pierna izquierda", "G83.1"),
        ("monoplejia", "G83.3"),
        ("diplejia espastica", "G83.0"),
    ],
)

# --- enfermedad renal crónica (subclasificar_cie10_renal) ---

RENAL = _casos(
    C.RENAL_CRONICA,
    [
        ("trasplante renal", "Z94.0"),
        ("hemodialisis tres veces por semana", "Z99.2"),
        ("erc estadio 4", "N18"),
        ("nefropatia hipertensiva", "I12.0"),
        ("glomerulonefritis cronica", "N18"),
        ("nefropatia por iga", "N03.9"),
        ("osteodistrofia renal", "N25.0"),
    ],
)

# --- tumor maligno (subclasificar_cie10_cancer) ---

CANCER = _casos(
    C.TUMOR_MALIGNO,
    [
        ("leucemia linfoblastica aguda", "C91"),
        ("leucemia mieloide cronica", "C92"),
        ("leucemia", "C95"),
        ("linfoma de hodgkin", "C81"),
        ("linfoma no hodgkin", "C85"),
        ("mieloma multiple", "C90"),
        ("sarcoma de kaposi", "C46"),
        ("sarcoma de partes blandas", "C49"),
        ("carcinoma ductal de mama", "C50"),
        ("cancer de cervix", "C53"),
        ("adenocarcinoma de endometrio", "C54"),
        ("cancer de ovario", "C56"),
        ("adenocarcinoma de prostata", "C61"),
        ("cancer de testiculo", "C62"),
        ("adenocarcinoma de colon", "C18"),
        ("ca gastrico", "C16"),
        ("carcinoma de esofago", "C15"),
        ("adenocarcinoma de pulmon", "C34"),
        ("cancer de laringe", "C32"),
        ("carcinoma de lengua", "C14"),
        ("carcinoma urotelial de vejiga", "C67"),
        ("carcinoma de celulas claras", "C64"),
        ("cancer de tiroides", "C73"),
        ("adenocarcinoma de pancreas", "C25"),
        ("hepatocarcinoma", "C22"),
        ("glioblastoma", "C71"),
        ("sarcoma", "C49"),
        ("cancer de hueso", "C41"),
        ("melanoma", "C43"),
        ("neoplasia maligna", "C80"),
    ],
)

# --- enfermedad hepática severa (subclasificar_cie10_hepatica_severa) ---

HEPATICA_SEVERA = _casos(
    C.HEPATICA_SEVERA,
    [
        ("varices esofagicas con sangrado", "I85.0"),
        ("varices esofagicas pequeñas", "I85.9"),
        ("varices gastricas", "I86.4"),
        ("varices paraesofagicas por cirrosis", "I98.2"),
        ("hipertension portal", "K76.6"),
        ("sindrome hepatorrenal", "K76.7"),
        ("falla hepatica alcoholica", "K70.4"),
        ("necrosis hepatica", "K71.1"),
        ("enfermedad veno-oclusiva hepatica", "K76.5"),
        ("encefalopatia hepatica grado 2", "K72.1"),
        ("insuficiencia hepatica aguda", "K72.9"),
        ("varices paraesofagicas", "K72.9"),
        ("west haven 3", "K72.1"),
    ],
)

# --- tumor metastásico (subclasificar_cie10_metastasis) ---

METASTASIS = _casos(
    C.TUMOR_METASTASICO,
    [
        ("metastasis ganglionares", "C77"),
        ("metastasis hepaticas", "C78"),
        ("metastasis en hueso", "C79"),
        ("enfermedad diseminada", "C80"),
    ],
)

# --- VIH / SIDA (subclasificar_cie10_vih) ---

VIH = _casos(
    C.VIH_SIDA,
    [
        ("vih con tuberculosis pulmonar", "B20"),
        ("sida con sindrome de desgaste", "B22"),
        ("VIH en tratamiento antirretroviral", "B24"),
    ],
)


@pytest.mark.parametrize(
    ("texto", "categoria", "cie10"),
    DIABETES
    + DIABETES_COMPLICADA
    + HEMIPLEJIA
    + RENAL
    + CANCER
    + HEPATICA_SEVERA
    + METASTASIS
    + VIH,
)
def test_mapea_categoria_y_codigo(texto, categoria, cie10):
    assert mapear(texto) == (categoria, cie10)


# --- exclusiones: el texto cumple la inclusión pero cae en la exclusión ---

EXCLUSIONES = [
    # diabetes: obstetricia, parentesco, negación
    ("diabetes gestacional", C.DIABETES),
    ("madre con diabetes", C.DIABETES),
    ("niega diabetes", C.DIABETES),
    # diabetes complicada: prediabetes, negación
    ("prediabetes con compromiso renal", C.DIABETES_COMPLICADA),
    ("diabetes sin nefropatia ni complicacion renal", C.DIABETES_COMPLICADA),
    # hemiplejía: territorio facial, negación
    ("hemiparesia facial", C.HEMIPLEJIA),
    ("sin hemiparesia", C.HEMIPLEJIA),
    # renal: lesión aguda, cáncer, negación
    ("erc con lra", C.RENAL_CRONICA),
    ("tumor renal en hemodialisis", C.RENAL_CRONICA),
    ("niega erc", C.RENAL_CRONICA),
    # tumor maligno: piel, sospecha, in situ, parentesco
    ("carcinoma basocelular", C.TUMOR_MALIGNO),
    ("cancer de mama?", C.TUMOR_MALIGNO),
    ("carcinoma in situ de cervix", C.TUMOR_MALIGNO),
    ("padre con cancer de colon", C.TUMOR_MALIGNO),
    # hepática severa: várices de miembros, sospecha, negación
    ("varices en piernas y varices esofagicas", C.HEPATICA_SEVERA),
    ("probable falla hepatica", C.HEPATICA_SEVERA),
    ("sin varices esofagicas", C.HEPATICA_SEVERA),
    # metástasis: diferencial, negación
    ("metastasis vs primario", C.TUMOR_METASTASICO),
    ("libre de metastasis", C.TUMOR_METASTASICO),
    # vih: resultado negativo, hemorragia intraventricular
    ("vih negativo", C.VIH_SIDA),
    ("hiv grado ii", C.VIH_SIDA),
]


@pytest.mark.parametrize(("texto", "categoria"), EXCLUSIONES, ids=[t for t, _ in EXCLUSIONES])
def test_exclusion_bloquea_la_regla(texto, categoria):
    regla = next(r for r in REGLAS_PARTE2 if r.categoria is categoria)
    limpio = normalizar(texto)
    assert regla.inclusion.search(limpio), "el caso debe cumplir la inclusión"
    assert regla.exclusion.search(limpio), "el caso debe caer en la exclusión"
    resultado = mapear(texto)
    assert resultado is None or resultado[0] is not categoria


# --- prioridad entre reglas ---


def test_diabetes_complicada_sobrescribe_a_diabetes_simple():
    # En la tesis la regla de diabetes complicada no respeta `Mapeado_Charlson`.
    assert mapear("dm2 con enfermedad renal") == (C.DIABETES_COMPLICADA, "E11.2")


def test_metastasis_sobrescribe_a_tumor_maligno():
    assert mapear("adenocarcinoma de colon con siembra peritoneal") == (
        C.TUMOR_METASTASICO,
        "C79",
    )


def test_hepatica_severa_sobrescribe_a_renal_cronica():
    assert mapear("sindrome hepatorrenal en hemodialisis") == (
        C.HEPATICA_SEVERA,
        "K76.7",
    )


def test_regla_no_prioritaria_respeta_lo_ya_mapeado():
    # Cumple renal crónica y VIH: gana la primera (renal), como en la tesis.
    assert mapear("vih con erc estadio 3") == (C.RENAL_CRONICA, "N18")


# --- tildes y mayúsculas ---


@pytest.mark.parametrize(
    ("con_tildes", "sin_tildes"),
    [
        ("Nefropatía Diabética", "nefropatia diabetica"),
        ("HEMIPLEJÍA IZQUIERDA", "hemiplejia izquierda"),
        ("Enfermedad Renal Crónica", "enfermedad renal cronica"),
        ("Cáncer de Próstata", "cancer de prostata"),
        ("Várices Esofágicas", "varices esofagicas"),
        ("Metástasis Óseas", "metastasis oseas"),
        ("Diabetes Mellitus Tipo 2", "diabetes mellitus tipo 2"),
        ("VIH/SIDA", "vih/sida"),
    ],
)
def test_tildes_y_mayusculas_mapean_igual(con_tildes, sin_tildes):
    resultado = mapear(con_tildes)
    assert resultado is not None
    assert resultado == mapear(sin_tildes)


# --- ramas de respaldo que la inclusión nunca alcanza ---
# Todo texto que cumple la inclusión de hemiplejía o de renal crónica cae en alguna rama
# específica; los códigos de respaldo se prueban llamando la subclasificación directo.


def test_subclasificar_hemiplejia_sin_tipo_reconocido_da_g83_9():
    assert subclasificar_cie10_hemiplejia("paralisis") == "G83.9"


def test_subclasificar_renal_sin_tipo_reconocido_da_n19():
    assert subclasificar_cie10_renal("insuficiencia renal") == "N19"
