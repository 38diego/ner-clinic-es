"""Tests de las 17 reglas de Charlson unidas: orden, sobrescritura y casos entre partes.

Todos los textos son sintéticos. Los resultados esperados de los casos entre partes son
los de la tesis, salvo las correcciones de #24.
"""

import pytest

from ner_clinic.charlson.reglas import REGLAS, mapear
from ner_clinic.tipos import CategoriaCharlson as C


def test_reglas_son_las_17_categorias_en_el_orden_de_la_tesis():
    assert isinstance(REGLAS, tuple)
    assert [r.categoria for r in REGLAS] == list(C)


def test_solo_sobrescriben_diabetes_complicada_hepatica_severa_y_metastasis():
    assert [r.categoria for r in REGLAS if r.sobrescribe] == [
        C.DIABETES_COMPLICADA,
        C.HEPATICA_SEVERA,
        C.TUMOR_METASTASICO,
    ]


@pytest.mark.parametrize(
    ("texto", "categoria", "cie10"),
    [
        # Hepática leve (parte 1) asigna; hepática severa (parte 2) la sobrescribe
        ("varices paraesofagicas por cirrosis", C.HEPATICA_SEVERA, "I98.2"),
        # IAM (parte 1) asigna; diabetes complicada (parte 2) lo sobrescribe
        ("infarto agudo de miocardio con nefropatia diabetica", C.DIABETES_COMPLICADA, "E11.2"),
        # IAM (parte 1) asigna; tumor metastásico (parte 2) lo sobrescribe.
        # La tesis daba C80 porque buscaba "oseo" y no "oseas"; corregido en #24.
        ("infarto agudo de miocardio con metastasis oseas", C.TUMOR_METASTASICO, "C79"),
    ],
)
def test_casos_entre_partes_dan_el_resultado_de_la_tesis(texto, categoria, cie10):
    assert mapear(texto) == (categoria, cie10)


def test_regla_normal_de_la_parte2_no_sobrescribe_a_la_parte1():
    # IAM (parte 1) asigna primero; VIH (parte 2) es normal y no lo cambia.
    assert mapear("infarto agudo de miocardio en paciente con vih") == (
        C.INFARTO_MIOCARDIO,
        "I21",
    )
