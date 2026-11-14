"""Tests de la expansión de siglas clínicas. Todos los textos son sintéticos."""

import pytest

from ner_clinic.siglas import SIGLAS, expandir


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("HTA", ["hipertensión arterial"]),
        ("hta", ["hipertensión arterial"]),
        ("HTA descompensada", ["hipertensión arterial descompensada"]),
        ("DM2", ["diabetes mellitus tipo 2"]),
        ("DM tipo 2", ["diabetes mellitus tipo 2"]),
        ("EPOC", ["enfermedad pulmonar obstructiva crónica"]),
        ("IAM", ["infarto agudo de miocardio"]),
        ("VIH (+)", ["virus de la inmunodeficiencia humana (+)"]),
        ("HTA y DM2", ["hipertensión arterial y diabetes mellitus tipo 2"]),
    ],
)
def test_expande_las_siglas_conservando_el_resto(texto, esperado):
    assert expandir(texto) == esperado


def test_sigla_con_varios_nombres_da_una_expansion_por_nombre():
    assert expandir("TEP") == ["tromboembolismo pulmonar", "embolia pulmonar"]


@pytest.mark.parametrize("texto", ["", "diabetes mellitus", "hipertensión", "HTAP2"])
def test_sin_siglas_no_hay_expansiones(texto):
    assert expandir(texto) == []


def test_siglas_que_son_palabras_comunes_exigen_mayusculas():
    # "les" es también un pronombre: solo cuenta como sigla en mayúsculas.
    assert expandir("les") == []
    assert expandir("LES") == ["lupus eritematoso sistémico"]


@pytest.mark.parametrize(
    "sigla",
    [
        "HTA",
        "DM",
        "DM1",
        "DM2",
        "EPOC",
        "IAM",
        "ACV",
        "ECV",
        "ICC",
        "IC",
        "ERC",
        "IRC",
        "FA",
        "TEP",
        "TVP",
        "VIH",
        "LES",
    ],
)
def test_tabla_incluye_las_siglas_frecuentes(sigla):
    assert SIGLAS[sigla]
