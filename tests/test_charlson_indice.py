"""Tests del cálculo del índice de Charlson (issue #13)."""

import pytest

from ner_clinic.charlson.indice import PESOS, calcular, calcular_desde_categorias
from ner_clinic.tipos import CategoriaCharlson as C
from ner_clinic.tipos import Entidad, Estado


def _entidad(categoria: C | None, estado: Estado = Estado.CONFIRMADA, texto: str = "x") -> Entidad:
    return Entidad(texto=texto, menciones=[], estado=estado, categoria_charlson=categoria)


# --- Tabla de pesos ---------------------------------------------------------


def test_pesos_cubre_exactamente_las_17_categorias():
    assert set(PESOS) == set(C)
    assert len(PESOS) == len(list(C)) == 17


@pytest.mark.parametrize(
    ("categoria", "peso"),
    [
        (C.INFARTO_MIOCARDIO, 1),
        (C.INSUFICIENCIA_CARDIACA, 1),
        (C.VASCULAR_PERIFERICA, 1),
        (C.CEREBROVASCULAR, 1),
        (C.DEMENCIA, 1),
        (C.PULMONAR_CRONICA, 1),
        (C.REUMATOLOGICA, 1),
        (C.ULCERA_PEPTICA, 1),
        (C.HEPATICA_LEVE, 1),
        (C.DIABETES, 1),
        (C.DIABETES_COMPLICADA, 2),
        (C.HEMIPLEJIA, 2),
        (C.RENAL_CRONICA, 2),
        (C.TUMOR_MALIGNO, 2),
        (C.HEPATICA_SEVERA, 3),
        (C.TUMOR_METASTASICO, 6),
        (C.VIH_SIDA, 6),
    ],
)
def test_pesos_estandar_charlson_1987(categoria, peso):
    assert PESOS[categoria] == peso
    assert calcular_desde_categorias([categoria]) == peso


# --- Cálculo desde categorías -----------------------------------------------


def test_sin_categorias_da_cero():
    assert calcular_desde_categorias([]) == 0


def test_suma_categorias_distintas():
    assert calcular_desde_categorias([C.INFARTO_MIOCARDIO, C.RENAL_CRONICA, C.VIH_SIDA]) == 9


def test_cada_categoria_cuenta_una_sola_vez():
    assert calcular_desde_categorias([C.DEMENCIA, C.DEMENCIA, C.DEMENCIA]) == 1


@pytest.mark.parametrize(
    ("leve", "severa", "peso_severa"),
    [
        (C.DIABETES, C.DIABETES_COMPLICADA, 2),
        (C.HEPATICA_LEVE, C.HEPATICA_SEVERA, 3),
        (C.TUMOR_MALIGNO, C.TUMOR_METASTASICO, 6),
    ],
)
def test_jerarquia_cuenta_solo_la_forma_de_mas_peso(leve, severa, peso_severa):
    assert calcular_desde_categorias([leve, severa]) == peso_severa
    assert calcular_desde_categorias([severa, leve]) == peso_severa


def test_jerarquias_combinadas_con_otras_categorias():
    categorias = [
        C.DIABETES,
        C.DIABETES_COMPLICADA,
        C.HEPATICA_LEVE,
        C.HEPATICA_SEVERA,
        C.TUMOR_MALIGNO,
        C.TUMOR_METASTASICO,
        C.INSUFICIENCIA_CARDIACA,
    ]
    assert calcular_desde_categorias(categorias) == 2 + 3 + 6 + 1


def test_todas_las_categorias_da_el_maximo_sin_formas_leves():
    # 17 categorías suman 33; se descuentan diabetes (1), hepática leve (1) y tumor (2).
    assert calcular_desde_categorias(list(C)) == 29


# --- Cálculo desde entidades ------------------------------------------------


def test_calcular_suma_entidades_confirmadas():
    entidades = [_entidad(C.INFARTO_MIOCARDIO), _entidad(C.HEMIPLEJIA)]
    assert calcular(entidades) == 3


def test_negadas_y_familiares_no_suman():
    entidades = [
        _entidad(C.INFARTO_MIOCARDIO),
        _entidad(C.VIH_SIDA, Estado.NEGADA),
        _entidad(C.TUMOR_METASTASICO, Estado.FAMILIAR),
    ]
    assert calcular(entidades) == 1


def test_forma_severa_negada_no_anula_la_leve_confirmada():
    entidades = [
        _entidad(C.DIABETES),
        _entidad(C.DIABETES_COMPLICADA, Estado.NEGADA),
    ]
    assert calcular(entidades) == 1


def test_posibles_no_suman_al_indice():
    entidades = [_entidad(C.DIABETES), _entidad(C.TUMOR_METASTASICO, Estado.POSIBLE)]
    assert calcular(entidades) == 1


def test_con_posibles_suma_confirmadas_y_posibles():
    entidades = [
        _entidad(C.DIABETES),
        _entidad(C.TUMOR_METASTASICO, Estado.POSIBLE),
        _entidad(C.VIH_SIDA, Estado.NEGADA),
        _entidad(C.DEMENCIA, Estado.FAMILIAR),
    ]
    assert calcular(entidades, con_posibles=True) == 7


def test_con_posibles_aplica_la_misma_jerarquia():
    # La forma severa posible deja sin peso a la leve confirmada, y al revés no suma doble.
    entidades = [_entidad(C.TUMOR_MALIGNO), _entidad(C.TUMOR_METASTASICO, Estado.POSIBLE)]
    assert calcular(entidades) == 2
    assert calcular(entidades, con_posibles=True) == 6
    entidades = [_entidad(C.DIABETES_COMPLICADA), _entidad(C.DIABETES, Estado.POSIBLE)]
    assert calcular(entidades, con_posibles=True) == 2


def test_con_solo_confirmadas_los_dos_indices_son_iguales():
    entidades = [_entidad(C.INFARTO_MIOCARDIO), _entidad(C.HEMIPLEJIA, Estado.NEGADA)]
    assert calcular(entidades) == calcular(entidades, con_posibles=True) == 1


def test_entidades_sin_categoria_se_ignoran():
    entidades = [_entidad(None), _entidad(C.DEMENCIA), _entidad(None, Estado.NEGADA)]
    assert calcular(entidades) == 1


def test_jerarquia_entre_entidades_distintas_de_la_misma_nota():
    entidades = [
        _entidad(C.TUMOR_MALIGNO, texto="adenocarcinoma de colon"),
        _entidad(C.TUMOR_METASTASICO, texto="metastasis hepaticas"),
    ]
    assert calcular(entidades) == 6


def test_calcular_acepta_cualquier_iterable():
    assert calcular(e for e in [_entidad(C.ULCERA_PEPTICA)]) == 1
