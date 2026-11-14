"""Tests de las métricas de NER: emparejamiento, micro P/R/F1, bootstrap y matriz de confusión.

Todos los casos son spans escritos a mano, sin texto de pacientes.
"""

import pytest

from evaluacion.metricas import (
    PRF,
    Conteos,
    Emparejamiento,
    Intervalo,
    Span,
    bootstrap_diferencia,
    bootstrap_ic,
    emparejar,
    limites,
    matriz_confusion,
    micro_prf,
)
from ner_clinic.tipos import Mencion

# --- limites ---------------------------------------------------------------------------


def test_limites_acepta_tuplas_y_objetos_con_inicio_y_fin():
    assert limites((3, 8)) == Span(3, 8)
    assert limites(Span(3, 8)) == Span(3, 8)
    assert limites(Mencion("asma", 10, 14)) == Span(10, 14)


# --- emparejar: estricto ---------------------------------------------------------------


def test_estricto_spans_identicos_son_verdaderos_positivos():
    gold = [Span(0, 4), Span(10, 20)]

    resultado = emparejar(gold, [Span(10, 20), Span(0, 4)])

    assert resultado.conteos == Conteos(vp=2, fp=0, fn=0)
    assert resultado.verdaderos == [(Span(0, 4), Span(0, 4)), (Span(10, 20), Span(10, 20))]


def test_estricto_span_desplazado_es_un_fp_y_un_fn():
    resultado = emparejar([Span(0, 10)], [Span(1, 11)])

    assert resultado.conteos == Conteos(vp=0, fp=1, fn=1)
    assert resultado.falsos_positivos == [Span(1, 11)]
    assert resultado.falsos_negativos == [Span(0, 10)]


def test_estricto_span_contenido_no_cuenta():
    assert emparejar([Span(0, 20)], [Span(5, 10)]).conteos == Conteos(0, 1, 1)


def test_estricto_es_el_modo_por_defecto():
    assert emparejar([Span(0, 10)], [Span(0, 9)]) == emparejar(
        [Span(0, 10)], [Span(0, 9)], modo="estricto"
    )


def test_predicciones_duplicadas_cuentan_una_vez():
    resultado = emparejar([Span(0, 4)], [Span(0, 4), Span(0, 4), (0, 4)])

    assert resultado.conteos == Conteos(vp=1, fp=0, fn=0)


def test_predicciones_duplicadas_sin_gold_son_un_solo_fp():
    resultado = emparejar([], [Mencion("asma", 0, 4), Mencion("Asma", 0, 4)], modo="laxo")

    assert resultado.conteos == Conteos(vp=0, fp=1, fn=0)
    assert resultado.falsos_positivos == [Mencion("asma", 0, 4)]


def test_gold_duplicado_no_se_deduplica():
    resultado = emparejar([Span(0, 4), Span(0, 4)], [Span(0, 4)])

    assert resultado.conteos == Conteos(vp=1, fp=0, fn=1)


def test_conserva_las_menciones_recibidas():
    gold = Mencion("diabetes", 0, 8)
    pred = Mencion("diabetes", 0, 8)

    resultado = emparejar([gold], [pred])

    assert resultado.verdaderos[0][0] is gold
    assert resultado.verdaderos[0][1] is pred


# --- emparejar: laxo -------------------------------------------------------------------


def test_laxo_span_desplazado_con_solapamiento_cuenta():
    assert emparejar([Span(0, 10)], [Span(5, 15)], modo="laxo").conteos == Conteos(1, 0, 0)


def test_laxo_span_contenido_cuenta():
    assert emparejar([Span(0, 20)], [Span(5, 10)], modo="laxo").conteos == Conteos(1, 0, 0)


def test_laxo_spans_que_solo_se_tocan_no_se_solapan():
    resultado = emparejar([Span(0, 10)], [Span(10, 20)], modo="laxo")

    assert resultado.conteos == Conteos(0, 1, 1)


def test_laxo_una_prediccion_que_cubre_dos_gold_acierta_solo_una():
    gold = [Span(0, 8), Span(10, 30)]

    resultado = emparejar(gold, [Span(0, 30)], modo="laxo")

    assert resultado.conteos == Conteos(vp=1, fp=0, fn=1)
    # Se prefiere el mayor solapamiento: 20 caracteres contra 8
    assert resultado.verdaderos == [(Span(10, 30), Span(0, 30))]
    assert resultado.falsos_negativos == [Span(0, 8)]


def test_laxo_un_gold_cubierto_por_dos_predicciones_empareja_con_una():
    resultado = emparejar([Span(0, 30)], [Span(0, 5), Span(10, 30)], modo="laxo")

    assert resultado.conteos == Conteos(vp=1, fp=1, fn=0)
    assert resultado.verdaderos == [(Span(0, 30), Span(10, 30))]
    assert resultado.falsos_positivos == [Span(0, 5)]


def test_laxo_prefiere_el_mayor_solapamiento_para_no_perder_parejas():
    # La predicción larga se solapa más con el segundo gold; la corta solo con el primero
    gold = [Span(0, 10), Span(12, 30)]
    pred = [Span(2, 8), Span(5, 30)]

    resultado = emparejar(gold, pred, modo="laxo")

    assert resultado.conteos == Conteos(vp=2, fp=0, fn=0)
    assert resultado.verdaderos == [(Span(0, 10), Span(2, 8)), (Span(12, 30), Span(5, 30))]


def test_laxo_empate_de_solapamiento_gana_el_primer_gold():
    resultado = emparejar([Span(0, 10), Span(20, 30)], [Span(5, 25)], modo="laxo")

    assert resultado.verdaderos == [(Span(0, 10), Span(5, 25))]
    assert resultado.falsos_negativos == [Span(20, 30)]


def test_documentos_vacios():
    for modo in ("estricto", "laxo"):
        assert emparejar([], [], modo=modo) == Emparejamiento()
        assert emparejar([], [], modo=modo).conteos == Conteos(0, 0, 0)
        assert emparejar([Span(0, 4)], [], modo=modo).conteos == Conteos(0, 0, 1)
        assert emparejar([], [Span(0, 4)], modo=modo).conteos == Conteos(0, 1, 0)


def test_modo_desconocido_falla():
    with pytest.raises(ValueError, match="modo desconocido"):
        emparejar([], [], modo="parcial")


# --- micro_prf -------------------------------------------------------------------------


def test_micro_prf_suma_los_conteos_de_todos_los_documentos():
    precision, recall, f1 = micro_prf([Conteos(3, 1, 0), (1, 1, 4)])

    # vp=4, fp=2, fn=4
    assert precision == pytest.approx(4 / 6)
    assert recall == pytest.approx(4 / 8)
    assert f1 == pytest.approx(2 * (4 / 6) * (4 / 8) / (4 / 6 + 4 / 8))


def test_micro_prf_perfecto():
    assert micro_prf([Conteos(5, 0, 0)]) == PRF(1.0, 1.0, 1.0)


def test_micro_prf_con_denominador_cero_da_cero():
    assert micro_prf([]) == PRF(0.0, 0.0, 0.0)
    assert micro_prf([Conteos(0, 0, 0)]) == PRF(0.0, 0.0, 0.0)
    assert micro_prf([Conteos(0, 3, 0)]) == PRF(0.0, 0.0, 0.0)
    assert micro_prf([Conteos(0, 0, 2)]) == PRF(0.0, 0.0, 0.0)


# --- bootstrap_ic ----------------------------------------------------------------------

DOCUMENTOS = [Conteos(3, 1, 0), Conteos(1, 2, 2), Conteos(4, 0, 1), Conteos(0, 1, 1)]


def test_bootstrap_es_determinista_con_la_misma_semilla():
    assert bootstrap_ic(DOCUMENTOS, n=200, semilla=7) == bootstrap_ic(DOCUMENTOS, n=200, semilla=7)


def test_bootstrap_cambia_con_otra_semilla():
    assert bootstrap_ic(DOCUMENTOS, n=200, semilla=1) != bootstrap_ic(DOCUMENTOS, n=200, semilla=2)


def test_bootstrap_el_intervalo_contiene_la_estimacion_puntual():
    puntual = micro_prf(DOCUMENTOS)
    intervalos = bootstrap_ic(DOCUMENTOS, n=500)

    for estimacion, intervalo in zip(puntual, intervalos, strict=True):
        assert 0.0 <= intervalo.inferior <= estimacion <= intervalo.superior <= 1.0


def test_bootstrap_nivel_mayor_da_intervalo_mas_ancho():
    angosto = bootstrap_ic(DOCUMENTOS, n=500, nivel=0.5)
    ancho = bootstrap_ic(DOCUMENTOS, n=500, nivel=0.99)

    assert ancho.f1.superior - ancho.f1.inferior >= angosto.f1.superior - angosto.f1.inferior


def test_bootstrap_con_un_documento_da_intervalo_de_ancho_cero():
    intervalos = bootstrap_ic([Conteos(3, 1, 1)], n=50)

    assert intervalos.precision == Intervalo(0.75, 0.75)
    assert intervalos.recall == Intervalo(0.75, 0.75)


def test_bootstrap_sin_documentos_da_ceros():
    cero = Intervalo(0.0, 0.0)
    assert tuple(bootstrap_ic([])) == (cero, cero, cero)


@pytest.mark.parametrize(("n", "nivel"), [(0, 0.95), (100, 0.0), (100, 1.0)])
def test_bootstrap_parametros_invalidos_fallan(n, nivel):
    with pytest.raises(ValueError):
        bootstrap_ic(DOCUMENTOS, n=n, nivel=nivel)


def test_bootstrap_acepta_los_conteos_de_emparejar():
    conteos = [
        emparejar([Span(0, 4)], [Span(0, 4)]).conteos,
        emparejar([Span(0, 4)], [Span(5, 9)]).conteos,
    ]

    assert bootstrap_ic(conteos, n=100) == bootstrap_ic([(1, 0, 0), (0, 1, 1)], n=100)


# --- matriz_confusion ------------------------------------------------------------------

CLASES = ["confirmada", "negada", "familiar"]


def test_matriz_confusion_cuenta_los_pares():
    esperados = ["confirmada", "confirmada", "negada", "familiar", "negada"]
    obtenidos = ["confirmada", "negada", "negada", "confirmada", "negada"]

    matriz = matriz_confusion(esperados, obtenidos, CLASES)

    assert matriz == {
        "confirmada": {"confirmada": 1, "negada": 1, "familiar": 0},
        "negada": {"confirmada": 0, "negada": 2, "familiar": 0},
        "familiar": {"confirmada": 1, "negada": 0, "familiar": 0},
    }
    assert list(matriz) == CLASES
    assert list(matriz["negada"]) == CLASES


def test_matriz_confusion_vacia_tiene_ceros():
    matriz = matriz_confusion([], [], CLASES)

    assert all(valor == 0 for fila in matriz.values() for valor in fila.values())


def test_matriz_confusion_largos_distintos_falla():
    with pytest.raises(ValueError, match="largo"):
        matriz_confusion(["negada"], [], CLASES)


@pytest.mark.parametrize(
    ("esperados", "obtenidos"), [(["otra"], ["negada"]), (["negada"], ["otra"])]
)
def test_matriz_confusion_clase_desconocida_falla(esperados, obtenidos):
    with pytest.raises(ValueError, match="clase desconocida"):
        matriz_confusion(esperados, obtenidos, CLASES)


def test_bootstrap_diferencia_pareado():
    a = [(1, 0, 1), (1, 0, 2), (2, 0, 0)]
    b = [(2, 0, 0), (2, 0, 1), (2, 0, 0)]
    ic = bootstrap_diferencia(a, b, n=200, semilla=0)
    # b nunca es peor que a en un documento: en ninguna réplica pierde recall
    assert ic.recall.inferior >= 0 and ic.recall.superior > 0
    assert ic.precision == (0.0, 0.0)  # ninguno tiene falsos positivos
    igual = bootstrap_diferencia(a, a, n=50)
    assert igual.f1 == (0.0, 0.0)
    assert bootstrap_diferencia([], [], n=5).f1 == (0.0, 0.0)
    with pytest.raises(ValueError, match="mismos documentos"):
        bootstrap_diferencia(a, b[:2])
