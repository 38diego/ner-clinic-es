"""Tests de `ner_clinic.normalizacion` (#163), sin red: codificador de trigramas e índice chico."""

import json
import subprocess
import sys

import pytest

from evaluacion import medicion_final as mf
from evaluacion.clasificador import decisor as decisor_evaluacion
from evaluacion.dos_etapas import mapeador_de
from ner_clinic.normalizacion import Normalizador, SegundaEtapa, modelo_a_json, modelo_de_json
from ner_clinic.normalizacion import decision as d
from ner_clinic.normalizacion.indice import ORIGEN_CATALOGO
from ner_clinic.normalizacion.jerarquia import bloque, capitulo
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_dos_etapas import TRAIN_6, _indice
from tests.test_normalizador import DEV

VARIANTES = ("original", "minusculas", "errores_10")


@pytest.fixture
def rejilla(monkeypatch):
    from tests.test_clasificador import _rejilla_chica

    _rejilla_chica(monkeypatch)


@pytest.fixture
def etapa(rejilla):
    """La segunda etapa con el modelo guardado y vuelto a leer de JSON, y lo de la
    evaluación para comparar."""
    indice, codificador = _indice()
    modelo, eleccion = mf.entrenar_final(indice, codificador, TRAIN_6, [], {}, 3, d.K)
    guardado = modelo_de_json(json.loads(json.dumps(modelo_a_json(modelo, eleccion["umbral"]))))
    final = indice.con_sinonimos(mf.sinonimos_de(TRAIN_6), codificador)
    return SegundaEtapa(Normalizador(final, codificador), *guardado), (
        indice,
        codificador,
        modelo,
        eleccion,
    )


def test_import_ner_clinic_no_carga_torch():
    codigo = "import sys, ner_clinic; print('torch' in sys.modules)"
    salida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert salida.stdout.strip() == "False"


def test_la_libreria_decide_igual_que_la_evaluacion(etapa):
    """`SegundaEtapa` (lo que usará el pipeline) da las mismas categorías y códigos que el
    método medido en #154: `mapeador_de` con el decisor del clasificador y las exclusiones."""
    segunda, (indice, codificador, modelo, eleccion) = etapa
    pliegue, _ = mf.topk_desarrollo(indice, codificador, TRAIN_6, DEV, [], ["x-1"], VARIANTES)
    textos = list(pliegue.topk)
    en_dev = mapeador_de(decisor_evaluacion(modelo, eleccion["umbral"]), pliegue.topk)
    decisiones = segunda.decidir(textos)
    obtenido = [None if (x := decisiones[t]) is None else (x.categoria, x.codigo) for t in textos]
    assert obtenido == [en_dev(t) for t in textos]
    assert any(x is not None for x in obtenido)


def test_equivalencia_con_una_exclusion(etapa, monkeypatch):
    """Si la exclusión quita la categoría ganadora, las dos rutas dejan la mención sin
    categoría (no pasan a la segunda más probable)."""
    segunda, (indice, codificador, modelo, eleccion) = etapa
    pliegue, _ = mf.topk_desarrollo(indice, codificador, TRAIN_6, DEV, [], ["x-1"], VARIANTES)
    textos = list(pliegue.topk)
    antes = segunda.decidir(textos)
    victima = next(t for t in textos if antes[t] is not None)
    ganadora = antes[victima].categoria
    monkeypatch.setattr(d, "excluida", lambda t, c: t == victima and c is ganadora)
    en_dev = mapeador_de(decisor_evaluacion(modelo, eleccion["umbral"]), pliegue.topk)
    despues = segunda.decidir(textos)
    assert despues[victima] is None and en_dev(victima) is None
    assert [None if (x := despues[t]) is None else (x.categoria, x.codigo) for t in textos] == [
        en_dev(t) for t in textos
    ]


def test_la_decision_trae_el_candidato_del_indice(etapa):
    segunda, _ = etapa
    decisiones = segunda.decidir(["infarto cerebral", "infarto cerebral", "tos"])
    assert set(decisiones) == {"infarto cerebral", "tos"}  # un lote, sin repetir
    infarto = decisiones["infarto cerebral"]
    assert infarto is not None and infarto.categoria is C.CEREBROVASCULAR
    assert infarto.candidato is not None and infarto.candidato.codigo.startswith("I6")
    assert segunda.umbral <= infarto.probabilidad <= 1
    assert segunda.decidir([]) == {}


def test_no_guarda_los_textos_entre_llamadas(etapa):
    segunda, _ = etapa
    segunda.decidir(["VIH avanzado", "cirrosis hepática"])
    assert segunda.normalizador._memoria == {}


def test_la_exclusion_quita_la_categoria(etapa, monkeypatch):
    segunda, _ = etapa
    monkeypatch.setattr(d, "excluida", lambda texto, categoria: True)
    assert segunda.decidir(["VIH"]) == {"VIH": None}


def test_par_solo_de_las_reglas_no_tiene_candidato():
    from ner_clinic.normalizacion.indice import Candidato

    top = [d.Propuesta(Candidato("R05", "Tos", 0.9, ORIGEN_CATALOGO), "R05", None)]
    pares = d.pares("diabetes", top, (C.DIABETES, "E11.9"))
    assert [(p.categoria, p.candidato) for p in pares] == [(C.DIABETES, None)]
    assert len(pares[0].atributos) == len(d.ATRIBUTOS)


def test_reglas_de_con_memoria():
    memoria: d.ReglasDe = {}
    assert d.reglas_de("diabetis mellitus", memoria)[0] is C.DIABETES
    assert d.reglas_de("tos", memoria) is None
    assert set(memoria) == {"diabetis mellitus", "tos"}


def test_modelo_de_json_rechaza_otros_atributos(etapa):
    segunda, _ = etapa
    datos = modelo_a_json(segunda.modelo, segunda.umbral)
    datos["atributos"] = list(reversed(datos["atributos"]))
    with pytest.raises(ValueError, match="atributos"):
        modelo_de_json(datos)


def test_jerarquia_en_la_libreria():
    assert (bloque("I63.9"), capitulo("I63.9")) == ("I60-I69", "IX")
    assert capitulo("CAP.02") == "II"
    assert bloque("ZZZ") is None


def test_alternativas_cie10_sin_repetir_y_con_la_descripcion_del_catalogo(etapa):
    """#230: hasta 5 códigos distintos, de mayor a menor similitud, con la descripción del
    catálogo aunque la fila más parecida sea un sinónimo."""
    segunda, _ = etapa
    indice = segunda.normalizador.indice
    salida = segunda.alternativas(["infarto cerebral", "infarto cerebral", "tos"], k=5, minimo=0)
    assert set(salida) == {"infarto cerebral", "tos"}
    for alternativas in salida.values():
        codigos = [a.codigo for a in alternativas]
        assert 1 <= len(codigos) <= 5 and len(set(codigos)) == len(codigos)
        similitudes = [a.similitud for a in alternativas]
        assert similitudes == sorted(similitudes, reverse=True)
        for a in alternativas:
            catalogo = indice.descripcion_catalogo(a.codigo)
            assert catalogo is None or a.descripcion == catalogo
    # La primera alternativa es el código del mejor candidato del índice.
    primero = indice.buscar(segunda.normalizador.codificador(["infarto cerebral"]), 1)[0][0]
    assert salida["infarto cerebral"][0].codigo == primero.codigo
    assert segunda.alternativas([]) == {}


def test_alternativas_respeta_k(etapa):
    segunda, _ = etapa
    assert len(segunda.alternativas(["infarto cerebral"], k=2, minimo=0)["infarto cerebral"]) <= 2


def test_alternativas_sin_ruido_debajo_del_minimo(etapa):
    segunda, _ = etapa
    todas = segunda.alternativas(["infarto cerebral"], minimo=0)["infarto cerebral"]
    corte = todas[0].similitud
    filtradas = segunda.alternativas(["infarto cerebral"], minimo=corte)["infarto cerebral"]
    assert filtradas and all(a.similitud >= corte for a in filtradas)
    assert segunda.alternativas(["infarto cerebral"], minimo=1.01) == {"infarto cerebral": []}
