"""Tests de la segunda etapa entrenada (#151), sin red."""

import json

import pytest
import torch

from evaluacion import clasificador as cl
from evaluacion import dos_etapas as de
from evaluacion.__main__ import main
from evaluacion.normalizador import Candidato
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_dos_etapas import TRAIN_6, _indice, _mini
from tests.test_normalizador import DEV, Trigramas


def _p(codigo, similitud, categoria, origen="catalogo"):
    return de.Propuesta(Candidato(codigo, "-", similitud, origen), codigo, categoria)


def _atributo(par, nombre):
    return par.atributos[cl.ATRIBUTOS.index(nombre)]


TOP = [
    _p("D49.2", 0.80, None),
    _p("C34.9", 0.78, C.TUMOR_MALIGNO, "mención de CodiEsp entrenamiento"),
    _p("C34.1", 0.77, C.TUMOR_MALIGNO),
    _p("C78.0", 0.75, C.TUMOR_METASTASICO),
]


def test_pares_de_cada_categoria_del_top_y_de_las_reglas():
    pares = cl.pares("ca de pulmón", TOP, (C.DEMENCIA, "F03"))
    assert [p.categoria for p in pares] == [C.TUMOR_MALIGNO, C.TUMOR_METASTASICO, C.DEMENCIA]
    assert all(len(p.atributos) == len(cl.ATRIBUTOS) for p in pares)
    maligno, metastasico, demencia = pares
    assert maligno.codigo == "C34.9"
    assert _atributo(maligno, "similitud") == 0.78
    assert _atributo(maligno, "inverso_rango") == 0.5
    assert _atributo(maligno, "margen_primero") == pytest.approx(-0.02)
    assert _atributo(maligno, "margen_otra") == pytest.approx(-0.02)  # el D49.2 ("ninguna")
    assert _atributo(maligno, "fraccion_top5") == 0.4
    assert _atributo(maligno, "mejor_es_sinonimo") == 1.0
    assert _atributo(maligno, "fraccion_sinonimos") == 0.5
    assert _atributo(maligno, "reglas_otra") == 1.0
    assert _atributo(maligno, "categoria_tumor_maligno") == 1.0
    assert _atributo(metastasico, "voto_suma") == pytest.approx(0.75 / 3.10)
    # La de las reglas no está en el top: sin similitud, con su código
    assert demencia.codigo == "F03"
    assert _atributo(demencia, "en_topk") == 0.0
    assert _atributo(demencia, "reglas_esta") == 1.0
    assert _atributo(demencia, "similitud") == 0.0


def test_votos_de_los_pares_suman_uno_con_ninguna():
    maligno, metastasico = cl.pares("x", TOP, None)
    ninguna = _atributo(maligno, "ninguna_t005")
    total = _atributo(maligno, "voto_t005") + _atributo(metastasico, "voto_t005") + ninguna
    assert total == pytest.approx(1.0)


def test_sin_categorias_no_hay_pares():
    assert cl.pares("tos", [_p("R05", 0.9, None)], None) == []


def test_atributos_de_la_mencion():
    par = cl.pares("EPOC", [_p("J44.9", 1.0, C.PULMONAR_CRONICA)], None)[0]
    assert _atributo(par, "sigla") == 1.0
    assert _atributo(par, "exacta") == 1.0
    assert _atributo(par, "palabras") == 1.0
    excluida = cl.pares("mioma uterino", [_p("C90.0", 0.9, C.TUMOR_MALIGNO)], None)[0]
    assert _atributo(excluida, "excluida") == 1.0


def test_entrenar_es_determinista_y_separa():
    x = torch.tensor([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.2]])
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    uno = cl.entrenar(x, y, 1e-3)
    dos = cl.entrenar(x, y, 1e-3)
    assert torch.equal(uno.pesos, dos.pesos)
    probabilidades = uno.probabilidades(x)
    assert probabilidades[2] > 0.5 > probabilidades[0]
    # Con mucha penalización los pesos se encogen
    assert cl.entrenar(x, y, 10.0).pesos.abs().sum() < uno.pesos.abs().sum()


def _pliegues():
    indice, codificador = _indice()
    return de.pliegues_topk(indice, codificador, TRAIN_6, n=3, k=5)


def test_el_f1_rapido_es_el_del_decisor_en_el_arnes():
    pliegues = _pliegues()
    reglas = {}
    datos = cl.Datos.de(pliegues, reglas)
    modelo = cl.entrenar(*datos.matriz(), 1e-2)
    for umbral in (0.1, 0.5, 0.9):
        rapido = datos.f1(datos.mejores(modelo), umbral)
        lista = de.casos(pliegues, cl.decisor(modelo, umbral, reglas))
        assert rapido == pytest.approx(de._prf(lista, de._ids(pliegues))["f1"], abs=1e-4)


def test_pesos_por_atributo():
    datos = cl.Datos.de(_pliegues(), {})
    modelo = cl.entrenar(*datos.matriz(), 1e-2)
    assert list(modelo.pesos_por_atributo()) == list(cl.ATRIBUTOS)


def test_elegir_dentro_de_la_rejilla(monkeypatch):
    monkeypatch.setattr(cl, "PENALIZACIONES", (1e-3, 1e-1))
    monkeypatch.setattr(cl, "UMBRALES", (0.3, 0.6))
    datos = [cl.Datos.de([p], {}) for p in _pliegues()]
    penalizacion, umbral, f1, curva = cl.elegir(datos)
    assert penalizacion in (1e-3, 1e-1) and umbral in (0.3, 0.6) and 0 <= f1 <= 1
    assert set(curva) == {1e-3, 1e-1} and max(curva.values()) == f1


def test_mejores_sin_pares():
    datos = cl.Datos(["tos"], [None], [[]])
    modelo = cl.Modelo(torch.zeros(len(cl.ATRIBUTOS)), torch.ones(len(cl.ATRIBUTOS)),
                       torch.zeros(len(cl.ATRIBUTOS)), 0.0)  # fmt: skip
    assert datos.mejores(modelo) == [None]
    assert datos.f1([None], 0.5) == 0.0


def _rejilla_chica(monkeypatch):
    from evaluacion import votos as vt

    monkeypatch.setattr(cl, "PENALIZACIONES", (1e-2,))
    monkeypatch.setattr(cl, "UMBRALES", (0.3, 0.6))
    monkeypatch.setattr(vt, "KS", (1, 3))
    monkeypatch.setattr(vt, "TEMPERATURAS", (None,))
    monkeypatch.setattr(vt, "UMBRALES", (0.3, 0.6))


def test_evaluar(monkeypatch):
    _rejilla_chica(monkeypatch)
    indice, codificador = _indice()
    resultados = cl.evaluar(
        "sapbert", indice, codificador, TRAIN_6, {}, pliegues=3, k=5, replicas=5
    )
    assert set(resultados["metodos"]) == {"hibrido_exclusiones", "votos", "clasificador"}
    assert len(resultados["por_pliegue"]) == 3
    assert set(resultados["final"]["pesos"]) == set(cl.ATRIBUTOS)
    texto = json.dumps(resultados, ensure_ascii=False).casefold()
    assert "infarto cerebral" not in texto and "ictus" not in texto


def test_subcomando_clasificador(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    _rejilla_chica(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV})
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    main(
        ["clasificador", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
        + ["--pliegues", "3", "--bootstrap", "5", "--salida", "c.json"]
    )
    assert "| clasificador |" in capsys.readouterr().out
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["pliegues"] == 3


def test_elegir_a_igual_f1_mas_penalizacion_y_umbral_mas_alto(monkeypatch):
    monkeypatch.setattr(cl, "PENALIZACIONES", (1e-3, 1e-1))
    monkeypatch.setattr(cl, "UMBRALES", (0.3, 0.6))
    monkeypatch.setattr(cl.Datos, "f1", lambda self, mejores, umbral: 0.5)  # todo empata
    datos = [cl.Datos.de([p], {}) for p in _pliegues()]
    assert cl.elegir(datos)[:3] == (1e-1, 0.6, 0.5)


def test_evaluar_entrena_sin_el_pliegue_externo(monkeypatch):
    _rejilla_chica(monkeypatch)
    vistos = []
    original = cl.Datos.de

    def espiar(pliegues, reglas):
        vistos.append({d.id for p in pliegues for d in p.documentos})
        return original(pliegues, reglas)

    monkeypatch.setattr(cl.Datos, "de", staticmethod(espiar))
    indice, codificador = _indice()
    cl.evaluar("sapbert", indice, codificador, TRAIN_6, {}, pliegues=3, k=5, replicas=5)
    externos = [{"t-1", "t-4"}, {"t-2", "t-5"}, {"t-3", "t-6"}]
    # Las 2 primeras tandas (de 2 internos cada una) son del externo 0, etc.
    for i, externo in enumerate(externos):
        assert not any(v & externo for v in vistos[2 * i : 2 * i + 2])
