"""Tests de la segunda etapa con reordenador (#152), sin red: puntuador de juguete."""

import json

import pytest
import torch

from evaluacion import clasificador as cl
from evaluacion import dos_etapas as de
from evaluacion import reordenador as rr
from evaluacion import votos as vt
from evaluacion.__main__ import main
from evaluacion.normalizador import Candidato
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_dos_etapas import TRAIN_6, _indice, _mini
from tests.test_normalizador import DEV, Trigramas, _documento


def _p(codigo, descripcion, similitud, categoria):
    return de.Propuesta(Candidato(codigo, descripcion, similitud), codigo, categoria)


def _letras_comunes(pares):
    """Puntuador de juguete: fracción de letras de la mención que están en la descripción."""
    return [len(set(a.lower()) & set(b.lower())) / max(len(set(a.lower())), 1) for a, b in pares]


TOP = [
    _p("R20", "parestesia", 0.9, None),
    _p("I63.9", "infarto cerebral", 0.8, C.CEREBROVASCULAR),
    _p("I63.0", "ictus trombotico", 0.7, C.CEREBROVASCULAR),
    _p("F03", "demencia", 0.6, C.DEMENCIA),
]
PUNTOS = {"parestesia": 0.2, "infarto cerebral": 0.9, "ictus trombotico": 0.5, "demencia": 0.1}


def _fijo(pares):
    return [PUNTOS[d] for _, d in pares]


def test_candidatos_con_revision_y_licencia():
    assert set(rr.CANDIDATOS) == {"bge_m3", "bge_base", "mmarco"}
    for modelo in rr.CANDIDATOS.values():
        assert len(modelo.revision) == 40 and modelo.licencia and modelo.datos
    assert "no comercial" in rr.CANDIDATOS["mmarco"].datos


def test_reordenar_usa_la_puntuacion_y_corta_en_k():
    documento = _documento("a", "Tuvo ictus.", ("I63.9", "ictus"))
    nuevo = rr.reordenar(de.Pliegue([documento], {"ictus": TOP}), _fijo, 3)
    top = nuevo.topk["ictus"]
    assert [p.candidato.codigo for p in top] == ["I63.9", "I63.0", "R20"]
    assert [p.similitud for p in top] == [0.9, 0.5, 0.2]
    assert top[0].categoria is C.CEREBROVASCULAR  # la categoría no cambia
    decisor = vt.votar(vt.Configuracion(k=1, umbral=0.5))
    assert decisor("ictus", top) == (C.CEREBROVASCULAR, "I63.9")


def test_pares_de():
    pliegue = de.Pliegue([], {"ictus": TOP})
    assert rr.pares_de([pliegue], 2) == [("ictus", "parestesia"), ("ictus", "infarto cerebral")]


def test_atributos():
    extra = rr.atributos(_fijo, 3)
    mejor, margen, fraccion = extra("ictus", TOP, C.CEREBROVASCULAR)
    assert mejor == 0.9
    assert margen == pytest.approx(0.7)  # contra el R20 (0,2)
    assert fraccion == pytest.approx(1.4 / 1.6)
    # El F03 queda fuera de los 3 primeros: es como una categoría solo de las reglas
    assert extra("ictus", TOP, C.DEMENCIA) == (0.0, pytest.approx(-0.9), 0.0)


def test_el_clasificador_acepta_atributos_extra():
    pares = cl.pares("ictus", TOP, None, rr.atributos(_fijo, 3))
    assert all(len(p.atributos) == len(cl.ATRIBUTOS) + len(rr.ATRIBUTOS) for p in pares)
    assert pares[0].atributos[-3] == 0.9
    # Sin `extra` no cambian
    assert all(len(p.atributos) == len(cl.ATRIBUTOS) for p in cl.pares("ictus", TOP, None))


def test_puntuaciones_en_disco(tmp_path):
    creados, llamadas = [], []

    def crear():
        creados.append(1)

        def puntuar(pares):
            llamadas.append(list(pares))
            return _letras_comunes(pares)

        return puntuar

    ruta = tmp_path / "puntos.pt"
    uno = rr.PuntuacionesEnDisco(crear, ruta)
    pares = [("ictus", "ictus"), ("tos", "ictus"), ("ictus", "ictus")]
    assert uno(pares) == _letras_comunes(pares)
    assert llamadas == [[("ictus", "ictus"), ("tos", "ictus")]]  # sin repetir
    uno.guardar()

    dos = rr.PuntuacionesEnDisco(crear, ruta)
    assert dos(pares) == _letras_comunes(pares)
    assert len(creados) == 1 and dos.nuevos == 0  # no carga el modelo
    dos.guardar()  # sin pares nuevos no escribe ni falla


def test_rutas_por_modelo(tmp_path):
    rutas = {rr.ruta_puntuaciones(m, tmp_path) for m in rr.CANDIDATOS.values()}
    assert len(rutas) == 3
    assert rr.ruta_resultados("mmarco").name == "dos_etapas_reordenador_mmarco.json"


def test_latencia_con_notas_de_menciones_reales():
    llamadas = []

    def puntuar(pares):
        llamadas.append(list(pares))
        return [0.0] * len(pares)

    pliegue = de.Pliegue([], {"ictus": TOP, "acv": TOP})
    hilos = torch.get_num_threads()
    resultado = rr.latencia(puntuar, pliegue, ks=(2, 3), notas=2, entidades=2, hilos=1)
    assert set(resultado) == {"k2", "k3"}
    assert resultado["k3"]["pares_por_nota"] == 6
    assert resultado["k2"]["maximo_s"] >= resultado["k2"]["mediana_s"]
    assert len(llamadas) == 6  # 2 notas más la de calentar, por cada k
    assert llamadas[0][:2] == [("ictus", "parestesia"), ("ictus", "infarto cerebral")]
    assert torch.get_num_threads() == hilos


def _rejilla_chica(monkeypatch):
    monkeypatch.setattr(cl, "PENALIZACIONES", (1e-2,))
    monkeypatch.setattr(cl, "UMBRALES", (0.3, 0.6))
    monkeypatch.setattr(rr, "UMBRALES", (0.3, 0.6))
    monkeypatch.setattr(vt, "TEMPERATURAS", (None,))
    monkeypatch.setattr(vt, "NIVELES", ("categoria",))


def test_evaluar_sin_texto_del_corpus(monkeypatch):
    _rejilla_chica(monkeypatch)
    indice, codificador = _indice()
    resultados = rr.evaluar(
        "mmarco",
        "sapbert",
        indice,
        codificador,
        _letras_comunes,
        TRAIN_6,
        {},
        pliegues=3,
        k=5,
        replicas=5,
        ks=(3,),
    )
    assert set(resultados["metodos"]) == {
        "hibrido_exclusiones",
        "clasificador",
        "solo_k3",
        "clasificador_k3",
    }
    assert set(resultados["contra_clasificador"]) == {"solo_k3", "clasificador_k3"}
    assert len(resultados["elegidas"]["solo_k3"]) == 3
    assert set(resultados["f1_por_pliegue"]) == set(resultados["metodos"])
    assert all(len(f1) == 3 for f1 in resultados["f1_por_pliegue"].values())
    assert resultados["reordenador"]["licencia"] == "Apache-2.0"
    texto = json.dumps(resultados, ensure_ascii=False).casefold()
    assert "infarto cerebral" not in texto and "ictus" not in texto
    assert "| clasificador k3 |" in rr.tabla(resultados)


def test_subcomando_reordenador(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    _rejilla_chica(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV})
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(rr, "ReordenadorHF", lambda modelo: _letras_comunes)
    main(
        ["reordenador", "--reordenador", "mmarco", "--raiz", ".", "--catalogo", "."]
        + ["--indices", "indices", "--pliegues", "3", "--bootstrap", "5", "--salida", "r.json"]
    )
    assert "| clasificador k5 |" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["pliegues"] == 3
    assert list((tmp_path / "indices").glob("reordenador_*.pt"))
    guardado = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert set(guardado["costo"]["latencia"]["por_k"]) == {"k5", "k10"}
    assert guardado["costo"]["memoria_modelo_mib"] >= 0
