"""De punta a punta en CodiEsp desarrollo (#167), sin red ni modelos."""

import json

import pytest

from evaluacion import extremo_codiesp as ex
from evaluacion.codiesp import DocumentoCodificado
from ner_clinic.extraccion import ExtractorFalso
from ner_clinic.normalizacion.decision import Decision
from ner_clinic.normalizacion.indice import ORIGEN_CATALOGO, Candidato
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_api import salidas_para
from tests.test_pipeline_normalizador import SegundaFalsa

DOCS = [
    DocumentoCodificado("d-1", "Paciente con falla renal crónica. Niega EPOC.", ["N18.4", "R05"]),
    DocumentoCodificado("d-2", "Antecedente de EPOC y eppoc. Refiere tos.", ["J44.9"]),
    DocumentoCodificado("d-3", "Dolor abdominal.", ["R10.9"]),
]
PALABRAS = ("falla renal crónica", "EPOC", "eppoc", "tos")


def _extractor():
    salidas = {}
    for d in DOCS:
        salidas |= salidas_para(d.texto, *PALABRAS)
    return ExtractorFalso(salidas)


def _segunda():
    return SegundaFalsa(
        {
            "falla renal crónica": Decision(
                C.RENAL_CRONICA, "N18.4", 0.9, Candidato("N18.4", "ERC estadio 4", 0.8, "x")
            ),
            "EPOC": Decision(
                C.PULMONAR_CRONICA, "J44.9", 0.9, Candidato("J44.9", "EPOC", 1.0, ORIGEN_CATALOGO)
            ),
        }
    )


def test_gold_por_documento():
    assert ex.gold_de(DOCS[0]) == {C.RENAL_CRONICA}
    assert ex.gold_de(DOCS[2]) == set()


def test_conteos():
    assert ex.conteos({C.DIABETES, C.DEMENCIA}, {C.DIABETES, C.VIH_SIDA}) == (1, 1, 1)
    assert ex.conteos({C.DIABETES}, {C.VIH_SIDA}, C.DIABETES) == (0, 1, 0)


def test_evaluar_compara_reglas_y_normalizador():
    resultados = ex.evaluar(DOCS, _extractor(), _segunda(), replicas=20)
    reglas, normalizador, hibrido = (resultados["metodos"][m] for m in ex.METODOS)
    # Las reglas no conocen "falla renal crónica"; el normalizador sí
    assert reglas["micro"]["vp"] == 1 and reglas["micro"]["fn"] == 1
    assert normalizador["micro"]["vp"] == 2 and normalizador["micro"]["fn"] == 0
    assert normalizador["indice"]["exacto"] == 1.0
    assert resultados["delta_f1"] > 0
    # El híbrido toma EPOC de las reglas y "falla renal crónica" del normalizador
    assert hibrido["micro"]["vp"] == 2 and hibrido["micro"]["fn"] == 0
    assert resultados["delta_f1_hibrido"] > 0
    assert resultados["pares_gold"] == 2 and resultados["documentos_con_categoria_gold"] == 2
    # En d-2 la fusión nombra la entidad "eppoc": la categoría llega por la mención "EPOC"
    assert resultados["respaldo_menciones"]["entidades"] == 1
    assert resultados["respaldo_menciones"]["en_el_gold"] == 1
    texto = json.dumps(resultados, ensure_ascii=False)
    assert "Paciente" not in texto and "d-1" not in texto
    assert "| reglas |" in ex.tabla(resultados) and "Δ F1 híbrido" in ex.tabla(resultados)


def test_respaldo_por_una_mencion():
    doc = DocumentoCodificado("d-4", "Con eppoc, EPOC y epoc.", ["J44.9"])
    extractor = ExtractorFalso(salidas_para(doc.texto, "eppoc", "EPOC", "epoc"))
    resultados = ex.evaluar([doc], extractor, _segunda(), replicas=5)
    assert resultados["respaldo_menciones"] == {
        "entidades": 1,
        "en_el_gold": 1,
        "por_categoria": {"pulmonar_cronica": 1},
    }


def test_subcomando_extremo(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli
    from evaluacion.__main__ import main
    from ner_clinic.normalizacion import artefactos
    from tests.test_dos_etapas import _mini

    monkeypatch.chdir(tmp_path)
    _mini(tmp_path, {"train": DOCS, "dev": DOCS, "test": DOCS})
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [])
    monkeypatch.setattr(cli, "ExtractorHF", lambda: _extractor())
    monkeypatch.setattr(artefactos, "cargar", lambda ruta: _segunda())
    argumentos = ["extremo", "--raiz", ".", "--cache", "cache", "--bootstrap", "5"]
    main([*argumentos, "--salida", "e.json"])
    guardado = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert guardado["normalizador"]["huella_indice"]
    assert "Δ F1" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="una sola vez"):
        main([*argumentos, "--salida", "e.json"])
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [" M x.py"])
    with pytest.raises(SystemExit, match="árbol limpio"):
        main([*argumentos, "--salida", "otro.json"])


def test_sintetico_con_normalizador(monkeypatch):
    from evaluacion import contexto_charlson

    nota = {"id": "n-1", "texto": "Paciente con EPOC.", "entidades": []}
    extractor = ExtractorFalso(salidas_para(nota["texto"], "EPOC"))
    resultados = contexto_charlson.evaluar([nota], extractor, normalizador=_segunda())
    assert resultados["mapeo"] == "hibrido"
    solo = contexto_charlson.evaluar(
        [nota], extractor, normalizador=_segunda(), modo_mapeo="normalizador"
    )
    assert solo["mapeo"] == "normalizador"
    assert contexto_charlson.evaluar([nota], extractor)["mapeo"] == "reglas"


def test_sintetico_con_otra_configuracion_de_extraccion():
    from evaluacion import contexto_charlson
    from ner_clinic.extraccion import ConfiguracionExtraccion

    nota = {"id": "n-1", "texto": "dx: demencia mixta + HTA.", "entidades": []}
    extractor = ExtractorFalso(salidas_para(nota["texto"], "demencia mixta", "HTA"))
    tesis = ConfiguracionExtraccion(max_distancia_fusion=18)
    resultados = contexto_charlson.evaluar([nota], extractor, configuracion=tesis)
    assert resultados["configuracion"]["max_distancia_fusion"] == 18
    corrida = contexto_charlson.correr([nota], extractor, configuracion=tesis)
    assert [e.texto for e in corrida.resultados[0].entidades] == ["demencia mixta + HTA"]
    corrida = contexto_charlson.correr([nota], extractor)  # sin fusión (#55)
    assert [e.texto for e in corrida.resultados[0].entidades] == ["demencia mixta", "HTA"]


def test_sintetico_con_variante_desconocida_o_sin_configuracion(tmp_path, monkeypatch):
    from evaluacion import __main__ as cli
    from evaluacion.__main__ import main

    monkeypatch.setattr(cli, "ExtractorHF", lambda: _extractor())
    for variante, mensaje in (("no_existe", "desconocida"), ("crudo", "no cambia")):
        with pytest.raises(SystemExit, match=mensaje):
            main(["sintetico", "--variante", variante, "--salida", str(tmp_path / "s.json")])


def test_sintetico_normalizador_e_hibrido_son_excluyentes(tmp_path):
    from evaluacion.__main__ import main

    with pytest.raises(SystemExit):
        main(["sintetico", "--normalizador", "--hibrido", "--salida", str(tmp_path / "s.json")])
