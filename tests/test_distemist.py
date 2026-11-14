"""Tests del runner de DisTEMIST (`evaluacion.distemist` y el subcomando `distemist`).

Sin red ni modelo: usan el mini corpus de `tests/datos/distemist_mini/` y un
`ExtractorFalso` que marca palabras fijas en cada fragmento. El modelo "falso" parte
"diabetes mellitus tipo 2" en dos entidades, así el modelo crudo y el pipeline (que las
fusiona) dan cifras distintas y conocidas.
"""

import json
from pathlib import Path

import pytest

from evaluacion import distemist
from evaluacion.__main__ import main
from evaluacion.cache import ExtractorConCache
from evaluacion.corpus import leer_particion
from evaluacion.distemist import (
    MODOS,
    VARIANTES,
    VARIANTES_DETALLADAS,
    Variante,
    correr,
    evaluar,
    medir_tiempos,
    ruta_detalle,
    ruta_resultados,
    tabla,
)
from ner_clinic.extraccion import ConfiguracionExtraccion, ExtractorFalso
from ner_clinic.limpieza import limpiar
from ner_clinic.segmentacion import segmentar

MINI = Path(__file__).parent / "datos" / "distemist_mini"
PALABRAS = (
    "diabetes mellitus",
    "tipo 2",
    "hipertensión arterial",
    "metástasis",
    "EPOC",
    "asma",
    "insuficiencia cardiaca",
)


def extractor_para(textos, palabras=PALABRAS) -> ExtractorFalso:
    """`ExtractorFalso` que marca cada aparición de `palabras` en los fragmentos."""
    salidas = {}
    for texto in textos:
        for fragmento in segmentar(limpiar(texto).texto):
            entidades = []
            for palabra in palabras:
                desde = 0
                while (inicio := fragmento.texto.find(palabra, desde)) != -1:
                    fin = inicio + len(palabra)
                    entidades.append(
                        {
                            "entity_group": "ENFERMEDAD",
                            "score": 0.99,
                            "word": palabra,
                            "start": inicio,
                            "end": fin,
                        }
                    )
                    desde = fin
            salidas[fragmento.texto] = sorted(entidades, key=lambda e: e["start"])
    return ExtractorFalso(salidas)


@pytest.fixture
def documentos():
    return leer_particion(MINI, "test")


@pytest.fixture
def falso(documentos):
    return extractor_para(d.texto for d in documentos)


def _prf(resultados, variante, modo):
    m = resultados["variantes"][variante][modo]
    return m["precision"], m["recall"], m["f1"]


# --- Variantes -----------------------------------------------------------------------


def test_tabla_de_variantes_cubre_la_ablacion():
    esperadas = {
        "crudo",
        "pipeline",
        "fusion_0",
        "fusion_10",
        "fusion_18",
        "fusion_30",
        "sin_fusion",
        "fusion_barreras",
        "sin_expansion",
        "filtro_tesis",
        "filtro_minimo",
        "filtro_lista_blanca",
        "filtro_ninguno",
        "sin_lista_negra",
        "sin_sufijos",
        "sin_deduplicacion",
        "sin_cortar_parentesis",
    }
    assert set(VARIANTES) == esperadas
    assert VARIANTES["pipeline"] == Variante()
    assert VARIANTES["sin_fusion"].configuracion == ConfiguracionExtraccion(
        max_distancia_fusion=None
    )
    assert not VARIANTES["sin_deduplicacion"].deduplicar
    assert VARIANTES["crudo"].crudo


def test_todas_las_variantes_corren(documentos, falso):
    corrida = correr(documentos, falso, VARIANTES)
    assert corrida.documentos == ["caso-1", "caso-2", "caso-3"]
    for nombre in VARIANTES:
        for modo in MODOS:
            assert len(corrida.emparejamientos[nombre][modo]) == len(documentos)


def test_variante_desconocida_falla(documentos, falso):
    with pytest.raises(ValueError, match="Variantes desconocidas"):
        correr(documentos, falso, ["pipeline", "no_existe"])


def test_cifras_conocidas_del_mini_corpus(documentos, falso):
    resultados, _ = evaluar(documentos, falso, "test", replicas=50)
    # La fusión de la tesis une "diabetes mellitus" + "tipo 2" y acierta las 6 menciones.
    assert _prf(resultados, "fusion_18", "estricto") == (1.0, 1.0, 1.0)
    # El pipeline ya no fusiona (#55): en este corpus se comporta como sin_fusion.
    assert _prf(resultados, "pipeline", "estricto") == _prf(resultados, "sin_fusion", "estricto")
    # El modelo crudo deja las dos partes: 2 falsos positivos y 1 falso negativo.
    crudo = resultados["variantes"]["crudo"]
    assert (crudo["estricto"]["vp"], crudo["estricto"]["fp"], crudo["estricto"]["fn"]) == (5, 2, 1)
    # En modo laxo "diabetes mellitus" acierta; "tipo 2" sigue sobrando.
    assert (crudo["laxo"]["vp"], crudo["laxo"]["fp"], crudo["laxo"]["fn"]) == (6, 1, 0)
    # Sin fusión el pipeline se comporta como el crudo en este corpus.
    assert _prf(resultados, "sin_fusion", "estricto") == _prf(resultados, "crudo", "estricto")
    # El filtro de la tesis descarta "EPOC" y "asma" (4 caracteres); sin la fusión (#55),
    # además, "diabetes mellitus" y "tipo 2" quedan partidas.
    tesis = resultados["variantes"]["filtro_tesis"]["estricto"]
    assert (tesis["vp"], tesis["fp"], tesis["fn"]) == (3, 2, 3)
    # Los IC contienen la cifra puntual.
    ic = crudo["estricto"]["ic95"]
    for metrica in ("precision", "recall", "f1"):
        assert ic[metrica][0] <= crudo["estricto"][metrica] <= ic[metrica][1]


def test_json_de_resultados_tiene_la_forma_esperada(documentos, falso):
    resultados, _ = evaluar(documentos, falso, "test", replicas=20, limite=3)
    assert resultados["particion"] == "test"
    assert resultados["limite"] == 3
    assert (resultados["documentos"], resultados["menciones_gold"]) == (3, 6)
    entorno = resultados["entorno"]
    for clave in ("modelo", "stride_tokens", "transformers", "torch", "python", "commit", "cpu"):
        assert clave in entorno
    assert set(resultados["variantes"]) == set(VARIANTES)
    for por_modo in resultados["variantes"].values():
        assert set(por_modo) == set(MODOS)
        for m in por_modo.values():
            assert set(m) == {"precision", "recall", "f1", "ic95", "vp", "fp", "fn"}
            assert set(m["ic95"]) == {"precision", "recall", "f1"}
    assert resultados["tiempos"] is None
    assert resultados["cache"] is None
    assert resultados["memoria_pico_mib"] > 0


def test_json_de_resultados_no_trae_texto_del_corpus(documentos, falso):
    resultados, detalle = evaluar(documentos, falso, "test", replicas=20)
    volcado = json.dumps(resultados, ensure_ascii=False)
    for documento in documentos:
        for mencion in documento.menciones:
            assert mencion.texto not in volcado
    # El detalle sí lo trae, y es aparte.
    assert "insuficiencia cardiaca" in json.dumps(detalle, ensure_ascii=False)


def test_detalle_solo_de_crudo_y_pipeline(documentos, falso):
    _, detalle = evaluar(documentos, falso, "test", replicas=20)
    assert set(detalle["variantes"]) == set(VARIANTES_DETALLADAS)
    caso1 = detalle["variantes"]["crudo"]["estricto"][0]
    assert caso1["documento"] == "caso-1"
    assert [m["texto"] for m in caso1["falsos_positivos"]] == ["diabetes mellitus", "tipo 2"]
    assert caso1["falsos_negativos"] == [
        {"inicio": 13, "fin": 37, "texto": "diabetes mellitus tipo 2"}
    ]
    par = caso1["verdaderos"][0]
    hta = {"inicio": 40, "fin": 61, "texto": "hipertensión arterial"}
    assert par["gold"] == par["prediccion"] == hta


def test_train_por_defecto_solo_crudo_y_pipeline(falso):
    documentos = leer_particion(MINI, "test")
    resultados, _ = evaluar(documentos, falso, "train", replicas=10)
    assert list(resultados["variantes"]) == ["crudo", "pipeline"]


# --- Caché y tiempos -----------------------------------------------------------------


def test_la_cache_evita_una_segunda_llamada_al_extractor(documentos, falso, tmp_path):
    primera = ExtractorConCache(falso, carpeta=tmp_path)
    r1, _ = evaluar(documentos, primera, "test", replicas=20, cache=primera)
    llamadas = len(falso.llamadas)
    # Una inferencia por documento con texto, aunque corran 14 variantes.
    assert llamadas == len(documentos)
    assert r1["cache"]["fallos"] > 0

    segunda = ExtractorConCache(falso, carpeta=tmp_path)
    r2, _ = evaluar(documentos, segunda, "test", replicas=20, cache=segunda)
    assert len(falso.llamadas) == llamadas
    assert r2["cache"] == {"aciertos": r2["cache"]["aciertos"], "fallos": 0, "llamadas": 0}
    assert r2["variantes"] == r1["variantes"]


def test_tiempos_con_el_extractor_sin_cache(documentos, falso):
    sin_cache = extractor_para(d.texto for d in documentos)
    resultados, _ = evaluar(documentos, falso, "test", replicas=10, extractor_tiempos=sin_cache)
    # Calentamiento más una llamada por documento.
    assert len(sin_cache.llamadas) == 1 + len(documentos)
    assert sin_cache.llamadas[0] == [distemist.TEXTO_CALENTAMIENTO]
    tiempos = resultados["tiempos"]
    assert tiempos["documentos"] == 3
    for clave in ("segundos_por_documento", "caracteres_por_documento"):
        assert set(tiempos[clave]) == {"mediana", "p95", "maximo"}
    c = tiempos["caracteres_por_documento"]
    assert c["maximo"] == max(len(d.texto) for d in documentos)
    assert c["mediana"] <= c["p95"] <= c["maximo"]


def test_tiempos_sin_documentos():
    tiempos = medir_tiempos([], ExtractorFalso({}))
    assert tiempos["segundos_por_documento"] == {"mediana": 0.0, "p95": 0.0, "maximo": 0.0}
    assert tiempos["ms_por_1000_caracteres"] == 0.0


# --- Salida y línea de comandos ------------------------------------------------------


def test_rutas_de_salida(tmp_path):
    assert ruta_resultados(None, "test") == Path("evaluacion/resultados/distemist_test.json")
    assert ruta_resultados(tmp_path, "train") == tmp_path / "distemist_train.json"
    assert ruta_resultados(tmp_path / "x.json", "test") == tmp_path / "x.json"
    assert ruta_detalle(None, "test") == Path("data/evaluacion/distemist_test_errores.json")
    assert ruta_detalle(tmp_path, "test") == tmp_path / "distemist_test_errores.json"
    assert ruta_detalle(tmp_path / "d.json", "test") == tmp_path / "d.json"


def test_tabla_markdown(documentos, falso):
    resultados, _ = evaluar(documentos, falso, "test", variantes=["fusion_18"], replicas=10)
    assert "| fusion_18 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |" in tabla(resultados)


def test_entorno_sin_git(monkeypatch):
    def falla(*args, **kwargs):
        raise OSError("sin git")

    monkeypatch.setattr(distemist.subprocess, "run", falla)
    assert distemist.entorno()["commit"] is None


def test_subcomando_distemist(tmp_path, documentos, monkeypatch, capsys):
    falso = extractor_para(d.texto for d in documentos)
    monkeypatch.setattr("evaluacion.__main__.ExtractorHF", lambda: falso)
    argumentos = [
        "distemist",
        "--raiz",
        str(MINI),
        "--limite",
        "2",
        "--salida",
        str(tmp_path / "resultados"),
        "--detalle",
        str(tmp_path / "detalle"),
        "--cache",
        str(tmp_path / "cache"),
        "--bootstrap",
        "20",
    ]
    main(argumentos)
    salida = capsys.readouterr().out
    assert "DisTEMIST test: 2 documentos" in salida
    assert "| crudo |" in salida
    assert "Tiempos del pipeline sin caché (2 documentos)" in salida

    resultados = json.loads((tmp_path / "resultados" / "distemist_test.json").read_text())
    assert resultados["documentos"] == 2
    assert resultados["limite"] == 2
    assert resultados["tiempos"]["documentos"] == 2
    detalle = json.loads((tmp_path / "detalle" / "distemist_test_errores.json").read_text())
    assert set(detalle["variantes"]) == {"crudo", "pipeline"}
    assert list((tmp_path / "cache").glob("*.json"))

    # Segunda corrida: todo sale de la caché y las métricas no cambian.
    main([*argumentos, "--sin-tiempos"])
    salida = capsys.readouterr().out
    assert "Tiempos: no medidos" in salida
    segunda = json.loads((tmp_path / "resultados" / "distemist_test.json").read_text())
    assert segunda["cache"]["fallos"] == 0
    assert segunda["variantes"] == resultados["variantes"]


def test_subcomando_train_con_variantes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("evaluacion.__main__.ExtractorHF", lambda: ExtractorFalso({}))
    monkeypatch.setattr(
        "evaluacion.__main__.leer_particion",
        lambda raiz, particion: leer_particion(MINI, "test"),
    )
    main(
        [
            "distemist",
            "--particion",
            "train",
            "--variantes",
            "crudo",
            "--salida",
            str(tmp_path / "r.json"),
            "--detalle",
            str(tmp_path / "d.json"),
            "--cache",
            str(tmp_path / "cache"),
            "--sin-tiempos",
            "--bootstrap",
            "10",
        ]
    )
    resultados = json.loads((tmp_path / "r.json").read_text())
    assert resultados["particion"] == "train"
    assert list(resultados["variantes"]) == ["crudo"]


def test_contra_pipeline_es_pareado_por_documento(documentos, falso):
    resultados, _ = evaluar(
        documentos, falso, "test", variantes=["crudo", "pipeline", "sin_fusion"], replicas=20
    )
    comparacion = resultados["contra_pipeline"]
    assert set(comparacion) == {"sin_fusion"}
    estricto = comparacion["sin_fusion"]["estricto"]
    pipeline = resultados["variantes"]["pipeline"]["estricto"]["f1"]
    sin = resultados["variantes"]["sin_fusion"]["estricto"]["f1"]
    assert estricto["diferencia_f1"] == pytest.approx(sin - pipeline, abs=1e-4)
    assert set(estricto["ic95"]) == {"precision", "recall", "f1"}
    sin_pipeline, _ = evaluar(documentos, falso, "test", variantes=["sin_fusion"], replicas=5)
    assert sin_pipeline["contra_pipeline"] == {}


def test_las_variantes_de_fusion_fusionan():
    """Desde #55 el pipeline no fusiona: las filas de la ablación con fusión la piden."""
    for nombre in ("fusion_10", "fusion_18", "fusion_30", "fusion_barreras"):
        assert VARIANTES[nombre].configuracion.max_distancia_fusion is not None, nombre
    assert VARIANTES["fusion_barreras"].configuracion.barreras_fusion == "ampliadas"
    assert VARIANTES["pipeline"].configuracion.max_distancia_fusion is None
