"""Tests de la medición final en desarrollo (#154), sin red."""

import json

import pytest

from evaluacion import medicion_final as mf
from evaluacion.__main__ import main
from evaluacion.sinonimos_publicos import MencionExterna
from tests.test_dos_etapas import TRAIN_6, _indice, _mini
from tests.test_normalizador import DEV, Trigramas, _documento
from tests.test_sinonimos_publicos import _cantemist, _distemist, _preparar

VARIANTES = ("original", "minusculas", "errores_10")


@pytest.fixture
def rejilla(monkeypatch):
    from tests.test_clasificador import _rejilla_chica

    _rejilla_chica(monkeypatch)


def _evaluar(perdidas=None):
    fuentes, reporte = _preparar()
    indice, codificador = _indice()
    return mf.evaluar(
        "sapbert", indice, codificador, TRAIN_6, DEV, {}, fuentes, reporte, ["x-1"],
        perdidas, pliegues=3, k=5, replicas=5, variantes=VARIANTES,
    )  # fmt: skip


def test_evaluar_sin_texto_de_los_corpus(rejilla):
    resultados = _evaluar()
    dev = resultados["desarrollo"]
    assert list(dev["metodos"]) == list(mf.METODOS)
    assert set(dev["comparaciones"]) == set(mf.METODOS) - {mf.REFERENCIA}
    # Por categoría solo el elegido: es la única comparación de la decisión
    assert "por_categoria" in dev["comparaciones"][mf.ELEGIDO]
    assert "por_categoria" not in dev["comparaciones"]["hibrido_fuentes"]
    assert set(dev["comparaciones"][mf.ELEGIDO]["variantes"]) == {"minusculas", "errores_10"}
    for m in dev["metodos"].values():
        assert set(m["categoria"]) == set(VARIANTES)
        assert set(m["no_vistas"]) == {"sinonimos_codiesp", "sinonimos_codiesp_y_fuentes"}
        assert "por_categoria" in m["categoria"]["original"]
    assert len(resultados["metas"]) == 6
    assert resultados["clasificador_final"][mf.ELEGIDO]["atributos"]
    assert resultados["indice_dev"]["con_fuentes"]["sinonimos_externos"] > 0
    assert resultados["indice_dev"]["sin_fuentes"]["sinonimos_externos"] == 0
    assert dev["perdidas_hibrido"]["reproduce_149"] is None
    texto = json.dumps(resultados, ensure_ascii=False).casefold()
    for palabra in ("infarto cerebral", "ictus", "carcinoma", "cirrosis", '"vih"'):
        assert palabra not in texto  # "vih_sida" es el nombre de una categoría


def test_las_perdidas_se_comparan_con_las_del_arnes(rejilla):
    resultados = _evaluar(perdidas=[])
    perdidas = resultados["desarrollo"]["perdidas_hibrido"]
    assert perdidas["reproduce_149"] is (perdidas["perdidas"] == 0)


def test_el_indice_de_desarrollo_no_tiene_desarrollo_ni_prueba():
    indice, codificador = _indice()
    externas = [
        MencionExterna("VIH avanzado", "B20", "d-1", "distemist"),  # documento de desarrollo
        MencionExterna("tos crónica", "R05", "x-1", "distemist"),  # documento de prueba
        MencionExterna("cirrosis hepática", "K74.6", "S1", "distemist"),
    ]
    pliegue, info = mf.topk_desarrollo(
        indice, codificador, TRAIN_6, DEV, externas, ["x-1"], VARIANTES, k=5
    )
    assert info["textos_sinonimos"] - info["textos_codiesp"] == {"cirrosis hepática"}
    assert info["verificacion_origen"]["ids_dev"] == 2
    assert info["verificacion_origen"]["ids_test"] == 1
    # El top-k trae también las variantes de estrés
    assert {"VIH", "vih"} <= set(pliegue.topk)
    assert all(len(top) == 5 for top in pliegue.topk.values())


def test_el_modelo_va_y_vuelve_por_json(rejilla):
    indice, codificador = _indice()
    modelo, eleccion = mf.entrenar_final(indice, codificador, TRAIN_6, [], {}, 3, 5)
    datos = json.loads(json.dumps(mf.modelo_a_json(modelo, eleccion["umbral"])))
    otro, umbral = mf.modelo_de_json(datos)
    assert umbral == eleccion["umbral"]
    assert otro.pesos.tolist() == pytest.approx(modelo.pesos.tolist())
    datos["atributos"] = datos["atributos"][:-1]
    with pytest.raises(ValueError, match="atributos"):
        mf.modelo_de_json(datos)


def test_evaluar_pide_la_variante_de_estres():
    with pytest.raises(ValueError, match="errores_10"):
        mf.evaluar("sapbert", None, None, [], [], {}, {}, {}, [], variantes=("original",))


def _metas_de(f1, precision, f1_ref=0.885, precision_ref=0.9, ic=(0.01, 0.02), top5=(13, 26),
              caidas=(0.02, 0.03)):  # fmt: skip
    def metodo(f1_, p, caida):
        return {
            "categoria": {"original": {"micro": {"f1": f1_, "precision": p}}},
            "caida_errores_10": caida,
        }

    return mf.metas(
        {
            "validacion_cruzada": {
                "comparacion": {"micro": {"f1": {"diferencia": 0.04, "ic95": list(ic)}}}
            },
            "desarrollo": {
                "metodos": {
                    mf.ELEGIDO: metodo(f1, precision, caidas[0]),
                    mf.REFERENCIA: metodo(f1_ref, precision_ref, caidas[1]),
                },
                "recuperadas": {
                    "recuperadas": {"top5": top5[0]},
                    "perdidas_base": {"top5": top5[1]},
                },
            },
        }
    )


def test_metas():
    cumple = [m["cumple"] for m in _metas_de(0.91, 0.895)]
    assert cumple == [True, True, True, True, True, True]
    cumple = [m["cumple"] for m in _metas_de(0.885, 0.88, ic=(-0.01, 0.02), top5=(12, 26),
                                            caidas=(0.04, 0.03))]  # fmt: skip
    assert cumple == [False, False, False, False, False, False]


def test_subcomando_final(tmp_path, monkeypatch, capsys, rejilla):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    prueba = [_documento("x-1", "Con tos.", ("R05", "tos"))]
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV, "test": prueba})
    _cantemist(tmp_path / "cantemist")
    _distemist(tmp_path / "distemist")
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [])
    argumentos = (
        ["final", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
        + ["--cantemist", "cantemist", "--distemist", "distemist", "--arnes", "no.json"]
        + ["--pliegues", "3", "--bootstrap", "5", "--salida", "f.json"]
    )
    main(argumentos)
    assert "| Meta | Valor | Cumple |" in capsys.readouterr().out
    guardado = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert guardado["eleccion"]["fuentes"] == list(mf.FUENTES_ELEGIDAS)
    # Una sola vez: no sobrescribe
    with pytest.raises(SystemExit, match="una sola vez"):
        main(argumentos)
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [" M x.py"])
    with pytest.raises(SystemExit, match="árbol limpio"):
        main([*argumentos[:-1], "otro.json"])


def test_decidir_entidades_aplica_la_segunda_etapa_y_las_exclusiones():
    from evaluacion import costo_normalizacion as cn
    from evaluacion.normalizador import Normalizador
    from ner_clinic.tipos import CategoriaCharlson as C

    indice, codificador = _indice()
    normalizador = Normalizador(indice, codificador, 5)
    vistos = []

    def decidir(texto, top):
        vistos.append(len(top))
        return C.TUMOR_MALIGNO, "C80.1"

    salida = cn.decidir_entidades(normalizador, ["linfoma", "mioma uterino"], decidir)
    assert salida == [(C.TUMOR_MALIGNO, "C80.1"), None]  # el mioma lo quita la exclusión
    assert vistos == [5, 5]


def test_la_segunda_etapa_del_contenedor_es_la_de_desarrollo(rejilla):
    """`decidir_entidades` (costo en Docker) decide igual que `mapeador_de` (desarrollo),
    con el modelo guardado en el JSON."""
    from evaluacion import costo_normalizacion as cn
    from evaluacion.clasificador import decisor
    from evaluacion.dos_etapas import mapeador_de
    from evaluacion.normalizador import Normalizador

    indice, codificador = _indice()
    modelo, eleccion = mf.entrenar_final(indice, codificador, TRAIN_6, [], {}, 3, 5)
    guardado = mf.modelo_de_json(
        json.loads(json.dumps(mf.modelo_a_json(modelo, eleccion["umbral"])))
    )
    pliegue, _ = mf.topk_desarrollo(indice, codificador, TRAIN_6, DEV, [], ["x-1"], VARIANTES, k=5)
    textos = list(pliegue.topk)
    en_dev = mapeador_de(decisor(modelo, eleccion["umbral"]), pliegue.topk)
    final = indice.con_sinonimos(mf.sinonimos_de(TRAIN_6), codificador)
    normalizador = Normalizador(final, codificador, 5)
    normalizador.precalcular(textos)
    assert cn.decidir_entidades(normalizador, textos, decisor(*guardado)) == [
        en_dev(t) for t in textos
    ]
