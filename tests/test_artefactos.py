"""Tests de los artefactos del normalizador (#164), sin red ni modelos."""

import json

import pytest

from evaluacion import artefactos as ev
from evaluacion import medicion_final as mf
from evaluacion.__main__ import main
from evaluacion.clasificador import decisor
from evaluacion.dos_etapas import mapeador_de
from ner_clinic.normalizacion import artefactos as lib
from tests.test_dos_etapas import TRAIN_6, _indice, _mini
from tests.test_normalizador import DEV, Trigramas, _documento

PROHIBIDOS = {"dev": [d.id for d in DEV], "test": ["x-1"]}


@pytest.fixture
def rejilla(monkeypatch):
    from tests.test_clasificador import _rejilla_chica

    _rejilla_chica(monkeypatch)


@pytest.fixture
def construidos(rejilla, tmp_path):
    indice, codificador = _indice()
    final, meta, clasificador = ev.construir(indice, codificador, TRAIN_6, PROHIBIDOS, 3)
    ruta_indice, ruta_clasificador = tmp_path / "indice.pt", tmp_path / "clasificador.json"
    lib.guardar_indice(final, ruta_indice, meta)
    ruta_clasificador.write_text(json.dumps(clasificador), encoding="utf-8")
    return indice, codificador, final, meta, clasificador, ruta_indice, ruta_clasificador


def test_el_indice_final_es_catalogo_mas_sinonimos_de_entrenamiento(construidos):
    indice, _, final, meta, clasificador, *_ = construidos
    assert (
        meta["filas"] == len(final.codigos) == len(indice.codigos) + len(mf.sinonimos_de(TRAIN_6))
    )
    assert meta["verificacion_origen"]["ids_dev"] == len(DEV)
    assert (
        meta["huella_indice"] == lib.huella_indice(final) == clasificador["meta"]["huella_indice"]
    )


def test_el_indice_va_y_vuelve(construidos):
    *_, final, _, _, ruta_indice, _ = construidos
    leido, meta = lib.cargar_indice(ruta_indice)
    assert leido.codigos == final.codigos and leido.descripciones == final.descripciones
    assert [leido.origen(i) for i in range(len(leido.codigos))] == final.origenes
    assert lib.huella_indice(leido) == meta["huella_indice"]


def test_cargar_decide_como_el_metodo_medido(construidos):
    """Lo guardado y cargado por la librería decide igual que el clasificador de la
    evaluación con el mismo entrenamiento (lo que se midió en #154)."""
    indice, codificador, _, _, clasificador, ruta_indice, ruta_clasificador = construidos
    segunda = lib.cargar(ruta_indice, ruta_clasificador, lambda: codificador)
    modelo, eleccion = mf.entrenar_final(indice, codificador, TRAIN_6, [], {}, 3, ev.K)
    assert clasificador["umbral"] == eleccion["umbral"]
    pliegue, _ = mf.topk_desarrollo(indice, codificador, TRAIN_6, DEV, [], ["x-1"], ("original",))
    en_dev = mapeador_de(decisor(modelo, eleccion["umbral"]), pliegue.topk)
    textos = list(pliegue.topk)
    decisiones = segunda.decidir(textos)
    assert [None if (x := decisiones[t]) is None else (x.categoria, x.codigo) for t in textos] == [
        en_dev(t) for t in textos
    ]
    comprobacion = ev.comprobar(segunda, DEV, None)
    assert comprobacion["coincide"] is False
    micro = {"f1": comprobacion["f1_desarrollo"], **comprobacion["conteos"]}
    assert ev.comprobar(segunda, DEV, micro)["coincide"] is True
    # Mismo F1 con otros conteos no basta
    otro = {**micro, "vp": micro["vp"] + 1}
    assert ev.comprobar(segunda, DEV, otro)["coincide"] is False


def test_el_clasificador_no_tiene_texto_de_corpus(construidos):
    *_, clasificador, _, _ = construidos
    texto = json.dumps(clasificador, ensure_ascii=False).casefold()
    # Los nombres de las categorías ("categoria_diabetes") sí están: son atributos
    for palabra in ("infarto cerebral", '"diabetes"', "epoc", '"vih"', "tuvo un"):
        assert palabra not in texto
    assert len(clasificador["pesos"]) == len(clasificador["atributos"]) == 39


def test_cargar_falla_si_el_indice_es_otro(construidos):
    *_, ruta_indice, ruta_clasificador = construidos
    datos = json.loads(ruta_clasificador.read_text(encoding="utf-8"))
    datos["meta"]["huella_indice"] = "0" * 64
    ruta_clasificador.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluacion artefactos"):
        lib.cargar(ruta_indice, ruta_clasificador, Trigramas)


def test_cargar_falla_si_alteran_las_filas_del_indice(construidos):
    """La huella se recalcula al cargar: no basta con copiar la meta del archivo viejo."""
    *_, ruta_indice, ruta_clasificador = construidos
    indice, meta = lib.cargar_indice(ruta_indice)
    indice.descripciones[-1] = "otro texto"
    lib.guardar_indice(indice, ruta_indice, meta)
    with pytest.raises(ValueError, match="no es aquel"):
        lib.cargar(ruta_indice, ruta_clasificador, Trigramas)


def test_cargar_falla_si_cambia_la_configuracion_de_sapbert(construidos):
    *_, ruta_indice, ruta_clasificador = construidos
    datos = json.loads(ruta_clasificador.read_text(encoding="utf-8"))
    datos["meta"]["configuracion_embeddings"] = "otra"
    ruta_clasificador.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="configuración de SapBERT"):
        lib.cargar(ruta_indice, ruta_clasificador, Trigramas)


def test_cargar_falla_si_faltan_artefactos(tmp_path, construidos):
    *_, ruta_indice, ruta_clasificador = construidos
    with pytest.raises(FileNotFoundError, match="evaluacion artefactos"):
        lib.cargar(tmp_path / "no.pt", ruta_clasificador, Trigramas)
    with pytest.raises(FileNotFoundError, match="evaluacion artefactos"):
        lib.cargar(ruta_indice, tmp_path / "no.json", Trigramas)


def test_el_clasificador_versionado_carga(tmp_path):
    """El JSON del paquete (si ya se construyó) tiene los atributos de la librería y su
    comprobación contra #154."""
    if not lib.RUTA_CLASIFICADOR.is_file():
        pytest.skip("todavía no se ha construido")
    datos = lib.leer_clasificador()
    from ner_clinic.normalizacion.decision import modelo_de_json

    modelo_de_json(datos)
    comprobacion = datos["meta"]["comprobacion_desarrollo"]
    assert comprobacion["coincide"] and comprobacion["eleccion_coincide"]


def test_subcomando_artefactos(tmp_path, monkeypatch, capsys, rejilla):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    prueba = [_documento("x-1", "Con tos.", ("R05", "tos"))]
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV, "test": prueba})
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [])
    argumentos = ["artefactos", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
    argumentos += ["--pliegues", "3", "--indice-final", "i.pt", "--clasificador", "c.json"]
    # Sin la medición de #154 no hay con qué comparar: guarda, pero falla
    with pytest.raises(SystemExit, match="no reproducen"):
        main([*argumentos, "--medicion-154", "no.json"])
    guardado = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert guardado["meta"]["comprobacion_desarrollo"]["f1_154"] is None
    assert "Comprobación en desarrollo" in capsys.readouterr().out
    # Con una medición de #154 igual a lo obtenido, pasa
    comprobacion = guardado["meta"]["comprobacion_desarrollo"]
    eleccion = guardado["meta"]["eleccion"]
    micro = {"f1": comprobacion["f1_desarrollo"], **comprobacion["conteos"]}
    medicion = {
        "desarrollo": {"metodos": {ev.METODO_154: {"categoria": {"original": {"micro": micro}}}}},
        "clasificador_final": {ev.METODO_154: eleccion},
    }
    (tmp_path / "m.json").write_text(json.dumps(medicion), encoding="utf-8")
    main([*argumentos, "--medicion-154", "m.json"])
    assert lib.cargar(tmp_path / "i.pt", tmp_path / "c.json", Trigramas)
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [" M x.py"])
    with pytest.raises(SystemExit, match="árbol limpio"):
        main([*argumentos[:-1], "otro.json"])
    assert not (tmp_path / "otro.json").exists()  # no escribe nada
