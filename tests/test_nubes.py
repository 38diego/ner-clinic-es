"""Tests del lector de NUBes y de la evaluación de la negación (`evaluacion.nubes`).

Sin red ni modelo: usan el mini corpus BRAT de `tests/datos/nubes_mini/` (frases
sintéticas escritas a mano con el formato de NUBes) y un `ExtractorFalso` que marca
palabras fijas en cada frase.
"""

import json
import zipfile
from pathlib import Path

import pytest

from evaluacion import nubes
from evaluacion.__main__ import main
from evaluacion.cache import ExtractorConCache
from evaluacion.corpus import ErrorCorpus
from evaluacion.nubes import Evaluada, Frase, Region
from ner_clinic.extraccion import ExtractorFalso
from ner_clinic.limpieza import limpiar
from ner_clinic.segmentacion import segmentar
from ner_clinic.tipos import Estado

MINI = Path(__file__).parent / "datos" / "nubes_mini"
PALABRAS = (
    "fiebre",
    "disnea",
    "diabetes mellitus",
    "neumonía",
    "derrame pleural",
    "Dolor abdominal",
    "vómitos",
    "Hipertensión arterial",
    "alteraciones",
    "Edema",
    "edema",
    "cefalea",
    "asma bronquial",
    "dolor torácico",
)


def extractor_para(frases, palabras=PALABRAS) -> ExtractorFalso:
    """`ExtractorFalso` que marca cada aparición de `palabras` en los fragmentos."""
    salidas = {}
    for fragmento in nubes.fragmentos_de(frases):
        entidades = []
        for palabra in palabras:
            desde = 0
            while (inicio := fragmento.find(palabra, desde)) != -1:
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
        salidas[fragmento] = sorted(entidades, key=lambda e: e["start"])
    return ExtractorFalso(salidas)


@pytest.fixture
def frases():
    return nubes.leer(MINI)


@pytest.fixture
def falso(frases):
    return extractor_para(frases)


# --- Lectura -------------------------------------------------------------------------


def test_lee_una_frase_por_linea_sin_las_vacias(frases):
    ids = [f.id for f in frases]
    assert ids[0] == "sample-001.mini.chico:1"
    assert "sample-001.mini.chico:5" not in ids  # línea vacía
    assert ids[-1] == "sample-002.mini.preil:2"
    assert len(frases) == 12


def test_lee_solo_las_muestras_pedidas():
    frases = nubes.leer(MINI, ["SAMPLE-002"])
    assert [f.texto for f in frases] == ["Niega dolor torácico .", "Probable derrame pleural ."]


def test_muestra_que_no_existe_falla():
    with pytest.raises(ErrorCorpus, match="No hay archivos"):
        nubes.leer(MINI, ["SAMPLE-010"])


def test_regiones_en_posiciones_de_la_frase(frases):
    for frase in frases:
        for region in frase.regiones:
            assert 0 <= region.inicio < region.fin <= len(frase.texto)
    primera = frases[0]
    (region,) = primera.regiones
    assert primera.texto[region.inicio : region.fin] == "refiere fiebre ni disnea"
    assert (region.clase, region.clave) == ("negacion", "No")


def test_negacion_morfologica_es_la_palabra_misma(frases):
    frase = next(f for f in frases if f.id == "sample-001.mini.chico:2")
    (region,) = frase.regiones
    assert frase.texto[region.inicio : region.fin] == "afebril"
    assert region.clase == "negacion"


def test_clave_de_incertidumbre_con_alcance(frases):
    frase = next(f for f in frases if f.id == "sample-002.mini.preil:2")
    (region,) = frase.regiones
    assert frase.texto[region.inicio : region.fin] == "derrame pleural"
    assert (region.clase, region.clave) == ("incertidumbre", "Probable")


def test_alcance_discontinuo_toma_la_clase_de_la_clave(frases):
    frase = next(f for f in frases if f.id == "sample-001.mini.chico:7")
    (region,) = frase.regiones
    assert frase.texto[region.inicio : region.fin] == "alteraciones"
    assert (region.clase, region.clave) == ("incertidumbre", "aparentes")


def _escribir(carpeta: Path, texto: str, ann: str) -> Path:
    muestra = carpeta / "NUBes" / "SAMPLE-001"
    muestra.mkdir(parents=True)
    (muestra / "x.txt").write_text(texto, encoding="utf-8")
    (muestra / "x.ann").write_text(ann, encoding="utf-8")
    return muestra / "x.txt"


def test_offset_que_no_cuadra_falla(tmp_path):
    ruta = _escribir(tmp_path, "No fiebre .\n", "T1\tNegSynMarker 0 2\tSi\n")
    with pytest.raises(ErrorCorpus, match="no coincide"):
        nubes.leer_archivo(ruta)


def test_anotacion_discontinua_falla(tmp_path):
    ruta = _escribir(tmp_path, "No fiebre .\n", "T1\tDISO 0 2;3 9\tNo fiebre\n")
    with pytest.raises(ErrorCorpus, match="discontinua"):
        nubes.leer_archivo(ruta)


def test_alcance_que_cruza_un_salto_de_linea_falla(tmp_path):
    ann = "T1\tNegSynMarker 0 2\tNo\nT2\tPhrase 3 15\tfiebre . Tos\nR1\tScope Arg1:T1 Arg2:T2\n"
    ruta = _escribir(tmp_path, "No fiebre .\nTos .\n", ann)
    with pytest.raises(ErrorCorpus, match="cruza un salto de línea"):
        nubes.leer_archivo(ruta)


def test_alcance_que_no_sale_de_una_clave_falla(tmp_path):
    ann = "T1\tDISO 0 2\tNo\nT2\tDISO 3 9\tfiebre\nR1\tScope Arg1:T1 Arg2:T2\n"
    ruta = _escribir(tmp_path, "No fiebre .\n", ann)
    with pytest.raises(ErrorCorpus, match="no es una clave"):
        nubes.leer_archivo(ruta)


# --- Estado esperado -----------------------------------------------------------------


def test_region_de_exige_la_mitad_de_la_mencion():
    regiones = [Region(10, 20, "negacion", "no")]
    assert nubes.region_de(10, 20, regiones) == regiones[0]
    assert nubes.region_de(5, 15, regiones) == regiones[0]  # justo la mitad
    assert nubes.region_de(4, 15, regiones) is None
    assert nubes.region_de(0, 5, regiones) is None


def test_region_de_prefiere_la_mas_corta():
    incierta = Region(0, 30, "incertidumbre", "posible")
    negada = Region(15, 30, "negacion", "sin")
    assert nubes.region_de(15, 30, [incierta, negada]) == negada
    assert nubes.region_de(0, 8, [incierta, negada]) == incierta
    # A igual largo, gana la negación.
    otra = Region(15, 30, "incertidumbre", "o")
    assert nubes.region_de(15, 30, [otra, negada]) == negada


def test_esperado_de():
    assert nubes.esperado_de(None) == "afirmada"
    assert nubes.esperado_de(Region(0, 1, "negacion", "no")) == "negada"
    assert nubes.esperado_de(Region(0, 1, "incertidumbre", "o")) == "incierta"


def test_estado_por_texto_reproduce_la_clasificacion_de_la_tesis():
    # La palabra está negada en una parte de la frase: sale negada aunque esté confirmada
    # en otra (lo que corrigió #73).
    assert nubes.estado_por_texto("Edema maleolar , no edema pulmonar .", "Edema") == "negada"
    assert nubes.estado_por_texto("Edema maleolar .", "Edema") == "confirmada"
    assert nubes.estado_por_texto("Madre con asma . Niega asma .", "asma") == "familiar"
    assert nubes.estado_por_texto("Niega asma .", "   ") == "confirmada"
    assert nubes.estado_por_texto("   ", "asma") == "confirmada"


def test_estado_por_texto_usa_la_prioridad_del_contexto():
    assert nubes.estado_por_texto("sospecha de TEP .", "TEP") == "posible"
    assert nubes.estado_por_texto("Madre con sospecha de TEP .", "TEP") == "familiar"


def test_estado_por_texto_usa_el_texto_limpio():
    assert nubes.estado_por_texto("  no fiebre", "fiebre") == "negada"


# --- Corrida -------------------------------------------------------------------------


def test_cada_frase_se_analiza_sola(frases, falso):
    nubes.correr(frases, falso)
    fragmentos = {f for llamada in falso.llamadas for f in llamada}
    assert fragmentos == {f.texto.strip() for f in frases}


def test_fragmentos_de_sin_repetir_y_del_mas_corto(frases):
    fragmentos = nubes.fragmentos_de([*frases, frases[0]])
    assert len(fragmentos) == len(set(fragmentos)) == len(frases)
    assert [len(f) for f in fragmentos] == sorted(len(f) for f in fragmentos)
    for frase in frases:
        assert all(f.texto in fragmentos for f in segmentar(limpiar(frase.texto).texto))


def test_estados_del_mini_corpus(frases, falso):
    evaluadas = [e for lista in nubes.correr(frases, falso) for e in lista]
    estados = {(e.frase.rsplit(":", 1)[1], e.texto): (e.esperado, e.obtenido) for e in evaluadas}
    # Sin la fusión (#55), "fiebre" y "disnea" salen por separado, las dos negadas
    assert estados[("1", "fiebre")] == ("negada", "negada")
    assert estados[("1", "disnea")] == ("negada", "negada")
    assert estados[("2", "diabetes mellitus")] == ("afirmada", "confirmada")
    # "neumonía" está en el alcance de "Posible"; "derrame pleural", en el de "sin", que
    # corta el alcance posible (#83).
    assert estados[("3", "neumonía")] == ("incierta", "posible")
    assert estados[("3", "derrame pleural")] == ("negada", "negada")
    # "Sin ... aparentes" es incertidumbre en el gold: "aparentes" hacia atrás (#83).
    assert estados[("7", "alteraciones")] == ("incierta", "posible")
    # Por posición (#73): "Edema" queda confirmada aunque "edema" esté negada después.
    assert estados[("8", "Edema")] == ("afirmada", "confirmada")
    assert estados[("8", "edema")] == ("negada", "negada")
    # "Tampoco presenta" es una clave del pipeline desde #65.
    assert estados[("9", "cefalea")] == ("negada", "negada")
    assert estados[("10", "asma bronquial")] == ("afirmada", "familiar")
    # "Ha cedido" no es clave (resuelta no es negada, PRD de negación, FR5).
    assert estados[("11", "cefalea")] == ("negada", "confirmada")
    # "Probable" es clave de incertidumbre desde #83.
    probable = next(e for e in evaluadas if e.frase == "sample-002.mini.preil:2")
    assert (probable.texto, probable.esperado, probable.obtenido) == (
        "derrame pleural",
        "incierta",
        "posible",
    )


def test_precalentar_llena_la_cache(frases, falso, tmp_path):
    cache = ExtractorConCache(falso, carpeta=tmp_path, version_transformers="x")
    assert nubes.precalentar(frases, cache, lote=3) >= 0
    assert cache.fallos == len(frases)
    assert cache.llamadas == 4  # 12 fragmentos en lotes de 3
    nubes.correr(frases, cache)
    assert cache.fallos == len(frases)  # el análisis solo lee la caché
    assert cache.aciertos >= len(frases)


def test_precalentar_con_siglas_en_minuscula(tmp_path):
    """La caché se llena con el texto que ve el modelo, con las siglas en mayúscula (#114)."""
    frases = [
        nubes.Frase("a:1", "Paciente con epoc y hta ."),
        nubes.Frase("a:2", "Paciente con asma ."),
    ]
    cache = ExtractorConCache(ExtractorFalso({}), carpeta=tmp_path, version_transformers="x")
    nubes.precalentar(frases, cache)
    assert (cache.fallos, cache.llamadas) == (2, 1)
    nubes.correr(frases, cache)
    assert (cache.fallos, cache.llamadas) == (2, 1)  # el análisis solo lee la caché


# --- Métricas ------------------------------------------------------------------------


def _evaluada(esperado, obtenido, por_texto=None, inicio=0):
    return Evaluada(
        frase="f:1",
        texto="x",
        inicio=inicio,
        fin=inicio + 1,
        entidad="x",
        esperado=esperado,
        obtenido=obtenido,
        por_texto=por_texto or obtenido,
        clave_gold=None,
    )


def test_conteos_negada_ignora_las_inciertas():
    evaluadas = [
        _evaluada("negada", "negada"),
        _evaluada("negada", "confirmada"),
        _evaluada("negada", "familiar"),
        _evaluada("afirmada", "negada"),
        _evaluada("afirmada", "confirmada"),
        _evaluada("incierta", "negada"),
    ]
    assert tuple(nubes.conteos_negada(evaluadas)) == (1, 1, 2)


def test_conteos_negada_cuenta_posible_como_no_negada():
    evaluadas = [_evaluada("negada", "posible"), _evaluada("afirmada", "posible")]
    assert tuple(nubes.conteos_negada(evaluadas)) == (0, 0, 1)


def test_conteos_posible_sobre_todas_las_menciones():
    evaluadas = [
        _evaluada("incierta", "posible"),
        _evaluada("incierta", "posible"),
        _evaluada("incierta", "negada"),
        _evaluada("incierta", "confirmada"),
        _evaluada("negada", "posible"),
        _evaluada("afirmada", "posible"),
        _evaluada("afirmada", "confirmada"),
        _evaluada("negada", "negada"),
    ]
    assert tuple(nubes.conteos_posible(evaluadas)) == (2, 2, 2)


def test_metricas_posible_con_ic():
    por_frase = [
        [_evaluada("incierta", "posible")],
        [_evaluada("incierta", "confirmada"), _evaluada("afirmada", "posible")],
        [_evaluada("incierta", "posible")],
    ]
    posible = nubes.metricas_posible(por_frase, replicas=50)
    assert (posible["vp"], posible["fp"], posible["fn"]) == (2, 1, 1)
    assert posible["precision"] == posible["recall"] == round(2 / 3, 4)
    assert set(posible["ic95"]) == {"precision", "recall", "f1"}
    for inferior, superior in posible["ic95"].values():
        assert 0 <= inferior <= superior <= 1


def test_metricas_posible_sin_el_estado_dan_cero():
    posible = nubes.metricas_posible([[_evaluada("incierta", "confirmada")]], replicas=10)
    assert (posible["vp"], posible["fp"], posible["fn"]) == (0, 0, 1)
    assert posible["f1"] == 0.0


def test_matriz_tiene_filas_esperadas_y_columnas_obtenidas():
    m = nubes.matriz(
        [
            _evaluada("negada", "negada"),
            _evaluada("incierta", "familiar"),
            _evaluada("incierta", "posible"),
        ]
    )
    assert list(m) == ["negada", "incierta", "afirmada"]
    assert list(m["negada"]) == ["confirmada", "negada", "posible", "familiar"]
    assert m["negada"]["negada"] == 1
    assert m["incierta"]["familiar"] == 1
    assert m["incierta"]["posible"] == 1
    assert sum(sum(fila.values()) for fila in m.values()) == 3


def test_matriz_con_un_estado_desconocido_falla():
    with pytest.raises(ValueError, match="clase desconocida"):
        nubes.matriz([_evaluada("negada", "dudosa")])


def test_obtenidos_cubren_los_estados_del_pipeline():
    assert {e.value for e in Estado} <= set(nubes.OBTENIDOS)


def test_por_texto_compara_el_estado_obtenido_con_el_de_la_clasificacion_por_texto():
    evaluadas = [
        _evaluada("afirmada", "confirmada", por_texto="negada"),
        _evaluada("negada", "negada"),
        _evaluada("incierta", "confirmada", por_texto="negada"),
    ]
    resultado = nubes.por_texto(evaluadas)
    assert resultado["menciones_que_cambian"] == 2
    assert resultado["transiciones"] == {"negada->confirmada": 2}
    assert resultado["aciertan_por_texto"] == 0
    assert resultado["aciertan_por_posicion"] == 1
    assert resultado["metricas_por_texto"]["fp"] == 1


# --- Muestra de discrepancias --------------------------------------------------------


def test_muestrear_es_reproducible_y_corto():
    evaluadas = [_evaluada("negada", "confirmada", inicio=i) for i in range(40)] + [
        _evaluada("afirmada", "negada", inicio=100 + i) for i in range(5)
    ]
    textos = {"f:1": "x" * 300}
    a = nubes.muestrear(evaluadas, textos, n=10)
    assert a == nubes.muestrear(evaluadas, textos, n=10)
    assert [d["esperado"] for d in a].count("negada") == 5
    assert [d["esperado"] for d in a].count("afirmada") == 5
    assert all(len(d["contexto"]) <= 120 for d in a)
    assert all(d["categoria"] is None for d in a)


def test_conservar_clasificacion():
    muestra = [{"frase": "f:1", "inicio": 0, "fin": 1, "contexto": "x", "categoria": None}]
    previa = {"frase": "f:1", "inicio": 0, "fin": 1, "contexto": "x", "categoria": "otra"}
    anterior = {"discrepancias": [{**previa, "nota": "n"}]}
    assert nubes.conservar_clasificacion(muestra, None) is muestra
    (item,) = nubes.conservar_clasificacion(muestra, anterior)
    assert (item["categoria"], item["nota"]) == ("otra", "n")
    # Si el contexto cambió, no se copia.
    cambiada = [{**muestra[0], "contexto": "y", "categoria": None}]
    assert nubes.conservar_clasificacion(cambiada, anterior)[0]["categoria"] is None


def test_validar():
    buenas = {"discrepancias": [{"frase": "f:1", "categoria": "otra", "contexto": "x"}]}
    assert nubes.validar(buenas) == []
    malas = {"discrepancias": [{"frase": "f:1", "categoria": None, "contexto": "x" * 121}]}
    assert len(nubes.validar(malas)) == 2


# --- Evaluación completa -------------------------------------------------------------


def test_evaluar_el_mini_corpus(frases, falso, tmp_path):
    cache = ExtractorConCache(falso, carpeta=tmp_path, version_transformers="x")
    resultados, detalle = nubes.evaluar(
        frases, cache, ["SAMPLE-001", "SAMPLE-002"], cache=cache, replicas=50
    )
    assert resultados["corpus"]["frases"] == 12
    assert resultados["corpus"]["frases_con_negacion"] == 8
    assert resultados["corpus"]["frases_con_incertidumbre"] == 3
    assert resultados["menciones_evaluadas"] == 16
    assert resultados["matriz"] == {
        "negada": {"confirmada": 1, "negada": 7, "posible": 0, "familiar": 0},
        "incierta": {"confirmada": 0, "negada": 0, "posible": 3, "familiar": 0},
        "afirmada": {"confirmada": 4, "negada": 0, "posible": 0, "familiar": 1},
    }
    negada = resultados["negada"]
    # Por posición (#73), "Edema" de la frase 8 ya no sale negada: el FP desaparece.
    # Desde #65, "Tampoco presenta" niega la cefalea de la frase 9. Queda el FN de la
    # frase 11: "Ha cedido" es resuelta, no negada, y no es clave del pipeline.
    assert (negada["vp"], negada["fp"], negada["fn"]) == (7, 0, 1)
    assert negada["precision"] == 1.0
    assert negada["recall"] == round(7 / 8, 4)
    # Con las claves de #83, las 3 inciertas salen posibles y no hay FP.
    posible = resultados["posible"]
    assert (posible["vp"], posible["fp"], posible["fn"]) == (3, 0, 0)
    assert posible["f1"] == 1.0
    assert resultados["salio_posible"] == 3
    assert resultados["salio_familiar"] == 1
    assert list(resultados["negada_por_muestra"]) == ["SAMPLE-001", "SAMPLE-002"]
    assert resultados["negada_por_muestra"]["SAMPLE-002"]["vp"] == 1
    assert resultados["por_texto"]["menciones_que_cambian"] == 1
    assert resultados["por_texto"]["transiciones"] == {"negada->confirmada": 1}
    assert resultados["por_texto"]["metricas_por_texto"]["fp"] == 1
    assert resultados["cache"]["fallos"] == 12
    assert len(resultados["discrepancias"]) == 1
    assert all(len(d["contexto"]) <= 120 for d in resultados["discrepancias"])
    assert len(detalle["menciones"]) == 16
    assert "NUBes" in resultados["atribucion"]

    tabla = nubes.tabla_matriz(resultados)
    assert "| Esperado \\ obtenido | confirmada | negada | posible | familiar | Total |" in tabla
    assert "| negada | 1 | 7 | 0 | 0 | 8 |" in tabla
    assert "| incierta | 0 | 0 | 3 | 0 | 3 |" in tabla
    assert "P 1.000 R 0.875" in nubes.resumen(resultados)
    assert "Posible: P 1.000 R 1.000 F1 1.000 (VP 3, FP 0, FN 0)" in nubes.resumen(resultados)
    for item in resultados["discrepancias"]:
        item["categoria"] = "otra"
    assert "| Otra | 1 | 0 | 1 |" in nubes.tabla_discrepancias(resultados)


def test_evaluar_sin_cache_no_precalienta(frases, falso):
    resultados, _ = nubes.evaluar(frases, falso, ["SAMPLE-001"], replicas=10)
    assert resultados["tiempos"]["segundos_inferencia"] == 0.0
    assert resultados["cache"] is None


def test_rutas_por_defecto_y_carpeta(tmp_path):
    assert nubes.ruta_resultados(None) == Path("evaluacion/resultados/nubes.json")
    assert nubes.ruta_resultados(tmp_path) == tmp_path / "nubes.json"
    assert nubes.ruta_resultados(tmp_path / "a.json") == tmp_path / "a.json"
    assert nubes.ruta_detalle(None) == Path("data/evaluacion/nubes_menciones.json")
    assert nubes.ruta_detalle(tmp_path) == tmp_path / "nubes_menciones.json"
    assert nubes.ruta_detalle(tmp_path / "b.json") == tmp_path / "b.json"


# --- Descarga ------------------------------------------------------------------------


def _zip_falso(tmp_path: Path) -> Path:
    """Zip con la estructura del de GitHub y el mini corpus dentro."""
    ruta = tmp_path / "origen.zip"
    with zipfile.ZipFile(ruta, "w") as z:
        for archivo in sorted(MINI.rglob("*")):
            if archivo.is_file():
                z.write(archivo, f"{nubes.CARPETA_CORPUS}/{archivo.relative_to(MINI)}")
    return ruta


def test_descargar_verifica_el_hash_y_descomprime(tmp_path):
    origen = _zip_falso(tmp_path)
    sha = nubes.sha256_de(origen)
    destino = tmp_path / "nubes"
    corpus = nubes.descargar(destino, url=origen.as_uri(), sha256=sha)
    assert corpus == destino / nubes.CARPETA_CORPUS
    assert len(nubes.leer(corpus)) == 12
    # Segunda vez: no baja nada (la URL ya no existe) ni vuelve a descomprimir.
    assert nubes.descargar(destino, url="file:///no/existe.zip", sha256=sha) == corpus


def test_descargar_con_hash_distinto_falla_y_borra(tmp_path):
    origen = _zip_falso(tmp_path)
    destino = tmp_path / "nubes"
    with pytest.raises(ErrorCorpus, match="SHA-256"):
        nubes.descargar(destino, url=origen.as_uri(), sha256="0" * 64)
    assert not any(destino.iterdir())


# --- Línea de comandos ---------------------------------------------------------------


def test_subcomando_nubes(monkeypatch, tmp_path, capsys, frases):
    falso = extractor_para(frases)
    monkeypatch.setattr("evaluacion.__main__.ExtractorHF", lambda batch_size: falso)
    monkeypatch.setattr(nubes, "descargar", lambda raiz: raiz)
    salida = tmp_path / "res.json"
    argumentos = [
        "nubes",
        "--raiz",
        str(MINI),
        "--muestras",
        "SAMPLE-001",
        "SAMPLE-002",
        "--salida",
        str(salida),
        "--detalle",
        str(tmp_path / "det.json"),
        "--cache",
        str(tmp_path / "cache"),
    ]
    main(argumentos)
    consola = capsys.readouterr().out
    assert "12 frases" in consola
    assert "Negada: P 1.000 R 0.875" in consola
    resultados = json.loads(salida.read_text(encoding="utf-8"))
    assert resultados["corpus"]["muestras"] == ["SAMPLE-001", "SAMPLE-002"]

    # La clasificación a mano sobrevive a una nueva corrida.
    for item in resultados["discrepancias"]:
        item["categoria"] = "otra"
    salida.write_text(json.dumps(resultados, ensure_ascii=False), encoding="utf-8")
    main([*argumentos, "--limite", "10"])
    assert "0 discrepancias de la muestra sin clasificar" in capsys.readouterr().out


def test_particiones_del_prd():
    assert nubes.MUESTRAS_DESARROLLO == (
        "SAMPLE-002",
        "SAMPLE-004",
        "SAMPLE-006",
        "SAMPLE-008",
        "SAMPLE-010",
    )
    assert nubes.MUESTRAS_RESERVADAS == (
        "SAMPLE-001",
        "SAMPLE-003",
        "SAMPLE-005",
        "SAMPLE-007",
        "SAMPLE-009",
    )
    assert set(nubes.MUESTRAS_DESARROLLO).isdisjoint(nubes.MUESTRAS_RESERVADAS)
    assert sorted(nubes.MUESTRAS_DESARROLLO + nubes.MUESTRAS_RESERVADAS) == list(nubes.MUESTRAS)


def test_muestras_de():
    assert nubes.muestras_de("desarrollo") == list(nubes.MUESTRAS_DESARROLLO)
    assert nubes.muestras_de("reservada") == list(nubes.MUESTRAS_RESERVADAS)
    assert nubes.muestras_de("todo") == list(nubes.MUESTRAS)
    assert nubes.muestras_de(None) == list(nubes.MUESTRAS)
    assert nubes.muestras_de(None, ["SAMPLE-003"]) == ["SAMPLE-003"]
    with pytest.raises(ValueError, match="no las dos"):
        nubes.muestras_de("desarrollo", ["SAMPLE-003"])


def _argumentos_nubes(tmp_path: Path, *seleccion: str) -> list[str]:
    return [
        "nubes",
        "--raiz",
        str(MINI),
        *seleccion,
        "--salida",
        str(tmp_path / "res.json"),
        "--detalle",
        str(tmp_path / "det.json"),
        "--cache",
        str(tmp_path / "cache"),
    ]


@pytest.mark.parametrize(
    ("particion", "muestras", "frases_esperadas"),
    [("reservada", ["SAMPLE-001"], 10), ("desarrollo", ["SAMPLE-002"], 2)],
)
def test_subcomando_nubes_con_particion(
    monkeypatch, tmp_path, capsys, frases, particion, muestras, frases_esperadas
):
    # El mini corpus solo tiene SAMPLE-001 y SAMPLE-002: se recorta cada partición a esas.
    monkeypatch.setitem(nubes.PARTICIONES, particion, tuple(muestras))
    monkeypatch.setattr(
        "evaluacion.__main__.ExtractorHF", lambda batch_size: extractor_para(frases)
    )
    monkeypatch.setattr(nubes, "descargar", lambda raiz: raiz)
    main(_argumentos_nubes(tmp_path, "--particion", particion))
    consola = capsys.readouterr().out
    assert f"NUBes partición {particion}: {muestras[0]}: {frases_esperadas} frases" in consola
    corpus = json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))["corpus"]
    assert corpus["particion"] == particion
    assert corpus["muestras"] == muestras


def test_subcomando_nubes_sin_seleccion_es_todo(monkeypatch, tmp_path, frases):
    leidas = []
    original = nubes.leer

    def leer(raiz, muestras):
        leidas.append(list(muestras))
        return original(raiz, ["SAMPLE-001", "SAMPLE-002"])

    monkeypatch.setattr(nubes, "leer", leer)
    monkeypatch.setattr(
        "evaluacion.__main__.ExtractorHF", lambda batch_size: extractor_para(frases)
    )
    monkeypatch.setattr(nubes, "descargar", lambda raiz: raiz)
    main(_argumentos_nubes(tmp_path))
    assert leidas == [list(nubes.MUESTRAS)]
    corpus = json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))["corpus"]
    assert corpus["particion"] == "todo"


def test_subcomando_nubes_con_muestras_no_registra_particion(monkeypatch, tmp_path, frases):
    monkeypatch.setattr(
        "evaluacion.__main__.ExtractorHF", lambda batch_size: extractor_para(frases)
    )
    monkeypatch.setattr(nubes, "descargar", lambda raiz: raiz)
    main(_argumentos_nubes(tmp_path, "--muestras", "SAMPLE-002"))
    corpus = json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))["corpus"]
    assert corpus["particion"] is None
    assert corpus["muestras"] == ["SAMPLE-002"]


def test_particion_y_muestras_a_la_vez_falla(capsys):
    with pytest.raises(SystemExit):
        main(["nubes", "--particion", "desarrollo", "--muestras", "SAMPLE-002"])
    assert "--muestras" in capsys.readouterr().err


def test_frase_es_inmutable():
    frase = Frase("a:1", "texto")
    with pytest.raises(AttributeError):
        frase.texto = "otro"  # type: ignore[misc]
