"""Tests de la normalización por embeddings (#131) y de los sinónimos de CodiEsp (#132).

Sin red: el codificador de prueba convierte cada texto en un vector de trigramas de
caracteres, así que textos parecidos quedan cerca. Los tests marcados `modelo` usan los
modelos reales (`pytest -m modelo`).
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluacion import codiesp
from evaluacion import normalizacion_codiesp as nc
from evaluacion import normalizador as nz
from evaluacion import sinonimos_codiesp as sc
from evaluacion.__main__ import main
from evaluacion.charlson_codiesp import Caso
from evaluacion.codiesp import DocumentoCodificado, MencionCodificada
from ner_clinic.tipos import CategoriaCharlson as C

DIMENSION = 256

CATALOGO = {
    "E11": "Diabetes mellitus tipo 2",
    "E11.9": "Diabetes mellitus tipo 2 sin complicaciones",
    "I25": "Enfermedad isquémica crónica del corazón",
    "I63.9": "Infarto cerebral, no especificado",
    "I10": "Hipertensión esencial (primaria)",
    "J44.9": "Enfermedad pulmonar obstructiva crónica, no especificada",
    "C81.90": "Linfoma de Hodgkin, no especificado, sitio no especificado",
    "R05": "Tos",
}


class Trigramas:
    """Codificador de prueba: trigramas de caracteres en minúscula, en `DIMENSION` cubetas."""

    def __init__(self):
        self.llamadas = 0
        self.textos = 0

    def __call__(self, textos):
        self.llamadas += 1
        self.textos += len(textos)
        vectores = torch.zeros(len(textos), DIMENSION)
        for i, texto in enumerate(textos):
            t = f"  {texto.lower()} "
            for j in range(len(t) - 2):
                vectores[i, sum(map(ord, t[j : j + 3])) % DIMENSION] += 1
        return torch.nn.functional.normalize(vectores, dim=-1)


MODELO = nz.ModeloEmbeddings("prueba/trigramas", "abc123def456789", "media", "MIT")


def _normalizador(k: int = 3) -> nz.Normalizador:
    codificador = Trigramas()
    return nz.Normalizador(nz.construir_indice(CATALOGO, codificador), codificador, k)


# --- Índice y búsqueda ---------------------------------------------------------------


def test_buscar_ordena_por_similitud():
    normalizador = _normalizador(k=3)
    candidatos = normalizador.candidatos("diabetes mellitus tipo 2 sin complicaciones")
    assert candidatos[0].codigo == "E11.9"
    assert candidatos[0].descripcion == CATALOGO["E11.9"]
    similitudes = [c.similitud for c in candidatos]
    assert similitudes == sorted(similitudes, reverse=True)
    assert len(candidatos) == 3


def test_k_mayor_que_el_catalogo():
    normalizador = _normalizador(k=50)
    assert len(normalizador.candidatos("tos")) == len(CATALOGO)


def test_la_busqueda_por_lotes_da_lo_mismo():
    codificador = Trigramas()
    indice = nz.construir_indice(CATALOGO, codificador)
    consultas = codificador(["tos", "infarto cerebral", "epoc", "hipertension"])
    assert indice.buscar(consultas, 2, lote=1) == indice.buscar(consultas, 2, lote=1024)


def test_precalcular_codifica_cada_texto_una_vez():
    normalizador = _normalizador()
    codificador = normalizador.codificador
    normalizador.precalcular(["tos", "tos", "infarto cerebral"])
    assert codificador.textos == len(CATALOGO) + 2
    normalizador.precalcular(["tos"])
    normalizador.candidatos("infarto cerebral")
    assert codificador.textos == len(CATALOGO) + 2
    normalizador.candidatos("epoc")
    assert codificador.textos == len(CATALOGO) + 3


def test_mapeador_con_umbral_y_categoria_de_quan():
    normalizador = _normalizador()
    assert normalizador.mapeador(0.0)("diabetes mellitus tipo 2") == (C.DIABETES, "E11.9")
    assert normalizador.mapeador(0.0)("hipertension esencial") is None  # I10: sin categoría
    assert normalizador.mapeador(1.01)("diabetes mellitus tipo 2") is None  # bajo el umbral


def test_un_codigo_padre_se_lee_como_su_hijo_no_especificado():
    normalizador = _normalizador()
    assert normalizador.codigo_para_categoria("E11") == "E11.9"
    assert normalizador.codigo_para_categoria("I63.9") == "I63.9"  # ya tiene categoría
    assert normalizador.codigo_para_categoria("I25") == "I25"  # sin hijo .9 en el catálogo
    assert normalizador.mapeador(0.0)("diabetes mellitus tipo 2") == (C.DIABETES, "E11.9")
    assert normalizador.mapeador(0.0)("enfermedad isquemica cronica del corazon") is None


def test_huella_del_catalogo():
    assert nz.huella_catalogo(CATALOGO) == nz.huella_catalogo(dict(reversed(CATALOGO.items())))
    assert nz.huella_catalogo(CATALOGO) != nz.huella_catalogo({**CATALOGO, "R05": "Tos seca"})


def test_indice_en_disco_se_guarda_y_se_reusa(tmp_path):
    construidos = []

    def fabrica():
        construidos.append(1)
        return Trigramas()

    indice, meta = nz.indice_en_disco(CATALOGO, MODELO, fabrica, raiz=tmp_path)
    assert construidos == [1]
    assert meta["codigos"] == len(CATALOGO)
    assert meta["dimension"] == DIMENSION
    assert meta["bytes"] > 0

    otra_vez, meta2 = nz.indice_en_disco(CATALOGO, MODELO, fabrica, raiz=tmp_path)
    assert construidos == [1]
    assert otra_vez.codigos == indice.codigos
    assert otra_vez.descripciones == indice.descripciones
    assert torch.equal(otra_vez.vectores, indice.vectores)
    assert meta2 == meta

    # Otro catálogo u otra revisión: se reconstruye
    nz.indice_en_disco({**CATALOGO, "Z99": "Otra"}, MODELO, fabrica, raiz=tmp_path)
    assert construidos == [1, 1]
    con_otro_meta = tmp_path / next(p.name for p in tmp_path.iterdir())
    guardado = torch.load(con_otro_meta, weights_only=True)
    guardado["meta"]["configuracion"] = "otra"
    torch.save(guardado, con_otro_meta)
    nz.indice_en_disco({**CATALOGO, "Z99": "Otra"}, MODELO, fabrica, raiz=tmp_path)
    assert construidos == [1, 1, 1]


def test_cambiar_la_configuracion_del_modelo_reconstruye_el_indice(tmp_path):
    construidos = []

    def fabrica():
        construidos.append(1)
        return Trigramas()

    nz.indice_en_disco(CATALOGO, MODELO, fabrica, raiz=tmp_path)
    nz.indice_en_disco(CATALOGO, MODELO, fabrica, raiz=tmp_path)
    assert construidos == [1]
    for cambio in ({"pooling": "cls"}, {"minusculas": True}, {"tokenizador": ("t/z", "r")}):
        nz.indice_en_disco(CATALOGO, replace(MODELO, **cambio), fabrica, raiz=tmp_path)
    assert construidos == [1, 1, 1, 1]
    assert nz.huella_configuracion(MODELO) != nz.huella_configuracion(
        replace(MODELO, prefijo="query: ")
    )


# --- Codificador de Hugging Face, con un modelo de mentira ---------------------------


class _Tokenizador:
    def __init__(self):
        self.recibidos = []

    def __call__(self, textos, **_):
        self.recibidos.extend(textos)
        largo = max(len(t.split()) for t in textos)
        mascara = torch.tensor(
            [[1] * len(t.split()) + [0] * (largo - len(t.split())) for t in textos]
        )
        return {"input_ids": mascara.clone(), "attention_mask": mascara}


class _Red:
    def __call__(self, input_ids, attention_mask):
        # Token j de cada texto: el vector (j + 1, 1)
        n, largo = input_ids.shape
        posiciones = torch.arange(1, largo + 1, dtype=torch.float).expand(n, largo)
        return SimpleNamespace(
            last_hidden_state=torch.stack([posiciones, torch.ones(n, largo)], -1)
        )


def _codificador_hf(monkeypatch, modelo, lote=2):
    import transformers

    tokenizador = _Tokenizador()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizador)
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: SimpleNamespace(eval=_Red)
    )
    return nz.CodificadorHF(modelo, lote=lote), tokenizador


def test_codificador_hf_media_ignora_el_relleno(monkeypatch):
    modelo = nz.ModeloEmbeddings("x/y", "r", "media", "MIT", "query: ")
    codificador, tokenizador = _codificador_hf(monkeypatch, modelo)
    vectores = codificador(["a", "a b c", "a b"])
    assert tokenizador.recibidos == ["query: a", "query: a b c", "query: a b"]
    # "query: a" son 2 tokens: media de (1,1) y (2,1) = (1.5, 1), sin el relleno del lote
    esperado = torch.nn.functional.normalize(torch.tensor([1.5, 1.0]), dim=0)
    assert torch.allclose(vectores[0], esperado)
    assert torch.allclose(vectores.norm(dim=1), torch.ones(3))


def test_codificador_hf_minusculas_y_tokenizador_de_otro_repo(monkeypatch):
    import transformers

    pedidos = []
    modelo = nz.ModeloEmbeddings("x/y", "r", "cls", "?", minusculas=True, tokenizador=("t/z", "rt"))
    codificador, tokenizador = _codificador_hf(monkeypatch, modelo)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda nombre, revision: pedidos.append((nombre, revision)) or tokenizador,
    )
    codificador = nz.CodificadorHF(modelo)
    codificador(["Ictus Isquémico"])
    assert pedidos == [("t/z", "rt")]
    assert tokenizador.recibidos[-1] == "ictus isquémico"


def test_codificador_hf_cls(monkeypatch):
    modelo = nz.ModeloEmbeddings("x/y", "r", "cls", "?")
    codificador, _ = _codificador_hf(monkeypatch, modelo)
    vectores = codificador(["a b", "c"])
    esperado = torch.nn.functional.normalize(torch.tensor([1.0, 1.0]), dim=0)
    assert torch.allclose(vectores, esperado.expand(2, 2))
    assert codificador([]).numel() == 0


# --- Evaluación en CodiEsp -----------------------------------------------------------


def _documento(id_, texto, *menciones):
    lista = []
    for codigo, fragmento in menciones:
        a = texto.index(fragmento)
        lista.append(
            MencionCodificada("DIAGNOSTICO", codigo, fragmento, ((a, a + len(fragmento)),))
        )
    return DocumentoCodificado(id_, texto, [m.codigo for m in lista], lista)


DOC_TRAIN = _documento(
    "t-1",
    "Varón con diabetes mellitus tipo 2 y tos.",
    ("E11.9", "diabetes mellitus tipo 2"),
    ("R05", "tos"),
)
DOC_DEV = _documento(
    "d-1",
    "Mujer con infarto cerebral y EPOC.",
    ("I63.9", "infarto cerebral"),
    ("J44.9", "EPOC"),
)


def test_elegir_umbral():
    curva = [
        {"umbral": 0.1, "f1": 0.5},
        {"umbral": 0.2, "f1": 0.7},
        {"umbral": 0.3, "f1": 0.7},
        {"umbral": 0.4, "f1": 0.6},
    ]
    assert nc.elegir_umbral(curva) == 0.3


def test_curva_umbral():
    curva = nc.curva_umbral([(_normalizador(), [DOC_DEV])])
    assert [p["umbral"] for p in curva] == list(nc.UMBRALES)
    assert curva[-1]["recall"] == 0.0  # con umbral 1, solo pasa una coincidencia perfecta
    assert curva[0]["recall"] > 0


def test_metricas_codigo():
    r = nc.metricas_codigo(_normalizador(), [DOC_TRAIN, DOC_DEV])
    assert r == {
        "menciones": 4,
        "top1": 0.75,
        "top5": 1.0,
        "menciones_charlson": 3,
        "top1_charlson": 0.6667,
        "top5_charlson": 1.0,
        "categoria_top5": 1.0,
    }
    assert nc.metricas_codigo(_normalizador(), [])["top1"] == 0.0


def test_categoria_top5_lee_el_codigo_padre_como_su_hijo():
    # Con k=1 el único candidato de "diabetes mellitus tipo 2" es E11, que cuenta como E11.9
    doc = _documento("t-2", "Con diabetes mellitus tipo 2.", ("E11.9", "diabetes mellitus tipo 2"))
    normalizador = _normalizador(k=1)
    assert normalizador.candidatos("diabetes mellitus tipo 2")[0].codigo == "E11"
    assert nc.metricas_codigo(normalizador, [doc])["categoria_top5"] == 1.0


def test_latencia():
    r = nc.latencia(_normalizador(), ["tos", "tos", "epoc"])
    assert r["menciones"] == 2
    assert r["ms_mediana"] >= 0
    assert nc.latencia(_normalizador(), [])["ms_mediana"] == 0.0


def test_evaluar():
    resultados = nc.evaluar(
        "e5_small",
        _normalizador(),
        {"train": [DOC_TRAIN], "dev": [DOC_DEV]},
        {"codigos": len(CATALOGO)},
        bytes_modelo=123,
        replicas=5,
        variantes=("original", "minusculas", "errores_10"),
    )
    assert resultados["conjunto_principal"] == "train+dev"
    assert resultados["umbral"]["particion"] == "dev"
    assert resultados["umbral"]["elegido"] in nc.UMBRALES
    assert set(resultados["categoria"]) == {"original", "minusculas", "errores_10"}
    assert "por_categoria" in resultados["categoria"]["original"]["train+dev"]
    assert resultados["modelo"]["licencia"] == "MIT"
    assert resultados["costo"]["textos_unicos"] >= 4
    # Sin texto del corpus
    serializado = json.dumps(resultados, ensure_ascii=False)
    assert "infarto cerebral" not in serializado
    assert "diabetes mellitus" not in serializado
    tabla = nc.tabla([resultados])
    assert "| e5_small |" in tabla


def test_evaluar_exige_desarrollo_y_rechaza_prueba():
    with pytest.raises(ValueError, match="hace falta"):
        nc.evaluar("e5_small", _normalizador(), {"train": [DOC_TRAIN]}, {}, replicas=5)
    with pytest.raises(ValueError, match="#134"):
        nc.evaluar("e5_small", _normalizador(), {"dev": [DOC_DEV], "test": [DOC_DEV]}, {})


def test_tabla_con_las_reglas():
    from evaluacion import charlson_codiesp

    reglas, _ = charlson_codiesp.evaluar({"dev": [DOC_DEV]}, replicas=5)
    tabla = nc.tabla([], reglas)
    assert "| reglas + corrector | — |" in tabla


# --- Línea de comandos ---------------------------------------------------------------


def _mini_codiesp(raiz: Path) -> None:
    for particion, doc in {"train": DOC_TRAIN, "dev": DOC_DEV}.items():
        carpeta = raiz / codiesp.CARPETA_CODIESP / particion
        (carpeta / "text_files").mkdir(parents=True)
        (carpeta / "text_files" / f"{doc.id}.txt").write_text(doc.texto, encoding="utf-8")
        filas = [
            f"{doc.id}\tDIAGNOSTICO\t{m.codigo.lower()}\t{m.texto}\t{m.inicio} {m.fin}\n"
            for m in doc.menciones
        ]
        (carpeta / f"{particion}X.tsv").write_text("".join(filas), encoding="utf-8")
        (carpeta / f"{particion}D.tsv").write_text(
            "".join(f"{doc.id}\t{m.codigo.lower()}\n" for m in doc.menciones), encoding="utf-8"
        )
    carpeta = raiz / codiesp.CARPETA_CATALOGO
    carpeta.mkdir()
    (carpeta / codiesp.ARCHIVO_CATALOGO).write_text(
        "".join(f"{c}\t{d}\t-\n" for c, d in CATALOGO.items()), encoding="utf-8"
    )


def test_subcomando_normalizacion(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    _mini_codiesp(tmp_path)
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(cli, "_bytes_modelo", lambda modelo: 1)
    monkeypatch.setattr(nc, "CARPETA_RESULTADOS", tmp_path / "resultados")
    main(
        [
            "normalizacion",
            "--modelo",
            "e5_small",
            "--raiz",
            ".",
            "--catalogo",
            ".",
            "--indices",
            "indices",
            "--bootstrap",
            "5",
        ]
    )
    salida = capsys.readouterr().out
    assert "| e5_small |" in salida
    assert (tmp_path / "resultados" / "normalizacion_e5_small_desarrollo.json").is_file()
    assert list((tmp_path / "indices").glob("*.pt"))

    monkeypatch.setattr(cli.charlson_codiesp, "RUTA_RESULTADOS", tmp_path / "no_hay.json")
    main(["normalizacion", "--tabla"])
    assert "| e5_small |" in capsys.readouterr().out


def test_subcomando_normalizacion_sin_modelo():
    with pytest.raises(SystemExit, match="Falta --modelo"):
        main(["normalizacion"])


# --- Modelos reales ------------------------------------------------------------------


# Casos de #120 y #123, solo con el catálogo. "ictus" no está en el catálogo como sinónimo
# (solo en "Historia familiar de ictus" y en la escala NIHSS): ningún modelo lo resuelve
# así; SapBERT lo deja más cerca de "Terremoto". Con los sinónimos de #132 sí (test de
# abajo). El RoBERTa sin afinar tampoco resuelve "accidente cerebrovascular". Son `xfail`
# estrictos: si pasan, el test avisa.
_SIN_SINONIMOS = pytest.mark.xfail(strict=True, reason="solo con el catálogo (ver #132)")
_CASOS_REALES = [
    pytest.param(
        clave,
        texto,
        categoria,
        marks=_SIN_SINONIMOS
        if texto == "ictus" or (clave == "bsc_bio_ehr" and texto == "accidente cerebrovascular")
        else (),
        id=f"{clave}-{texto}",
    )
    for clave in nz.CANDIDATOS
    for texto, categoria in [
        ("accidente cerebrovascular", C.CEREBROVASCULAR),
        ("ictus", C.CEREBROVASCULAR),
        ("linfoma de Hodgkin sin metástasis", C.TUMOR_MALIGNO),
    ]
]


@pytest.mark.modelo
@pytest.mark.parametrize(("clave", "texto", "categoria"), _CASOS_REALES)
def test_casos_de_los_issues_con_el_modelo_real(clave, texto, categoria):
    """Con el catálogo real, la categoría del primer candidato (sin umbral)."""
    modelo = nz.CANDIDATOS[clave]
    catalogo = codiesp.leer_catalogo()
    codificador = nz.CodificadorHF(modelo)
    indice, _ = nz.indice_en_disco(catalogo, modelo, lambda: codificador)
    normalizador = nz.Normalizador(indice, codificador)
    assert normalizador.mapeador(0.0)(texto)[0] is categoria


@pytest.mark.modelo
def test_ictus_con_los_sinonimos_de_entrenamiento():
    """Con los sinónimos de CodiEsp entrenamiento, SapBERT resuelve "ictus" (#132)."""
    modelo = nz.CANDIDATOS["sapbert"]
    codificador = nz.CodificadorHF(modelo)
    indice, _ = nz.indice_en_disco(codiesp.leer_catalogo(), modelo, lambda: codificador)
    sinonimos = sc.sinonimos_de(codiesp.leer_particion(codiesp.RAIZ_CODIESP, "train"))
    normalizador = nz.Normalizador(indice.con_sinonimos(sinonimos, codificador), codificador)
    assert normalizador.mapeador(0.0)("ictus")[0] is C.CEREBROVASCULAR
    assert normalizador.candidatos("ictus")[0].origen == sc.ORIGEN_CODIESP


# --- Sinónimos de CodiEsp entrenamiento (#132) ---------------------------------------


def test_codificador_con_memoria_codifica_cada_texto_una_vez():
    base = Trigramas()
    codificador = nz.CodificadorConMemoria(base)
    primero = codificador(["tos", "epoc", "tos"])
    assert base.textos == 2
    assert torch.allclose(primero[0], primero[2])
    assert torch.allclose(codificador(["epoc"])[0], base(["epoc"])[0])
    assert base.textos == 3  # la llamada directa de arriba; "epoc" no se volvió a pedir
    assert codificador([]).numel() == 0


def test_indice_con_sinonimos_y_explicacion():
    codificador = Trigramas()
    indice = nz.construir_indice(CATALOGO, codificador)
    assert indice.con_sinonimos([], codificador) is indice
    sinonimo = nz.Sinonimo("ictus", "I63.9", "mención de prueba", ("t-1",))
    ampliado = indice.con_sinonimos([sinonimo], codificador)
    assert len(ampliado.codigos) == len(CATALOGO) + 1
    assert ampliado.vectores.shape[0] == len(ampliado.codigos)
    assert ampliado.origenes[: len(CATALOGO)] == [nz.ORIGEN_CATALOGO] * len(CATALOGO)
    assert len(indice.codigos) == len(CATALOGO)  # el original no cambia

    primero = ampliado.buscar(codificador(["ictus"]), k=1)[0][0]
    assert (primero.codigo, primero.origen) == ("I63.9", "mención de prueba")
    assert primero.explicacion() == "I63.9: parecido a 'ictus', mención de prueba"
    tos = ampliado.buscar(codificador(["tos"]), k=1)[0][0]
    assert tos.explicacion() == "R05: Tos (catálogo CIE-10)"


TRAIN = [
    _documento("t-1", "Ingresa por ictus. Tiene DM.", ("I63.9", "ictus"), ("E11.9", "DM")),
    _documento("t-2", "Antecedente de  Ictus y tos.", ("I63.9", "Ictus"), ("R05", "tos")),
    _documento("t-3", "Con EPOC y Tos.", ("J44.9", "EPOC"), ("R06", "Tos")),
    _documento("t-4", "Refiere tos.", ("R05", "tos")),
]
DEV = [
    _documento("d-1", "Mujer con ictus y VIH.", ("I63.9", "ictus"), ("B20", "VIH")),
    _documento("d-2", "Varón con CMV y DM.", ("B25.9", "CMV"), ("E11.9", "DM")),
]


def test_sinonimos_de_deduplica_sin_distinguir_mayusculas():
    sinonimos = {s.texto.casefold(): s for s in sc.sinonimos_de(TRAIN)}
    assert set(sinonimos) == {"ictus", "dm", "tos", "epoc"}
    # "tos" está dos veces con R05 y una con R06; la escritura más frecuente es "tos"
    assert sinonimos["tos"] == nz.Sinonimo(
        "tos", "R05", sc.ORIGEN_CODIESP, ("t-2", "t-3", "t-4"), 3
    )
    # Empate de escrituras: queda la menor
    assert sinonimos["ictus"].texto == "Ictus"
    assert sinonimos["ictus"].documentos == ("t-1", "t-2")


def test_sinonimos_de_normaliza_espacios_y_desempata_por_codigo():
    doc = _documento(
        "t-9", "Un ictus\n  isquémico y ictus isquémico.", ("I63.9", "ictus\n  isquémico")
    )
    doc.menciones.append(MencionCodificada("DIAGNOSTICO", "I63.5", "ictus isquémico", ((23, 38),)))
    # Una mención que solo tiene espacios no da sinónimo
    doc.menciones.append(MencionCodificada("DIAGNOSTICO", "R69", "\n ", ((8, 10),)))
    (sinonimo,) = sc.sinonimos_de([doc])
    assert (sinonimo.texto, sinonimo.codigo, sinonimo.menciones) == ("ictus isquémico", "I63.5", 2)


def test_resumen_sinonimos():
    assert sc.resumen_sinonimos(TRAIN) == {
        "documentos": 4,
        "menciones": 7,
        "sinonimos": 4,
        "textos_con_varios_codigos": 1,
        "menciones_con_otro_codigo": 1,
    }


def test_verificar_origen():
    sinonimos = sc.sinonimos_de(TRAIN)
    assert sc.verificar_origen(sinonimos, {"dev": ["d-1"], "test": ["x", "y"]}) == {
        "documentos_origen": 4,
        "ids_dev": 1,
        "ids_test": 2,
    }
    with pytest.raises(ValueError, match="1 documentos de 'test'"):
        sc.verificar_origen(sinonimos, {"dev": ["d-1"], "test": ["t-3"]})


def test_pliegues_de():
    pliegues = sc.pliegues_de(list(reversed(TRAIN)), 2)
    assert [[d.id for d in p] for p in pliegues] == [["t-1", "t-3"], ["t-2", "t-4"]]
    with pytest.raises(ValueError, match="al menos 2"):
        sc.pliegues_de(TRAIN, 1)


def test_cada_pliegue_solo_tiene_los_sinonimos_de_los_otros():
    codificador = Trigramas()
    indice = nz.construir_indice(CATALOGO, codificador)
    for normalizador, pliegue in sc._pliegues_con_sinonimos(indice, codificador, TRAIN, 2):
        indice_pliegue = normalizador.indice
        sinonimos = {
            texto.casefold()
            for texto, origen in zip(
                indice_pliegue.descripciones, indice_pliegue.origenes, strict=True
            )
            if origen == sc.ORIGEN_CODIESP
        }
        # "DM" y "EPOC" solo están en t-1 y t-3: no pueden estar en el índice de ese pliegue
        esperados = {"t-1": {"ictus", "tos"}, "t-2": {"ictus", "dm", "epoc", "tos"}}
        assert sinonimos == esperados[pliegue[0].id]


def test_metricas_siglas():
    casos = [
        Caso("d", "VIH", "B20", C.VIH_SIDA, "B20", C.VIH_SIDA),  # en la lista, acierto
        Caso("d", "DM ", "E11.9", C.DIABETES, None, None),  # en la lista, fallo
        Caso("d", "CMV", "B25.9", None, None, None),  # fuera, acierto sin categoría
        Caso("d", "HCC", "C22.0", C.TUMOR_MALIGNO, "C22.0", C.TUMOR_MALIGNO),  # fuera, acierto
        Caso("d", "ictus", "I63.9", C.CEREBROVASCULAR, None, None),  # no es sigla
        Caso("d", "Dm", "E11.9", C.DIABETES, None, None),  # tiene minúscula: no es sigla
    ]
    r = sc.metricas_siglas(casos)
    assert r["todas"] == {
        "menciones": 4,
        "siglas_distintas": 4,
        "aciertos": 3,
        "gold_charlson": 3,
        "aciertos_charlson": 2,
    }
    assert r["en_lista"]["menciones"] == 2 and r["en_lista"]["aciertos"] == 1
    assert r["fuera_de_lista"]["aciertos"] == 2

    normalizador = _normalizador(k=1)
    con_codigo = sc.metricas_siglas(casos, normalizador)
    # Con trigramas solo "DM" acierta el código (E11.9); VIH, CMV y HCC no tienen el suyo
    assert [con_codigo[g]["codigo_top1"] for g in ("todas", "en_lista", "fuera_de_lista")] == [
        1,
        1,
        0,
    ]


def test_siglas_en_sinonimos():
    casos = [
        Caso("d", "DM", "E11.9", C.DIABETES, None, None),
        Caso("d", "VIH", "B20", C.VIH_SIDA, None, None),
        Caso("d", "tos", "R05", None, None, None),  # está, pero no es sigla
    ]
    assert sc.siglas_en_sinonimos(casos, sc.sinonimos_de(TRAIN)) == 1


def _evaluar_con_prueba(ids_prueba, **kwargs):
    codificador = Trigramas()
    parametros = {
        "replicas": 5,
        "variantes": ("original", "minusculas", "errores_10"),
        "pliegues": 2,
    } | kwargs
    return sc.evaluar(
        "sapbert",
        nz.construir_indice(CATALOGO, codificador),
        codificador,
        {"train": TRAIN, "dev": DEV},
        {"codigos": len(CATALOGO)},
        ids_prueba,
        **parametros,
    )


def _evaluar(**kwargs):
    return _evaluar_con_prueba(["x-1", "x-2"], **kwargs)


def test_evaluar_con_y_sin_sinonimos():
    r = _evaluar()
    assert list(r["metodos"]) == ["reglas_corrector", *sc.CONFIGURACIONES, *sc.HIBRIDOS]
    reglas = r["metodos"]["reglas_corrector"]
    sin, con = r["metodos"]["sin_sinonimos"], r["metodos"]["con_sinonimos"]
    assert sin["filas_indice"] == len(CATALOGO)
    assert con["filas_indice"] == len(CATALOGO) + 4
    assert (sin["umbral"]["pliegues"], con["umbral"]["pliegues"]) == (1, 2)
    assert con["umbral"]["particion"] == "train"
    assert set(con["categoria"]) == {"original", "minusculas", "errores_10"}
    assert "por_categoria" in con["categoria"]["original"]
    assert "por_categoria" not in con["categoria"]["minusculas"]
    # "ictus" y "DM" de desarrollo salen de un sinónimo, que es idéntico
    assert con["origen_top1"][sc.ORIGEN_CODIESP] >= 2
    assert sin["origen_top1"] == {nz.ORIGEN_CATALOGO: 4}
    assert r["verificacion_origen"] == {"documentos_origen": 4, "ids_dev": 2, "ids_test": 2}
    assert r["sinonimos"]["sinonimos"] == 4
    assert r["siglas"]["lista_114"] == 30
    assert reglas["siglas"]["todas"]["menciones"] == 3
    assert "por_categoria" in reglas["categoria"]["original"]
    # "ictus" y "DM" de desarrollo están entre los sinónimos; "VIH" y "CMV", no
    assert reglas["no_vistas"]["menciones"] == 2
    assert con["no_vistas"]["menciones"] == 2
    assert r["siglas"]["menciones_en_sinonimos"] == 1
    assert r["costo"]["latencia_con_sinonimos"]["menciones"] == 4
    # Sin texto del corpus
    serializado = json.dumps(r, ensure_ascii=False)
    for texto in ("Mujer con", "ictus", "Ingresa por"):
        assert texto not in serializado

    tabla = sc.tabla(r)
    assert "| sapbert con sinonimos |" in tabla
    assert "| reglas + corrector | — |" in tabla
    assert "| sapbert hibrido exclusiones |" in tabla
    assert tabla.count("\n") == 6
    # Los híbridos usan el umbral del modelo con sinónimos
    for nombre in sc.HIBRIDOS:
        assert r["metodos"][nombre]["umbral"]["elegido"] == con["umbral"]["elegido"]
    assert set(r["exclusiones"]) == {"train_validacion_cruzada", "dev"}
    for particion in ("train_validacion_cruzada", "dev"):
        assert set(r["exclusiones"][particion]) == set(sc.EFECTOS)
    # El respaldo solo agrega categorías: tiene al menos las de las reglas y las del híbrido
    respaldo = r["metodos"]["hibrido_respaldo"]["categoria"]["original"]
    for otro in ("reglas_corrector", "hibrido_exclusiones"):
        otras = r["metodos"][otro]["categoria"]["original"]
        assert respaldo["menciones_con_categoria"] >= otras["menciones_con_categoria"]


def test_evaluar_rechaza_particiones_y_fuga():
    codificador = Trigramas()
    indice = nz.construir_indice(CATALOGO, codificador)
    with pytest.raises(ValueError, match="#134"):
        sc.evaluar("sapbert", indice, codificador, {"train": TRAIN, "test": DEV}, {}, [])
    with pytest.raises(ValueError, match="Hacen falta"):
        sc.evaluar("sapbert", indice, codificador, {"dev": DEV}, {}, [])
    with pytest.raises(ValueError, match="'test'"):
        _evaluar_con_prueba(["t-1"])


def test_ids_de(tmp_path):
    _mini_codiesp(tmp_path)
    assert codiesp.ids_de(tmp_path, "dev") == ["d-1"]
    with pytest.raises(codiesp.ErrorCorpus, match="'test'"):
        codiesp.ids_de(tmp_path, "test")
    with pytest.raises(ValueError, match="desconocida"):
        codiesp.ids_de(tmp_path, "otra")


def test_subcomando_sinonimos(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    _mini_codiesp(tmp_path)
    prueba = tmp_path / codiesp.CARPETA_CODIESP / "test" / "text_files"
    prueba.mkdir(parents=True)
    (prueba / "x-1.txt").write_text("no se lee", encoding="utf-8")
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(sc, "CARPETA_RESULTADOS", tmp_path / "resultados")
    main(
        ["sinonimos", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
        + ["--bootstrap", "5", "--pliegues", "2"]
    )
    salida = capsys.readouterr().out
    assert "| sapbert con sinonimos |" in salida
    guardado = json.loads(
        (tmp_path / "resultados" / "sinonimos_sapbert_desarrollo.json").read_text(encoding="utf-8")
    )
    assert guardado["verificacion_origen"]["ids_test"] == 1


# --- Híbrido (#133) ------------------------------------------------------------------


def _siempre(categoria, codigo):
    return lambda texto: (categoria, codigo)


def test_con_exclusiones_quita_la_categoria_si_la_regla_la_excluye():
    mapeador = nz.con_exclusiones(_siempre(C.TUMOR_MALIGNO, "C90.0"))
    assert mapeador("miomas") is None  # benigno, aunque el modelo lo acerque a "mieloma"
    assert mapeador("Adenoma de próstata") is None
    assert mapeador("mieloma") == (C.TUMOR_MALIGNO, "C90.0")
    assert nz.con_exclusiones(lambda texto: None)("miomas") is None
    # "metástasis" en la exclusión de tumor maligno cede la mención a otra regla: no se
    # aplica si la otra regla tampoco toma el texto (#144)
    linfoma = nz.con_exclusiones(_siempre(C.TUMOR_MALIGNO, "C81.90"))
    assert linfoma("linfoma de Hodgkin sin metástasis") == (C.TUMOR_MALIGNO, "C81.90")
    assert linfoma("cáncer de mama sin metástasis a distancia") == (C.TUMOR_MALIGNO, "C81.90")


@pytest.mark.parametrize(
    ("texto", "categoria"),
    [
        # La otra regla sí acepta el texto: la exclusión se aplica
        ("metástasis hepáticas", C.TUMOR_MALIGNO),
        ("infarto cerebral", C.INFARTO_MIOCARDIO),
        ("carcinoma renal", C.RENAL_CRONICA),
        # La exclusión no es de precedencia, o no toda cae en otra regla
        ("adenoma de próstata sin metástasis", C.TUMOR_MALIGNO),
        ("madre con cáncer metastásico", C.TUMOR_MALIGNO),
    ],
)
def test_con_exclusiones_aplica_las_que_si_excluyen(texto, categoria):
    assert nz.con_exclusiones(_siempre(categoria, "X"))(texto) is None


def test_con_respaldo():
    respaldo = _siempre(C.DIABETES, "E11.9")
    assert nz.con_respaldo(lambda texto: None, respaldo)("dm") == (C.DIABETES, "E11.9")
    modelo = _siempre(C.VIH_SIDA, "B20")
    assert nz.con_respaldo(modelo, respaldo)("vih") == (C.VIH_SIDA, "B20")


def test_efecto_exclusiones():
    antes = [
        Caso("d", "miomas", "D25.9", None, "C90.0", C.TUMOR_MALIGNO),  # arregla
        Caso("d", "x", "C61", C.TUMOR_MALIGNO, "C61", C.TUMOR_MALIGNO),  # rompe
        Caso("d", "tos", "R05", None, None, None),  # no aplica
        Caso("d", "y", "C61", C.TUMOR_MALIGNO, "C61", C.TUMOR_MALIGNO),  # queda igual
        Caso("d", "z", "C78.3", C.TUMOR_METASTASICO, "C31.0", C.TUMOR_MALIGNO),  # confusión
    ]
    despues = [replace(c, codigo=None, prediccion=None) for c in antes]
    despues[3] = antes[3]
    assert sc.efecto_exclusiones(antes, despues) == {
        "aplica": 3,
        "arregla": 1,
        "quita_confusion": 1,
        "rompe": 1,
    }


def test_evaluar_con_una_exclusion_que_se_activa():
    """El modelo lleva "miomas" (benigno) a un código de tumor maligno; el híbrido lo quita."""
    codificador = Trigramas()
    catalogo = {"C90.0": "Miomas malignos", "R05": "Tos"}
    train = [_documento("t-1", "Con mieloma y tos.", ("C90.0", "mieloma"), ("R05", "tos"))]
    train.append(_documento("t-2", "Refiere tos.", ("R05", "tos")))
    dev = [_documento("d-1", "Útero con miomas.", ("D25.9", "miomas"))]
    r = sc.evaluar(
        "sapbert",
        nz.construir_indice(catalogo, codificador),
        codificador,
        {"train": train, "dev": dev},
        {},
        [],
        replicas=5,
        variantes=("original", "minusculas", "errores_10"),
        pliegues=2,
    )
    assert r["exclusiones"]["dev"] == {"aplica": 1, "arregla": 1, "quita_confusion": 0, "rompe": 0}
    original = {m: r["metodos"][m]["categoria"]["original"]["micro"] for m in r["metodos"]}
    assert original["con_sinonimos"]["fp"] == 1
    assert original["hibrido_exclusiones"]["fp"] == 0


# --- Medición final en CodiEsp prueba (#134) ------------------------------------------


def _evaluar_prueba(prueba=DEV):
    from evaluacion import prueba_codiesp as pc

    codificador = Trigramas()
    return pc.evaluar(
        "sapbert",
        nz.construir_indice(CATALOGO, codificador),
        codificador,
        TRAIN,
        prueba,
        {"codigos": len(CATALOGO)},
        replicas=5,
        variantes=("original", "minusculas", "errores_10"),
    )


def test_evaluar_prueba():
    from evaluacion import prueba_codiesp as pc

    r = _evaluar_prueba()
    assert list(r["metodos"]) == list(pc.METODOS)
    assert r["umbral"]["valor"] == pc.UMBRAL == 0.69
    assert r["particiones"] == {"sinonimos": "train", "medicion": "test"}
    assert r["verificacion_origen"] == {"documentos_origen": 4, "ids_test": 2}
    # El código CIE-10 solo se mide para el modelo
    assert [m for m in r["metodos"] if "codigo" in r["metodos"][m]] == ["sapbert_sinonimos"]
    assert "codigo_top1" in r["metodos"]["sapbert_sinonimos"]["siglas"]["todas"]
    assert "codigo_top1" not in r["metodos"]["reglas_corrector"]["siglas"]["todas"]
    for m in r["metodos"].values():
        assert set(m["casos_issues"]) == set(pc.CASOS_ISSUES)
        assert "por_categoria" in m["categoria"]["original"]
    categorias = {c.value for c in C} | {None}
    assert all(set(m["casos_issues"].values()) <= categorias for m in r["metodos"].values())
    # Sin texto del corpus (los casos de los issues son frases sintéticas)
    serializado = json.dumps(r, ensure_ascii=False)
    for texto in ("Mujer con", "Ingresa por", "Varón con"):
        assert texto not in serializado

    tabla = pc.tabla(r)
    assert "| hibrido exclusiones |" in tabla
    assert tabla.count("\n") == 5


def test_evaluar_prueba_rechaza_sinonimos_de_prueba():
    with pytest.raises(ValueError, match="'test'"):
        _evaluar_prueba(prueba=[TRAIN[0]])


def _mini_codiesp_con_prueba(raiz: Path) -> None:
    _mini_codiesp(raiz)
    carpeta = raiz / codiesp.CARPETA_CODIESP / "test"
    (carpeta / "text_files").mkdir(parents=True)
    doc = replace(DOC_DEV, id="p-1")
    (carpeta / "text_files" / "p-1.txt").write_text(doc.texto, encoding="utf-8")
    (carpeta / "testX.tsv").write_text(
        "".join(
            f"p-1\tDIAGNOSTICO\t{m.codigo.lower()}\t{m.texto}\t{m.inicio} {m.fin}\n"
            for m in doc.menciones
        ),
        encoding="utf-8",
    )
    (carpeta / "testD.tsv").write_text(
        "".join(f"p-1\t{m.codigo.lower()}\n" for m in doc.menciones), encoding="utf-8"
    )


def test_subcomando_prueba(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    _mini_codiesp_con_prueba(tmp_path)
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [])
    argumentos = ["prueba", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
    argumentos += ["--bootstrap", "5", "--salida", "final.json"]
    main(argumentos)
    assert "| hibrido exclusiones |" in capsys.readouterr().out
    assert json.loads((tmp_path / "final.json").read_text())["documentos"] == {
        "train": 1,
        "test": 1,
    }
    # Una sola vez: no sobrescribe
    with pytest.raises(SystemExit, match="una sola vez"):
        main(argumentos)


def test_subcomando_prueba_exige_el_arbol_limpio(tmp_path, monkeypatch):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_cambios_sin_commitear", lambda: [" M pyproject.toml"])
    with pytest.raises(SystemExit, match="árbol limpio"):
        main(["prueba", "--salida", "final.json"])
    assert not (tmp_path / "final.json").exists()


def test_cambios_sin_commitear_usa_git(monkeypatch):
    from evaluacion import __main__ as cli

    salida = SimpleNamespace(stdout=" M a.py\n?? b/\n")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: salida)
    assert cli._cambios_sin_commitear() == [" M a.py", "?? b/"]


# --- Costo de despliegue (#134) ------------------------------------------------------


def test_notas_de_2000():
    from evaluacion import costo_normalizacion as cn

    corta, larga = "a " * 500, "palabra " * 400
    notas = cn.notas_de_2000([corta, larga])
    assert len(notas) == 1
    assert 1800 <= len(notas[0]) <= 2000


def test_dockerfile_candidata():
    from evaluacion import costo_normalizacion as cn

    texto = cn.dockerfile_candidata("base:1")
    modelo = nz.CANDIDATOS["sapbert"]
    assert texto.startswith("FROM base:1\n")
    assert f"revision='{modelo.revision}'" in texto
    assert f"revision='{modelo.tokenizador[1]}'" in texto
    assert f"COPY indice.pt {cn.RUTA_INDICE_EN_IMAGEN}" in texto
    assert texto.rstrip().endswith("USER ner")
