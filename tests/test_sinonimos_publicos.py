"""Tests de los sinónimos de CANTEMIST y DisTEMIST (#153), con archivos de juguete."""

import json

import pytest

from evaluacion import dos_etapas as de
from evaluacion import fuentes_sinonimos as fs
from evaluacion import normalizador as nz
from evaluacion import sinonimos_publicos as sp
from evaluacion.__main__ import main
from evaluacion.sinonimos_codiesp import ORIGEN_CODIESP
from ner_clinic.charlson import quan
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_dos_etapas import TRAIN_6, _indice, _mini
from tests.test_normalizador import DEV, Trigramas, _documento


@pytest.mark.parametrize(
    ("cieo", "codigo", "categoria"),
    [
        ("8000/6", "C79.9", C.TUMOR_METASTASICO),
        ("8010/9", "C80.0", C.TUMOR_METASTASICO),
        ("8140/3", "C76.8", C.TUMOR_MALIGNO),
        ("8140/33", "C76.8", C.TUMOR_MALIGNO),  # el grado se ignora
        ("8010/3/H", "C76.8", C.TUMOR_MALIGNO),
        ("8090/3", "C44.91", None),  # basocelular: piel, Quan lo excluye
        ("9680/3", "C85.9", C.TUMOR_MALIGNO),
        ("9650/3", "C81.9", C.TUMOR_MALIGNO),
        ("9732/3", "C90.0", C.TUMOR_MALIGNO),
        ("9861/3", "C95.9", C.TUMOR_MALIGNO),
        ("8720/3", "C43.9", C.TUMOR_MALIGNO),
        ("9989/3", "D47.9", None),  # mielodisplásico: D45-D47
        ("8500/2", "D09.9", None),
        ("8000/1", "D48.9", None),
        ("8000/0", "D36.9", None),
    ],
)
def test_codigo_cieo(cieo, codigo, categoria):
    assert sp.codigo_cieo(cieo) == codigo
    assert quan.categoria_de(codigo) is categoria


def test_codigo_cieo_ilegible():
    assert sp.codigo_cieo("8000") is None
    assert sp.codigo_cieo("8000/5") is None


@pytest.mark.parametrize(
    "texto",
    [
        "M0",
        "m1",
        "pT2",
        "N1",
        "cM0",
        "T4b",
        "T3N0M0",
        "pT2N0M0",
        "ypT0N0(0/5)M0",
        "pT1-2N0M0",
        "T1aNoMx",
        "pT2pN1miM0",
        "T4N2M1",
    ],
)
def test_tnm(texto):
    assert sp.es_tnm(texto)


@pytest.mark.parametrize(
    "texto",
    [
        "metástasis",
        "tumor",
        "MTX",
        "TNM",
        "M0 y N1",
        "no",
        "Adenocarcinoma rectal T3N1M1",
        "M1 ganglionar por carcinoma",
    ],
)
def test_no_tnm(texto):
    assert not sp.es_tnm(texto)


def test_lexicas():
    menciones = [
        sp.MencionExterna(t, "C79.9", "cantemist/a", "cantemist")
        for t in ("lesiones hepáticas", "metástasis", "Adenocarcinoma", "linfomas", "nódulo")
    ]
    assert [m.texto for m in sp.lexicas(menciones)] == ["metástasis", "Adenocarcinoma", "linfomas"]


def _cantemist(raiz):
    for particion in sp.PARTICIONES_CANTEMIST:
        (raiz / particion / "cantemist-norm").mkdir(parents=True)
    carpeta = raiz / "train-set" / "cantemist-norm"
    (carpeta / "cc_onco1.ann").write_text(
        "T1\tMORFOLOGIA_NEOPLASIA 0 10\tCarcinoma  microcítico\n"
        "#1\tAnnotatorNotes T1\t8041/3\n"
        "T2\tMORFOLOGIA_NEOPLASIA 20 22\tM0\n"
        "#2\tAnnotatorNotes T2\t8000/6\n"
        "T3\tMORFOLOGIA_NEOPLASIA 30 40\tmetástasis\n"
        "#3\tAnnotatorNotes T3\t8000/6\n"
        "T4\tMORFOLOGIA_NEOPLASIA 50 55\traro\n"
        "#4\tAnnotatorNotes T4\tXXXX\n",
        encoding="utf-8",
    )
    (carpeta / "cc_onco1.txt").write_text("Nota de oncología de juguete.", encoding="utf-8")


def test_leer_cantemist(tmp_path):
    _cantemist(tmp_path)
    menciones, descartes = sp.leer_cantemist(tmp_path)
    assert menciones == [
        sp.MencionExterna("Carcinoma microcítico", "C76.8", "cantemist/cc_onco1", "cantemist"),
        sp.MencionExterna("metástasis", "C79.9", "cantemist/cc_onco1", "cantemist"),
    ]
    assert descartes == {"tnm": 1, "codigo_ilegible": 1}
    assert sp.textos_cantemist(tmp_path) == {"cantemist/cc_onco1": "Nota de oncología de juguete."}


def test_leer_cantemist_sin_datos(tmp_path):
    with pytest.raises(FileNotFoundError):
        sp.leer_cantemist(tmp_path)


def _distemist(raiz):
    carpeta = raiz / sp.CARPETA_CRUCES
    carpeta.mkdir(parents=True)
    filas = [
        ("es-S1", "T1", "cirrosis hepática", "EXACT", "K74.6"),
        ("es-S1", "T2", "deshidratación", "EXACT", "None"),
        ("es-S2", "T1", "HTA", "EXACT", "I10-I15.9"),
        ("es-S2", "T2", "fallo renal", "EXACT", "N17.0+R82.1"),
        ("es-S2", "T3", "DM", "EXACT", "E14"),
        ("es-S3", "T1", "EPOC  grave", "NARROW", "J44.9"),
        ("es-S3", "T2", "ictus y afasia", "COMPOSITE", "I63.9"),
    ]
    cabecera = "filename\tmark\tlabel\toff0\toff1\tspan\tcode\tsemantic_rel\tomim_code\t"
    cabecera += "mesh_code\ticd10_code\thpo_code\n"
    cuerpo = "".join(
        f"{f}\t{m}\tENFERMEDAD\t0\t1\t{s}\t123\t{r}\tNone\tNone\t{c}\tNone\n"
        for f, m, s, r, c in filas
    )
    (carpeta / "cruces.tsv").write_text(cabecera + cuerpo, encoding="utf-8")


def test_leer_distemist_solo_cie10_y_sin_snomed(tmp_path):
    _distemist(tmp_path)
    menciones, descartes = sp.leer_distemist(tmp_path)
    assert menciones == [
        sp.MencionExterna("cirrosis hepática", "K74.6", "S1", "distemist"),
        sp.MencionExterna("EPOC grave", "J44.9", "S3", "distemist"),
    ]
    assert descartes == {
        "sin_cie10": 1,
        "rango": 1,
        "compuesto": 1,
        "fuera_de_la_tabla": 1,
        "relacion": 1,
    }


def test_id_normalizado():
    assert sp.id_normalizado("es-S0210-56912007000900007-3") == "S0210-56912007000900007-3"
    assert sp.id_normalizado("S1") == "S1"


def _m(texto, codigo, documento, fuente="distemist"):
    return sp.MencionExterna(texto, codigo, documento, fuente)


def test_sin_documentos_quita_y_reporta():
    menciones = [_m("a", "K74.6", "d-1"), _m("b", "K74.6", "d-1"), _m("c", "J44.9", "t-1")]
    limpias, reporte = sp.sin_documentos(menciones, {"dev": ["d-1"], "test": ["x"]})
    assert limpias == [menciones[2]]
    assert reporte == {
        "dev": {"documentos": 1, "menciones": 2},
        "test": {"documentos": 0, "menciones": 0},
    }


def test_documentos_compartidos():
    nota = "uno dos tres cuatro cinco seis siete ocho nueve diez"
    compartidos = sp.documentos_compartidos(
        {"cantemist/a": nota, "cantemist/b": "otra cosa sin nada que ver con la nota"},
        {"S1": f"Antes. {nota}. Después."},
    )
    assert compartidos["compartidos"] == {"cantemist/a": "S1"}
    assert compartidos["fraccion_maxima"] == 1.0


def test_sinonimos_externos_codiesp_gana_y_sin_documentos_excluidos():
    menciones = [
        _m("Metástasis", "C79.9", "cantemist/a", "cantemist"),
        _m("metástasis", "C79.9", "cantemist/b", "cantemist"),
        _m("metástasis", "C76.8", "S2"),
        _m("ictus", "I63.9", "S3"),
        _m("cirrosis", "K74.6", "S4"),
    ]
    sinonimos = sp.sinonimos_externos(menciones, excluir_documentos=["S4"], ya_estan=["ICTUS"])
    assert sinonimos == [
        nz.Sinonimo(
            "metástasis", "C79.9", sp.ORIGENES["cantemist"], ("S2", "cantemist/a", "cantemist/b"), 3
        )
    ]


def test_los_pliegues_no_usan_externas_de_los_documentos_evaluados(monkeypatch):
    indice, codificador = _indice()
    vistos = []
    original = nz.Indice.con_sinonimos

    def espiar(self, sinonimos, codificador):
        vistos.append(sinonimos)
        return original(self, sinonimos, codificador)

    monkeypatch.setattr(nz.Indice, "con_sinonimos", espiar)
    # Una mención externa por documento de CodiEsp entrenamiento, y una de CANTEMIST
    externas = [_m(f"enfermedad {d.id}", "K74.6", d.id) for d in TRAIN_6]
    externas.append(_m("carcinoma", "C76.8", "cantemist/a", "cantemist"))
    pliegues = de.pliegues_topk(indice, codificador, TRAIN_6, n=3, k=5, externas=externas)
    for pliegue, sinonimos in zip(pliegues, vistos, strict=True):
        evaluados = {d.id for d in pliegue.documentos}
        origen = {i for s in sinonimos for i in s.documentos}
        assert not origen & evaluados
        assert "cantemist/a" in origen
        assert {f"enfermedad {i}" for i in evaluados}.isdisjoint(s.texto for s in sinonimos)
    vistos.clear()
    de.pliegues_internos(indice, codificador, TRAIN_6, 0, n=3, k=5, externas=externas)
    externo = {d.id for d in de.pliegues_de(TRAIN_6, 3)[0]}
    assert all(not externo & {i for s in sinonimos for i in s.documentos} for sinonimos in vistos)
    internos = [p for j, p in enumerate(de.pliegues_de(TRAIN_6, 3)) if j != 0]
    for pliegue, sinonimos in zip(internos, vistos, strict=True):
        assert not {d.id for d in pliegue} & {i for s in sinonimos for i in s.documentos}


def test_externa_choca_con_codiesp(monkeypatch):
    indice, codificador = _indice()
    vistos = []
    original = nz.Indice.con_sinonimos
    monkeypatch.setattr(
        nz.Indice,
        "con_sinonimos",
        lambda self, s, c: vistos.append(list(s)) or original(self, s, c),
    )
    # "diabetes" es sinónimo de CodiEsp (t-6, E11.9): la externa con otro código no entra
    externas = [_m("Diabetes", "E14.9", "cantemist/a", "cantemist")]
    de._buscar(indice, codificador, TRAIN_6, [], 5, externas=externas)
    diabetes = [s for s in vistos[0] if s.texto.casefold() == "diabetes"]
    assert [(s.codigo, s.origen) for s in diabetes] == [("E11.9", ORIGEN_CODIESP)]


def _fuentes():
    cantemist = [
        _m("carcinoma", "C76.8", "cantemist/a", "cantemist"),
        _m("lesiones hepáticas", "C79.9", "cantemist/a", "cantemist"),
    ]
    distemist = [_m("infarto cerebral", "I63.9", "t-5"), _m("ictus", "I63.9", "d-1")]
    return cantemist, distemist


def _preparar():
    cantemist, distemist = _fuentes()
    ids = {"train": [d.id for d in TRAIN_6], "dev": ["d-1", "d-2"], "test": ["x-1"]}
    return fs.preparar(
        cantemist,
        distemist,
        {"cantemist": {"tnm": 1}, "distemist": {"rango": 2}},
        ids,
        {"compartidos": {}, "fraccion_maxima": 0.01},
    )


def test_preparar_quita_desarrollo_y_prueba():
    fuentes, reporte = _preparar()
    assert [m.documento for m in fuentes["distemist"]] == ["t-5"]
    assert reporte["distemist"]["excluidas_por_codiesp"]["dev"] == {"documentos": 1, "menciones": 1}
    assert reporte["distemist"]["en_codiesp_train"] == {"documentos": 1, "menciones": 1}
    assert [m.texto for m in fuentes["cantemist_lexico"]] == ["carcinoma"]
    assert reporte["cantemist"]["casos_compartidos_con_codiesp"]["compartidos"] == 0


def test_evaluar_sin_texto_de_los_corpus(monkeypatch):
    from tests.test_clasificador import _rejilla_chica

    _rejilla_chica(monkeypatch)
    fuentes, reporte = _preparar()
    indice, codificador = _indice()
    resultados = fs.evaluar(
        "sapbert",
        indice,
        codificador,
        TRAIN_6,
        {},
        fuentes,
        reporte,
        pliegues=3,
        k=5,
        replicas=5,
    )
    configuraciones = resultados["configuraciones"]
    assert list(configuraciones) == list(fs.CONFIGURACIONES)
    assert configuraciones["base"]["contra_base"] == {}
    ambas = configuraciones["ambas"]
    assert set(ambas["contra_base"]["clasificador"]) == {"micro", "tumor_maligno",
                                                          "tumor_metastasico"}  # fmt: skip
    assert all(len(f1) == 3 for f1 in ambas["f1_por_pliegue"].values())
    # "infarto cerebral" ya es sinónimo de CodiEsp (t-5): solo entra "carcinoma"
    assert ambas["indice"]["sinonimos_externos"] == 1
    assert configuraciones["base"]["indice"]["sinonimos_externos"] == 0
    texto = json.dumps(resultados, ensure_ascii=False).casefold()
    for prohibido in ("carcinoma", "hepáticas", "infarto cerebral", "ictus"):
        assert prohibido not in texto
    assert "| ambas | clasificador |" in fs.tabla(resultados)


def test_subcomando_fuentes(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli
    from tests.test_clasificador import _rejilla_chica

    _rejilla_chica(monkeypatch)
    monkeypatch.chdir(tmp_path)
    prueba = [_documento("x-1", "Con tos.", ("R05", "tos"))]
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV, "test": prueba})
    _cantemist(tmp_path / "cantemist")
    _distemist(tmp_path / "distemist")
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    main(
        ["fuentes", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
        + ["--cantemist", "cantemist", "--distemist", "distemist"]
        + ["--pliegues", "3", "--bootstrap", "5", "--salida", "f.json"]
    )
    assert "| base | clasificador |" in capsys.readouterr().out
    guardado = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert guardado["fuentes"]["cantemist"]["menciones_leidas"] == 4  # con TNM e ilegible
    assert guardado["fuentes"]["cantemist"]["menciones_usables"] == 2
    # S1 y S3 no son de CodiEsp desarrollo ni prueba: entran
    assert guardado["fuentes"]["distemist"]["menciones_usables"] == 2
    assert ORIGEN_CODIESP not in json.dumps(guardado["fuentes"])
