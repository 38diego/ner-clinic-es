"""Tests de la descarga y lectura de CodiEsp y su catálogo (#129), sin red.

El mini corpus es sintético y copia la estructura del zip de Zenodo.
"""

import hashlib
import zipfile
from pathlib import Path

import pytest

from evaluacion import codiesp
from evaluacion.__main__ import main
from evaluacion.corpus import ErrorCorpus

TEXTOS = {
    "caso-1": "Varón de 70 años con diabetes mellitus tipo 2 y EPOC. Se pauta estreptomicina\n"
    "por vía intramuscular.",
    "caso-2": "Mujer con infarto cerebral previo. Sin otros antecedentes.",
}


def _posicion(id_: str, *fragmentos: str) -> str:
    """ "inicio fin" de cada fragmento en el texto, separados por ";"."""
    texto = TEXTOS[id_]
    return ";".join(f"{texto.index(f)} {texto.index(f) + len(f)}" for f in fragmentos)


# documento, tipo, código (en minúscula, como en CodiEsp), texto, posición
FILAS_X = [
    (
        "caso-1",
        "DIAGNOSTICO",
        "e11.9",
        "diabetes mellitus tipo 2",
        _posicion("caso-1", "diabetes mellitus tipo 2"),
    ),
    ("caso-1", "DIAGNOSTICO", "j44.9", "EPOC", _posicion("caso-1", "EPOC")),
    # discontinua: "estreptomicina" + "intramuscular", con un salto de línea en medio
    (
        "caso-1",
        "PROCEDIMIENTO",
        "3e0234z",
        "estreptomicina intramuscular",
        _posicion("caso-1", "estreptomicina", "intramuscular"),
    ),
    ("caso-2", "DIAGNOSTICO", "i63.9", "infarto cerebral", _posicion("caso-2", "infarto cerebral")),
]
FILAS_D = [("caso-1", "e11.9"), ("caso-1", "j44.9"), ("caso-2", "i63.9")]
CATALOGO = [
    ("E11.9", "Diabetes mellitus tipo 2 sin complicaciones", "Type 2 diabetes"),
    ("J44.9", "Enfermedad pulmonar obstructiva crónica,  no especificada ", "COPD"),
    ("I63.9", "Infarto cerebral, no especificado", ""),
]


def _tsv(ruta: Path, filas) -> None:
    ruta.write_text("".join("\t".join(f) + "\n" for f in filas), encoding="utf-8")


def _mini_corpus(raiz: Path, filas_x=FILAS_X, particiones=codiesp.PARTICIONES) -> Path:
    corpus = raiz / codiesp.CARPETA_CODIESP
    for particion in particiones:
        carpeta = corpus / particion
        (carpeta / "text_files").mkdir(parents=True)
        for id_, texto in TEXTOS.items():
            (carpeta / "text_files" / f"{id_}.txt").write_text(texto, encoding="utf-8")
        _tsv(carpeta / f"{particion}D.tsv", FILAS_D)
        _tsv(carpeta / f"{particion}X.tsv", filas_x)
    return corpus


def _mini_catalogo(raiz: Path, filas=CATALOGO) -> Path:
    carpeta = raiz / codiesp.CARPETA_CATALOGO
    carpeta.mkdir(parents=True)
    _tsv(carpeta / codiesp.ARCHIVO_CATALOGO, filas)
    return carpeta


def test_las_posiciones_del_mini_corpus_cuadran():
    for id_, _, _, texto, posicion in FILAS_X:
        tramos = codiesp._tramos(posicion)
        assert " ".join(TEXTOS[id_][a:b] for a, b in tramos) == texto


# --- Lectura -------------------------------------------------------------------------


def test_lee_codigos_menciones_y_tramos(tmp_path):
    _mini_corpus(tmp_path)
    documentos = codiesp.leer_particion(tmp_path, "train")

    assert [d.id for d in documentos] == ["caso-1", "caso-2"]
    caso1 = documentos[0]
    assert caso1.diagnosticos == ["E11.9", "J44.9"]
    assert [m.codigo for m in caso1.menciones] == ["E11.9", "J44.9", "3E0234Z"]
    assert [m.codigo for m in caso1.menciones_diagnostico] == ["E11.9", "J44.9"]

    discontinua = caso1.menciones[2]
    inicio = TEXTOS["caso-1"].index("estreptomicina")
    fin = TEXTOS["caso-1"].index("intramuscular") + len("intramuscular")
    assert len(discontinua.tramos) == 2
    assert (discontinua.inicio, discontinua.fin) == (inicio, fin)
    assert discontinua.texto == "estreptomicina intramuscular"


def test_inicio_y_fin_con_tramos_desordenados():
    # En CodiEsp el segundo tramo puede estar antes que el primero ("resección vejiga")
    mencion = codiesp.MencionCodificada("DIAGNOSTICO", "L98.9", "lesión piel", ((50, 56), (10, 14)))
    assert (mencion.inicio, mencion.fin) == (10, 56)


def test_acepta_la_carpeta_del_corpus_o_la_que_la_contiene(tmp_path):
    corpus = _mini_corpus(tmp_path)
    assert codiesp.leer_particion(corpus, "dev") == codiesp.leer_particion(tmp_path, "dev")


def test_mencion_que_no_cuadra_se_excluye_con_advertencia(tmp_path):
    filas = [
        *FILAS_X,
        (
            "caso-2",
            "DIAGNOSTICO",
            "z86.73",
            "infarto cerebral previo.",
            _posicion("caso-2", "infarto cerebral previo"),
        ),
    ]
    _mini_corpus(tmp_path, filas)
    with pytest.warns(UserWarning, match="1 de 5 menciones de 'test' no cuadran"):
        lectura = codiesp.leer_con_reporte(tmp_path, "test", tolerancia=0.5)
    assert [e.codigo for e in lectura.errados] == ["Z86.73"]
    assert len(lectura.documentos[1].menciones) == 1


def test_demasiadas_menciones_que_no_cuadran_fallan(tmp_path):
    filas = [("caso-2", "DIAGNOSTICO", "i63.9", "infarto", _posicion("caso-2", "Mujer co"))]
    _mini_corpus(tmp_path, filas)
    with pytest.raises(ErrorCorpus, match="no cuadran"):
        codiesp.leer_particion(tmp_path, "test")


def test_particion_desconocida(tmp_path):
    with pytest.raises(ValueError, match="Partición desconocida"):
        codiesp.leer_particion(tmp_path, "prueba")


def test_particion_que_falta(tmp_path):
    _mini_corpus(tmp_path, particiones=("train",))
    with pytest.raises(ErrorCorpus, match="No se encontró la partición 'test'"):
        codiesp.leer_particion(tmp_path, "test")


def test_tsv_que_falta(tmp_path):
    corpus = _mini_corpus(tmp_path)
    (corpus / "dev" / "devD.tsv").unlink()
    with pytest.raises(ErrorCorpus, match="No se encontró .*devD.tsv"):
        codiesp.leer_particion(tmp_path, "dev")


def test_fila_con_columnas_de_menos(tmp_path):
    _mini_corpus(tmp_path, [("caso-1", "DIAGNOSTICO", "e11.9")])
    with pytest.raises(ErrorCorpus, match="trainX.tsv, fila 1: se esperaban 5 columnas"):
        codiesp.leer_particion(tmp_path, "train")


@pytest.mark.parametrize("archivo", ["D", "X"])
def test_codigo_de_un_texto_que_no_esta(tmp_path, archivo):
    corpus = _mini_corpus(tmp_path)
    (corpus / "train" / "text_files" / "caso-2.txt").unlink()
    with pytest.raises(ErrorCorpus, match=f"train{archivo}.tsv apunta a un texto que no está"):
        if archivo == "X":
            _tsv(corpus / "train" / "trainD.tsv", FILAS_D[:2])
        codiesp.leer_particion(tmp_path, "train")


def test_lee_el_catalogo(tmp_path):
    _mini_catalogo(tmp_path)
    catalogo = codiesp.leer_catalogo(tmp_path)
    assert catalogo == {
        "E11.9": "Diabetes mellitus tipo 2 sin complicaciones",
        "J44.9": "Enfermedad pulmonar obstructiva crónica, no especificada",
        "I63.9": "Infarto cerebral, no especificado",
    }


def test_catalogo_con_codigo_repetido_falla(tmp_path):
    _mini_catalogo(tmp_path, [*CATALOGO, ("e11.9", "Otra descripción", "")])
    with pytest.raises(ErrorCorpus, match="E11.9 está dos veces"):
        codiesp.leer_catalogo(tmp_path)


# --- Descarga ------------------------------------------------------------------------


def _zip(origen: Path, carpeta: Path, nombre: str) -> tuple[str, str]:
    ruta = origen / nombre
    with zipfile.ZipFile(ruta, "w") as z:
        for archivo in sorted(carpeta.rglob("*")):
            if archivo.is_file():
                z.write(archivo, archivo.relative_to(carpeta.parent).as_posix())
    return ruta.as_uri(), hashlib.md5(ruta.read_bytes()).hexdigest()


@pytest.fixture
def zips(tmp_path, monkeypatch):
    """El mini corpus y el mini catálogo empaquetados como en Zenodo, con sus URL y MD5."""
    origen = tmp_path / "origen"
    url, md5 = _zip(origen, _mini_corpus(origen), "codiesp.zip")
    url_cat, md5_cat = _zip(origen, _mini_catalogo(origen), "codiesp_codes.zip")
    for nombre, valor in [
        ("URL_CODIESP", url),
        ("MD5_CODIESP", md5),
        ("URL_CATALOGO", url_cat),
        ("MD5_CATALOGO", md5_cat),
    ]:
        monkeypatch.setattr(codiesp, nombre, valor)


def test_descarga_corpus_y_catalogo(tmp_path, zips):
    corpus = codiesp.descargar_corpus(tmp_path / "data")
    catalogo = codiesp.descargar_catalogo(tmp_path / "codigos")

    assert corpus == tmp_path / "data" / codiesp.CARPETA_CODIESP
    assert len(codiesp.leer_particion(corpus, "test")) == 2
    assert len(codiesp.leer_catalogo(catalogo)) == 3


def test_md5_distinto_falla(tmp_path, zips, monkeypatch):
    monkeypatch.setattr(codiesp, "MD5_CODIESP", "0" * 32)
    with pytest.raises(ErrorCorpus, match="El MD5 .* no coincide"):
        codiesp.descargar_corpus(tmp_path / "data")


def test_subcomando_descargar_codiesp(tmp_path, zips, capsys):
    main(
        [
            "descargar",
            "--corpus",
            "codiesp",
            "--destino",
            str(tmp_path / "data"),
            "--destino-catalogo",
            str(tmp_path / "codigos"),
        ]
    )
    salida = capsys.readouterr().out
    assert "train: 2 documentos, 3 menciones de diagnóstico, 0 menciones que no cuadran" in salida
    assert "codiesp_codes: 3 códigos" in salida


# --- Corpus real ---------------------------------------------------------------------


@pytest.mark.modelo
def test_corpus_real():
    """Baja CodiEsp y su catálogo (si no están) y verifica sus cifras."""
    corpus = codiesp.descargar_corpus()
    catalogo = codiesp.leer_catalogo(codiesp.descargar_catalogo())
    cifras = {}
    for particion in codiesp.PARTICIONES:
        with pytest.warns(UserWarning, match="no cuadran"):
            lectura = codiesp.leer_con_reporte(corpus, particion)
        documentos = lectura.documentos
        diagnosticos = [m for d in documentos for m in d.menciones_diagnostico]
        assert all(m.codigo in catalogo for m in diagnosticos)
        cifras[particion] = (len(documentos), len(diagnosticos), len(lectura.errados))
    assert cifras == {"train": (500, 7207, 4), "dev": (250, 3430, 3), "test": (250, 3664, 1)}
    assert len(catalogo) == 98288
