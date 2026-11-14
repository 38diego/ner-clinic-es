"""Tests de la descarga y lectura de DisTEMIST (`evaluacion.corpus`).

Sin red: usan el mini corpus escrito a mano en `tests/datos/distemist_mini/`, con la
misma estructura de carpetas y TSV que el zip de Zenodo. Tiene un texto con saltos de
línea `\\r\\n`, un documento sin menciones, el TSV de entrenamiento partido en dos y una
anotación con el offset errado. La descarga se prueba con una URL `file://`.

El test marcado `modelo` baja el corpus real y verifica sus cifras.
"""

import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest

from evaluacion import corpus
from evaluacion.__main__ import main
from evaluacion.corpus import (
    Documento,
    ErrorCorpus,
    MencionGold,
    descargar,
    leer_con_reporte,
    leer_particion,
)

MINI = Path(__file__).parent / "datos" / "distemist_mini"


# --- Lectura -------------------------------------------------------------------------


def test_lee_documentos_y_menciones_de_prueba():
    documentos = leer_particion(MINI, "test")
    assert [d.id for d in documentos] == ["caso-1", "caso-2", "caso-3"]
    caso1 = documentos[0]
    # Ordenadas por posición aunque en el TSV vengan desordenadas.
    assert caso1.menciones == [
        MencionGold(13, 37, "diabetes mellitus tipo 2"),
        MencionGold(40, 61, "hipertensión arterial"),
        MencionGold(78, 88, "metástasis"),
    ]


def test_offsets_cuadran_con_el_texto():
    for documento in leer_particion(MINI, "test"):
        for m in documento.menciones:
            assert documento.texto[m.inicio : m.fin] == m.texto


def test_no_traduce_saltos_de_linea():
    caso2 = leer_particion(MINI, "test")[1]
    assert "\r\n" in caso2.texto
    assert [m.texto for m in caso2.menciones] == ["EPOC", "asma", "insuficiencia cardiaca"]


def test_documento_sin_menciones_se_conserva():
    caso3 = leer_particion(MINI, "test")[2]
    assert caso3 == Documento("caso-3", "Control rutinario sin hallazgos.\n", [])


def test_acepta_la_carpeta_del_corpus_o_la_que_la_contiene():
    assert leer_particion(MINI / "distemist_zenodo", "test") == leer_particion(MINI, "test")


def test_offset_errado_falla_si_supera_la_tolerancia():
    # 1 de 4 anotaciones de entrenamiento está errada: 25 % > 1 %.
    with pytest.raises(ErrorCorpus, match="1 de 4 anotaciones de 'train'.*miocardio"):
        leer_particion(MINI, "train")


def test_offset_errado_se_reporta_y_se_excluye_bajo_la_tolerancia():
    with pytest.warns(UserWarning, match="1 de 4 .*entreno-2 T2.*Se excluyen"):
        lectura = leer_con_reporte(MINI, "train", tolerancia=0.5)
    # Lee los dos TSV de entrenamiento.
    assert [d.id for d in lectura.documentos] == ["entreno-1", "entreno-2"]
    assert lectura.total_menciones == 3
    (errado,) = lectura.errados
    assert (errado.documento, errado.marca, errado.span, errado.en_texto) == (
        "entreno-2",
        "T2",
        "miocardio",
        "Anteceden",
    )
    assert lectura.documentos[1].menciones == [MencionGold(15, 41, "infarto agudo de miocardio")]


def test_particion_desconocida():
    with pytest.raises(ValueError, match="Partición desconocida"):
        leer_particion(MINI, "dev")


def test_particion_que_no_esta(tmp_path):
    with pytest.raises(ErrorCorpus, match="No se encontró la partición 'test'"):
        leer_particion(tmp_path, "test")


def test_carpeta_train_set_alternativa(tmp_path):
    shutil.copytree(MINI / "distemist_zenodo" / "training", tmp_path / "train_set")
    with pytest.warns(UserWarning):
        lectura = leer_con_reporte(tmp_path, "train", tolerancia=0.5)
    assert len(lectura.documentos) == 2


def test_sin_tsv_falla(tmp_path):
    carpeta = tmp_path / "test_annotated"
    shutil.copytree(MINI / "distemist_zenodo" / "test_annotated", carpeta)
    for tsv in (carpeta / "subtrack1_entities").glob("*.tsv"):
        tsv.unlink()
    with pytest.raises(ErrorCorpus, match="No hay TSV"):
        leer_particion(tmp_path, "test")


def test_tsv_sin_columnas_falla(tmp_path):
    carpeta = tmp_path / "test_annotated"
    shutil.copytree(MINI / "distemist_zenodo" / "test_annotated", carpeta)
    tsv = next((carpeta / "subtrack1_entities").glob("*.tsv"))
    tsv.write_text("filename\tmark\ncaso-1\tT1\n", encoding="utf-8")
    with pytest.raises(ErrorCorpus, match="le faltan columnas"):
        leer_particion(tmp_path, "test")


def test_anotacion_de_texto_inexistente_falla(tmp_path):
    carpeta = tmp_path / "test_annotated"
    shutil.copytree(MINI / "distemist_zenodo" / "test_annotated", carpeta)
    (carpeta / "text_files" / "caso-2.txt").unlink()
    with pytest.raises(ErrorCorpus, match="texto que no está: caso-2"):
        leer_particion(tmp_path, "test")


# --- Descarga ------------------------------------------------------------------------


@pytest.fixture
def zip_mini(tmp_path) -> tuple[str, str]:
    """El mini corpus empaquetado como el zip de Zenodo: (URL file://, MD5)."""
    ruta = tmp_path / "origen" / "distemist_zenodo.zip"
    ruta.parent.mkdir()
    with zipfile.ZipFile(ruta, "w") as z:
        for archivo in sorted(MINI.rglob("*")):
            if archivo.is_file():
                z.write(archivo, archivo.relative_to(MINI).as_posix())
    return ruta.as_uri(), hashlib.md5(ruta.read_bytes()).hexdigest()


def test_descarga_verifica_y_descomprime(tmp_path, zip_mini):
    url, md5 = zip_mini
    destino = tmp_path / "data"
    carpeta = descargar(destino, url=url, md5=md5)
    assert carpeta == destino / "distemist_zenodo"
    assert len(leer_particion(carpeta, "test")) == 3
    assert not list(destino.glob("*.parcial"))


def test_no_vuelve_a_bajar_si_el_md5_coincide(tmp_path, zip_mini, monkeypatch):
    url, md5 = zip_mini
    destino = tmp_path / "data"
    descargar(destino, url=url, md5=md5)

    def sin_red(*args, **kwargs):
        raise AssertionError("no debía descargar")

    monkeypatch.setattr(corpus.urllib.request, "urlopen", sin_red)
    assert descargar(destino, url=url, md5=md5) == destino / "distemist_zenodo"


def test_vuelve_a_bajar_si_el_zip_local_esta_danado(tmp_path, zip_mini):
    url, md5 = zip_mini
    destino = tmp_path / "data"
    destino.mkdir()
    (destino / "distemist_zenodo.zip").write_bytes(b"incompleto")
    descargar(destino, url=url, md5=md5)
    assert corpus.md5_de(destino / "distemist_zenodo.zip") == md5


def test_md5_distinto_falla_sin_descomprimir(tmp_path, zip_mini):
    url, _ = zip_mini
    destino = tmp_path / "data"
    with pytest.raises(ErrorCorpus, match="El MD5 .* no coincide.*No se descomprimió"):
        descargar(destino, url=url, md5="0" * 32)
    assert list(destino.iterdir()) == []


def test_rechaza_rutas_fuera_del_destino(tmp_path):
    ruta = tmp_path / "malo.zip"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("../fuera.txt", "x")
    md5 = hashlib.md5(ruta.read_bytes()).hexdigest()
    destino = tmp_path / "data"
    with pytest.raises(ErrorCorpus, match="fuera del destino"):
        descargar(destino, url=ruta.as_uri(), md5=md5)
    assert not (tmp_path / "fuera.txt").exists()


def test_subcomando_descargar(tmp_path, zip_mini, monkeypatch, capsys):
    url, md5 = zip_mini
    monkeypatch.setattr(
        "evaluacion.__main__.descargar",
        lambda destino: descargar(destino, url=url, md5=md5),
    )
    monkeypatch.setattr(
        "evaluacion.__main__.leer_con_reporte",
        lambda raiz, particion: leer_con_reporte(raiz, particion, tolerancia=0.5),
    )
    with pytest.warns(UserWarning):
        main(["descargar", "--destino", str(tmp_path / "data")])
    salida = capsys.readouterr().out
    assert "test: 3 documentos, 6 menciones, 0 offsets" in salida
    assert "train: 2 documentos, 3 menciones, 1 offsets" in salida


# --- Corpus real ---------------------------------------------------------------------


@pytest.mark.modelo
def test_corpus_real():
    """Baja DisTEMIST (si no está) y verifica sus cifras y todos sus offsets."""
    carpeta = descargar()
    prueba = leer_con_reporte(carpeta, "test", tolerancia=0)
    entrenamiento = leer_con_reporte(carpeta, "train", tolerancia=0)
    assert (len(prueba.documentos), prueba.total_menciones) == (250, 2598)
    assert (len(entrenamiento.documentos), entrenamiento.total_menciones) == (750, 8065)
    assert prueba.errados == entrenamiento.errados == []
