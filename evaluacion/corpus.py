"""Descarga y lectura del corpus DisTEMIST (FR1 y FR2 del PRD de evaluación).

DisTEMIST (Miranda-Escalada et al., 2022) es un corpus público de casos clínicos en
español con las enfermedades anotadas a mano. Se distribuye en Zenodo (registro 7614764)
con licencia CC BY 4.0. Aquí solo se usa el subtrack 1 (NER): textos y menciones.

Estructura del zip (las carpetas se descubren; los TSV se leen todos los que haya):

    distemist_zenodo/
        test_annotated/text_files/*.txt
        test_annotated/subtrack1_entities/*.tsv
        training/text_files/*.txt
        training/subtrack1_entities/*.tsv

Cada TSV tiene las columnas `filename`, `mark`, `label`, `off0`, `off1` y `span`, con
offsets en caracteres sobre el texto leído en UTF-8 sin traducir saltos de línea.

El corpus nunca se commitea: se descarga en `data/distemist/`, que está en `.gitignore`.
"""

import csv
import hashlib
import shutil
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

URL_DISTEMIST = "https://zenodo.org/records/7614764/files/distemist_zenodo.zip"
MD5_DISTEMIST = "bc84461ad07b59a94c79ca9c648952c2"
RAIZ_DATOS = Path("data/distemist")
NOMBRE_ZIP = "distemist_zenodo.zip"
CARPETA_CORPUS = "distemist_zenodo"

Particion = Literal["test", "train"]

# Nombres posibles de la carpeta de cada partición, en orden de preferencia. El zip de
# Zenodo usa `training`; algunas copias del corpus usan `train_set`.
CARPETAS_PARTICION: dict[str, tuple[str, ...]] = {
    "test": ("test_annotated",),
    "train": ("training", "train_set"),
}

# Fracción máxima de anotaciones con offsets que no apuntan al span antes de fallar.
TOLERANCIA_OFFSETS = 0.01

COLUMNAS_TSV = ("filename", "mark", "label", "off0", "off1", "span")


class ErrorCorpus(Exception):
    """El corpus no se pudo descargar, verificar o leer."""


@dataclass(frozen=True)
class MencionGold:
    """Mención anotada en el gold: `texto[inicio:fin] == texto` del documento."""

    inicio: int
    fin: int
    texto: str


@dataclass(frozen=True)
class Documento:
    """Un caso clínico con sus menciones gold, ordenadas por posición."""

    id: str
    texto: str
    menciones: list[MencionGold] = field(default_factory=list)


@dataclass(frozen=True)
class OffsetErrado:
    """Anotación cuyo `texto[inicio:fin]` no coincide con el span del TSV."""

    documento: str
    marca: str
    inicio: int
    fin: int
    span: str
    en_texto: str


@dataclass(frozen=True)
class Lectura:
    """Resultado de leer una partición: documentos válidos y anotaciones excluidas."""

    documentos: list[Documento]
    errados: list[OffsetErrado]

    @property
    def total_menciones(self) -> int:
        return sum(len(d.menciones) for d in self.documentos)


# --- Descarga ------------------------------------------------------------------------


def md5_de(ruta: Path) -> str:
    """MD5 en hexadecimal de un archivo, leído por bloques."""
    h = hashlib.md5()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _descomprimir(zip_path: Path, destino: Path) -> None:
    """Descomprime `zip_path` en `destino`, rechazando rutas que salgan de él."""
    destino_resuelto = destino.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for nombre in z.namelist():
            ruta = (destino / nombre).resolve()
            if not ruta.is_relative_to(destino_resuelto):
                raise ErrorCorpus(f"El zip trae una ruta fuera del destino: {nombre!r}")
        z.extractall(destino)


def descargar(
    destino: Path = RAIZ_DATOS,
    url: str = URL_DISTEMIST,
    md5: str = MD5_DISTEMIST,
    nombre_zip: str = NOMBRE_ZIP,
    carpeta: str = CARPETA_CORPUS,
) -> Path:
    """Baja DisTEMIST a `destino`, verifica su MD5 y lo descomprime.

    No vuelve a bajar el zip si ya está y su MD5 coincide, ni a descomprimir si la
    carpeta del corpus ya existe. Si el MD5 del archivo bajado no coincide, lo borra y
    falla sin descomprimir. Con `nombre_zip` y `carpeta` sirve para otros zips de
    Zenodo (CodiEsp, en `evaluacion/codiesp.py`).

    Devuelve la carpeta del corpus (`destino/distemist_zenodo`, o `destino/carpeta`).
    """
    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / nombre_zip

    if not (zip_path.exists() and md5_de(zip_path) == md5):
        parcial = zip_path.with_suffix(".zip.parcial")
        with urllib.request.urlopen(url) as respuesta, parcial.open("wb") as f:
            shutil.copyfileobj(respuesta, f)
        obtenido = md5_de(parcial)
        if obtenido != md5:
            parcial.unlink()
            raise ErrorCorpus(
                f"El MD5 de {url} no coincide: se esperaba {md5} y se obtuvo {obtenido}. "
                "No se descomprimió nada."
            )
        parcial.replace(zip_path)

    corpus = destino / carpeta
    if not corpus.is_dir():
        _descomprimir(zip_path, destino)
    return corpus


# --- Lectura -------------------------------------------------------------------------


def _carpeta_corpus(raiz: Path) -> Path:
    """Acepta tanto la carpeta del corpus como la que la contiene."""
    anidada = raiz / CARPETA_CORPUS
    return anidada if anidada.is_dir() else raiz


def carpeta_particion(raiz: Path, particion: Particion) -> Path:
    """Carpeta de la partición (`test_annotated`, `training`...) dentro de `raiz`."""
    if particion not in CARPETAS_PARTICION:
        raise ValueError(f"Partición desconocida: {particion!r}. Use 'test' o 'train'.")
    corpus = _carpeta_corpus(Path(raiz))
    for nombre in CARPETAS_PARTICION[particion]:
        carpeta = corpus / nombre
        if (carpeta / "text_files").is_dir() and (carpeta / "subtrack1_entities").is_dir():
            return carpeta
    raise ErrorCorpus(
        f"No se encontró la partición {particion!r} en {corpus}: se buscó "
        f"{', '.join(CARPETAS_PARTICION[particion])} con text_files/ y subtrack1_entities/."
    )


def _leer_textos(carpeta: Path) -> dict[str, str]:
    textos = {}
    for ruta in sorted((carpeta / "text_files").glob("*.txt")):
        with ruta.open(encoding="utf-8", newline="") as f:
            textos[ruta.stem] = f.read()
    return textos


def _leer_anotaciones(carpeta: Path) -> list[dict[str, str]]:
    """Filas de todos los TSV del subtrack 1 (el de entrenamiento puede venir partido)."""
    tsvs = sorted((carpeta / "subtrack1_entities").glob("*.tsv"))
    if not tsvs:
        raise ErrorCorpus(f"No hay TSV de anotaciones en {carpeta / 'subtrack1_entities'}")
    filas = []
    for tsv in tsvs:
        with tsv.open(encoding="utf-8", newline="") as f:
            lector = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            faltan = set(COLUMNAS_TSV) - set(lector.fieldnames or ())
            if faltan:
                raise ErrorCorpus(f"A {tsv.name} le faltan columnas: {sorted(faltan)}")
            filas.extend(lector)
    return filas


def leer_con_reporte(
    raiz: Path, particion: Particion, tolerancia: float = TOLERANCIA_OFFSETS
) -> Lectura:
    """Lee una partición y devuelve los documentos y las anotaciones excluidas.

    Cada anotación se verifica con `texto[inicio:fin] == span`. Las que no coinciden se
    excluyen y se reportan con una advertencia. Si son más de `tolerancia` (fracción del
    total), la lectura falla: no se sigue con offsets errados.
    """
    carpeta = carpeta_particion(raiz, particion)
    textos = _leer_textos(carpeta)
    filas = _leer_anotaciones(carpeta)

    menciones: dict[str, list[MencionGold]] = {id_: [] for id_ in textos}
    errados: list[OffsetErrado] = []
    for fila in filas:
        id_ = fila["filename"]
        if id_ not in textos:
            raise ErrorCorpus(f"La anotación {fila['mark']} apunta a un texto que no está: {id_}")
        inicio, fin = int(fila["off0"]), int(fila["off1"])
        en_texto = textos[id_][inicio:fin]
        if en_texto != fila["span"]:
            errados.append(OffsetErrado(id_, fila["mark"], inicio, fin, fila["span"], en_texto))
            continue
        menciones[id_].append(MencionGold(inicio, fin, fila["span"]))

    if errados:
        detalle = "; ".join(
            f"{e.documento} {e.marca} [{e.inicio}:{e.fin}] {e.span!r} != {e.en_texto!r}"
            for e in errados
        )
        mensaje = (
            f"{len(errados)} de {len(filas)} anotaciones de {particion!r} con offsets "
            f"que no apuntan al span: {detalle}"
        )
        if len(errados) > tolerancia * len(filas):
            raise ErrorCorpus(mensaje)
        warnings.warn(mensaje + ". Se excluyen.", stacklevel=2)

    documentos = [
        Documento(id_, texto, sorted(menciones[id_], key=lambda m: (m.inicio, m.fin)))
        for id_, texto in textos.items()
    ]
    return Lectura(documentos, errados)


def leer_particion(
    raiz: Path, particion: Particion, tolerancia: float = TOLERANCIA_OFFSETS
) -> list[Documento]:
    """Documentos de la partición `test` o `train`, con sus menciones gold verificadas."""
    return leer_con_reporte(raiz, particion, tolerancia).documentos
