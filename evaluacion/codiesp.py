"""Descarga y lectura de CodiEsp y de su catálogo CIE-10 (PRD `normalizacion`, FR1).

CodiEsp (Miranda-Escalada et al., 2020) son 1.000 casos clínicos en español codificados
a mano con CIE-10-ES, la versión española de la CIE-10-CM. Se distribuye en Zenodo
(registro 3837305, versión 1.4) con licencia CC BY 4.0. Particiones: 500 casos de
entrenamiento (`train`), 250 de desarrollo (`dev`) y 250 de prueba (`test`).

Estructura del zip:

    final_dataset_v4_to_publish/
        <particion>/text_files/*.txt
        <particion>/<particion>D.tsv   códigos de diagnóstico por documento
        <particion>/<particion>X.tsv   código, texto y posición de cada mención

Los TSV no tienen encabezado. El de CodiEsp-D tiene `documento` y `código`; el de
CodiEsp-X, `documento`, `tipo` (DIAGNOSTICO o PROCEDIMIENTO), `código`, `texto` y
`posición`. La posición es "inicio fin", o varios tramos separados por ";" cuando la
mención es discontinua ("2787 2801;2810 2823"). En ese caso el texto son los tramos
unidos por un espacio.

El catálogo (registro 3706838, versión 1.1, CC BY 4.0) es la lista de códigos válidos de
la tarea: CIE-10-ES Diagnósticos 2018, con su descripción en español y en inglés.

Nada de esto se commitea: va a `data/codiesp/` y `data/codiesp_codes/`, que están en
`.gitignore`. Fuentes y licencias en `docs/datos.md`.
"""

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from evaluacion.corpus import TOLERANCIA_OFFSETS, ErrorCorpus, descargar

URL_CODIESP = "https://zenodo.org/records/3837305/files/codiesp.zip"
MD5_CODIESP = "0642277ab2e7234f7053316b57c5e28c"
RAIZ_CODIESP = Path("data/codiesp")
CARPETA_CODIESP = "final_dataset_v4_to_publish"

URL_CATALOGO = "https://zenodo.org/records/3706838/files/codiesp_codes.zip"
MD5_CATALOGO = "531d29c58447e86820517a5a0c437a2d"
RAIZ_CATALOGO = Path("data/codiesp_codes")
CARPETA_CATALOGO = "codiesp_codes"
ARCHIVO_CATALOGO = "codiesp-D_codes.tsv"

Particion = Literal["train", "dev", "test"]
PARTICIONES: tuple[Particion, ...] = ("train", "dev", "test")
DIAGNOSTICO = "DIAGNOSTICO"


@dataclass(frozen=True)
class MencionCodificada:
    """Una mención de CodiEsp-X: `texto` es la unión de sus `tramos` con un espacio.

    Los tramos van en el orden del texto de la mención, que no siempre es el de la nota:
    en "resección vejiga" el segundo tramo puede estar antes que el primero. `inicio` y
    `fin` son los límites de todos los tramos.
    """

    tipo: str
    codigo: str
    texto: str
    tramos: tuple[tuple[int, int], ...]

    @property
    def inicio(self) -> int:
        return min(a for a, _ in self.tramos)

    @property
    def fin(self) -> int:
        return max(b for _, b in self.tramos)


@dataclass(frozen=True)
class DocumentoCodificado:
    """Un caso de CodiEsp: sus códigos de diagnóstico (D) y sus menciones (X)."""

    id: str
    texto: str
    diagnosticos: list[str] = field(default_factory=list)
    menciones: list[MencionCodificada] = field(default_factory=list)

    @property
    def menciones_diagnostico(self) -> list[MencionCodificada]:
        return [m for m in self.menciones if m.tipo == DIAGNOSTICO]


@dataclass(frozen=True)
class MencionErrada:
    """Mención cuyos tramos no reproducen su texto."""

    documento: str
    codigo: str
    texto: str
    en_texto: str


@dataclass(frozen=True)
class LecturaCodiEsp:
    documentos: list[DocumentoCodificado]
    errados: list[MencionErrada]


def codigo_en_mayuscula(codigo: str) -> str:
    """CodiEsp escribe los códigos en minúscula ("e11.9"); aquí van en mayúscula."""
    return codigo.strip().upper()


# --- Descarga ------------------------------------------------------------------------


def descargar_corpus(destino: Path = RAIZ_CODIESP) -> Path:
    """Baja CodiEsp, verifica su MD5 y lo descomprime. Devuelve la carpeta del corpus."""
    return descargar(destino, URL_CODIESP, MD5_CODIESP, "codiesp.zip", CARPETA_CODIESP)


def descargar_catalogo(destino: Path = RAIZ_CATALOGO) -> Path:
    """Baja el catálogo de códigos de CodiEsp, verifica su MD5 y lo descomprime."""
    return descargar(destino, URL_CATALOGO, MD5_CATALOGO, "codiesp_codes.zip", CARPETA_CATALOGO)


# --- Lectura -------------------------------------------------------------------------


def _carpeta(raiz: Path, anidada: str) -> Path:
    """Acepta tanto la carpeta descomprimida como la que la contiene."""
    return raiz / anidada if (raiz / anidada).is_dir() else raiz


def _filas(ruta: Path, columnas: int) -> list[list[str]]:
    if not ruta.is_file():
        raise ErrorCorpus(f"No se encontró {ruta}")
    with ruta.open(encoding="utf-8", newline="") as f:
        filas = [fila for fila in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if fila]
    for numero, fila in enumerate(filas, 1):
        if len(fila) < columnas:
            raise ErrorCorpus(f"{ruta.name}, fila {numero}: se esperaban {columnas} columnas")
    return filas


def _tramos(posicion: str) -> tuple[tuple[int, int], ...]:
    """ "2787 2801;2810 2823" → ((2787, 2801), (2810, 2823))."""
    tramos = []
    for parte in posicion.split(";"):
        inicio, fin = (int(x) for x in parte.split())
        tramos.append((inicio, fin))
    return tuple(tramos)


def leer_con_reporte(
    raiz: Path, particion: Particion, tolerancia: float = TOLERANCIA_OFFSETS
) -> LecturaCodiEsp:
    """Lee una partición de CodiEsp y reporta las menciones que no cuadran.

    Cada mención de CodiEsp-X se verifica uniendo sus tramos del texto. Las que no
    coinciden se excluyen con una advertencia; si son más de `tolerancia` (fracción del
    total), la lectura falla. En el corpus real hay 4 de 9.181 en `train`, por un
    signo de puntuación de más o de menos en el borde.
    """
    if particion not in PARTICIONES:
        raise ValueError(f"Partición desconocida: {particion!r}. Use {', '.join(PARTICIONES)}.")
    carpeta = _carpeta(Path(raiz), CARPETA_CODIESP) / particion
    if not (carpeta / "text_files").is_dir():
        raise ErrorCorpus(f"No se encontró la partición {particion!r} en {carpeta}")

    textos = {}
    for ruta in sorted((carpeta / "text_files").glob("*.txt")):
        with ruta.open(encoding="utf-8", newline="") as f:
            textos[ruta.stem] = f.read()

    diagnosticos: dict[str, list[str]] = {id_: [] for id_ in textos}
    for id_, codigo, *_ in _filas(carpeta / f"{particion}D.tsv", 2):
        if id_ not in textos:
            raise ErrorCorpus(f"{particion}D.tsv apunta a un texto que no está: {id_}")
        diagnosticos[id_].append(codigo_en_mayuscula(codigo))

    menciones: dict[str, list[MencionCodificada]] = {id_: [] for id_ in textos}
    errados: list[MencionErrada] = []
    filas_x = _filas(carpeta / f"{particion}X.tsv", 5)
    for id_, tipo, codigo, texto, posicion, *_ in filas_x:
        if id_ not in textos:
            raise ErrorCorpus(f"{particion}X.tsv apunta a un texto que no está: {id_}")
        tramos = _tramos(posicion)
        en_texto = " ".join(textos[id_][a:b] for a, b in tramos)
        if en_texto != texto:
            errados.append(MencionErrada(id_, codigo_en_mayuscula(codigo), texto, en_texto))
            continue
        menciones[id_].append(MencionCodificada(tipo, codigo_en_mayuscula(codigo), texto, tramos))

    if errados:
        detalle = "; ".join(
            f"{e.documento} {e.codigo} {e.texto!r} != {e.en_texto!r}" for e in errados
        )
        mensaje = (
            f"{len(errados)} de {len(filas_x)} menciones de {particion!r} no cuadran: {detalle}"
        )
        if len(errados) > tolerancia * len(filas_x):
            raise ErrorCorpus(mensaje)
        warnings.warn(mensaje + ". Se excluyen.", stacklevel=2)

    documentos = [
        DocumentoCodificado(
            id_,
            texto,
            diagnosticos[id_],
            sorted(menciones[id_], key=lambda m: (m.inicio, m.fin)),
        )
        for id_, texto in textos.items()
    ]
    return LecturaCodiEsp(documentos, errados)


def leer_particion(
    raiz: Path, particion: Particion, tolerancia: float = TOLERANCIA_OFFSETS
) -> list[DocumentoCodificado]:
    """Los documentos de una partición, con sus menciones verificadas."""
    return leer_con_reporte(raiz, particion, tolerancia).documentos


def ids_de(raiz: Path, particion: Particion) -> list[str]:
    """Los ids de los documentos de una partición, por el nombre de sus archivos.

    No lee los textos: sirve para verificar que nada de una partición entra a otra sin
    abrir la de prueba (#132).
    """
    if particion not in PARTICIONES:
        raise ValueError(f"Partición desconocida: {particion!r}. Use {', '.join(PARTICIONES)}.")
    carpeta = _carpeta(Path(raiz), CARPETA_CODIESP) / particion / "text_files"
    if not carpeta.is_dir():
        raise ErrorCorpus(f"No se encontró la partición {particion!r} en {carpeta}")
    return sorted(ruta.stem for ruta in carpeta.glob("*.txt"))


def leer_catalogo(raiz: Path = RAIZ_CATALOGO) -> dict[str, str]:
    """Código CIE-10-ES de diagnóstico (en mayúscula, con punto) → descripción en español."""
    ruta = _carpeta(Path(raiz), CARPETA_CATALOGO) / ARCHIVO_CATALOGO
    catalogo = {}
    for codigo, descripcion, *_ in _filas(ruta, 2):
        codigo = codigo_en_mayuscula(codigo)
        if codigo in catalogo:
            raise ErrorCorpus(f"{ARCHIVO_CATALOGO}: el código {codigo} está dos veces")
        catalogo[codigo] = " ".join(descripcion.split())
    return catalogo
