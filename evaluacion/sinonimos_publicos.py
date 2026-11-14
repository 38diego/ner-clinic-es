"""Más sinónimos públicos para el índice: CANTEMIST y DisTEMIST (PRD `normalizacion-f1`,
FR5, #153).

Los sinónimos de CodiEsp entrenamiento (#132) cubren lo que los codificadores de CodiEsp
anotaron en 500 notas. Aquí se suman menciones de otros dos corpus públicos, sobre todo
de tumores, que eran 27 de las 52 pérdidas en desarrollo (#149).

**CANTEMIST** (Zenodo 3978041, v1.6, CC BY 4.0): 1.301 casos clínicos de oncología con las
menciones de tumor codificadas en CIE-O-3 **morfología** (`8140/3`). El código no trae la
topografía, así que la categoría sale del comportamiento, el dígito tras la barra
(`codigo_cieo`):

- `/6` (metastásico) → C79.9; `/9` (maligno, incierto si primario o metastásico; en la
  práctica "carcinomatosis") → C80.0. Las dos son tumor metastásico en Quan (C77-C80).
- `/3` (maligno) → tumor maligno, con un código representativo de la morfología:
  linfoma de Hodgkin C81.9, otros linfomas C85.9, mieloma C90.0, leucemias C95.9,
  melanoma C43.9 y el resto C76.8. Sin categoría: el basocelular (8090-8098) va a C44.91,
  piel, que Quan excluye, y los mielodisplásicos y mieloproliferativos (9950-9989) a
  D47.9, porque la CIE-10 los pone en D45-D47.
- `/0` (benigno) → D36.9, `/1` (incierto) → D48.9, `/2` (*in situ*) → D09.9: sin
  categoría. Entran igual, porque también enseñan qué **no** es un tumor maligno.
- El segundo dígito (grado) y el sufijo `/H` se ignoran.
- Las etiquetas TNM, simples o compuestas ("M0", "pT2N0M0"), se descartan (`es_tnm`):
  CANTEMIST las anota con `/6` aunque "M0" dice que **no** hay metástasis.

Variante **`cantemist_lexico`**, declarada antes de medir: CANTEMIST anota en contexto,
así que "lesiones hepáticas" o "nódulos pulmonares" llevan `/6` porque en esa nota eran
metástasis. Como sinónimos sueltos enseñarían que todo nódulo pulmonar es metastásico. La
variante deja solo los textos con una raíz oncológica (`PATRON_LEXICO`).

**DisTEMIST** (Zenodo 7614764, CC BY 4.0): sus `cross-mappings` traen, para las
enfermedades de entrenamiento enlazadas a SNOMED CT, el código CIE-10 que los
organizadores sacaron por el metatesauro UMLS. Se usa solo esa columna CIE-10, nunca los
códigos SNOMED, así que no hace falta una tabla SNOMED → CIE-10 propia. Para la
evaluación el índice vive en `data/` y no se redistribuye; si llegara a la imagen
publicada, habría que revisar la atribución (ver `docs/datos.md`). Se descartan los rangos
("I10-I15.9"), los compuestos ("N17.0+R82.1"), las relaciones `COMPOSITE` y los códigos
cuyo capítulo no está en la tabla de `jerarquia_cie10` ("E14", de la CIE-10 de la OMS).

**Fuga.** Los 1.000 documentos de DisTEMIST son los de CodiEsp con el prefijo `es-`
(`id_normalizado`), y las 583 notas con CIE-10 están todas en CodiEsp. Así que:

- Las menciones de documentos de CodiEsp desarrollo o prueba **nunca** entran
  (`sin_documentos`, que reporta cuántas quitó).
- En la validación cruzada tampoco entran las de los pliegues que se evalúan
  (`dos_etapas._buscar`, con `verificar_origen` sobre el id normalizado).
- CANTEMIST no comparte casos con CodiEsp: `documentos_compartidos` compara los textos
  por 8-gramas de palabras y lo reporta.

**Choques.** Si un texto ya es sinónimo de CodiEsp en ese pliegue, gana el de CodiEsp.
Entre las fuentes externas, el texto se queda con su código más frecuente, como en #132.
"""

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluacion.corpus import descargar
from evaluacion.jerarquia_cie10 import jerarquia_de
from evaluacion.normalizador import Sinonimo

URL_CANTEMIST = "https://zenodo.org/records/3978041/files/cantemist.zip"
MD5_CANTEMIST = "a219fcca9078f19490471c920a3b5816"
RAIZ_CANTEMIST = Path("data/cantemist")
PARTICIONES_CANTEMIST = ("train-set", "dev-set1", "dev-set2", "test-set")
RAIZ_DISTEMIST = Path("data/distemist/distemist_zenodo")
CARPETA_CRUCES = "cross-mappings"

ORIGENES = {
    "cantemist": "mención de CANTEMIST",
    "distemist": "mención de DisTEMIST entrenamiento",
}
FUENTES = ("cantemist", "cantemist_lexico", "distemist")

PATRON_CIEO = re.compile(r"(\d{4})/(\d)\d?(?:/H)?")
# Un componente TNM ("pT2a", "N1mi", "M0", "No", "Mx"); una etiqueta los encadena ("T3N0M0")
PATRON_TNM = re.compile(r"[cpyra]{0,3}[tnm](?:[0-4xo]|is)[a-d]?(?:mi)?", re.IGNORECASE)
PATRON_LEXICO = re.compile(
    r"neopl|tumor|c[aá]ncer|carcin|sarcom|linfom|leuc|mielom|melanom|blastom|"
    r"met[aá]st|malign|omas?\b",
    re.IGNORECASE,
)
RELACIONES_DISTEMIST = ("EXACT", "NARROW")
NGRAMA = 8
UMBRAL_COMPARTIDO = 0.2


@dataclass(frozen=True)
class MencionExterna:
    """Una mención de un corpus externo con su código CIE-10-ES. `documento` es el id
    normalizado (el de CodiEsp si es el mismo caso)."""

    texto: str
    codigo: str
    documento: str
    fuente: str


def id_normalizado(id_: str) -> str:
    """El id de DisTEMIST sin el prefijo `es-`, que es el de CodiEsp."""
    return id_[3:] if id_.startswith("es-") else id_


def normalizar(texto: str) -> str:
    return " ".join(texto.split())


# --- CANTEMIST -----------------------------------------------------------------------


def descargar_cantemist(destino: Path = RAIZ_CANTEMIST) -> Path:
    """Baja CANTEMIST, verifica el MD5 y lo descomprime en `destino`."""
    descargar(destino, URL_CANTEMIST, MD5_CANTEMIST, "cantemist.zip", PARTICIONES_CANTEMIST[0])
    return Path(destino)


def codigo_cieo(codigo: str) -> str | None:
    """El código CIE-10-ES que representa una morfología CIE-O-3, o `None` si no se lee."""
    m = PATRON_CIEO.fullmatch(codigo.strip())
    if not m:
        return None
    morfologia, comportamiento = int(m[1]), m[2]
    fijos = {"6": "C79.9", "9": "C80.0", "2": "D09.9", "1": "D48.9", "0": "D36.9"}
    if comportamiento in fijos:
        return fijos[comportamiento]
    if comportamiento != "3":
        return None
    rangos = (
        (8090, 8098, "C44.91"),  # basocelular: piel, sin categoría en Quan
        (9650, 9667, "C81.9"),
        (9590, 9729, "C85.9"),
        (9730, 9739, "C90.0"),
        (9800, 9949, "C95.9"),
        (9950, 9989, "D47.9"),  # mielodisplásicos y mieloproliferativos: D45-D47, sin categoría
        (8720, 8799, "C43.9"),
    )
    return next((c for a, b, c in rangos if a <= morfologia <= b), "C76.8")


def es_tnm(texto: str) -> bool:
    """Si la mención es solo una etiqueta TNM, simple o compuesta ("M0", "pT2N0M0",
    "ypT0N0(0/5)M0"): sin sus componentes no le queda ninguna letra, y tiene algún dígito o
    una x. "Adenocarcinoma rectal T3N1M1" no lo es: es un tumor con su estadio."""
    if not re.search(r"[\dx]", texto, re.IGNORECASE):
        return False
    resto = PATRON_TNM.sub("", texto)
    return resto != texto and not re.search(r"[^\W\d_]", resto)


def _ann(ruta: Path) -> Iterable[tuple[str, str]]:
    """(texto, código CIE-O) de cada mención de un `.ann` de CANTEMIST."""
    textos: dict[str, str] = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        partes = linea.split("\t")
        if partes[0].startswith("T") and len(partes) == 3:
            textos[partes[0]] = partes[2]
        elif partes[0].startswith("#") and len(partes) == 3:
            marca = partes[1].split()[-1]
            if marca in textos:
                yield textos[marca], partes[2]


def leer_cantemist(raiz: Path = RAIZ_CANTEMIST) -> tuple[list[MencionExterna], Counter]:
    """Las menciones de las cuatro particiones de CANTEMIST, y cuántas se descartaron y
    por qué. CANTEMIST no se usa para medir, así que todas sus particiones sirven."""
    salida, descartes = [], Counter()
    for particion in PARTICIONES_CANTEMIST:
        carpeta = Path(raiz) / particion / "cantemist-norm"
        if not carpeta.is_dir():
            raise FileNotFoundError(f"No está CANTEMIST en {carpeta}: corra `descargar`.")
        for ruta in sorted(carpeta.glob("*.ann")):
            for texto, cieo in _ann(ruta):
                texto = normalizar(texto)
                codigo = codigo_cieo(cieo)
                if codigo is None:
                    descartes["codigo_ilegible"] += 1
                elif es_tnm(texto):
                    descartes["tnm"] += 1
                elif texto:
                    salida.append(
                        MencionExterna(texto, codigo, f"cantemist/{ruta.stem}", "cantemist")
                    )
    return salida, descartes


def lexicas(menciones: Sequence[MencionExterna]) -> list[MencionExterna]:
    """Solo las que tienen una raíz oncológica: la variante `cantemist_lexico`."""
    return [m for m in menciones if PATRON_LEXICO.search(m.texto)]


def textos_cantemist(raiz: Path = RAIZ_CANTEMIST) -> dict[str, str]:
    """Id → texto de las notas, para buscar casos compartidos con CodiEsp."""
    return {
        f"cantemist/{ruta.stem}": ruta.read_text(encoding="utf-8")
        for particion in PARTICIONES_CANTEMIST
        for ruta in sorted((Path(raiz) / particion / "cantemist-norm").glob("*.txt"))
    }


# --- DisTEMIST -----------------------------------------------------------------------


def _codigo_distemist(codigo: str, relacion: str) -> tuple[str | None, str | None]:
    """(código, None) si sirve, o (None, motivo del descarte)."""
    if codigo in ("", "None"):
        return None, "sin_cie10"
    if relacion not in RELACIONES_DISTEMIST:
        return None, "relacion"
    if "+" in codigo or "|" in codigo:
        return None, "compuesto"
    if "-" in codigo:
        return None, "rango"
    if jerarquia_de(codigo) is None:
        return None, "fuera_de_la_tabla"
    return codigo.upper(), None


def leer_distemist(raiz: Path = RAIZ_DISTEMIST) -> tuple[list[MencionExterna], Counter]:
    """Las menciones de los `cross-mappings` de DisTEMIST con un código CIE-10 usable."""
    carpeta = Path(raiz) / CARPETA_CRUCES
    rutas = sorted(carpeta.glob("*.tsv"))
    if not rutas:
        raise FileNotFoundError(f"No están los cross-mappings de DisTEMIST en {carpeta}.")
    salida, descartes = [], Counter()
    for ruta in rutas:
        with ruta.open(encoding="utf-8", newline="") as f:
            for fila in csv.DictReader(f, delimiter="\t"):
                codigo, motivo = _codigo_distemist(fila["icd10_code"], fila["semantic_rel"])
                texto = normalizar(fila["span"])
                if motivo:
                    descartes[motivo] += 1
                elif texto:
                    documento = id_normalizado(fila["filename"])
                    salida.append(MencionExterna(texto, codigo, documento, "distemist"))
    return salida, descartes


# --- Fuga y choques ------------------------------------------------------------------


def sin_documentos(
    menciones: Sequence[MencionExterna], prohibidos: dict[str, Iterable[str]]
) -> tuple[list[MencionExterna], dict[str, dict[str, int]]]:
    """Quita las menciones de documentos de `prohibidos` (partición → ids de CodiEsp) y
    reporta cuántos documentos y menciones de cada partición se cruzaron."""
    por_particion = {p: set(ids) for p, ids in prohibidos.items()}
    todos = set().union(*por_particion.values()) if por_particion else set()
    reporte = {
        p: {
            "documentos": len({m.documento for m in menciones} & ids),
            "menciones": sum(m.documento in ids for m in menciones),
        }
        for p, ids in por_particion.items()
    }
    return [m for m in menciones if m.documento not in todos], reporte


def _ngramas(texto: str) -> set[str]:
    palabras = re.findall(r"\w+", texto.casefold())
    return {" ".join(palabras[i : i + NGRAMA]) for i in range(max(len(palabras) - NGRAMA + 1, 1))}


def documentos_compartidos(
    externos: dict[str, str], codiesp: dict[str, str], umbral: float = UMBRAL_COMPARTIDO
) -> dict[str, Any]:
    """Casos externos que comparten más de `umbral` de sus 8-gramas con una nota de
    CodiEsp, y la mayor fracción vista (para reportarla aunque no pase el umbral)."""
    indice: dict[str, set[str]] = defaultdict(set)
    for id_, texto in codiesp.items():
        for g in _ngramas(texto):
            indice[g].add(id_)
    compartidos, maxima = {}, 0.0
    for id_, texto in externos.items():
        gramas = _ngramas(texto)
        conteo = Counter(otro for g in gramas for otro in indice.get(g, ()))
        if not conteo:
            continue
        otro, n = conteo.most_common(1)[0]
        fraccion = n / len(gramas)
        maxima = max(maxima, fraccion)
        if fraccion > umbral:
            compartidos[id_] = otro
    return {"compartidos": compartidos, "fraccion_maxima": round(maxima, 4)}


def _mas_frecuente(conteo: Counter) -> str:
    return min(conteo, key=lambda clave: (-conteo[clave], clave))


def sinonimos_externos(
    menciones: Sequence[MencionExterna],
    excluir_documentos: Iterable[str] = (),
    ya_estan: Iterable[str] = (),
) -> list[Sinonimo]:
    """Un sinónimo por texto distinto, sin las menciones de `excluir_documentos` ni los
    textos que ya están (`ya_estan`, los de CodiEsp: ganan en un choque).

    Como en `sinonimos_codiesp.sinonimos_de`: el código y la escritura más frecuentes. El
    origen es el de la fuente con más menciones de ese texto.
    """
    excluir = set(excluir_documentos)
    ocupados = {t.casefold() for t in ya_estan}
    grupos: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {
            "codigos": Counter(),
            "escrituras": Counter(),
            "documentos": Counter(),
            "fuentes": Counter(),
        }  # fmt: skip
    )
    for m in menciones:
        clave = m.texto.casefold()
        if m.documento in excluir or clave in ocupados:
            continue
        grupo = grupos[clave]
        grupo["codigos"][m.codigo] += 1
        grupo["escrituras"][m.texto] += 1
        grupo["documentos"][m.documento] += 1
        grupo["fuentes"][m.fuente] += 1
    return [
        Sinonimo(
            _mas_frecuente(g["escrituras"]),
            _mas_frecuente(g["codigos"]),
            ORIGENES[_mas_frecuente(g["fuentes"])],
            tuple(sorted(g["documentos"])),
            sum(g["codigos"].values()),
        )
        for _, g in sorted(grupos.items())
    ]
