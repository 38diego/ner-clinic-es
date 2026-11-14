"""Jerarquía de la CIE-10: código de 3 caracteres → bloque → capítulo (#150).

El catálogo CIE-10-ES 2018 no trae los bloques ("I60-I69 Enfermedades cerebrovasculares")
ni los capítulos. La tabla `RUTA_TABLA` (solo códigos, sin descripciones) sale de la tabla
tabular de la CIE-10-CM 2018 del CDC/NCHS, de dominio público; cómo se regenera está en
`evaluacion.jerarquia_cie10`.
"""

from functools import cache
from pathlib import Path
from typing import NamedTuple

RUTA_TABLA = Path(__file__).resolve().parent / "jerarquia_cie10cm_2018.tsv"
ROMANOS = tuple(
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII".split()  # noqa: SIM905
)


class Jerarquia(NamedTuple):
    bloque: str
    capitulo: str


@cache
def leer(ruta: Path = RUTA_TABLA) -> dict[str, Jerarquia]:
    lineas = ruta.read_text(encoding="utf-8").splitlines()[1:]
    return {c: Jerarquia(b, k) for c, b, k in (linea.split("\t") for linea in lineas)}


def jerarquia_de(codigo: str) -> Jerarquia | None:
    """El bloque y el capítulo de un código ("I63.9" → ("I60-I69", "IX")), o `None`.

    El catálogo de CodiEsp trae además 20 cabeceras de capítulo ("CAP.02"): su capítulo
    sale del número y su bloque es la propia cabecera.
    """
    codigo = codigo.upper()
    if codigo.startswith("CAP.") and codigo[4:].isdigit() and 1 <= int(codigo[4:]) <= 22:
        return Jerarquia(codigo, ROMANOS[int(codigo[4:]) - 1])
    return leer().get(codigo[:3])


def capitulo(codigo: str) -> str | None:
    """El capítulo de la CIE-10 de un código ("I63.9" → "IX"), o `None` si no está en la
    tabla."""
    jerarquia = jerarquia_de(codigo)
    return None if jerarquia is None else jerarquia.capitulo


def bloque(codigo: str) -> str | None:
    """El bloque de la CIE-10 de un código ("I63.9" → "I60-I69"), o `None`."""
    jerarquia = jerarquia_de(codigo)
    return None if jerarquia is None else jerarquia.bloque
