"""Jerarquía de la CIE-10: código de 3 caracteres → bloque → capítulo (#150).

El catálogo de CodiEsp (CIE-10-ES 2018) no trae los bloques ("I60-I69 Enfermedades
cerebrovasculares") ni los capítulos. La CIE-10-ES es la traducción de la CIE-10-CM de
Estados Unidos y comparte su estructura, así que la jerarquía sale de la **tabla tabular
de la CIE-10-CM 2018** del CDC/NCHS, que es de dominio público (obra del gobierno de
Estados Unidos):

- Archivo: `ICD-10-CM-Codes-Tables-and-Index-2018.zip`, en `URL_TABULAR`, con su MD5.
- De su `icd10cm_tabular_2018.xml` se toma cada `diag` de 3 caracteres con la sección
  (bloque) y el capítulo que lo contienen.

La tabla derivada se versiona en `RUTA_TABLA`, dentro de `ner_clinic.normalizacion` (sin
descripciones, solo códigos). Cubre los 1.910 códigos de 3 caracteres del catálogo de
CodiEsp. Se regenera con:

    uv run python -m evaluacion.jerarquia_cie10 [--destino data/cie10cm]
"""

import argparse
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from evaluacion.corpus import md5_de
from ner_clinic.normalizacion.jerarquia import (  # noqa: F401  (se reexportan)
    ROMANOS,
    RUTA_TABLA,
    Jerarquia,
    jerarquia_de,
    leer,
)

URL_TABULAR = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2018/"
    "ICD-10-CM-Codes-Tables-and-Index-2018.zip"
)
MD5_TABULAR = "b6186cc0a1d6d2503bb5b3957e3a9a32"
XML_TABULAR = "icd10cm_tabular_2018.xml"
RAIZ_DESCARGA = Path("data/cie10cm")


def tabla_de_xml(contenido: bytes) -> dict[str, Jerarquia]:
    """Código de 3 caracteres → (bloque, capítulo en romanos) desde el XML tabular.

    Cada `diag` de primer nivel cuelga de la sección que lo contiene; las secciones que
    solo agrupan otras (como "C00-C96") no tienen `diag` propios y no aparecen.
    """
    raiz = ET.fromstring(contenido)
    tabla = {}
    for capitulo in raiz.findall("chapter"):
        romano = ROMANOS[int(capitulo.findtext("name")) - 1]
        for seccion in capitulo.findall("section"):
            for diag in seccion.findall("diag"):
                tabla[diag.findtext("name")] = Jerarquia(seccion.get("id"), romano)
    return tabla


def descargar(destino: Path = RAIZ_DESCARGA) -> Path:
    """Baja el zip del CDC a `destino` y verifica su MD5 (no lo vuelve a bajar si ya está)."""
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / Path(URL_TABULAR).name
    if ruta.exists() and md5_de(ruta) == MD5_TABULAR:
        return ruta
    parcial = ruta.with_suffix(".zip.parcial")
    with urllib.request.urlopen(URL_TABULAR) as respuesta, parcial.open("wb") as f:
        shutil.copyfileobj(respuesta, f)
    obtenido = md5_de(parcial)
    if obtenido != MD5_TABULAR:
        parcial.unlink()
        raise ValueError(f"El MD5 no coincide: se esperaba {MD5_TABULAR} y se obtuvo {obtenido}.")
    parcial.replace(ruta)
    return ruta


def escribir(tabla: dict[str, Jerarquia], ruta: Path = RUTA_TABLA) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    filas = ["codigo_3\tbloque\tcapitulo\n"]
    filas += [f"{c}\t{j.bloque}\t{j.capitulo}\n" for c, j in sorted(tabla.items())]
    ruta.write_text("".join(filas), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - usa la red
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--destino", type=Path, default=RAIZ_DESCARGA)
    args = parser.parse_args(argv)
    with zipfile.ZipFile(descargar(args.destino)) as z:
        tabla = tabla_de_xml(z.read(XML_TABULAR))
    escribir(tabla)
    print(f"{len(tabla)} códigos de 3 caracteres en {RUTA_TABLA}")


if __name__ == "__main__":  # pragma: no cover
    main()
