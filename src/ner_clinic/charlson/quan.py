"""Códigos CIE-10 de cada categoría de Charlson según Quan et al. (2005).

    Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, et al. Coding
    algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data.
    Med Care. 2005;43(11):1130-1139. doi:10.1097/01.mlr.0000182534.20362.5d

`TABLA_QUAN` copia la tabla 1 del artículo (columna "ICD-10"), con su misma notación:
"I21" es la categoría de 3 caracteres con todas sus subcategorías, "I25.2" una
subcategoría y "I42.5-I42.9" o "C00-C26" un rango. Está cruzada en
`tests/test_charlson_quan.py` contra la implementación del paquete `coder` de rOpenSci.

La tabla se escribió para la CIE-10 de la OMS. CodiEsp codifica con CIE-10-ES (la
versión española de la CIE-10-CM), que comparte los códigos de 3 y 4 caracteres pero
tiene algunos propios. Los que quedan sin categoría están en `docs/datos.md`.
"""

from ner_clinic.tipos import CategoriaCharlson

__all__ = ["TABLA_QUAN", "categoria_de", "codigos_de", "normalizar_codigo"]

C = CategoriaCharlson

TABLA_QUAN: dict[CategoriaCharlson, tuple[str, ...]] = {
    C.INFARTO_MIOCARDIO: ("I21", "I22", "I25.2"),
    C.INSUFICIENCIA_CARDIACA: (
        "I09.9",
        "I11.0",
        "I13.0",
        "I13.2",
        "I25.5",
        "I42.0",
        "I42.5-I42.9",
        "I43",
        "I50",
        "P29.0",
    ),
    C.VASCULAR_PERIFERICA: (
        "I70",
        "I71",
        "I73.1",
        "I73.8",
        "I73.9",
        "I77.1",
        "I79.0",
        "I79.2",
        "K55.1",
        "K55.8",
        "K55.9",
        "Z95.8",
        "Z95.9",
    ),
    C.CEREBROVASCULAR: ("G45", "G46", "H34.0", "I60-I69"),
    C.DEMENCIA: ("F00-F03", "F05.1", "G30", "G31.1"),
    C.PULMONAR_CRONICA: (
        "I27.8",
        "I27.9",
        "J40-J47",
        "J60-J67",
        "J68.4",
        "J70.1",
        "J70.3",
    ),
    C.REUMATOLOGICA: ("M05", "M06", "M31.5", "M32-M34", "M35.1", "M35.3", "M36.0"),
    C.ULCERA_PEPTICA: ("K25-K28",),
    C.HEPATICA_LEVE: (
        "B18",
        "K70.0-K70.3",
        "K70.9",
        "K71.3-K71.5",
        "K71.7",
        "K73",
        "K74",
        "K76.0",
        "K76.2-K76.4",
        "K76.8",
        "K76.9",
        "Z94.4",
    ),
    C.DIABETES: tuple(
        f"{tipo}.{sub}" for tipo in ("E10", "E11", "E12", "E13", "E14") for sub in "01689"
    ),
    C.DIABETES_COMPLICADA: tuple(
        f"{tipo}.{sub}" for tipo in ("E10", "E11", "E12", "E13", "E14") for sub in "23457"
    ),
    C.HEMIPLEJIA: ("G04.1", "G11.4", "G80.1", "G80.2", "G81", "G82", "G83.0-G83.4", "G83.9"),
    C.RENAL_CRONICA: (
        "I12.0",
        "I13.1",
        "N03.2-N03.7",
        "N05.2-N05.7",
        "N18",
        "N19",
        "N25.0",
        "Z49.0-Z49.2",
        "Z94.0",
        "Z99.2",
    ),
    C.TUMOR_MALIGNO: (
        "C00-C26",
        "C30-C34",
        "C37-C41",
        "C43",
        "C45-C58",
        "C60-C76",
        "C81-C85",
        "C88",
        "C90-C97",
    ),
    C.HEPATICA_SEVERA: (
        "I85.0",
        "I85.9",
        "I86.4",
        "I98.2",
        "K70.4",
        "K71.1",
        "K72.1",
        "K72.9",
        "K76.5",
        "K76.6",
        "K76.7",
    ),
    C.TUMOR_METASTASICO: ("C77-C80",),
    C.VIH_SIDA: ("B20-B22", "B24"),
}


def normalizar_codigo(codigo: str) -> str:
    """Mayúsculas, sin punto ni espacios: "e11 .9 " → "E119"."""
    return "".join(codigo.split()).upper().replace(".", "")


def _expandir(entrada: str) -> list[str]:
    """Los prefijos (sin punto) de una entrada de la tabla: "K70.0-K70.3" → K700…K703."""
    desde, _, hasta = entrada.partition("-")
    desde, hasta = normalizar_codigo(desde), normalizar_codigo(hasta or desde)
    # "C00-C26" varía los dos dígitos; "K70.0-K70.3", solo el cuarto carácter.
    fijo = {3: 1, 4: 3}.get(len(desde))
    if fijo is None or len(hasta) != len(desde) or desde[:fijo] != hasta[:fijo]:
        raise ValueError(f"Entrada de la tabla de Quan mal escrita: {entrada!r}")
    base, ancho = desde[:fijo], len(desde) - fijo
    inicio, fin = int(desde[fijo:]), int(hasta[fijo:])
    if inicio > fin:
        raise ValueError(f"Rango de Quan al revés: {entrada!r}")
    return [f"{base}{n:0{ancho}d}" for n in range(inicio, fin + 1)]


def _prefijos() -> dict[str, CategoriaCharlson]:
    prefijos: dict[str, CategoriaCharlson] = {}
    for categoria, entradas in TABLA_QUAN.items():
        for entrada in entradas:
            for prefijo in _expandir(entrada):
                if prefijo in prefijos:
                    raise ValueError(f"{prefijo} está en dos categorías de la tabla de Quan")
                prefijos[prefijo] = categoria
    return prefijos


_PREFIJOS = _prefijos()


def categoria_de(codigo: str) -> CategoriaCharlson | None:
    """La categoría de Charlson de un código CIE-10, o `None` si no tiene.

    Acepta el código con o sin punto y en cualquier caja ("e11.9", "E119"). Un código
    de 3 caracteres que la tabla solo cubre en parte (p. ej. "E11", que reparte sus
    subcategorías entre diabetes con y sin complicación) no tiene categoría.
    """
    normal = normalizar_codigo(codigo)
    return _PREFIJOS.get(normal[:4]) or _PREFIJOS.get(normal[:3])


def codigos_de(categoria: CategoriaCharlson) -> list[str]:
    """Los prefijos (sin punto) de la tabla para `categoria`, en orden."""
    return sorted(p for p, c in _PREFIJOS.items() if c is categoria)
