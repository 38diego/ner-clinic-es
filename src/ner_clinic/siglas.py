"""Siglas clínicas frecuentes en las notas en español y su nombre completo.

Se usan para cruzar las entidades con los diagnósticos registrados (#24): "HTA" no
comparte ninguna palabra con "Hipertensión esencial (primaria)", pero su nombre completo,
"hipertensión arterial", sí. Los nombres están en minúsculas y con tildes.
"""

import itertools
import re

# Sigla en mayúsculas -> nombre(s) completo(s). Si hay varios, se prueba cada uno.
SIGLAS: dict[str, tuple[str, ...]] = {
    "HTA": ("hipertensión arterial",),
    "DM": ("diabetes mellitus",),
    "DM1": ("diabetes mellitus tipo 1",),
    "DM2": ("diabetes mellitus tipo 2",),
    "DMID": ("diabetes mellitus insulinodependiente",),
    "DMNID": ("diabetes mellitus no insulinodependiente",),
    "EPOC": ("enfermedad pulmonar obstructiva crónica",),
    "IAM": ("infarto agudo de miocardio",),
    "SCA": ("síndrome coronario agudo",),
    "ACV": ("accidente cerebrovascular",),
    "ECV": ("enfermedad cerebrovascular",),
    "AIT": ("ataque isquémico transitorio",),
    "ICC": ("insuficiencia cardiaca congestiva",),
    "IC": ("insuficiencia cardiaca",),
    "ERC": ("enfermedad renal crónica",),
    "IRC": ("insuficiencia renal crónica",),
    "FA": ("fibrilación auricular",),
    "TEP": ("tromboembolismo pulmonar", "embolia pulmonar"),
    "TVP": ("trombosis venosa profunda",),
    "VIH": ("virus de la inmunodeficiencia humana",),
    "SIDA": ("síndrome de inmunodeficiencia adquirida",),
    "LES": ("lupus eritematoso sistémico",),
    "AR": ("artritis reumatoide",),
    "ERGE": ("enfermedad por reflujo gastroesofágico",),
    "HBP": ("hiperplasia benigna de próstata",),
    "IVU": ("infección de vías urinarias",),
    "SAHOS": ("síndrome de apnea hipopnea obstructiva del sueño",),
}

# Siglas que en minúsculas son palabras comunes ("les" es un pronombre, "ar" una sílaba):
# solo cuentan escritas en mayúsculas.
_SOLO_MAYUSCULAS = frozenset({"LES", "AR"})

_PALABRA = re.compile(r"\w+")


def _nombres(palabra: str) -> tuple[str, ...] | None:
    sigla = palabra.upper()
    if sigla not in SIGLAS or (sigla in _SOLO_MAYUSCULAS and palabra != sigla):
        return None
    return SIGLAS[sigla]


def expandir(texto: str) -> list[str]:
    """Las versiones de `texto` con cada sigla reemplazada por su nombre completo.

    "HTA descompensada" da `["hipertensión arterial descompensada"]`. Si una sigla tiene
    varios nombres, hay una versión por cada combinación. Sin siglas, devuelve `[]`.
    """
    partes: list[tuple[str, ...]] = []
    desde = 0
    hay_siglas = False
    for palabra in _PALABRA.finditer(texto):
        nombres = _nombres(palabra.group())
        if nombres is None:
            continue
        hay_siglas = True
        partes.append((texto[desde : palabra.start()],))
        partes.append(nombres)
        desde = palabra.end()
    if not hay_siglas:
        return []
    partes.append((texto[desde:],))
    return ["".join(combinacion) for combinacion in itertools.product(*partes)]
