"""Corrección de errores de ortografía antes de las reglas de Charlson (#115).

El modelo NER tolera los errores ("diabetis mellitus", "eppoc" salen detectados), pero
las reglas de Charlson son regex y no les asignan categoría: la enfermedad aparece en la
lista y no suma al índice. `mapear` corre las reglas y, solo si ninguna aplica, corrige
las palabras del texto contra `VOCABULARIO` y vuelve a correrlas sobre el texto corregido.

Reglas de la corrección (PRD `robustez`, FR4), estrictas para no inventar enfermedades:

- Solo palabras de 5 letras o más (en el texto normalizado: minúscula y sin tildes). Las
  claves de 4 letras ("epoc", "sida") solo aceptan una letra repetida ("eppoc").
- Solo a distancia 1 de Damerau-Levenshtein de una palabra del vocabulario: una letra
  cambiada, sobrante, faltante o dos vecinas intercambiadas.
- Una palabra que ya está en el vocabulario, o en `PALABRAS_COMUNES` (palabras reales,
  que aparecen en texto clínico y cambiarían de sentido: "séptica", "isquemia",
  "señal"), no se toca.
- `correcciones` lista solo las que hicieron falta para la categoría.
- Sin ambigüedad: si queda a distancia 1 de dos palabras del vocabulario que no son la
  misma palabra en otro género ("cardiaco"/"cardiaca"), no se corrige. El cambio de
  género tampoco es un error ("cronico" no se corrige a "cronica").
- Las exclusiones de las reglas siguen aplicando sobre el texto corregido.

El resultado dice qué palabras se corrigieron (`Mapeo.correcciones`), para que la API y la
demo muestren que la categoría no salió de una coincidencia exacta.

Límites conocidos:

- Un error en una palabra de una exclusión ("diabetes gestacinal") no se corrige, porque
  la inclusión ya aplica sin corregir nada: la exclusión no se ve y la entidad sale con
  categoría. Pasaba igual antes de este módulo.
- Los errores en las claves de negación o de familiar ("niegh", "majre") no se corrigen
  aquí: los corrige `contexto` (#126).

El vocabulario son las palabras que disparan una regla por sí solas o en su frase típica
(`FRASES_DEL_VOCABULARIO`, que los tests verifican regla por regla), sin siglas: una sigla
mal escrita ("hra") está a distancia 1 de demasiadas cosas.
"""

import re
from dataclasses import dataclass

from rapidfuzz.distance import DamerauLevenshtein

from ner_clinic.charlson.base import normalizar
from ner_clinic.charlson.reglas import mapear as mapear_exacto
from ner_clinic.tipos import CategoriaCharlson

__all__ = [
    "FRASES_DEL_VOCABULARIO",
    "PALABRAS_COMUNES",
    "VOCABULARIO",
    "Mapeo",
    "corregir",
    "mapear",
]

_PALABRA = re.compile(r"[^\W\d_]+")
LONGITUD_MINIMA = 5
DISTANCIA_MAXIMA = 1

# Palabra del vocabulario -> una frase típica en la que la regla aplica. Los tests
# comprueban que cada frase mapea; las palabras que solo cuentan acompañadas
# ("insuficiencia", "cronica") llevan la frase completa.
FRASES_DEL_VOCABULARIO: dict[str, str] = {
    # Infarto de miocardio
    "infarto": "infarto agudo de miocardio",
    "miocardio": "infarto agudo de miocardio",
    # Insuficiencia cardiaca
    "insuficiencia": "insuficiencia cardiaca",
    "cardiaca": "insuficiencia cardiaca",
    "cardiaco": "infarto cardiaco",
    "cardiopatia": "cardiopatia dilatada",
    "isquemica": "miocardiopatia isquemica",
    # Vascular periférica
    "claudicacion": "claudicacion intermitente",
    "aneurisma": "aneurisma de aorta",
    "vascular": "enfermedad vascular periferica",
    "periferica": "enfermedad vascular periferica",
    # Cerebrovascular
    "cerebral": "infarto cerebral",
    # Demencia
    "demencia": "demencia",
    "alzheimer": "alzheimer",
    # Pulmonar crónica
    "epoc": "epoc",
    "enfisema": "enfisema",
    "bronquiectasias": "bronquiectasias",
    "bronquitis": "bronquitis cronica",
    "pulmonar": "enfermedad pulmonar obstructiva cronica",
    "obstructiva": "enfermedad pulmonar obstructiva cronica",
    "cronica": "enfermedad pulmonar obstructiva cronica",
    # Reumatológica
    "artritis": "artritis reumatoide",
    "reumatoide": "artritis reumatoide",
    "lupus": "lupus eritematoso sistemico",
    "esclerodermia": "esclerodermia",
    "polimiositis": "polimiositis",
    # Úlcera péptica
    "ulcera": "ulcera peptica",
    "peptica": "ulcera peptica",
    "gastrica": "ulcera gastrica",
    "duodenal": "ulcera duodenal",
    # Hepática
    "cirrosis": "cirrosis",
    "hepatitis": "hepatitis cronica",
    "hepatopatia": "hepatopatia cronica",
    "varices": "varices esofagicas",
    "esofagicas": "varices esofagicas",
    "encefalopatia": "encefalopatia hepatica",
    "hepatica": "encefalopatia hepatica",
    # Diabetes
    "diabetes": "diabetes",
    "diabetico": "diabetico",
    "diabetica": "nefropatia diabetica",
    "cetoacidosis": "cetoacidosis",
    "nefropatia": "nefropatia diabetica",
    "retinopatia": "retinopatia diabetica",
    # Hemiplejia
    "hemiplejia": "hemiplejia",
    "hemiparesia": "hemiparesia",
    "paraplejia": "paraplejia",
    "tetraplejia": "tetraplejia",
    "cuadriplejia": "cuadriplejia",
    # Renal crónica
    "renal": "insuficiencia renal cronica",
    "hemodialisis": "hemodialisis",
    # Tumor maligno y metastásico
    "cancer": "cancer de mama",
    "carcinoma": "carcinoma",
    "adenocarcinoma": "adenocarcinoma",
    "linfoma": "linfoma",
    "leucemia": "leucemia",
    "mieloma": "mieloma",
    "sarcoma": "sarcoma",
    "melanoma": "melanoma",
    "maligno": "tumor maligno",
    "maligna": "neoplasia maligna",
    "neoplasia": "neoplasia maligna",
    "metastasis": "metastasis",
    "metastasico": "metastasico",
    # VIH/SIDA
    "sida": "sida",
}

VOCABULARIO: frozenset[str] = frozenset(FRASES_DEL_VOCABULARIO)

# Palabras reales que quedan a distancia 1 de una del vocabulario y no deben corregirse
# porque cambiarían de sentido. Salen de dos barridos (revisiones del PR #124): las
# palabras de `/usr/share/dict/spanish` y del hunspell `es_CO` (95.278) y las del texto de
# DisTEMIST **entrenamiento** (18.227; la partición de prueba no se usa para ajustar).
# Solo se protegen las que aparecen en texto clínico. Las palabras raras del diccionario
# ("bascular", "malino", "pertica", "reinal") en una nota son casi siempre errores de
# digitación (la confusión b/v es muy común), así que se corrigen. Las variantes de la
# misma palabra (plurales, "hemiplejica", "metastatico") también, porque dan la misma
# categoría. "época" y "sidra" las cubre la regla de las claves de 4 letras. Un test
# recorre la lista y comprueba que cada palabra sigue protegiendo algo.
PALABRAS_COMUNES: frozenset[str] = frozenset(
    {
        "arteritis",  # arteritis temporal: no es artritis
        "clonica",  # crisis tónico-clónica: no es crónica
        "gastrina",  # la hormona: no es gástrica
        "hematica",  # serie hemática: no es hepática
        "infanto",  # infanto-juvenil: no es infarto
        "isquemia",  # se corregiría a "isquemica" y cambiaría la regla que aplica
        "leucemica",  # infiltración leucémica: ya lo dice la palabra "leucemia" si está
        "neuropatia",  # no es nefropatía (E11.4 frente a E11.2)
        "penal",  # no es renal
        "periferia",  # "lesión en la periferia": no es periférica
        "senal",  # "señal" sin tilde: no es renal
        "septica",  # "úlcera séptica" no es úlcera péptica
    }
)


@dataclass(frozen=True)
class Mapeo:
    """Categoría y CIE-10 de un texto, y las palabras que hubo que corregir para lograrlo
    (`(escrita, corregida)`, normalizadas); vacío si las reglas aplicaron sin corregir."""

    categoria: CategoriaCharlson
    cie10: str
    correcciones: tuple[tuple[str, str], ...] = ()

    @property
    def por_correccion(self) -> bool:
        return bool(self.correcciones)


def _raiz(palabra: str) -> str:
    """La palabra sin la vocal de género final: "cardiaco" y "cardiaca" son la misma."""
    return palabra[:-1] if palabra[-1] in "oa" else palabra


def _con_letra_duplicada(palabra: str, clave: str) -> bool:
    """Si `palabra` es `clave` con una letra repetida ("eppoc" de "epoc").

    Es el único error que se acepta para las claves de 4 letras ("epoc", "sida"): con
    cualquier otro, demasiadas palabras quedan a distancia 1 ("sidra", "saida").
    """
    return any(
        palabra[i] == palabra[i + 1] and palabra[:i] + palabra[i + 1 :] == clave
        for i in range(len(palabra) - 1)
    )


def _corregir_palabra(palabra: str) -> str | None:
    if len(palabra) < LONGITUD_MINIMA or palabra in VOCABULARIO or palabra in PALABRAS_COMUNES:
        return None
    cercanas = sorted(
        clave
        for clave in VOCABULARIO
        if abs(len(clave) - len(palabra)) <= DISTANCIA_MAXIMA
        and DamerauLevenshtein.distance(palabra, clave, score_cutoff=DISTANCIA_MAXIMA)
        <= DISTANCIA_MAXIMA
        and (len(clave) >= LONGITUD_MINIMA or _con_letra_duplicada(palabra, clave))
    )
    if (
        not cercanas
        or len({_raiz(c) for c in cercanas}) > 1
        or _raiz(palabra) == _raiz(cercanas[0])
    ):
        return None
    return cercanas[0]


def corregir(texto: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """El texto normalizado con las palabras corregidas y la lista de correcciones."""
    correcciones: list[tuple[str, str]] = []

    def reemplazar(coincidencia: re.Match[str]) -> str:
        palabra = coincidencia.group()
        corregida = _corregir_palabra(palabra)
        if corregida is None:
            return palabra
        correcciones.append((palabra, corregida))
        return corregida

    corregido = _PALABRA.sub(reemplazar, normalizar(texto))
    return corregido, tuple(dict.fromkeys(correcciones))  # una vez cada palabra


def mapear(texto: str) -> Mapeo | None:
    """`reglas.mapear` y, si ninguna regla aplica, lo mismo sobre el texto corregido."""
    exacto = mapear_exacto(texto)
    if exacto is not None:
        return Mapeo(*exacto)
    corregido, correcciones = corregir(texto)
    if not correcciones:
        return None
    resultado = mapear_exacto(corregido)
    if resultado is None:
        return None
    # Se quitan las correcciones que no hicieron falta: si sin una el resultado es el
    # mismo, no se reporta (la demo diría "¿quiso decir…?" de una palabra que no influyó).
    necesarias = list(correcciones)
    for correccion_ in correcciones:
        sin_ella = [c for c in necesarias if c != correccion_]
        if mapear_exacto(_aplicar(texto, sin_ella)) == resultado:
            necesarias = sin_ella
    return Mapeo(*resultado, correcciones=tuple(necesarias))


def _aplicar(texto: str, correcciones: list[tuple[str, str]]) -> str:
    """El texto normalizado con solo `correcciones` aplicadas."""
    reemplazos = dict(correcciones)
    return _PALABRA.sub(lambda m: reemplazos.get(m.group(), m.group()), normalizar(texto))
