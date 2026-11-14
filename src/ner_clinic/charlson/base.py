"""Piezas comunes de las reglas de Charlson: `normalizar` y `Regla`."""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from ner_clinic.tipos import CategoriaCharlson

# La metástasis afirmada en el texto (ya normalizado): coincide si aparece "metastasis" o
# "metastasico" y ninguna mención está negada ("sin metastasis", "libre de metastasis",
# "sin evidencia de metastasis", "metastasis descartadas"). Las exclusiones de tumor
# maligno y hepatopatía leve la usan para no perder la categoría de "linfoma sin
# metastasis" (#120). "micrometastasis" no cuenta. Con `search`, el `^` ancla al inicio
# del texto (sin `re.MULTILINE`).
METASTASIS_AFIRMADA = (
    r"^(?!.*\b(sin|no|ni|libre\s+de|ausencia\s+de|negativ[oa]\s+para|descart\w*)\s+"
    r"(?:\w+\s+){0,5}(metastasis|metastasic[oa]s?)\b)"
    r"(?!.*\bmetastasis\s+(descartad|negativ)\w*)"
    r"(?=.*\b(metastasis|metastasic[oa]))"
)
_PATRON_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PATRON_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin caracteres de control y con espacios simples.

    Igual que `limpiar_texto_clinico` de la tesis. La `ñ` también pierde la tilde y
    queda como `n`.
    """
    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    texto = _PATRON_CONTROL.sub("", texto)
    return _PATRON_ESPACIOS.sub(" ", texto).strip()


@dataclass(frozen=True)
class Regla:
    """Una categoría de Charlson con sus regex y su subclasificación CIE-10.

    `inclusion` y `exclusion` se buscan (`search`) en el texto normalizado.
    `subclasificar` recibe el texto normalizado y devuelve el código CIE-10.

    `sobrescribe=True` marca las reglas que en la tesis no respetan la máscara
    `Mapeado_Charlson`: si aplican, reemplazan la categoría que otra regla anterior
    haya asignado. Las demás solo asignan si nada se asignó antes.
    """

    categoria: CategoriaCharlson
    inclusion: re.Pattern[str]
    exclusion: re.Pattern[str]
    subclasificar: Callable[[str], str]
    sobrescribe: bool = False

    def aplica(self, texto_normalizado: str) -> bool:
        """La inclusión coincide y la exclusión no."""
        return bool(self.inclusion.search(texto_normalizado)) and not self.exclusion.search(
            texto_normalizado
        )
