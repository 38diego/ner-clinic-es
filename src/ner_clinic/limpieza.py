"""Limpieza de la nota clínica antes del NER, sin perder las posiciones del original.

Porta `limpiar_encabezados_historia` de la tesis: quita los encabezados de folio, las
líneas de guiones y los saltos de línea repetidos. A diferencia de la tesis, guarda para
cada carácter del texto limpio su posición en el original, para poder resaltar las
entidades en la nota que pegó el usuario.

Corrección frente a la tesis (#24): antes de limpiar normaliza los saltos de línea de
Windows (`\r\n`) y de Mac antiguo (`\r`) a `\n`. La tesis solo colapsaba `\n`, así que
un texto pegado desde Windows conservaba los saltos repetidos (`\r\n\r\n`).
"""

import re
from dataclasses import dataclass, field

from ner_clinic.tipos import Mencion

PATRON_ENCABEZADO = re.compile(
    r"Folio:\s*\d+\s*\|\s*Fecha:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s*",
    flags=re.IGNORECASE,
)
PATRON_GUIONES = re.compile(r"-{3,}")
PATRON_SALTOS = re.compile(r"\n+")
# "\r" de un "\r\n" (se elimina) y "\r" suelto (se reemplaza por "\n").
PATRON_CR_DE_CRLF = re.compile(r"\r(?=\n)")
PATRON_CR_SUELTO = re.compile(r"\r")


@dataclass(frozen=True)
class TextoLimpio:
    """Texto limpio con el mapa de posiciones hacia el texto original.

    `posiciones[i]` es la posición en `original` del carácter `texto[i]`. Un carácter que
    reemplaza un tramo eliminado (el espacio que deja un encabezado o una línea de
    guiones, el `\n` que reemplaza un `\r` suelto) apunta al inicio de ese tramo.
    """

    texto: str
    original: str
    posiciones: list[int] = field(repr=False)

    def a_original(self, inicio: int, fin: int) -> tuple[int, int]:
        """Traduce el rango `[inicio, fin)` del texto limpio a un rango del original.

        Si el rango no cruza un tramo eliminado, `original[a:b] == texto[inicio:fin]`.
        Si lo cruza, el rango del original abarca también lo eliminado.
        """
        if not 0 <= inicio <= fin <= len(self.texto):
            raise ValueError(f"rango inválido [{inicio}, {fin}) para un texto de {len(self.texto)}")
        if inicio == fin:
            if inicio < len(self.texto):
                pos = self.posiciones[inicio]
            else:
                pos = self.posiciones[-1] + 1 if self.posiciones else 0
            return pos, pos
        return self.posiciones[inicio], self.posiciones[fin - 1] + 1

    def mencion_a_original(self, mencion: Mencion) -> Mencion:
        """Traduce una mención del texto limpio a una del original.

        El texto se toma del original (`original[a:b]`): si la mención cruza un tramo
        eliminado, incluye lo que estaba escrito ahí (p. ej. la línea de guiones).
        """
        inicio, fin = self.a_original(mencion.inicio, mencion.fin)
        return Mencion(self.original[inicio:fin], inicio, fin)


def _sustituir(
    texto: str, posiciones: list[int], patron: re.Pattern, reemplazo: str
) -> tuple[str, list[int]]:
    """Como `patron.sub(reemplazo, texto)`, pero arrastrando el mapa de posiciones.

    `reemplazo` es de un carácter, que queda apuntando al inicio del tramo que reemplaza,
    o vacío, y entonces el tramo se elimina sin dejar nada.
    """
    partes, nuevas, previo = [], [], 0
    for coincidencia in patron.finditer(texto):
        inicio, fin = coincidencia.span()
        partes.append(texto[previo:inicio])
        nuevas.extend(posiciones[previo:inicio])
        if reemplazo:
            partes.append(reemplazo)
            nuevas.append(posiciones[inicio])
        previo = fin
    partes.append(texto[previo:])
    nuevas.extend(posiciones[previo:])
    return "".join(partes), nuevas


def limpiar(texto: str | None) -> TextoLimpio:
    """Quita encabezados de folio, líneas de guiones y saltos de línea repetidos.

    Antes normaliza `\r\n` y `\r` a `\n`. Con saltos `\n` produce el mismo texto que la
    función de la tesis. Si `texto` no es un string (p. ej. `None`), devuelve un
    `TextoLimpio` vacío.
    """
    if not isinstance(texto, str):
        return TextoLimpio(texto="", original="", posiciones=[])

    limpio, posiciones = texto, list(range(len(texto)))
    limpio, posiciones = _sustituir(limpio, posiciones, PATRON_CR_DE_CRLF, "")
    limpio, posiciones = _sustituir(limpio, posiciones, PATRON_CR_SUELTO, "\n")
    limpio, posiciones = _sustituir(limpio, posiciones, PATRON_ENCABEZADO, " ")
    limpio, posiciones = _sustituir(limpio, posiciones, PATRON_GUIONES, " ")
    limpio, posiciones = _sustituir(limpio, posiciones, PATRON_SALTOS, "\n")

    inicio = len(limpio) - len(limpio.lstrip())
    fin = len(limpio.rstrip())
    if fin <= inicio:
        return TextoLimpio(texto="", original=texto, posiciones=[])
    return TextoLimpio(texto=limpio[inicio:fin], original=texto, posiciones=posiciones[inicio:fin])
