"""Segmentación del texto limpio en fragmentos que el modelo NER puede procesar.

Porta `segmentar_historia` de la tesis: agrupa párrafos hasta `max_caracteres` y, si un
párrafo solo ya supera el máximo, lo corta por oraciones (separadas por `". "`).

Diferencia con la tesis: la tesis reconstruía cada fragmento concatenando oraciones con
`". "` y párrafos con `"\\n"`, así que el fragmento no siempre era una subcadena del texto
(agregaba un punto al final de cada párrafo largo y perdía el salto de línea siguiente).
Aquí cada fragmento es una rebanada del texto limpio:
`texto[f.desplazamiento : f.desplazamiento + len(f.texto)] == f.texto`. Los puntos de
corte son los mismos que en la tesis, porque se decide con la misma cuenta de longitud.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fragmento:
    """Un trozo del texto limpio y su posición de inicio en él."""

    texto: str
    desplazamiento: int


class _Acumulador:
    """Lleva el fragmento en construcción como un rango `[inicio, fin)` del texto.

    `longitud` replica la cuenta de la tesis (oración + 2, párrafo + 1), que es la que
    decide dónde cortar; el texto real del rango siempre mide lo mismo o menos.
    """

    def __init__(self, texto: str):
        self.texto = texto
        self.fragmentos: list[Fragmento] = []
        self.inicio: int | None = None
        self.fin = 0
        self.longitud = 0

    def agregar(self, inicio: int, fin: int, longitud: int) -> None:
        if self.inicio is None:
            self.inicio = inicio
        self.fin = fin
        self.longitud += longitud

    def cerrar(self) -> None:
        if self.inicio is not None:
            crudo = self.texto[self.inicio : self.fin]
            limpio = crudo.strip()
            if limpio:
                desplazamiento = self.inicio + len(crudo) - len(crudo.lstrip())
                self.fragmentos.append(Fragmento(texto=limpio, desplazamiento=desplazamiento))
        self.inicio = None
        self.longitud = 0


def segmentar(texto_limpio: str | None, max_caracteres: int = 2000) -> list[Fragmento]:
    """Divide el texto limpio en fragmentos de hasta `max_caracteres` caracteres.

    Ningún fragmento supera el máximo, salvo una oración que por sí sola sea más larga.
    Texto vacío o `None` devuelve una lista vacía.
    """
    if not texto_limpio:
        return []

    acumulador = _Acumulador(texto_limpio)
    inicio_parrafo = 0
    for parrafo in texto_limpio.split("\n"):
        fin_parrafo = inicio_parrafo + len(parrafo)

        if len(parrafo) > max_caracteres:
            oraciones = parrafo.split(". ")
            inicio_oracion = inicio_parrafo
            for i, oracion in enumerate(oraciones):
                fin_oracion = inicio_oracion + len(oracion)
                # El punto del separador ". " queda con la oración que lo precede.
                fin_con_punto = fin_oracion + 1 if i < len(oraciones) - 1 else fin_oracion
                if acumulador.longitud + len(oracion) >= max_caracteres:
                    acumulador.cerrar()
                acumulador.agregar(inicio_oracion, fin_con_punto, len(oracion) + 2)
                inicio_oracion = fin_oracion + 2
        else:
            if acumulador.longitud + len(parrafo) + 1 > max_caracteres:
                acumulador.cerrar()
            acumulador.agregar(inicio_parrafo, fin_parrafo, len(parrafo) + 1)

        inicio_parrafo = fin_parrafo + 1

    acumulador.cerrar()
    return acumulador.fragmentos
