"""Métricas de NER a nivel de mención, con intervalos de confianza por bootstrap.

Funciones puras, sin modelo ni corpus:

- `emparejar` compara las menciones gold y las predichas de un documento, en modo
  estricto (mismo `inicio` y `fin`) o laxo (hay solapamiento), y devuelve los verdaderos
  positivos, falsos positivos y falsos negativos con sus menciones.
- `micro_prf` suma los conteos de todos los documentos y calcula precisión, recall y F1.
- `bootstrap_ic` remuestrea documentos para dar el intervalo de confianza de las tres.
- `matriz_confusion` cruza clases esperadas y obtenidas (p. ej. el estado de la entidad).

Una mención es cualquier objeto con atributos `inicio` y `fin` (como `ner_clinic.tipos.
Mencion` o `Span`) o una tupla `(inicio, fin)`. Así el módulo no depende del lector del
corpus.
"""

import random
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

Modo = Literal["estricto", "laxo"]


class Span(NamedTuple):
    """Límites de una mención en el texto: `texto[inicio:fin]`."""

    inicio: int
    fin: int


class Conteos(NamedTuple):
    """Verdaderos positivos, falsos positivos y falsos negativos de un documento."""

    vp: int
    fp: int
    fn: int


class PRF(NamedTuple):
    """Precisión, recall y F1."""

    precision: float
    recall: float
    f1: float


class Intervalo(NamedTuple):
    """Límites inferior y superior de un intervalo de confianza."""

    inferior: float
    superior: float


class IntervalosPRF(NamedTuple):
    """Intervalo de confianza de la precisión, el recall y el F1."""

    precision: Intervalo
    recall: Intervalo
    f1: Intervalo


@dataclass
class Emparejamiento:
    """Resultado de emparejar las menciones de un documento.

    `verdaderos` guarda los pares `(gold, prediccion)`; `falsos_positivos` y
    `falsos_negativos`, las menciones sin pareja. Las menciones son las mismas que se
    recibieron, para el análisis de errores.
    """

    verdaderos: list[tuple[Any, Any]] = field(default_factory=list)
    falsos_positivos: list[Any] = field(default_factory=list)
    falsos_negativos: list[Any] = field(default_factory=list)

    @property
    def conteos(self) -> Conteos:
        return Conteos(
            vp=len(self.verdaderos),
            fp=len(self.falsos_positivos),
            fn=len(self.falsos_negativos),
        )


def limites(mencion: Any) -> Span:
    """Devuelve `(inicio, fin)` de una mención con atributos o de una tupla."""
    if hasattr(mencion, "inicio") and hasattr(mencion, "fin"):
        return Span(mencion.inicio, mencion.fin)
    inicio, fin = mencion
    return Span(inicio, fin)


def _solapamiento(a: Span, b: Span) -> int:
    """Caracteres en común entre dos spans; 0 si no se tocan."""
    return max(0, min(a.fin, b.fin) - max(a.inicio, b.inicio))


def _sin_duplicados(menciones: Iterable[Any]) -> list[Any]:
    """Deja la primera mención de cada span `(inicio, fin)`, en el orden recibido."""
    vistos: set[Span] = set()
    unicas = []
    for mencion in menciones:
        span = limites(mencion)
        if span not in vistos:
            vistos.add(span)
            unicas.append(mencion)
    return unicas


def emparejar(
    gold: Iterable[Any], prediccion: Iterable[Any], modo: Modo = "estricto"
) -> Emparejamiento:
    """Empareja las menciones gold y predichas de un documento.

    - **Estricto**: una predicción acierta si tiene el mismo `(inicio, fin)` que un gold.
    - **Laxo**: acierta si se solapa con un gold en al menos un carácter. El emparejamiento
      es uno a uno: se toman los pares de mayor a menor solapamiento (a igual solapamiento,
      primero el gold y luego la predicción que aparecen antes) y cada gold y cada
      predicción entran en un solo par. Una predicción que cubre dos gold acierta uno y el
      otro queda como falso negativo.

    Las predicciones con el mismo span cuentan una sola vez. Los gold no se deduplican.
    """
    if modo not in ("estricto", "laxo"):
        raise ValueError(f"modo desconocido: {modo!r}; use 'estricto' o 'laxo'")
    golds = list(gold)
    predichas = _sin_duplicados(prediccion)
    spans_gold = [limites(g) for g in golds]
    spans_pred = [limites(p) for p in predichas]

    if modo == "estricto":
        candidatos = [
            (i, j) for i, sg in enumerate(spans_gold) for j, sp in enumerate(spans_pred) if sg == sp
        ]
    else:
        solapes = [
            (_solapamiento(sg, sp), i, j)
            for i, sg in enumerate(spans_gold)
            for j, sp in enumerate(spans_pred)
        ]
        solapes.sort(key=lambda t: (-t[0], t[1], t[2]))
        candidatos = [(i, j) for solape, i, j in solapes if solape > 0]

    gold_usados: set[int] = set()
    pred_usadas: set[int] = set()
    pares: list[tuple[int, int]] = []
    for i, j in candidatos:
        if i not in gold_usados and j not in pred_usadas:
            gold_usados.add(i)
            pred_usadas.add(j)
            pares.append((i, j))

    pares.sort()
    return Emparejamiento(
        verdaderos=[(golds[i], predichas[j]) for i, j in pares],
        falsos_positivos=[p for j, p in enumerate(predichas) if j not in pred_usadas],
        falsos_negativos=[g for i, g in enumerate(golds) if i not in gold_usados],
    )


def _dividir(numerador: float, denominador: float) -> float:
    return numerador / denominador if denominador else 0.0


def micro_prf(conteos_por_documento: Iterable[Sequence[int]]) -> PRF:
    """Micro precisión, recall y F1: suma los `(vp, fp, fn)` de todos los documentos.

    Con denominador cero la métrica vale 0.0, sin excepción.
    """
    vp = fp = fn = 0
    for conteos in conteos_por_documento:
        vp_doc, fp_doc, fn_doc = conteos
        vp += vp_doc
        fp += fp_doc
        fn += fn_doc
    precision = _dividir(vp, vp + fp)
    recall = _dividir(vp, vp + fn)
    f1 = _dividir(2 * precision * recall, precision + recall)
    return PRF(precision, recall, f1)


def _percentil(valores_ordenados: Sequence[float], q: float) -> float:
    """Percentil `q` (entre 0 y 1) con interpolación lineal, como `numpy.percentile`."""
    posicion = q * (len(valores_ordenados) - 1)
    abajo = int(posicion)
    arriba = min(abajo + 1, len(valores_ordenados) - 1)
    fraccion = posicion - abajo
    return valores_ordenados[abajo] + (valores_ordenados[arriba] - valores_ordenados[abajo]) * (
        fraccion
    )


def bootstrap_ic(
    conteos_por_documento: Iterable[Sequence[int]],
    n: int = 1000,
    semilla: int = 0,
    nivel: float = 0.95,
) -> IntervalosPRF:
    """Intervalo de confianza de P, R y F1 por bootstrap sobre documentos.

    En cada una de las `n` réplicas se sortean, con reemplazo, tantos documentos como hay
    y se calcula `micro_prf`. El intervalo es el de percentiles: `(1 - nivel) / 2` y
    `(1 + nivel) / 2`. Con la misma semilla da el mismo resultado. Sin documentos, todos
    los intervalos son `(0.0, 0.0)`.
    """
    if n < 1:
        raise ValueError(f"n debe ser al menos 1; se recibió {n}")
    if not 0 < nivel < 1:
        raise ValueError(f"nivel debe estar entre 0 y 1; se recibió {nivel}")
    documentos = [tuple(c) for c in conteos_por_documento]
    if not documentos:
        cero = Intervalo(0.0, 0.0)
        return IntervalosPRF(cero, cero, cero)

    generador = random.Random(semilla)
    replicas = [micro_prf(generador.choices(documentos, k=len(documentos))) for _ in range(n)]
    cola = (1 - nivel) / 2

    def intervalo(valores: list[float]) -> Intervalo:
        valores.sort()
        return Intervalo(_percentil(valores, cola), _percentil(valores, 1 - cola))

    return IntervalosPRF(
        precision=intervalo([r.precision for r in replicas]),
        recall=intervalo([r.recall for r in replicas]),
        f1=intervalo([r.f1 for r in replicas]),
    )


def bootstrap_diferencia(
    conteos_a: Sequence[Sequence[int]],
    conteos_b: Sequence[Sequence[int]],
    n: int = 1000,
    semilla: int = 0,
    nivel: float = 0.95,
) -> IntervalosPRF:
    """Intervalo de la diferencia `b - a` en P, R y F1, por bootstrap **pareado**.

    `conteos_a[i]` y `conteos_b[i]` son el mismo documento medido con dos métodos. Cada
    réplica sortea los mismos documentos para los dos, así que la variación que comparten
    (documentos fáciles o difíciles) se cancela y el intervalo es más angosto que el de
    comparar los dos intervalos sueltos.
    """
    if len(conteos_a) != len(conteos_b):
        raise ValueError(
            f"Los dos métodos tienen que medirse en los mismos documentos: "
            f"{len(conteos_a)} y {len(conteos_b)}"
        )
    if n < 1:
        raise ValueError(f"n debe ser al menos 1; se recibió {n}")
    if not conteos_a:
        cero = Intervalo(0.0, 0.0)
        return IntervalosPRF(cero, cero, cero)
    pares = list(zip(map(tuple, conteos_a), map(tuple, conteos_b), strict=True))
    generador = random.Random(semilla)
    diferencias = []
    for _ in range(n):
        muestra = generador.choices(pares, k=len(pares))
        a = micro_prf(p[0] for p in muestra)
        b = micro_prf(p[1] for p in muestra)
        diferencias.append((b.precision - a.precision, b.recall - a.recall, b.f1 - a.f1))
    cola = (1 - nivel) / 2

    def intervalo(valores: list[float]) -> Intervalo:
        valores.sort()
        return Intervalo(_percentil(valores, cola), _percentil(valores, 1 - cola))

    return IntervalosPRF(*(intervalo([d[i] for d in diferencias]) for i in range(3)))


def matriz_confusion(
    esperados: Sequence[Hashable], obtenidos: Sequence[Hashable], clases: Sequence[Hashable]
) -> dict[Hashable, dict[Hashable, int]]:
    """Cuenta los pares `(esperado, obtenido)`: `matriz[esperado][obtenido]`.

    Filas y columnas siguen el orden de `clases`, con ceros donde no hay casos. Falla si
    las listas no tienen el mismo largo o si aparece una clase que no está en `clases`.
    """
    if len(esperados) != len(obtenidos):
        raise ValueError(
            f"esperados y obtenidos difieren en largo: {len(esperados)} y {len(obtenidos)}"
        )
    matriz = {fila: dict.fromkeys(clases, 0) for fila in clases}
    for esperado, obtenido in zip(esperados, obtenidos, strict=True):
        for clase in (esperado, obtenido):
            if clase not in matriz:
                raise ValueError(f"clase desconocida: {clase!r}")
        matriz[esperado][obtenido] += 1
    return matriz
