"""Métricas de contexto y Charlson sobre el set sintético reservado (FR9, issue #45).

Corre `analizar()` sobre cada nota de `evaluacion/sintetico/notas.jsonl` (el gold
adjudicado de #44, ver su README) y compara contra lo anotado:

- **Recall de entidades**: una entidad esperada se encuentra si alguna entidad de la salida
  coincide con ella sin distinguir mayúsculas y por contención en cualquier sentido, el
  criterio de `experimentos/errores_tesis/medir.py`. Si varias coinciden, se prefiere la
  que es igual (sin mayúsculas) y, si no hay, la primera en el texto. Una misma entidad de
  la salida puede emparejar con más de una esperada ("VIH" y "SIDA" en una sola mención).
- **Entidades de la salida que no están en el gold**: las que no coinciden con ninguna
  esperada de su nota. Solo se cuentan y se listan: el gold no pretende ser exhaustivo en
  el NER (no anota síntomas, signos ni hábitos), así que no es una precisión de NER.
- **Estado**: matriz de confusión (`metricas.matriz_confusion`) y exactitud por clase,
  sobre las entidades encontradas. Las filas son los estados que anota el gold
  (confirmada, negada y familiar); las columnas, además, `posible` (#88).
- **Categoría de Charlson**: precisión y recall por categoría y micro global, sobre las
  entidades encontradas que son confirmadas en el gold y en la salida. Las que el pipeline
  clasificó con otro estado ya cuentan como error de estado y no se mezclan aquí (se
  reporta cuántas son). Una entidad sin categoría esperada que sale con categoría es un
  falso positivo de esa categoría; una con categoría que sale sin ella es un falso
  negativo, y una que sale con otra cuenta como falso negativo de la esperada y falso
  positivo de la obtenida. Una precisión o un recall sin casos vale `None`.
- **Índice de Charlson**: notas con el índice exacto y con diferencia de 1 punto o menos.
- **`registrado`**: de las entidades encontradas que traen `registrado` en el gold,
  cuántas salen con ese valor. Se le pasa a `analizar()` `diagnosticos_registrados` cuando
  la nota trae `registrados`.

Cada error va listado con la nota, la entidad, lo esperado y lo obtenido: las notas son
sintéticas, así que el texto se puede versionar.

**Medición única**: el set es reservado. Se corre una vez con el modelo real y el
resultado queda en `evaluacion/resultados/sintetico.json`, con el commit y las versiones.
El subcomando no sobrescribe ese archivo; si se vuelve a medir después de cambiar algo,
el resultado va aparte (`--salida evaluacion/resultados/sintetico_v2.json`) y el reporte
lo dice.

Uso:

    uv run python -m evaluacion sintetico [--notas evaluacion/sintetico/notas.jsonl]
        [--salida ARCHIVO.json] [--cache data/cache]
"""

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion.cache import ExtractorConCache
from evaluacion.distemist import entorno
from evaluacion.metricas import matriz_confusion, micro_prf
from ner_clinic.extraccion import CONFIGURACION_POR_DEFECTO, ConfiguracionExtraccion, Extractor
from ner_clinic.pipeline import ModoMapeo, _analizar
from ner_clinic.tipos import CategoriaCharlson, Entidad, Estado, Resultado

RUTA_NOTAS = Path("evaluacion/sintetico/notas.jsonl")
RUTA_RESULTADOS = Path("evaluacion/resultados/sintetico.json")
# Columnas de la matriz: todo lo que puede dar el pipeline. Filas: lo que anota el gold,
# que no tiene `posible` (PRD `incertidumbre`): una entidad del gold que sale posible es
# un error de estado, en la columna `posible`.
ESTADOS = tuple(e.value for e in Estado)
ESTADOS_GOLD = tuple(e for e in ESTADOS if e != Estado.POSIBLE.value)
CATEGORIAS = tuple(c.value for c in CategoriaCharlson)


# --- Lectura -------------------------------------------------------------------------


def leer_notas(ruta: Path = RUTA_NOTAS) -> list[dict[str, Any]]:
    """Notas del set, una por línea de `ruta` (se saltan las líneas vacías)."""
    with Path(ruta).open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def huella(ruta: Path) -> str:
    """sha256 del archivo de notas, para saber contra qué versión del gold se midió."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# --- Emparejamiento ------------------------------------------------------------------


def coincide(esperado: str, obtenido: str) -> bool:
    """Criterio de `medir.py`: sin mayúsculas, uno contiene al otro."""
    e, o = esperado.lower(), obtenido.lower()
    return e in o or o in e


def buscar(esperado: str, entidades: Sequence[Entidad]) -> Entidad | None:
    """La entidad de la salida que corresponde a `esperado`, o `None` si no hay.

    Prefiere la igual sin mayúsculas; si no hay, la primera que coincide por contención.
    """
    candidatas = [e for e in entidades if coincide(esperado, e.texto)]
    iguales = [e for e in candidatas if e.texto.lower() == esperado.lower()]
    return (iguales or candidatas or [None])[0]


@dataclass
class Par:
    """Una entidad esperada y la que le corresponde en la salida (`None` si no salió)."""

    nota: str
    esperada: dict[str, Any]
    obtenida: Entidad | None


@dataclass
class Corrida:
    """Lo que salió de `analizar()` en cada nota, ya emparejado con el gold."""

    notas: list[dict[str, Any]]
    resultados: list[Resultado]
    pares: list[Par] = field(default_factory=list)
    sobrantes: list[tuple[str, Entidad]] = field(default_factory=list)
    segundos: float = 0.0


def correr(
    notas: Sequence[dict[str, Any]],
    extractor: Extractor,
    normalizador: Any = None,
    configuracion: ConfiguracionExtraccion = CONFIGURACION_POR_DEFECTO,
    modo_mapeo: ModoMapeo = "hibrido",
) -> Corrida:
    """Corre `analizar()` sobre cada nota y empareja su salida con lo esperado. Con
    `normalizador` (#167), la categoría de Charlson la da el normalizador según
    `modo_mapeo` (solo él o como respaldo de las reglas, #180); `configuracion` cambia una
    decisión de la extracción, como en la ablación (#55)."""
    inicio = time.perf_counter()
    corrida = Corrida(notas=list(notas), resultados=[])
    for nota in notas:
        resultado = _analizar(
            nota["texto"],
            diagnosticos_registrados=nota.get("registrados"),
            extractor=extractor,
            normalizador=normalizador,
            configuracion=configuracion,
            modo_mapeo=modo_mapeo,
        )
        corrida.resultados.append(resultado)
        for esperada in nota["entidades"]:
            corrida.pares.append(
                Par(nota["id"], esperada, buscar(esperada["texto"], resultado.entidades))
            )
        for entidad in resultado.entidades:
            if not any(coincide(e["texto"], entidad.texto) for e in nota["entidades"]):
                corrida.sobrantes.append((nota["id"], entidad))
    corrida.segundos = time.perf_counter() - inicio
    return corrida


# --- Métricas ------------------------------------------------------------------------


def _redondear(valor: float) -> float:
    return round(valor, 4)


def _fraccion(aciertos: int, total: int) -> dict[str, Any]:
    return {
        "aciertos": aciertos,
        "total": total,
        "exactitud": _redondear(aciertos / total) if total else 0.0,
    }


def _categoria(entidad: Entidad) -> str | None:
    return entidad.categoria_charlson.value if entidad.categoria_charlson else None


def _error(par: Par, esperado: Any, obtenido: Any) -> dict[str, Any]:
    return {
        "id": par.nota,
        "entidad": par.esperada["texto"],
        "salida": par.obtenida.texto if par.obtenida else None,
        "esperado": esperado,
        "obtenido": obtenido,
    }


def metricas_entidades(corrida: Corrida) -> dict[str, Any]:
    """Recall de las esperadas, con las perdidas, y las de la salida que no están en el gold."""
    encontradas = sum(p.obtenida is not None for p in corrida.pares)
    return {
        "esperadas": len(corrida.pares),
        "encontradas": encontradas,
        "recall": _redondear(encontradas / len(corrida.pares)) if corrida.pares else 0.0,
        "perdidas": [
            _error(p, p.esperada["estado"], None) for p in corrida.pares if p.obtenida is None
        ],
        "salida_total": sum(len(r.entidades) for r in corrida.resultados),
        "fuera_del_gold": len(corrida.sobrantes),
        "lista_fuera_del_gold": [
            {
                "id": nota,
                "entidad": entidad.texto,
                "esperado": None,
                "obtenido": entidad.estado.value,
                "categoria": _categoria(entidad),
            }
            for nota, entidad in corrida.sobrantes
        ],
    }


def metricas_estado(corrida: Corrida) -> dict[str, Any]:
    """Matriz de confusión del estado y exactitud global y por clase esperada."""
    pares = [p for p in corrida.pares if p.obtenida is not None]
    esperados = [p.esperada["estado"] for p in pares]
    obtenidos = [p.obtenida.estado.value for p in pares]
    completa = matriz_confusion(esperados, obtenidos, ESTADOS)
    matriz = {clase: completa[clase] for clase in ESTADOS_GOLD}
    por_clase = {
        clase: _fraccion(matriz[clase][clase], sum(matriz[clase].values()))
        for clase in ESTADOS_GOLD
    }
    return {
        "matriz": matriz,
        **_fraccion(sum(matriz[c][c] for c in ESTADOS_GOLD), len(pares)),
        "por_clase": por_clase,
        "errores": [
            _error(p, e, o) for p, e, o in zip(pares, esperados, obtenidos, strict=True) if e != o
        ],
    }


def metricas_categoria(corrida: Corrida) -> dict[str, Any]:
    """Precisión y recall por categoría de Charlson entre las confirmadas encontradas."""
    confirmada = Estado.CONFIRMADA.value
    encontradas = [
        p for p in corrida.pares if p.obtenida is not None and p.esperada["estado"] == confirmada
    ]
    pares = [p for p in encontradas if p.obtenida.estado is Estado.CONFIRMADA]
    conteos = {c: [0, 0, 0] for c in CATEGORIAS}  # vp, fp, fn
    errores = []
    for par in pares:
        esperada, obtenida = par.esperada.get("categoria"), _categoria(par.obtenida)
        if esperada == obtenida:
            if esperada is not None:
                conteos[esperada][0] += 1
            continue
        errores.append(_error(par, esperada, obtenida))
        if obtenida is not None:
            conteos[obtenida][1] += 1
        if esperada is not None:
            conteos[esperada][2] += 1

    def prf(vp: int, fp: int, fn: int) -> dict[str, Any]:
        """P, R y F1; `None` la que no tiene casos (sin predichas o sin esperadas)."""
        precision, recall, f1 = micro_prf([(vp, fp, fn)])
        return {
            "precision": _redondear(precision) if vp + fp else None,
            "recall": _redondear(recall) if vp + fn else None,
            "f1": _redondear(f1) if vp + fp and vp + fn else None,
            "vp": vp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "entidades": len(pares),
        "excluidas_por_estado": len(encontradas) - len(pares),
        "por_categoria": {c: prf(*conteos[c]) for c in CATEGORIAS if any(conteos[c])},
        "global": prf(*(sum(v[i] for v in conteos.values()) for i in range(3))),
        "errores": errores,
    }


def metricas_indice(corrida: Corrida) -> dict[str, Any]:
    """Notas con el índice de Charlson exacto y con 1 punto o menos de diferencia."""
    pares = [
        (nota, resultado.indice_charlson)
        for nota, resultado in zip(corrida.notas, corrida.resultados, strict=True)
        if "indice" in nota
    ]
    exactas = sum(nota["indice"] == obtenido for nota, obtenido in pares)
    cercanas = sum(abs(nota["indice"] - obtenido) <= 1 for nota, obtenido in pares)
    return {
        "notas": len(pares),
        "exacto": _fraccion(exactas, len(pares)),
        "a_1_punto": _fraccion(cercanas, len(pares)),
        "errores": [
            {"id": nota["id"], "esperado": nota["indice"], "obtenido": obtenido}
            for nota, obtenido in pares
            if nota["indice"] != obtenido
        ],
    }


def metricas_registrado(corrida: Corrida) -> dict[str, Any]:
    """Exactitud de `registrado` en las entidades encontradas que lo traen en el gold."""
    con_valor = [p for p in corrida.pares if "registrado" in p.esperada]
    pares = [p for p in con_valor if p.obtenida is not None]
    errores = [
        _error(p, p.esperada["registrado"], p.obtenida.registrado)
        for p in pares
        if p.obtenida.registrado != p.esperada["registrado"]
    ]
    return {
        "con_valor_esperado": len(con_valor),
        "no_encontradas": len(con_valor) - len(pares),
        **_fraccion(len(pares) - len(errores), len(pares)),
        "errores": errores,
    }


# --- Evaluación completa -------------------------------------------------------------


def evaluar(
    notas: Sequence[dict[str, Any]],
    extractor: Extractor,
    ruta_notas: Path | None = None,
    cache: ExtractorConCache | None = None,
    normalizador: Any = None,
    configuracion: ConfiguracionExtraccion = CONFIGURACION_POR_DEFECTO,
    modo_mapeo: ModoMapeo = "hibrido",
) -> dict[str, Any]:
    """Corre el set y devuelve el JSON de resultados, con el entorno y los errores.

    `ruta_notas`, si se pasa, queda en el JSON con su sha256. `cache`, si se pasa, es el
    `ExtractorConCache` cuyos contadores se reportan.
    """
    corrida = correr(notas, extractor, normalizador, configuracion, modo_mapeo)
    return {
        "mapeo": "reglas" if normalizador is None else modo_mapeo,
        "configuracion": asdict(configuracion),
        "set": {
            "notas": len(notas),
            "ruta": str(ruta_notas) if ruta_notas else None,
            "sha256": huella(ruta_notas) if ruta_notas else None,
        },
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": entorno(),
        "entidades": metricas_entidades(corrida),
        "estado": metricas_estado(corrida),
        "categoria": metricas_categoria(corrida),
        "indice": metricas_indice(corrida),
        "registrado": metricas_registrado(corrida),
        "segundos": round(corrida.segundos, 2),
        "cache": (
            None
            if cache is None
            else {"aciertos": cache.aciertos, "fallos": cache.fallos, "llamadas": cache.llamadas}
        ),
    }


# --- Reporte -------------------------------------------------------------------------


def tabla(resultados: dict[str, Any]) -> str:
    """Resumen en Markdown: métricas principales, matriz del estado y categorías."""
    ent = resultados["entidades"]
    est = resultados["estado"]
    cat = resultados["categoria"]
    ind = resultados["indice"]
    reg = resultados["registrado"]
    lineas = [
        "| Métrica | Valor |",
        "|---|---|",
        f"| Recall de entidades esperadas | {ent['recall']:.3f} "
        f"({ent['encontradas']}/{ent['esperadas']}) |",
        f"| Entidades de la salida fuera del gold | {ent['fuera_del_gold']} "
        f"de {ent['salida_total']} |",
        f"| Exactitud del estado | {est['exactitud']:.3f} ({est['aciertos']}/{est['total']}) |",
        f"| Categoría de Charlson: P / R global | {_cifra(cat['global']['precision'])} / "
        f"{_cifra(cat['global']['recall'])} ({cat['entidades']} confirmadas) |",
        f"| Índice exacto | {ind['exacto']['exactitud']:.3f} "
        f"({ind['exacto']['aciertos']}/{ind['notas']}) |",
        f"| Índice a 1 punto o menos | {ind['a_1_punto']['exactitud']:.3f} "
        f"({ind['a_1_punto']['aciertos']}/{ind['notas']}) |",
        f"| Exactitud de `registrado` | {reg['exactitud']:.3f} "
        f"({reg['aciertos']}/{reg['total']}) |",
        "",
        "| Esperado \\ obtenido | " + " | ".join(ESTADOS) + " | Exactitud |",
        "|---|" + "---|" * (len(ESTADOS) + 1),
    ]
    for clase in ESTADOS_GOLD:
        fila = " | ".join(str(est["matriz"][clase][o]) for o in ESTADOS)
        lineas.append(f"| {clase} | {fila} | {est['por_clase'][clase]['exactitud']:.3f} |")
    lineas += ["", "| Categoría | P | R | VP | FP | FN |", "|---|---|---|---|---|---|"]
    for nombre, m in cat["por_categoria"].items():
        lineas.append(
            f"| `{nombre}` | {_cifra(m['precision'])} | {_cifra(m['recall'])} | "
            f"{m['vp']} | {m['fp']} | {m['fn']} |"
        )
    return "\n".join(lineas)


def _cifra(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.3f}"
