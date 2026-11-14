"""Evaluación de NER sobre DisTEMIST: modelo crudo, pipeline y ablación (FR4, FR5, FR10).

Para cada documento de una partición corre cada variante, compara sus menciones con las
del gold (`metricas.emparejar`, en modo estricto y laxo) y calcula P/R/F1 micro con
intervalo de confianza por bootstrap sobre documentos.

Variantes (`VARIANTES`, una línea por fila; agregar una es agregar una línea):

- `crudo`: lo que devuelve el modelo (`pipeline.menciones_del_modelo`), sin fusión,
  expansión, recorte, filtros ni deduplicación.
- `pipeline`: el pipeline completo, como `analizar(texto)`.
- Ablación: el pipeline completo con un solo cambio de `ConfiguracionExtraccion` (fusión
  a otra distancia o sin fusión, sin expansión, otro filtro de longitud, sin lista negra,
  sin sufijos truncados) o sin la deduplicación interna.

Las predicciones de las variantes de pipeline son todas las menciones de todas las
entidades de `Resultado.entidades`, con cualquier estado: DisTEMIST anota también las
enfermedades negadas.

El modelo corre una sola vez por fragmento gracias a `cache.ExtractorConCache`. Los
tiempos se miden aparte, en una pasada del pipeline completo con el extractor sin caché,
documento por documento, después de calentar el modelo (la carga no cuenta).

Salidas:

- `evaluacion/resultados/distemist_<particion>.json` (versionado): versiones, CPU,
  métricas y conteos por variante y tiempos. No lleva texto del corpus.
- `data/evaluacion/distemist_<particion>_errores.json` (no versionado): los verdaderos
  positivos, falsos positivos y falsos negativos por documento, con sus spans y su texto,
  de las variantes que se analizan a mano (`VARIANTES_DETALLADAS`), para el análisis de
  errores (#46).
"""

import json
import os
import platform
import resource
import statistics
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion.cache import ExtractorConCache, version_de
from evaluacion.corpus import Documento, Particion
from evaluacion.metricas import (
    Emparejamiento,
    Modo,
    bootstrap_diferencia,
    bootstrap_ic,
    emparejar,
    micro_prf,
)
from ner_clinic.extraccion import (
    NOMBRE_MODELO,
    STRIDE_TOKENS,
    ConfiguracionExtraccion,
    Extractor,
)
from ner_clinic.pipeline import _analizar, menciones_del_modelo

CARPETA_RESULTADOS = Path("evaluacion/resultados")
CARPETA_DETALLE = Path("data/evaluacion")
MODOS: tuple[Modo, ...] = ("estricto", "laxo")
REPLICAS_BOOTSTRAP = 1000
SEMILLA_BOOTSTRAP = 0
# Texto con el que se calienta el modelo antes de medir tiempos (sintético).
TEXTO_CALENTAMIENTO = "Paciente con hipertensión arterial y diabetes mellitus tipo 2."


# --- Variantes -----------------------------------------------------------------------


@dataclass(frozen=True)
class Variante:
    """Cómo se obtienen las predicciones de una fila de la evaluación.

    `crudo=True` toma la salida del modelo sin pipeline; si no, corre el pipeline con
    `configuracion` y, si `deduplicar` es `False`, sin `depuracion.deduplicar_internas`.
    """

    configuracion: ConfiguracionExtraccion = field(default_factory=ConfiguracionExtraccion)
    deduplicar: bool = True
    crudo: bool = False

    def predecir(self, texto: str, extractor: Extractor) -> list[Any]:
        """Menciones predichas sobre `texto`, en posiciones del texto original."""
        if self.crudo:
            return menciones_del_modelo(texto, extractor=extractor)
        resultado = _analizar(
            texto,
            extractor=extractor,
            configuracion=self.configuracion,
            deduplicar=self.deduplicar,
        )
        return [m for entidad in resultado.entidades for m in entidad.menciones]


C = ConfiguracionExtraccion
VARIANTES: dict[str, Variante] = {
    "crudo": Variante(crudo=True),
    "pipeline": Variante(),
    "fusion_0": Variante(C(max_distancia_fusion=0)),
    "fusion_10": Variante(C(max_distancia_fusion=10)),
    "fusion_18": Variante(C(max_distancia_fusion=18)),  # la de la tesis (hasta #55)
    "fusion_30": Variante(C(max_distancia_fusion=30)),
    "sin_fusion": Variante(C(max_distancia_fusion=None)),
    # La fusión de la tesis con barreras ampliadas (#55); la distancia va explícita porque
    # el pipeline ya no fusiona
    "fusion_barreras": Variante(C(max_distancia_fusion=18, barreras_fusion="ampliadas")),
    "sin_expansion": Variante(C(expandir_palabras=False)),
    "filtro_tesis": Variante(C(filtro_longitud="tesis")),
    "filtro_minimo": Variante(C(filtro_longitud="minimo")),  # el del pipeline (#56)
    "filtro_lista_blanca": Variante(C(filtro_longitud="lista_blanca")),
    "filtro_ninguno": Variante(C(filtro_longitud="ninguno")),
    "sin_lista_negra": Variante(C(usar_lista_negra=False)),
    "sin_sufijos": Variante(C(descartar_sufijos_truncados=False)),
    "sin_deduplicacion": Variante(deduplicar=False),
    "sin_cortar_parentesis": Variante(C(cortar_parentesis=False)),  # #195
}

# Variantes cuyos TP/FP/FN por documento se guardan para el análisis de errores (#46).
VARIANTES_DETALLADAS = ("crudo", "pipeline")

# Variantes por defecto de cada partición: en `train` solo interesa el chequeo de
# contaminación (FR6), que compara el modelo crudo y el pipeline.
VARIANTES_POR_PARTICION: dict[str, tuple[str, ...]] = {
    "test": tuple(VARIANTES),
    "train": ("crudo", "pipeline"),
}


# --- Corrida -------------------------------------------------------------------------


@dataclass
class Corrida:
    """Emparejamientos de cada variante y modo, un elemento por documento."""

    documentos: list[str]
    emparejamientos: dict[str, dict[str, list[Emparejamiento]]]
    segundos: float = 0.0


def correr(
    documentos: Sequence[Documento], extractor: Extractor, variantes: Iterable[str]
) -> Corrida:
    """Corre `variantes` sobre `documentos` y empareja sus predicciones con el gold."""
    nombres = list(variantes)
    desconocidas = [n for n in nombres if n not in VARIANTES]
    if desconocidas:
        raise ValueError(f"Variantes desconocidas: {desconocidas}. Opciones: {list(VARIANTES)}")

    inicio = time.perf_counter()
    emparejamientos: dict[str, dict[str, list[Emparejamiento]]] = {
        n: {modo: [] for modo in MODOS} for n in nombres
    }
    for documento in documentos:
        for nombre in nombres:
            predichas = VARIANTES[nombre].predecir(documento.texto, extractor)
            for modo in MODOS:
                emparejamientos[nombre][modo].append(
                    emparejar(documento.menciones, predichas, modo)
                )
    return Corrida(
        documentos=[d.id for d in documentos],
        emparejamientos=emparejamientos,
        segundos=time.perf_counter() - inicio,
    )


def _redondear(valor: float) -> float:
    return round(valor, 4)


def metricas_de(
    emparejamientos: Sequence[Emparejamiento],
    replicas: int = REPLICAS_BOOTSTRAP,
    semilla: int = SEMILLA_BOOTSTRAP,
) -> dict[str, Any]:
    """P/R/F1 micro, su intervalo del 95 % y los conteos totales de una variante y modo."""
    conteos = [e.conteos for e in emparejamientos]
    prf = micro_prf(conteos)
    ic = bootstrap_ic(conteos, n=replicas, semilla=semilla)
    return {
        "precision": _redondear(prf.precision),
        "recall": _redondear(prf.recall),
        "f1": _redondear(prf.f1),
        "ic95": {
            nombre: [_redondear(intervalo.inferior), _redondear(intervalo.superior)]
            for nombre, intervalo in ic._asdict().items()
        },
        "vp": sum(c.vp for c in conteos),
        "fp": sum(c.fp for c in conteos),
        "fn": sum(c.fn for c in conteos),
    }


def contra_pipeline(corrida: Corrida, replicas: int = REPLICAS_BOOTSTRAP) -> dict[str, Any]:
    """Diferencia pareada por documento (variante − `pipeline`) en P, R y F1 de cada modo,
    para las variantes de pipeline corridas junto a `pipeline` (#55). Vacío si no hay."""
    base = corrida.emparejamientos.get("pipeline")
    if base is None:
        return {}
    salida = {}
    for nombre, por_modo in corrida.emparejamientos.items():
        if nombre in ("pipeline", "crudo"):
            continue
        salida[nombre] = {}
        for modo in MODOS:
            a = [e.conteos for e in base[modo]]
            b = [e.conteos for e in por_modo[modo]]
            ic = bootstrap_diferencia(a, b, n=replicas, semilla=SEMILLA_BOOTSTRAP)
            salida[nombre][modo] = {
                "diferencia_f1": _redondear(micro_prf(b).f1 - micro_prf(a).f1),
                "ic95": {
                    k: [_redondear(v.inferior), _redondear(v.superior)]
                    for k, v in ic._asdict().items()
                },
            }
    return salida


def _mencion(mencion: Any) -> dict[str, Any]:
    return {"inicio": mencion.inicio, "fin": mencion.fin, "texto": mencion.texto}


def detalle_de(corrida: Corrida, variantes: Iterable[str] = VARIANTES_DETALLADAS) -> dict:
    """TP/FP/FN por documento de `variantes`, con spans y texto (contiene el corpus)."""
    detalle: dict[str, Any] = {}
    for nombre in variantes:
        if nombre not in corrida.emparejamientos:
            continue
        detalle[nombre] = {
            modo: [
                {
                    "documento": documento,
                    "verdaderos": [
                        {"gold": _mencion(g), "prediccion": _mencion(p)} for g, p in e.verdaderos
                    ],
                    "falsos_positivos": [_mencion(p) for p in e.falsos_positivos],
                    "falsos_negativos": [_mencion(g) for g in e.falsos_negativos],
                }
                for documento, e in zip(corrida.documentos, por_modo, strict=True)
            ]
            for modo, por_modo in corrida.emparejamientos[nombre].items()
        }
    return detalle


# --- Tiempos -------------------------------------------------------------------------


def _resumen(valores: Sequence[float], decimales: int) -> dict[str, float]:
    """Mediana, p95 y máximo de `valores` (p95 con interpolación lineal)."""
    if not valores:
        return {"mediana": 0.0, "p95": 0.0, "maximo": 0.0}
    ordenados = sorted(valores)
    posicion = 0.95 * (len(ordenados) - 1)
    abajo = int(posicion)
    arriba = min(abajo + 1, len(ordenados) - 1)
    p95 = ordenados[abajo] + (ordenados[arriba] - ordenados[abajo]) * (posicion - abajo)
    return {
        "mediana": round(statistics.median(ordenados), decimales),
        "p95": round(p95, decimales),
        "maximo": round(ordenados[-1], decimales),
    }


def memoria_pico_mib() -> float:
    """Memoria residente pico del proceso en MiB (`ru_maxrss` viene en KiB en Linux)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def medir_tiempos(documentos: Sequence[Documento], extractor: Extractor) -> dict[str, Any]:
    """Tiempo por documento del pipeline completo con `extractor` (que debe ir sin caché).

    Primero calienta el modelo con `TEXTO_CALENTAMIENTO`, para que su carga no cuente.
    Luego corre `analizar` documento por documento y mide cada uno por separado.
    """
    extractor.extraer([TEXTO_CALENTAMIENTO])
    segundos = []
    for documento in documentos:
        inicio = time.perf_counter()
        _analizar(documento.texto, extractor=extractor)
        segundos.append(time.perf_counter() - inicio)
    caracteres = [len(d.texto) for d in documentos]
    total_segundos = sum(segundos)
    total_caracteres = sum(caracteres)
    return {
        "documentos": len(documentos),
        "segundos_por_documento": _resumen(segundos, 3),
        "caracteres_por_documento": _resumen(caracteres, 0),
        "segundos_totales": round(total_segundos, 2),
        "ms_por_1000_caracteres": (
            round(1000 * 1000 * total_segundos / total_caracteres, 1) if total_caracteres else 0.0
        ),
    }


# --- Entorno -------------------------------------------------------------------------


def _commit() -> dict[str, Any]:
    """Commit del repo y si hay cambios sin commitear; `None` si no se puede saber."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        cambios = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "cambios_sin_commitear": None}
    return {"commit": commit, "cambios_sin_commitear": bool(cambios)}


def _cpu() -> dict[str, Any]:
    """Nombre del procesador (de `/proc/cpuinfo` en Linux) y número de núcleos lógicos."""
    nombre = platform.processor() or None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("model name"):
                    nombre = linea.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"nombre": nombre, "nucleos_logicos": os.cpu_count(), "maquina": platform.machine()}


def entorno() -> dict[str, Any]:
    """Versiones del modelo, las librerías y Python, el commit y la CPU."""
    return {
        "modelo": NOMBRE_MODELO,
        "stride_tokens": STRIDE_TOKENS,
        "transformers": version_de("transformers"),
        "torch": version_de("torch"),
        "python": platform.python_version(),
        **_commit(),
        "cpu": _cpu(),
    }


# --- Evaluación completa -------------------------------------------------------------


def evaluar(
    documentos: Sequence[Documento],
    extractor: Extractor,
    particion: Particion,
    variantes: Iterable[str] | None = None,
    extractor_tiempos: Extractor | None = None,
    replicas: int = REPLICAS_BOOTSTRAP,
    limite: int | None = None,
    cache: ExtractorConCache | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Corre la evaluación y devuelve `(resultados, detalle)`.

    `resultados` es el JSON versionado (sin texto del corpus). `detalle` trae los TP/FP/FN
    de `VARIANTES_DETALLADAS` con texto, para guardar fuera del repo.

    `extractor` es el que usan las variantes (normalmente con caché). Si se pasa
    `extractor_tiempos` (sin caché), se miden los tiempos del pipeline completo con él;
    si no, `resultados["tiempos"]` es `None`. `cache`, si se pasa, es el
    `ExtractorConCache` cuyos contadores se reportan.
    """
    nombres = list(variantes) if variantes is not None else list(VARIANTES_POR_PARTICION[particion])
    corrida = correr(documentos, extractor, nombres)

    tiempos = None
    if extractor_tiempos is not None:
        tiempos = medir_tiempos(documentos, extractor_tiempos)

    resultados = {
        "particion": particion,
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limite": limite,
        "documentos": len(documentos),
        "menciones_gold": sum(len(d.menciones) for d in documentos),
        "bootstrap": {"replicas": replicas, "semilla": SEMILLA_BOOTSTRAP, "nivel": 0.95},
        "entorno": entorno(),
        "variantes": {
            nombre: {
                modo: metricas_de(corrida.emparejamientos[nombre][modo], replicas) for modo in MODOS
            }
            for nombre in nombres
        },
        "contra_pipeline": contra_pipeline(corrida, replicas),
        "tiempos": tiempos,
        "segundos_variantes": round(corrida.segundos, 2),
        "cache": (
            None
            if cache is None
            else {"aciertos": cache.aciertos, "fallos": cache.fallos, "llamadas": cache.llamadas}
        ),
        "memoria_pico_mib": memoria_pico_mib(),
    }
    detalle = {"particion": particion, "variantes": detalle_de(corrida)}
    return resultados, detalle


def ruta_resultados(salida: Path | None, particion: Particion) -> Path:
    """Archivo de resultados: `salida` si es un `.json`; si es carpeta, el nombre estándar."""
    nombre = f"distemist_{particion}.json"
    if salida is None:
        return CARPETA_RESULTADOS / nombre
    return salida if salida.suffix == ".json" else salida / nombre


def ruta_detalle(detalle: Path | None, particion: Particion) -> Path:
    """Archivo de TP/FP/FN con texto: por defecto en `data/evaluacion/`."""
    nombre = f"distemist_{particion}_errores.json"
    if detalle is None:
        return CARPETA_DETALLE / nombre
    return detalle if detalle.suffix == ".json" else detalle / nombre


def guardar_json(datos: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
        f.write("\n")


def tabla(resultados: dict[str, Any]) -> str:
    """Resumen en Markdown: P/R/F1 estrictos y laxos de cada variante."""
    lineas = [
        "| Variante | P estricta | R estricta | F1 estricta | P laxa | R laxa | F1 laxa |",
        "|---|---|---|---|---|---|---|",
    ]
    for nombre, por_modo in resultados["variantes"].items():
        celdas = [
            f"{por_modo[modo][m]:.3f}" for modo in MODOS for m in ("precision", "recall", "f1")
        ]
        lineas.append(f"| {nombre} | " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


def resumen_tiempos(resultados: dict[str, Any]) -> str:
    tiempos = resultados["tiempos"]
    if tiempos is None:
        return "Tiempos: no medidos (--sin-tiempos)."
    s = tiempos["segundos_por_documento"]
    c = tiempos["caracteres_por_documento"]
    return (
        f"Tiempos del pipeline sin caché ({tiempos['documentos']} documentos): mediana "
        f"{s['mediana']} s, p95 {s['p95']} s, máximo {s['maximo']} s; caracteres: mediana "
        f"{c['mediana']:.0f}, p95 {c['p95']:.0f}, máximo {c['maximo']:.0f}; "
        f"{tiempos['ms_por_1000_caracteres']} ms por 1.000 caracteres. "
        f"Memoria pico: {resultados['memoria_pico_mib']} MiB."
    )
