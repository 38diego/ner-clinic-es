"""Normalización por embeddings en CodiEsp desarrollo, frente a las reglas (#131).

Mide cada modelo de `normalizador.CANDIDATOS` como `charlson_codiesp` mide las reglas:
sobre el texto de cada mención de diagnóstico del gold de CodiEsp-X, en entrenamiento y
desarrollo, con las variantes de la prueba de estrés. Las métricas de categoría son las
mismas (`charlson_codiesp.metricas`), así que se comparan directo con
`codiesp_reglas_desarrollo.json`.

**Umbral.** Se elige en **desarrollo**, con la variante original: el que da el mayor F1
micro de categoría en la grilla `UMBRALES` (a igual F1, el mayor umbral). Con ese umbral se
reporta todo. Entrenamiento no se usa para elegirlo, así que su cifra es la más limpia de
las dos; la de desarrollo es optimista por el ajuste de un parámetro. La curva completa
queda en el JSON.

**Código CIE-10** (`codigo`), sin umbral y solo con la variante original: si el primer
candidato (`top1`) o alguno de los cinco primeros (`top5`) coincide con el gold en sus 3
primeros caracteres, sobre todas las menciones de diagnóstico y sobre las que tienen
categoría de Charlson. `categoria_top5` es la fracción de menciones gold con categoría en
las que algún código de los cinco primeros tiene esa categoría (con
`Normalizador.codigo_para_categoria`, como el top-1).

**Costo**: tamaño del modelo en disco, memoria máxima del proceso (modelo, catálogo e
índice), tiempo de construir el índice, y tiempo por mención: de una en una (latencia,
mediana y p95) y en lotes.

Cada modelo se corre en su propio proceso para que la memoria sea la suya:

    uv run python -m evaluacion normalizacion --modelo e5_small
        [--salida evaluacion/resultados/normalizacion_e5_small_desarrollo.json]
"""

import resource
import statistics
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist, robustez
from evaluacion.cache import version_de
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.metricas import micro_prf
from evaluacion.normalizador import CANDIDATOS, Normalizador
from ner_clinic.charlson import quan

CARPETA_RESULTADOS = Path("evaluacion/resultados")
UMBRALES = tuple(round(u / 100, 2) for u in range(0, 101))
MENCIONES_LATENCIA = 200


def ruta_resultados(modelo: str) -> Path:
    return CARPETA_RESULTADOS / f"normalizacion_{modelo}_desarrollo.json"


def textos_de(documentos: Sequence[DocumentoCodificado], variantes: Sequence[str]) -> list[str]:
    """El texto de cada mención de diagnóstico en cada variante, para precalcularlos."""
    textos = []
    for variante in variantes:
        for documento in documentos:
            nota = robustez.variar(
                documento.texto, variante, charlson_codiesp.SEMILLA_ERRORES, documento.id
            )
            textos.extend(
                charlson_codiesp.texto_en(nota, m) for m in documento.menciones_diagnostico
            )
    return textos


def curva_umbral(
    pliegues: Iterable[tuple[Normalizador, Sequence[DocumentoCodificado]]],
) -> list[dict[str, float]]:
    """P/R/F1 micro de categoría (sin IC) en cada umbral de `UMBRALES`, variante original.

    Cada pliegue son unos documentos y el normalizador con que se miden; los conteos de
    todos se suman. Sin validación cruzada es un solo pliegue. Se recorren una vez, así que
    pueden venir de un generador que arma el índice de cada pliegue cuando le toca.
    """
    ids: list[str] = []
    por_umbral: dict[float, list[charlson_codiesp.Caso]] = {u: [] for u in UMBRALES}
    for normalizador, documentos in pliegues:
        ids.extend(d.id for d in documentos)
        normalizador.precalcular(textos_de(documentos, ["original"]))
        for umbral in UMBRALES:
            por_umbral[umbral].extend(
                charlson_codiesp.casos(documentos, normalizador.mapeador(umbral))
            )
    curva = []
    for umbral in UMBRALES:
        prf = micro_prf(charlson_codiesp._por_documento(por_umbral[umbral], ids))
        curva.append(
            {
                "umbral": umbral,
                "precision": round(prf.precision, 4),
                "recall": round(prf.recall, 4),
                "f1": round(prf.f1, 4),
            }
        )
    return curva


def elegir_umbral(curva: Sequence[dict[str, float]]) -> float:
    """El umbral de mayor F1; a igual F1, el mayor (el que deja menos códigos dudosos)."""
    return max(curva, key=lambda p: (p["f1"], p["umbral"]))["umbral"]


def metricas_codigo(
    normalizador: Normalizador, documentos: Sequence[DocumentoCodificado]
) -> dict[str, Any]:
    """Acierto del código a 3 caracteres (top-1 y top-5) y de la categoría en el top-5."""
    todas = charlson = 0
    top1 = top5 = top1_ch = top5_ch = categoria5 = 0
    for documento in documentos:
        for m in documento.menciones_diagnostico:
            candidatos = normalizador.candidatos(charlson_codiesp.texto_en(documento.texto, m))
            prefijos = [c.codigo[:3] for c in candidatos]
            acierto1 = prefijos[0] == m.codigo[:3]
            acierto5 = m.codigo[:3] in prefijos
            todas += 1
            top1 += acierto1
            top5 += acierto5
            gold = quan.categoria_de(m.codigo)
            if gold is not None:
                charlson += 1
                top1_ch += acierto1
                top5_ch += acierto5
                categoria5 += any(
                    quan.categoria_de(normalizador.codigo_para_categoria(c.codigo)) is gold
                    for c in candidatos
                )

    def fraccion(n: int, total: int) -> float:
        return round(n / total, 4) if total else 0.0

    return {
        "menciones": todas,
        "top1": fraccion(top1, todas),
        "top5": fraccion(top5, todas),
        "menciones_charlson": charlson,
        "top1_charlson": fraccion(top1_ch, charlson),
        "top5_charlson": fraccion(top5_ch, charlson),
        "categoria_top5": fraccion(categoria5, charlson),
    }


def latencia(normalizador: Normalizador, textos: Sequence[str]) -> dict[str, float]:
    """Milisegundos por mención de una en una (sin memoria) y en un lote."""
    unicos = list(dict.fromkeys(textos))[:MENCIONES_LATENCIA]
    tiempos = []
    for texto in unicos:
        inicio = time.perf_counter()
        normalizador.indice.buscar(normalizador.codificador([texto]), normalizador.k)
        tiempos.append(1000 * (time.perf_counter() - inicio))
    tiempos.sort()
    inicio = time.perf_counter()
    normalizador.indice.buscar(normalizador.codificador(unicos), normalizador.k)
    en_lote = 1000 * (time.perf_counter() - inicio) / len(unicos) if unicos else 0.0
    return {
        "menciones": len(unicos),
        "ms_mediana": round(statistics.median(tiempos), 1) if tiempos else 0.0,
        "ms_p95": round(tiempos[int(0.95 * (len(tiempos) - 1))], 1) if tiempos else 0.0,
        "ms_en_lote": round(en_lote, 2),
    }


def memoria_maxima_mib() -> float:
    """Memoria residente máxima del proceso (Linux: `ru_maxrss` en KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def evaluar(
    nombre_modelo: str,
    normalizador: Normalizador,
    por_particion: dict[str, Sequence[DocumentoCodificado]],
    meta_indice: dict[str, Any],
    bytes_modelo: int | None = None,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    variantes: Sequence[str] = robustez.VARIANTES_TEXTO,
) -> dict[str, Any]:
    """El JSON de resultados de un modelo, sin texto del corpus."""
    charlson_codiesp._validar_particiones(list(por_particion))
    if "dev" not in por_particion:
        raise ValueError("El umbral se elige en 'dev': hace falta esa partición.")
    grupos = dict(por_particion)
    if len(por_particion) > 1:
        grupos["+".join(por_particion)] = [d for docs in por_particion.values() for d in docs]
    conjunto = list(grupos)[-1]
    todos = grupos[conjunto]

    inicio = time.perf_counter()
    normalizador.precalcular(textos_de(todos, variantes))
    segundos_menciones = round(time.perf_counter() - inicio, 1)

    curva = curva_umbral([(normalizador, por_particion["dev"])])
    umbral = elegir_umbral(curva)
    mapeador = normalizador.mapeador(umbral)

    por_variante: dict[str, Any] = {}
    for variante in variantes:
        por_variante[variante] = {
            nombre: charlson_codiesp.metricas(
                charlson_codiesp.casos(docs, mapeador, variante),
                [d.id for d in docs],
                replicas,
                por_categoria=nombre == conjunto,
            )
            for nombre, docs in grupos.items()
        }

    modelo = CANDIDATOS[nombre_modelo]
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": modelo.nombre},
        "modelo": {
            "clave": nombre_modelo,
            "nombre": modelo.nombre,
            "revision": modelo.revision,
            "pooling": modelo.pooling,
            "prefijo": modelo.prefijo,
            "minusculas": modelo.minusculas,
            "tokenizador": list(modelo.tokenizador) if modelo.tokenizador else None,
            "licencia": modelo.licencia,
            "bytes": bytes_modelo,
            "transformers": version_de("transformers"),
        },
        "indice": meta_indice,
        "particiones": list(por_particion),
        "conjunto_principal": conjunto,
        "semilla_errores": charlson_codiesp.SEMILLA_ERRORES,
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": {nombre: len(docs) for nombre, docs in grupos.items()},
        "umbral": {"elegido": umbral, "particion": "dev", "curva": curva},
        "categoria": por_variante,
        "codigo": {nombre: metricas_codigo(normalizador, docs) for nombre, docs in grupos.items()},
        "costo": {
            "segundos_menciones": segundos_menciones,
            "textos_unicos": len(normalizador._memoria),
            "latencia": latencia(normalizador, textos_de(por_particion["dev"], ["original"])),
            "memoria_maxima_mib": memoria_maxima_mib(),
        },
    }


def tabla(resultados: Sequence[dict[str, Any]], reglas: dict[str, Any] | None = None) -> str:
    """Comparación en Markdown: F1 de categoría por variante, código y costo."""
    lineas = [
        "| Método | Umbral | F1 original | F1 minúsculas | F1 errores 10 % | P | R | "
        "Código top-1 / top-5 | ms por mención | Memoria MiB |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    if reglas is not None:
        conjunto = reglas["conjunto_principal"]
        r = reglas["metodos"]["reglas_corrector"]
        o = r["original"][conjunto]["micro"]
        lineas.append(
            f"| reglas + corrector | — | {o['f1']:.3f} | "
            f"{r['minusculas'][conjunto]['micro']['f1']:.3f} | "
            f"{r['errores_10'][conjunto]['micro']['f1']:.3f} | {o['precision']:.3f} | "
            f"{o['recall']:.3f} | — | — | — |"
        )
    for res in resultados:
        conjunto = res["conjunto_principal"]
        c = res["categoria"]
        o = c["original"][conjunto]["micro"]
        codigo = res["codigo"][conjunto]
        lineas.append(
            f"| {res['modelo']['clave']} | {res['umbral']['elegido']:.2f} | {o['f1']:.3f} | "
            f"{c['minusculas'][conjunto]['micro']['f1']:.3f} | "
            f"{c['errores_10'][conjunto]['micro']['f1']:.3f} | {o['precision']:.3f} | "
            f"{o['recall']:.3f} | {codigo['top1']:.3f} / {codigo['top5']:.3f} | "
            f"{res['costo']['latencia']['ms_mediana']} | {res['costo']['memoria_maxima_mib']} |"
        )
    return "\n".join(lineas)
