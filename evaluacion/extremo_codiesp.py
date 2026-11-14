"""De punta a punta en CodiEsp desarrollo: reglas, normalizador e híbrido, por documento.

#167 comparó reglas y normalizador; #180 suma el híbrido (primero las reglas y, para lo
que no mapean, el normalizador).

Hasta #154 todo se midió sobre las menciones gold: el texto anotado, ya cortado. Aquí corre
el pipeline completo (`analizar()`, con el NER) sobre cada caso clínico de desarrollo, una
vez con las reglas y otra con el normalizador (`artefactos.cargar()`), y se compara qué
categorías de Charlson salen confirmadas en la nota contra las de sus códigos gold:

- **Gold por documento**: las categorías de Quan de los códigos de diagnóstico del caso
  (CodiEsp-D). Se compara con las **confirmadas** del pipeline, que son las que suman al
  índice. Es un supuesto: la norma de codificación española codifica las sospechas como
  confirmadas, así que dejar fuera las posibles puede costar recall, a los dos métodos
  por igual.
- **P, R y F1** micro sobre los pares (documento, categoría), con IC por bootstrap de
  documentos y la diferencia pareada normalizador − reglas.
- **Índice de Charlson** del documento (con la jerarquía leve/severa) contra el de las
  categorías gold: exacto, a ±1 y error absoluto medio.
- **Respaldo a las menciones** (#165): cuántas entidades recibieron su categoría de una
  mención y no de su nombre, y cuántas de esas categorías están en el gold. #154 no lo
  midió, porque decidía menciones gold sueltas.

Desarrollo ya se miró en #132, #133 y #154: las cifras son optimistas para los dos
métodos. Prueba no se toca. Nada se ajusta con este resultado. El JSON no tiene texto:
solo conteos (y los ids de los documentos, que son públicos, no van).

    uv run python -m evaluacion extremo
"""

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import distemist
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.metricas import bootstrap_diferencia, bootstrap_ic, micro_prf
from ner_clinic.charlson import quan
from ner_clinic.charlson.indice import calcular_desde_categorias
from ner_clinic.extraccion import Extractor
from ner_clinic.pipeline import analizar
from ner_clinic.tipos import CategoriaCharlson, Estado, Resultado

RUTA_RESULTADOS = Path("evaluacion/resultados/extremo_codiesp_desarrollo.json")
METODOS = ("reglas", "normalizador", "hibrido")
CATEGORIAS_DETALLE = (CategoriaCharlson.TUMOR_MALIGNO, CategoriaCharlson.TUMOR_METASTASICO)


def gold_de(documento: DocumentoCodificado) -> set[CategoriaCharlson]:
    """Las categorías de Charlson de los códigos de diagnóstico del caso."""
    categorias = (quan.categoria_de(c.upper()) for c in documento.diagnosticos)
    return {c for c in categorias if c is not None}


def confirmadas(resultado: Resultado) -> set[CategoriaCharlson]:
    return {
        e.categoria_charlson
        for e in resultado.entidades
        if e.estado is Estado.CONFIRMADA and e.categoria_charlson is not None
    }


def conteos(
    predichas: set[CategoriaCharlson],
    gold: set[CategoriaCharlson],
    categoria: CategoriaCharlson | None = None,
) -> tuple[int, int, int]:
    if categoria is not None:
        predichas, gold = predichas & {categoria}, gold & {categoria}
    return len(predichas & gold), len(predichas - gold), len(gold - predichas)


def _respaldo(resultado: Resultado, normalizador: Any) -> list[CategoriaCharlson]:
    """Las categorías que el normalizador dio por una mención y no por el nombre."""
    con_categoria = [e for e in resultado.entidades if e.mapeo == "normalizador"]
    nombres = normalizador.decidir([e.texto for e in con_categoria])
    return [e.categoria_charlson for e in con_categoria if nombres[e.texto] is None]


def correr(
    documentos: Sequence[DocumentoCodificado], extractor: Extractor, normalizador: Any
) -> dict[str, Any]:
    """Las categorías confirmadas de cada documento con cada método, y el respaldo."""
    por_metodo: dict[str, list[set[CategoriaCharlson]]] = {m: [] for m in METODOS}
    respaldo: list[tuple[CategoriaCharlson, bool]] = []
    for documento in documentos:
        reglas = analizar(documento.texto, extractor=extractor)
        normalizado = analizar(
            documento.texto,
            extractor=extractor,
            normalizador=normalizador,
            modo_mapeo="normalizador",
        )
        hibrido = analizar(documento.texto, extractor=extractor, normalizador=normalizador)
        por_metodo["reglas"].append(confirmadas(reglas))
        por_metodo["normalizador"].append(confirmadas(normalizado))
        por_metodo["hibrido"].append(confirmadas(hibrido))
        gold = gold_de(documento)
        respaldo += [(c, c in gold) for c in _respaldo(normalizado, normalizador)]
    return {"predichas": por_metodo, "respaldo": respaldo}


def _prf(lista: Sequence[tuple[int, int, int]], replicas: int) -> dict[str, Any]:
    prf = micro_prf(lista)
    ic = bootstrap_ic(lista, n=replicas, semilla=distemist.SEMILLA_BOOTSTRAP)
    vp, fp, fn = (sum(c[i] for c in lista) for i in range(3))
    return {
        **{k: round(v, 4) for k, v in prf._asdict().items()},
        "ic95": {k: [round(x, 4) for x in v] for k, v in ic._asdict().items()},
        "vp": vp,
        "fp": fp,
        "fn": fn,
    }


def _indice(predichas: Sequence[set], gold: Sequence[set]) -> dict[str, Any]:
    diferencias = [
        calcular_desde_categorias(p) - calcular_desde_categorias(g)
        for p, g in zip(predichas, gold, strict=True)
    ]
    n = len(diferencias)
    return {
        "exacto": round(sum(d == 0 for d in diferencias) / n, 4),
        "a_1": round(sum(abs(d) <= 1 for d in diferencias) / n, 4),
        "error_absoluto_medio": round(sum(abs(d) for d in diferencias) / n, 3),
        "sesgo_medio": round(sum(diferencias) / n, 3),
    }


def evaluar(
    documentos: Sequence[DocumentoCodificado],
    extractor: Extractor,
    normalizador: Any,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    meta_normalizador: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corrida = correr(documentos, extractor, normalizador)
    gold = [gold_de(d) for d in documentos]
    metodos: dict[str, Any] = {}
    por_doc: dict[str, list[tuple[int, int, int]]] = {}
    for metodo in METODOS:
        predichas = corrida["predichas"][metodo]
        por_doc[metodo] = [conteos(p, g) for p, g in zip(predichas, gold, strict=True)]
        metodos[metodo] = {
            "micro": _prf(por_doc[metodo], replicas),
            "por_categoria": {
                c.value: _prf(
                    [conteos(p, g, c) for p, g in zip(predichas, gold, strict=True)], replicas
                )
                for c in CategoriaCharlson
            },
            "indice": _indice(predichas, gold),
        }
    diferencia, diferencia_hibrido = (
        bootstrap_diferencia(
            por_doc["reglas"], por_doc[m], n=replicas, semilla=distemist.SEMILLA_BOOTSTRAP
        )
        for m in ("normalizador", "hibrido")
    )
    respaldo = corrida["respaldo"]
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": distemist.entorno(),
        "normalizador": meta_normalizador,
        "particion": "CodiEsp desarrollo (ya mirada en #132, #133 y #154: optimista)",
        "documentos": len(documentos),
        "documentos_con_categoria_gold": sum(bool(g) for g in gold),
        "pares_gold": sum(len(g) for g in gold),
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "metodos": metodos,
        "diferencia_normalizador_menos_reglas": {
            k: [round(x, 4) for x in v] for k, v in diferencia._asdict().items()
        },
        "delta_f1": round(
            metodos["normalizador"]["micro"]["f1"] - metodos["reglas"]["micro"]["f1"], 4
        ),
        "diferencia_hibrido_menos_reglas": {
            k: [round(x, 4) for x in v] for k, v in diferencia_hibrido._asdict().items()
        },
        "delta_f1_hibrido": round(
            metodos["hibrido"]["micro"]["f1"] - metodos["reglas"]["micro"]["f1"], 4
        ),
        "respaldo_menciones": {
            "entidades": len(respaldo),
            "en_el_gold": sum(acierto for _, acierto in respaldo),
            "por_categoria": dict(Counter(c.value for c, _ in respaldo)),
        },
    }


def tabla(resultados: dict[str, Any]) -> str:
    lineas = [
        "| Método | P | R | F1 [IC 95 %] | Índice exacto | Índice ±1 | Error medio del índice |",
        "|---|---|---|---|---|---|---|",
    ]
    for metodo in METODOS:
        m = resultados["metodos"][metodo]
        micro, indice = m["micro"], m["indice"]
        ic = micro["ic95"]["f1"]
        lineas.append(
            f"| {metodo} | {micro['precision']:.3f} | {micro['recall']:.3f} | "
            f"{micro['f1']:.3f} [{ic[0]:.3f}; {ic[1]:.3f}] | {indice['exacto']:.3f} | "
            f"{indice['a_1']:.3f} | {indice['error_absoluto_medio']:.2f} |"
        )
    lineas.append("")
    for nombre, clave, delta in (
        ("normalizador", "diferencia_normalizador_menos_reglas", "delta_f1"),
        ("híbrido", "diferencia_hibrido_menos_reglas", "delta_f1_hibrido"),
    ):
        d = resultados[clave]["f1"]
        lineas.append(
            f"Δ F1 {nombre} − reglas: {resultados[delta]:+.3f} [{d[0]:+.3f}; {d[1]:+.3f}]"
        )
    return "\n".join(lineas)
