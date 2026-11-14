"""Medición final de la normalización en dos etapas en CodiEsp desarrollo (PRD
`normalizacion-f1`, FR6, #154).

**Qué se mide.** Lo elegido en entrenamiento (#151 y #153): el clasificador de #151 sobre el
top-20 de un índice con los sinónimos de CodiEsp entrenamiento más los de DisTEMIST y la
variante léxica de CANTEMIST (`FUENTES_ELEGIDAS`, la combinación `ambas`, declarada antes
de medir). Se compara con el híbrido de hoy (#133: primer candidato con umbral 0,69 y las
exclusiones de las reglas, solo con los sinónimos de CodiEsp), que da 0,885 en desarrollo.

**Una sola vez.** Desarrollo ya se miró en #132 y #133, así que su cifra es optimista. Con
esta corrida se mide una vez más y no se vuelve a elegir nada con ella. Prueba no se lee:
solo los nombres de sus archivos, para verificar el origen de los sinónimos.

**El modelo final** (`entrenar_final`) se entrena con los 5 pliegues de entrenamiento, cada
uno buscado en un índice sin sus propios sinónimos (`dos_etapas.pliegues_topk`, también sin
las menciones externas de sus documentos). La penalización y el umbral se eligen por
validación cruzada sobre esos 5 pliegues, como `final` en #151. En desarrollo el índice
tiene todos los sinónimos de entrenamiento y de las fuentes, sin ninguno de un documento de
desarrollo ni de prueba (`verificar_origen`).

**Qué se reporta** (`evaluar`), todo en desarrollo:

- P/R/F1 de categoría, micro y por categoría, con IC por bootstrap por documento.
- La diferencia pareada contra el híbrido (micro, por categoría y en cada variante).
- Las variantes de la prueba de estrés (`robustez.VARIANTES_TEXTO`) y su caída.
- El F1 en las menciones no vistas: las que no están tal cual entre los sinónimos de CodiEsp
  entrenamiento (como en #132), y las que tampoco están entre los de las fuentes.
- Cuántas de las menciones que pierde el híbrido (52 en #149) recupera, por causa.

Además, como contexto y no como elección, los dos pasos por separado (el híbrido con las
fuentes y el clasificador sin ellas), y la validación cruzada anidada de entrenamiento del
elegido contra el híbrido (la de #153, en una sola corrida).

    uv run python -m evaluacion final [--modelo sapbert]
"""

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist, fuentes_sinonimos, robustez
from evaluacion.cache import version_de
from evaluacion.clasificador import (
    Datos,
    Modelo,
    ReglasDe,
    _juntar,
    decisor,
    elegir,
    entrenar,
)
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.dos_etapas import (
    UMBRAL_HIBRIDO,
    K,
    Pliegue,
    comparar,
    errores,
    mapeador_de,
    pliegues_topk,
    primer_candidato,
    propuestas,
)
from evaluacion.normalizacion_codiesp import textos_de
from evaluacion.normalizador import CANDIDATOS, Codificador, Indice, Mapeador, Normalizador
from evaluacion.sinonimos_codiesp import (
    PLIEGUES,
    normalizar,
    sinonimos_de,
    verificar_origen,
)
from evaluacion.sinonimos_publicos import MencionExterna, sinonimos_externos
from evaluacion.votos import recuperadas
from ner_clinic.normalizacion.decision import modelo_a_json, modelo_de_json  # noqa: F401
from ner_clinic.tipos import CategoriaCharlson

RUTA_RESULTADOS = Path("evaluacion/resultados/normalizacion_final_desarrollo.json")
FUENTES_ELEGIDAS = fuentes_sinonimos.CONFIGURACIONES["ambas"]
REFERENCIA = "hibrido_actual"
ELEGIDO = "clasificador_fuentes"
# Nombre → (con las fuentes, segunda etapa). Los dos últimos son solo contexto.
METODOS = {
    REFERENCIA: (False, "hibrido"),
    ELEGIDO: (True, "clasificador"),
    "hibrido_fuentes": (True, "hibrido"),
    "clasificador_sin_fuentes": (False, "clasificador"),
}
F1_HIBRIDO_133 = 0.885
CAIDA_PRECISION_MAXIMA = 0.01
META_ORIENTATIVA = 0.90
VARIANTE_ESTRES = "errores_10"


def externas_de(
    fuentes: dict[str, Sequence[MencionExterna]], usadas: Sequence[str] = FUENTES_ELEGIDAS
) -> list[MencionExterna]:
    return [m for fuente in usadas for m in fuentes[fuente]]


def entrenar_final(
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    externas: Sequence[MencionExterna],
    reglas: ReglasDe,
    pliegues: int = PLIEGUES,
    k: int = K,
) -> tuple[Modelo, dict[str, Any]]:
    """El clasificador entrenado con los `pliegues` de entrenamiento, con la penalización y
    el umbral elegidos por validación cruzada sobre ellos."""
    externos = pliegues_topk(indice, codificador, entrenamiento, pliegues, k, externas)
    datos = [Datos.de([p], reglas) for p in externos]
    penalizacion, umbral, f1, curva = elegir(datos)
    todos = _juntar(datos)
    modelo = entrenar(*todos.matriz(), penalizacion)
    return modelo, {
        "penalizacion": penalizacion,
        "umbral": umbral,
        "f1_validacion_cruzada_optimista": round(f1, 4),
        "f1_por_penalizacion": {str(c): round(v, 4) for c, v in curva.items()},
        "pares_entrenamiento": int(todos.matriz()[0].shape[0]),
    }


def topk_desarrollo(
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    desarrollo: Sequence[DocumentoCodificado],
    externas: Sequence[MencionExterna],
    ids_prueba: Sequence[str],
    variantes: Sequence[str] = robustez.VARIANTES_TEXTO,
    k: int = K,
) -> tuple[Pliegue, dict[str, Any]]:
    """El top-k de las menciones de desarrollo (en todas las `variantes`) en un índice con
    los sinónimos de entrenamiento y de `externas`, sin ninguno de desarrollo ni de prueba."""
    ids_dev = [d.id for d in desarrollo]
    propios = sinonimos_de(entrenamiento)
    externos = sinonimos_externos(externas, [*ids_dev, *ids_prueba], (s.texto for s in propios))
    sinonimos = propios + externos
    verificacion = verificar_origen(sinonimos, {"dev": ids_dev, "test": ids_prueba})
    final = indice.con_sinonimos(sinonimos, codificador)
    normalizador = Normalizador(final, codificador, k)
    textos = textos_de(desarrollo, variantes)
    normalizador.precalcular(textos)
    pliegue = Pliegue(list(desarrollo), {t: propuestas(normalizador, t) for t in textos})
    return pliegue, {
        "filas_indice": len(final.codigos),
        "sinonimos_codiesp": len(propios),
        "sinonimos_externos": len(externos),
        "verificacion_origen": verificacion,
        "textos_sinonimos": {s.texto.casefold() for s in sinonimos},
        "textos_codiesp": {s.texto.casefold() for s in propios},
    }


def _medir(
    mapeador: Mapeador,
    desarrollo: Sequence[DocumentoCodificado],
    variantes: Sequence[str],
    replicas: int,
    vistas: dict[str, set[str]],
) -> tuple[dict[str, Any], dict[str, list[charlson_codiesp.Caso]]]:
    """Categoría en cada variante y en las menciones no vistas; y los casos por variante."""
    ids = [d.id for d in desarrollo]
    por_variante = {v: charlson_codiesp.casos(desarrollo, mapeador, v) for v in variantes}
    originales = por_variante["original"]
    medicion = {
        "categoria": {
            v: charlson_codiesp.metricas(lista, ids, replicas, por_categoria=v == "original")
            for v, lista in por_variante.items()
        },
        "no_vistas": {
            nombre: charlson_codiesp.metricas(
                [c for c in originales if normalizar(c.texto).casefold() not in textos],
                ids,
                replicas,
                por_categoria=False,
            )
            for nombre, textos in vistas.items()
        },
    }
    return medicion, por_variante


def _caida(medicion: dict[str, Any], variante: str = VARIANTE_ESTRES) -> float:
    categoria = medicion["categoria"]
    return round(categoria["original"]["micro"]["f1"] - categoria[variante]["micro"]["f1"], 4)


def metas(resultados: dict[str, Any]) -> list[dict[str, Any]]:
    """Las metas del epic #148 que se pueden juzgar con esta corrida (el costo va aparte)."""
    dev = resultados["desarrollo"]
    elegido = dev["metodos"][ELEGIDO]["categoria"]["original"]["micro"]
    referencia = dev["metodos"][REFERENCIA]["categoria"]["original"]["micro"]
    cv = resultados["validacion_cruzada"]["comparacion"]["micro"]["f1"]
    top5 = dev["recuperadas"]
    caida_elegido = dev["metodos"][ELEGIDO]["caida_" + VARIANTE_ESTRES]
    caida_referencia = dev["metodos"][REFERENCIA]["caida_" + VARIANTE_ESTRES]
    return [
        {
            "meta": "Validación cruzada: el IC de la diferencia de F1 contra el híbrido no "
            "incluye el cero",
            "valor": f"{cv['diferencia']:+.3f} [{cv['ic95'][0]:+.3f}; {cv['ic95'][1]:+.3f}]",
            "cumple": cv["ic95"][0] > 0,
        },
        {
            "meta": f"Desarrollo: F1 mayor que {F1_HIBRIDO_133}",
            "valor": f"{elegido['f1']:.3f} (híbrido {referencia['f1']:.3f})",
            "cumple": elegido["f1"] > F1_HIBRIDO_133,
        },
        {
            "meta": f"Desarrollo: la precisión cae {CAIDA_PRECISION_MAXIMA} como máximo",
            "valor": f"{elegido['precision'] - referencia['precision']:+.3f}",
            "cumple": elegido["precision"] >= referencia["precision"] - CAIDA_PRECISION_MAXIMA,
        },
        {
            "meta": f"Desarrollo: meta orientativa de F1 {META_ORIENTATIVA}",
            "valor": f"{elegido['f1']:.3f}",
            "cumple": elegido["f1"] >= META_ORIENTATIVA,
        },
        {
            "meta": "Desarrollo: recupera al menos la mitad de las perdidas del top-5",
            "valor": f"{top5['recuperadas']['top5']} de {top5['perdidas_base']['top5']}",
            "cumple": 2 * top5["recuperadas"]["top5"] >= top5["perdidas_base"]["top5"],
        },
        {
            "meta": "Errores al 10 %: no cae más que el híbrido",
            "valor": f"cae {caida_elegido:.3f} (híbrido {caida_referencia:.3f})",
            "cumple": caida_elegido <= caida_referencia,
        },
    ]


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    desarrollo: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    fuentes: dict[str, Sequence[MencionExterna]],
    reporte_fuentes: dict[str, Any],
    ids_prueba: Sequence[str],
    perdidas_arnes: Sequence[dict[str, Any]] | None = None,
    pliegues: int = PLIEGUES,
    k: int = K,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    variantes: Sequence[str] = robustez.VARIANTES_TEXTO,
) -> dict[str, Any]:
    """El JSON de #154, sin texto de ningún corpus (ver el docstring del módulo).

    `perdidas_arnes` es el detalle de las menciones que el híbrido pierde en desarrollo
    según #149 (documento y posición), para comprobar que aquí se reproducen.
    """
    if VARIANTE_ESTRES not in variantes or "original" not in variantes:
        raise ValueError(f"Hacen falta las variantes 'original' y {VARIANTE_ESTRES!r}.")
    reglas: ReglasDe = {}
    externas = externas_de(fuentes)
    ids_dev = [d.id for d in desarrollo]

    # 1. Validación cruzada anidada en entrenamiento: el elegido contra el híbrido
    inicio = time.perf_counter()
    ids_cv, _, casos_base, _ = fuentes_sinonimos.medir(
        indice, codificador, entrenamiento, (), reglas, pliegues, k
    )
    _, _, casos_fuentes, elegidas = fuentes_sinonimos.medir(
        indice, codificador, entrenamiento, externas, reglas, pliegues, k
    )
    hibrido_cv, elegido_cv = casos_base["hibrido_exclusiones"], casos_fuentes["clasificador"]
    validacion = {
        "metodos": {
            REFERENCIA: charlson_codiesp.metricas(hibrido_cv, ids_cv, replicas, False)["micro"],
            ELEGIDO: charlson_codiesp.metricas(elegido_cv, ids_cv, replicas, False)["micro"],
        },
        "comparacion": {
            "micro": comparar(hibrido_cv, elegido_cv, ids_cv, replicas),
            **{
                c.value: comparar(hibrido_cv, elegido_cv, ids_cv, replicas, c)
                for c in CategoriaCharlson
            },
        },
        "elegidas_por_pliegue": elegidas,
        "segundos": round(time.perf_counter() - inicio, 1),
    }

    # 2. Los clasificadores finales, con los 5 pliegues
    finales = {}
    for con_fuentes in (True, False):
        modelo, eleccion = entrenar_final(
            indice, codificador, entrenamiento, externas if con_fuentes else (), reglas,
            pliegues, k,
        )  # fmt: skip
        finales[con_fuentes] = (modelo, eleccion)

    # 3. Desarrollo, una vez
    tops = {}
    for con_fuentes in (False, True):
        tops[con_fuentes] = topk_desarrollo(
            indice, codificador, entrenamiento, desarrollo,
            externas if con_fuentes else (), ids_prueba, variantes, k,
        )  # fmt: skip
    vistas = {
        "sinonimos_codiesp": tops[False][1]["textos_codiesp"],
        "sinonimos_codiesp_y_fuentes": tops[True][1]["textos_sinonimos"],
    }
    metodos: dict[str, Any] = {}
    casos_dev: dict[str, dict[str, list[charlson_codiesp.Caso]]] = {}
    segundos_segunda: dict[str, float] = {}
    for nombre, (con_fuentes, etapa) in METODOS.items():
        pliegue = tops[con_fuentes][0]
        if etapa == "hibrido":
            decidir = primer_candidato(UMBRAL_HIBRIDO)
        else:
            modelo, eleccion = finales[con_fuentes]
            decidir = decisor(modelo, eleccion["umbral"], reglas)
        mapeador = mapeador_de(decidir, pliegue.topk)
        inicio = time.perf_counter()
        charlson_codiesp.casos(desarrollo, mapeador)
        segundos_segunda[nombre] = time.perf_counter() - inicio
        medicion, casos_dev[nombre] = _medir(mapeador, desarrollo, variantes, replicas, vistas)
        medicion["caida_" + VARIANTE_ESTRES] = _caida(medicion)
        metodos[nombre] = medicion

    referencia = casos_dev[REFERENCIA]
    comparaciones = {}
    for nombre in METODOS:
        if nombre == REFERENCIA:
            continue
        otro = casos_dev[nombre]
        comparaciones[nombre] = {
            "micro": comparar(referencia["original"], otro["original"], ids_dev, replicas),
            "variantes": {
                v: comparar(referencia[v], otro[v], ids_dev, replicas)
                for v in variantes
                if v != "original"
            },
        }
        if nombre == ELEGIDO:
            comparaciones[nombre]["por_categoria"] = {
                c.value: comparar(referencia["original"], otro["original"], ids_dev, replicas, c)
                for c in CategoriaCharlson
            }

    base_dev = tops[False][0]
    perdidas = errores([base_dev], referencia["original"], UMBRAL_HIBRIDO)
    posiciones = {(p["documento"], p["inicio"], p["fin"]) for p in perdidas.pop("detalle")}
    reproduce = None
    if perdidas_arnes is not None:
        reproduce = posiciones == {(p["documento"], p["inicio"], p["fin"]) for p in perdidas_arnes}
    rescate = recuperadas([base_dev], referencia["original"], casos_dev[ELEGIDO]["original"])

    menciones = sum(len(d.menciones_diagnostico) for d in desarrollo)
    modelo_elegido, eleccion_elegido = finales[True]
    modelo_embeddings = CANDIDATOS[nombre_modelo]
    resultados = {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": modelo_embeddings.nombre},
        "modelo": {
            "clave": nombre_modelo,
            "nombre": modelo_embeddings.nombre,
            "revision": modelo_embeddings.revision,
            "licencia": modelo_embeddings.licencia,
            "transformers": version_de("transformers"),
        },
        "indice": meta_indice,
        "particiones": {
            "eleccion_y_entrenamiento": "train (validación cruzada)",
            "medicion": "dev, una sola vez (ya mirada en #132 y #133: cifra optimista)",
            "prueba": "no se lee; solo los nombres de archivo",
        },
        "eleccion": {
            "segunda_etapa": "clasificador (#151)",
            "fuentes": list(FUENTES_ELEGIDAS),
            "declarada_en": "#153, antes de medir",
        },
        "pliegues": pliegues,
        "k": k,
        "umbral_hibrido": UMBRAL_HIBRIDO,
        "semilla_errores": charlson_codiesp.SEMILLA_ERRORES,
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": {"train": len(entrenamiento), "dev": len(desarrollo)},
        "menciones_dev": menciones,
        "fuentes": reporte_fuentes,
        "indice_dev": {
            ("con_fuentes" if con_fuentes else "sin_fuentes"): {
                clave: valor for clave, valor in info.items() if not clave.startswith("textos_")
            }
            for con_fuentes, (_, info) in tops.items()
        },
        "no_vistas": {nombre: len(textos) for nombre, textos in vistas.items()},
        "validacion_cruzada": validacion,
        "clasificador_final": {
            ELEGIDO: {
                **eleccion_elegido,
                **modelo_a_json(modelo_elegido, eleccion_elegido["umbral"]),
                "pesos_por_atributo": dict(
                    sorted(modelo_elegido.pesos_por_atributo().items(), key=lambda x: -abs(x[1]))
                ),
            },
            "clasificador_sin_fuentes": finales[False][1],
        },
        "desarrollo": {
            "metodos": metodos,
            "comparaciones": comparaciones,
            "perdidas_hibrido": {**perdidas, "reproduce_149": reproduce},
            "recuperadas": rescate,
        },
        # Sin codificar ni buscar (eso es la primera etapa, igual para todos)
        "costo_segunda_etapa": {
            nombre: {
                "milisegundos_por_mencion": round(1000 * s / max(menciones, 1), 3),
                "milisegundos_por_documento": round(1000 * s / len(ids_dev), 3),
            }
            for nombre, s in segundos_segunda.items()
        },
    }
    resultados["metas"] = metas(resultados)
    return resultados


def _f1(m: dict[str, Any]) -> str:
    return f"{m['f1']:.3f} [{m['ic95']['f1'][0]:.3f}; {m['ic95']['f1'][1]:.3f}]"


def _delta(d: dict[str, Any]) -> str:
    return f"{d['diferencia']:+.3f} [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]"


def tabla(resultados: dict[str, Any]) -> str:
    """Desarrollo: cada método contra el híbrido, el estrés, las no vistas y las metas."""
    dev = resultados["desarrollo"]
    lineas = [
        "| Método (dev) | P | R | F1 [IC 95 %] | Δ F1 vs híbrido [IC] | F1 errores 10 % | "
        "F1 no vistas |",
        "|---|---|---|---|---|---|---|",
    ]
    for nombre, m in dev["metodos"].items():
        o = m["categoria"]["original"]["micro"]
        delta = _delta(dev["comparaciones"][nombre]["micro"]["f1"]) if nombre != REFERENCIA else "—"
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {o['precision']:.3f} | {o['recall']:.3f} | "
            f"{_f1(o)} | {delta} | "
            f"{m['categoria'][VARIANTE_ESTRES]['micro']['f1']:.3f} | "
            f"{m['no_vistas']['sinonimos_codiesp']['micro']['f1']:.3f} |"
        )
    r = dev["recuperadas"]
    lineas += [
        "",
        f"Recuperadas de las {sum(r['perdidas_base'].values())} perdidas del híbrido: "
        f"{sum(r['recuperadas'].values())} ({r['recuperadas']}); aciertos perdidos: "
        f"{r['aciertos_base_perdidos']}.",
        "",
        "| Meta | Valor | Cumple |",
        "|---|---|---|",
    ]
    for meta in resultados["metas"]:
        lineas.append(f"| {meta['meta']} | {meta['valor']} | {'sí' if meta['cumple'] else 'no'} |")
    return "\n".join(lineas)
