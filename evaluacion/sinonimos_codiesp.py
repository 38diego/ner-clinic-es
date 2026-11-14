"""Sinónimos de CodiEsp entrenamiento en el índice de embeddings (PRD `normalizacion`, FR4b, #132).

Las menciones de diagnóstico de CodiEsp **entrenamiento**, con su código gold, se agregan
al índice del catálogo como sinónimos (`Indice.con_sinonimos`). Es la forma "sin regex" de
cubrir siglas y formas coloquiales: "ictus" no está en el catálogo, pero sí en las notas
de entrenamiento con su código I63.9.

**Sinónimos** (`sinonimos_de`). El texto de cada mención se lee de sus tramos, como las
consultas, y se normaliza: espacios colapsados. Se deduplica sin distinguir mayúsculas.
Si un mismo texto tiene varios códigos gold, queda el más frecuente (a igual frecuencia,
el menor); si tiene varias escrituras, la más frecuente.

**Sin fuga.** Desde aquí entrenamiento ya no sirve para medir: la cifra es la de
**desarrollo**. `verificar_origen` comprueba que ningún sinónimo sale de un documento de
desarrollo ni de prueba (de prueba solo se leen los nombres de los archivos).

**Umbral.** Se elige en entrenamiento, no en desarrollo, para que la cifra de desarrollo
no quede ajustada. Sin sinónimos, entrenamiento no está en el índice y se usa entero. Con
sinónimos, por validación cruzada de `PLIEGUES` pliegues por documento: cada pliegue se
mide con un índice que tiene solo los sinónimos de los otros. Los conteos de todos los
pliegues se suman y se elige el umbral como en #131 (`elegir_umbral`).

**Siglas** (`metricas_siglas`). Una mención gold es una sigla si su texto es una palabra
de 2 a 6 letras o dígitos que empieza en mayúscula y no tiene minúsculas ("VIH", "DM2",
"EPOC"). Se cuenta cuántas resuelve cada método (categoría de Charlson igual a la del
gold, incluido "sin categoría"), separando las de la lista de 30 siglas que el pipeline
sube a mayúscula para el modelo (`SIGLAS_PARA_EL_MODELO`, #114) de las demás.

**Híbridos** (#133), con el umbral del modelo con sinónimos:

- `hibrido_exclusiones`: después del modelo, la exclusión de la regla de la categoría que
  dio (`normalizador.con_exclusiones`). `exclusiones` cuenta, en entrenamiento (con los
  mismos pliegues) y en desarrollo, cuántas veces quita una categoría y si arregla o rompe.
- `hibrido_respaldo`: lo anterior y, si no queda categoría, las reglas + corrector. Se mide
  como alternativa que favorece el recall.

    uv run python -m evaluacion sinonimos --modelo sapbert
"""

import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist, normalizacion_codiesp, robustez
from evaluacion.cache import version_de
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.normalizador import (
    CANDIDATOS,
    TOP_K,
    Codificador,
    CodificadorConMemoria,
    Indice,
    Normalizador,
    Sinonimo,
    con_exclusiones,
    con_respaldo,
)
from ner_clinic.extraccion import SIGLAS_PARA_EL_MODELO
from ner_clinic.normalizacion.decision import PATRON_SIGLA  # noqa: F401

CARPETA_RESULTADOS = Path("evaluacion/resultados")
ORIGEN_CODIESP = "mención de CodiEsp entrenamiento"
PLIEGUES = 5
CONFIGURACIONES = ("sin_sinonimos", "con_sinonimos")
HIBRIDOS = ("hibrido_exclusiones", "hibrido_respaldo")
EFECTOS = ("aplica", "arregla", "quita_confusion", "rompe")


def ruta_resultados(modelo: str) -> Path:
    return CARPETA_RESULTADOS / f"sinonimos_{modelo}_desarrollo.json"


def normalizar(texto: str) -> str:
    return " ".join(texto.split())


def _agrupar(documentos: Sequence[DocumentoCodificado]) -> dict[str, dict[str, Counter]]:
    """Texto normalizado en minúscula → conteo de códigos, de escrituras y de documentos."""
    grupos: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"codigos": Counter(), "escrituras": Counter(), "documentos": Counter()}
    )
    for documento in documentos:
        for mencion in documento.menciones_diagnostico:
            texto = normalizar(charlson_codiesp.texto_en(documento.texto, mencion))
            if not texto:
                continue
            grupo = grupos[texto.casefold()]
            grupo["codigos"][mencion.codigo] += 1
            grupo["escrituras"][texto] += 1
            grupo["documentos"][documento.id] += 1
    return grupos


def _mas_frecuente(conteo: Counter) -> str:
    return min(conteo, key=lambda clave: (-conteo[clave], clave))


def sinonimos_de(documentos: Sequence[DocumentoCodificado]) -> list[Sinonimo]:
    """Un sinónimo por texto distinto (sin distinguir mayúsculas), en orden alfabético."""
    return [
        Sinonimo(
            _mas_frecuente(grupo["escrituras"]),
            _mas_frecuente(grupo["codigos"]),
            ORIGEN_CODIESP,
            tuple(sorted(grupo["documentos"])),
            sum(grupo["codigos"].values()),
        )
        for _, grupo in sorted(_agrupar(documentos).items())
    ]


def resumen_sinonimos(documentos: Sequence[DocumentoCodificado]) -> dict[str, int]:
    """Cuántas menciones dan cuántos sinónimos, y cuántos textos tienen más de un código."""
    grupos = _agrupar(documentos).values()
    return {
        "documentos": len(documentos),
        "menciones": sum(sum(g["codigos"].values()) for g in grupos),
        "sinonimos": len(grupos),
        "textos_con_varios_codigos": sum(len(g["codigos"]) > 1 for g in grupos),
        "menciones_con_otro_codigo": sum(
            sum(g["codigos"].values()) - max(g["codigos"].values()) for g in grupos
        ),
    }


def verificar_origen(
    sinonimos: Sequence[Sinonimo], prohibidos: dict[str, Iterable[str]]
) -> dict[str, int]:
    """Falla si algún sinónimo sale de un documento de `prohibidos` (partición → ids).

    Devuelve cuántos documentos de origen hay y cuántos ids de cada partición se
    revisaron.
    """
    origen = {id_ for s in sinonimos for id_ in s.documentos}
    revisados = {"documentos_origen": len(origen)}
    for particion, ids in prohibidos.items():
        ids = set(ids)
        cruce = origen & ids
        if cruce:
            raise ValueError(
                f"{len(cruce)} documentos de {particion!r} entraron al índice como sinónimos."
            )
        revisados[f"ids_{particion}"] = len(ids)
    return revisados


def pliegues_de(
    documentos: Sequence[DocumentoCodificado], n: int = PLIEGUES
) -> list[list[DocumentoCodificado]]:
    """`n` pliegues por documento, en orden de id (el documento `i` va al pliegue `i % n`).

    Con menos de 2 no hay validación cruzada: el único pliegue no tendría sinónimos.
    """
    if n < 2:
        raise ValueError(f"Hacen falta al menos 2 pliegues, no {n}.")
    ordenados = sorted(documentos, key=lambda d: d.id)
    return [ordenados[i::n] for i in range(n)]


def _pliegues_con_sinonimos(
    indice: Indice,
    codificador: Codificador,
    documentos: Sequence[DocumentoCodificado],
    n: int,
    k: int = TOP_K,
) -> Iterator[tuple[Normalizador, list[DocumentoCodificado]]]:
    """Cada pliegue con un índice que tiene solo los sinónimos de los otros pliegues.

    Es un generador para que haya un solo índice de pliegue en memoria a la vez. `k` es
    cuántos candidatos trae cada búsqueda.
    """
    for pliegue in pliegues_de(documentos, n):
        ids = {d.id for d in pliegue}
        resto = [d for d in documentos if d.id not in ids]
        sinonimos = sinonimos_de(resto)
        verificar_origen(sinonimos, {"pliegue": ids})
        yield Normalizador(indice.con_sinonimos(sinonimos, codificador), codificador, k), pliegue


def metricas_siglas(
    casos: Sequence[charlson_codiesp.Caso], normalizador: Normalizador | None = None
) -> dict[str, dict[str, Any]]:
    """Aciertos de categoría (y de código a 3 caracteres, si hay `normalizador`) en las
    menciones que son siglas, en total y según estén o no en la lista de #114."""
    grupos: dict[str, list[charlson_codiesp.Caso]] = {
        "todas": [],
        "en_lista": [],
        "fuera_de_lista": [],
    }
    for caso in casos:
        texto = caso.texto.strip()
        if PATRON_SIGLA.fullmatch(texto):
            grupos["todas"].append(caso)
            en_lista = texto.upper() in SIGLAS_PARA_EL_MODELO
            grupos["en_lista" if en_lista else "fuera_de_lista"].append(caso)
    salida = {}
    for nombre, lista in grupos.items():
        con_gold = [c for c in lista if c.gold is not None]
        salida[nombre] = {
            "menciones": len(lista),
            "siglas_distintas": len({c.texto.strip() for c in lista}),
            "aciertos": sum(c.prediccion is c.gold for c in lista),
            "gold_charlson": len(con_gold),
            "aciertos_charlson": sum(c.prediccion is c.gold for c in con_gold),
        }
        if normalizador is not None:
            salida[nombre]["codigo_top1"] = sum(
                normalizador.candidatos(c.texto)[0].codigo[:3] == c.codigo_gold[:3] for c in lista
            )
    return salida


def siglas_en_sinonimos(
    casos: Sequence[charlson_codiesp.Caso], sinonimos: Sequence[Sinonimo]
) -> int:
    """Menciones sigla cuyo texto está tal cual (sin distinguir mayúsculas) entre los
    sinónimos."""
    textos = {s.texto.casefold() for s in sinonimos}
    return sum(
        PATRON_SIGLA.fullmatch(c.texto.strip()) is not None and c.texto.strip().casefold() in textos
        for c in casos
    )


def efecto_exclusiones(
    antes: Sequence[charlson_codiesp.Caso], despues: Sequence[charlson_codiesp.Caso]
) -> Counter[str]:
    """Cuántas categorías quitó una exclusión (`aplica`) y qué pasó con cada una.

    - `arregla`: el gold no tiene categoría; se quitó un falso positivo.
    - `quita_confusion`: el gold tiene otra categoría; se quitó el falso positivo, pero el
      falso negativo sigue.
    - `rompe`: la categoría estaba bien.

    `antes` y `despues` son las mismas menciones, sin y con exclusiones.
    """
    efecto: Counter[str] = Counter({clave: 0 for clave in EFECTOS})
    for a, d in zip(antes, despues, strict=True):
        if a.prediccion is not None and d.prediccion is None:
            efecto["aplica"] += 1
            if a.prediccion is a.gold:
                efecto["rompe"] += 1
            else:
                efecto["arregla" if a.gold is None else "quita_confusion"] += 1
    return efecto


def _medir(
    mapeador: charlson_codiesp.Mapeador,
    desarrollo: Sequence[DocumentoCodificado],
    variantes: Sequence[str],
    replicas: int,
    vistas: set[str],
) -> tuple[dict[str, Any], list[charlson_codiesp.Caso]]:
    """Categoría en cada variante, en las menciones no vistas y en las siglas."""
    ids = [d.id for d in desarrollo]
    originales = charlson_codiesp.casos(desarrollo, mapeador)
    nuevas = [c for c in originales if normalizar(c.texto).casefold() not in vistas]
    medicion = {
        "categoria": {
            variante: charlson_codiesp.metricas(
                originales
                if variante == "original"
                else charlson_codiesp.casos(desarrollo, mapeador, variante),
                ids,
                replicas,
                por_categoria=variante == "original",
            )
            for variante in variantes
        },
        "no_vistas": charlson_codiesp.metricas(nuevas, ids, replicas, por_categoria=False),
    }
    return medicion, originales


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    por_particion: dict[str, Sequence[DocumentoCodificado]],
    meta_indice: dict[str, Any],
    ids_prueba: Sequence[str],
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    variantes: Sequence[str] = robustez.VARIANTES_TEXTO,
    pliegues: int = PLIEGUES,
) -> dict[str, Any]:
    """El JSON de resultados, sin texto del corpus: en desarrollo, las reglas y el modelo
    sin sinónimos y con ellos.

    `indice` es el del catálogo; `codificador`, el del modelo (sin memoria, para medir la
    latencia). `no_vistas` mide solo las menciones de desarrollo cuyo texto no está tal
    cual entre los sinónimos: lo que los sinónimos no resuelven por coincidencia exacta.
    """
    charlson_codiesp._validar_particiones(list(por_particion))
    if set(por_particion) != {"train", "dev"}:
        raise ValueError("Hacen falta 'train' (sinónimos y umbral) y 'dev' (medición).")
    entrenamiento, desarrollo = por_particion["train"], por_particion["dev"]
    con_memoria = CodificadorConMemoria(codificador)

    sinonimos = sinonimos_de(entrenamiento)
    verificacion = verificar_origen(
        sinonimos, {"dev": [d.id for d in desarrollo], "test": ids_prueba}
    )
    vistas = {s.texto.casefold() for s in sinonimos}
    inicio = time.perf_counter()
    con_memoria([s.texto for s in sinonimos])
    segundos_sinonimos = round(time.perf_counter() - inicio, 1)

    metodos: dict[str, Any] = {}
    medicion, reglas = _medir(
        charlson_codiesp.METODOS["reglas_corrector"], desarrollo, variantes, replicas, vistas
    )
    metodos["reglas_corrector"] = {**medicion, "siglas": metricas_siglas(reglas)}
    latencia: dict[str, float] = {}
    normalizadores: dict[str, Normalizador] = {}
    for nombre in CONFIGURACIONES:
        inicio = time.perf_counter()
        if nombre == "sin_sinonimos":
            curva = normalizacion_codiesp.curva_umbral(
                [(Normalizador(indice, con_memoria), entrenamiento)]
            )
            final = indice
        else:
            curva = normalizacion_codiesp.curva_umbral(
                _pliegues_con_sinonimos(indice, con_memoria, entrenamiento, pliegues)
            )
            final = indice.con_sinonimos(sinonimos, con_memoria)
        segundos_umbral = round(time.perf_counter() - inicio, 1)
        umbral = normalizacion_codiesp.elegir_umbral(curva)
        normalizador = normalizadores[nombre] = Normalizador(final, con_memoria)
        normalizador.precalcular(normalizacion_codiesp.textos_de(desarrollo, variantes))
        medicion, originales = _medir(
            normalizador.mapeador(umbral), desarrollo, variantes, replicas, vistas
        )
        metodos[nombre] = {
            "filas_indice": len(final.codigos),
            "umbral": {
                "elegido": umbral,
                "particion": "train",
                "pliegues": 1 if nombre == "sin_sinonimos" else pliegues,
                "segundos": segundos_umbral,
                "curva": curva,
            },
            **medicion,
            "codigo": normalizacion_codiesp.metricas_codigo(normalizador, desarrollo),
            "siglas": metricas_siglas(originales, normalizador),
            "origen_top1": dict(
                Counter(normalizador.candidatos(c.texto)[0].origen for c in originales)
            ),
        }
        if nombre == "con_sinonimos":
            latencia = normalizacion_codiesp.latencia(
                Normalizador(final, codificador),
                normalizacion_codiesp.textos_de(desarrollo, ["original"]),
            )

    # Híbridos (#133), con el umbral y el índice del modelo con sinónimos
    umbral = metodos["con_sinonimos"]["umbral"]["elegido"]
    modelo_con = normalizadores["con_sinonimos"].mapeador(umbral)
    hibridos = {
        "hibrido_exclusiones": con_exclusiones(modelo_con),
        "hibrido_respaldo": con_respaldo(
            con_exclusiones(modelo_con), charlson_codiesp.METODOS["reglas_corrector"]
        ),
    }
    for nombre, mapeador in hibridos.items():
        medicion, originales = _medir(mapeador, desarrollo, variantes, replicas, vistas)
        metodos[nombre] = {
            "umbral": {"elegido": umbral, "de": "con_sinonimos"},
            **medicion,
            "siglas": metricas_siglas(originales),
        }
    en_entrenamiento: Counter[str] = Counter({clave: 0 for clave in EFECTOS})
    for normalizador, pliegue in _pliegues_con_sinonimos(
        indice, con_memoria, entrenamiento, pliegues
    ):
        normalizador.precalcular(normalizacion_codiesp.textos_de(pliegue, ["original"]))
        mapeador = normalizador.mapeador(umbral)
        en_entrenamiento += efecto_exclusiones(
            charlson_codiesp.casos(pliegue, mapeador),
            charlson_codiesp.casos(pliegue, con_exclusiones(mapeador)),
        )
    exclusiones = {
        "train_validacion_cruzada": {k: en_entrenamiento[k] for k in EFECTOS},
        "dev": dict(
            efecto_exclusiones(
                charlson_codiesp.casos(desarrollo, modelo_con),
                charlson_codiesp.casos(desarrollo, hibridos["hibrido_exclusiones"]),
            )
        ),
    }

    modelo = CANDIDATOS[nombre_modelo]
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": modelo.nombre},
        "modelo": {
            "clave": nombre_modelo,
            "nombre": modelo.nombre,
            "revision": modelo.revision,
            "licencia": modelo.licencia,
            "transformers": version_de("transformers"),
        },
        "indice": meta_indice,
        "particiones": {"sinonimos_y_umbral": "train", "medicion": "dev"},
        "semilla_errores": charlson_codiesp.SEMILLA_ERRORES,
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": {"train": len(entrenamiento), "dev": len(desarrollo)},
        "sinonimos": {**resumen_sinonimos(entrenamiento), "origen": ORIGEN_CODIESP},
        "verificacion_origen": verificacion,
        "siglas": {
            "lista_114": len(SIGLAS_PARA_EL_MODELO),
            "menciones_en_sinonimos": siglas_en_sinonimos(reglas, sinonimos),
        },
        "metodos": metodos,
        "exclusiones": exclusiones,
        "costo": {
            "segundos_sinonimos": segundos_sinonimos,
            "latencia_con_sinonimos": latencia,
            "memoria_maxima_mib": normalizacion_codiesp.memoria_maxima_mib(),
        },
    }


def _f1(m: dict[str, Any]) -> str:
    f1 = m["micro"]
    return f"{f1['f1']:.3f} [{f1['ic95']['f1'][0]:.3f}; {f1['ic95']['f1'][1]:.3f}]"


def tabla(resultados: dict[str, Any]) -> str:
    """Desarrollo: reglas, el modelo sin sinónimos y con ellos (categoría, código, siglas)."""
    lineas = [
        "| Método | Umbral | F1 original [IC 95 %] | F1 minúsculas | F1 errores 10 % | P | R | "
        "F1 no vistas | Código top-1 / top-5 | Siglas de Charlson |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    clave = resultados["modelo"]["clave"]
    for nombre, m in resultados["metodos"].items():
        o = m["categoria"]["original"]
        s = m["siglas"]["todas"]
        if nombre == "reglas_corrector":
            metodo, umbral, codigo = "reglas + corrector", "—", "—"
        else:
            metodo = f"{clave} {nombre.replace('_', ' ')}"
            umbral = f"{m['umbral']['elegido']:.2f}"
            codigo = (
                f"{m['codigo']['top1']:.3f} / {m['codigo']['top5']:.3f}" if "codigo" in m else "—"
            )
        lineas.append(
            f"| {metodo} | {umbral} | {_f1(o)} | "
            f"{m['categoria']['minusculas']['micro']['f1']:.3f} | "
            f"{m['categoria']['errores_10']['micro']['f1']:.3f} | "
            f"{o['micro']['precision']:.3f} | {o['micro']['recall']:.3f} | "
            f"{m['no_vistas']['micro']['f1']:.3f} | {codigo} | "
            f"{s['aciertos_charlson']}/{s['gold_charlson']} |"
        )
    return "\n".join(lineas)
