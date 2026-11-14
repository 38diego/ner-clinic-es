"""Prueba de estrés: el pipeline con notas mal escritas (PRD `robustez`, FR1, issue #112).

Las notas reales se escriben rápido: en minúscula, sin tildes y con errores de digitación.
DisTEMIST y el set sintético están bien escritos, así que sus cifras son las de un caso
favorable. Este módulo genera **variantes del texto del mismo largo** que el original
(los spans anotados de DisTEMIST siguen valiendo) y mide con ellas:

- **DisTEMIST** (partición de prueba): P/R/F1 estrictos y laxos con IC del modelo crudo y
  del pipeline en cada variante. Es la cifra con peso: no se usa para ajustar nada.
  Además, el **recall en las siglas** (`siglas`): de las menciones del gold que son una
  sigla ("EPOC", "HTA", "DM"; `es_sigla`), cuántas se emparejan. DisTEMIST son casos
  clínicos publicados, bien redactados, y solo el 3,5 % de sus menciones son siglas
  (90 de 2.598 en prueba): en el F1 global no se ve el efecto de las minúsculas, que se
  concentra en las siglas. Con 90 menciones, su IC es ancho.
- **Set sintético**: por nota, el índice de Charlson exacto (y a 1 punto o menos) y las
  categorías de Charlson de las confirmadas: cuántas del gold salen (recall) y cuántas
  salen sin estar en el gold (sobrantes). Se compara por categoría y no por entidad
  porque, con errores o sin tildes, el texto de la entidad ya no coincide con el del gold.
  Es un conjunto de categorías por nota: no es la misma métrica que
  `contexto_charlson.metricas_categoria` (entidad por entidad) y no se comparan.
  El set no es ciego (ya se vio en #45, #74 y #89): sus cifras son de desarrollo.

Variantes (`VARIANTES_TEXTO`):

- `original`: el texto tal cual.
- `minusculas`: todo en minúscula.
- `minusculas_sin_tildes`: además, sin tildes ni diéresis. La `ñ` se conserva: está en el
  teclado en Colombia.
- `errores_5` y `errores_10`: el texto original con errores en el 5 % o el 10 % de las
  palabras de 4 letras o más. Cada palabra elegida recibe uno: la sustitución de una letra
  o el intercambio de dos letras vecinas, nunca en la primera letra (los errores de
  digitación casi nunca caen ahí). No se insertan ni borran letras, para no mover los
  offsets. Son errores uniformes: no son la distribución real de los errores de un médico.

Las tildes se quitan de los caracteres precompuestos (NFC, como viene DisTEMIST): un texto
en NFD (letra más tilde combinante) conserva la tilde, porque quitarla cambiaría el largo.

Cada transformación conserva el largo carácter por carácter (si un carácter cambiaría de
largo, como la `İ` en minúscula, se deja igual). Aun así se verifica por documento, y los
que cambien de largo se excluyen y se cuentan.

Los errores dependen de una semilla y del id del documento, así que una variante es la
misma aunque cambie el orden o el número de documentos. En el set sintético, los umbrales
del fallback de Charlson (#115) se ajustan con la semilla 0 (`SEMILLA_DESARROLLO`) y el
reporte usa las semillas 1 a 5 (`SEMILLAS_REPORTE`); DisTEMIST usa la semilla 1.

Uso:

    uv run python -m evaluacion robustez [--salida evaluacion/resultados/robustez_linea_base.json]
        [--solo distemist|sintetico] [--semillas 0 1 2 3 4 5] [--limite N]
"""

import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import contexto_charlson, distemist
from evaluacion.cache import ExtractorConCache
from evaluacion.corpus import Documento
from evaluacion.metricas import Emparejamiento, Modo, bootstrap_ic, emparejar
from ner_clinic.extraccion import Extractor
from ner_clinic.pipeline import analizar
from ner_clinic.tipos import Estado, Resultado

RUTA_RESULTADOS = Path("evaluacion/resultados/robustez_linea_base.json")
RUTA_CORRECCIONES = Path("data/evaluacion/robustez_correcciones.json")
VARIANTES_TEXTO = (
    "original",
    "minusculas",
    "minusculas_sin_tildes",
    "errores_5",
    "errores_10",
)
FILAS_DISTEMIST = ("crudo", "pipeline")
SEMILLA_DESARROLLO = 0
SEMILLAS_REPORTE = (1, 2, 3, 4, 5)
SEMILLA_DISTEMIST = 1

_PALABRA = re.compile(r"[^\W\d_]{4,}")
_SIGLA = re.compile(r"[A-ZÁÉÍÓÚÑ0-9]{2,6}")
_LETRAS = "abcdefghijklmnopqrstuvwxyz"


# --- Variantes del texto -------------------------------------------------------------


def _por_caracter(texto: str, transformar: Callable[[str], str]) -> str:
    """Aplica `transformar` a cada carácter y deja igual el que cambiaría de largo."""
    return "".join(t if len(t := transformar(c)) == 1 else c for c in texto)


def minusculas(texto: str) -> str:
    return _por_caracter(texto, str.lower)


def _sin_tilde(caracter: str) -> str:
    if caracter in "ñÑ":
        return caracter
    return "".join(
        c for c in unicodedata.normalize("NFD", caracter) if unicodedata.category(c) != "Mn"
    )


def sin_tildes(texto: str) -> str:
    return _por_caracter(texto, _sin_tilde)


def _sustituir(palabra: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(palabra))
    original = palabra[i]
    letra = rng.choice([c for c in _LETRAS if c != original.lower()])
    return palabra[:i] + (letra.upper() if original.isupper() else letra) + palabra[i + 1 :]


def _intercambiar(palabra: str, rng: random.Random) -> str | None:
    posiciones = [i for i in range(1, len(palabra) - 1) if palabra[i] != palabra[i + 1]]
    if not posiciones:
        return None
    i = rng.choice(posiciones)
    return palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2 :]


def con_errores(texto: str, fraccion: float, rng: random.Random) -> str:
    """Un error de digitación en una `fraccion` de las palabras de 4 letras o más."""

    def error(coincidencia: re.Match[str]) -> str:
        palabra = coincidencia.group()
        if rng.random() >= fraccion:
            return palabra
        if rng.random() < 0.5 and (intercambiada := _intercambiar(palabra, rng)) is not None:
            return intercambiada
        return _sustituir(palabra, rng)

    return _PALABRA.sub(error, texto)


def variar(texto: str, variante: str, semilla: int = SEMILLA_DESARROLLO, clave: str = "") -> str:
    """`texto` en la `variante` pedida. Los errores dependen de `semilla` y de `clave`
    (el id del documento o de la nota)."""
    if variante == "original":
        return texto
    if variante == "minusculas":
        return minusculas(texto)
    if variante == "minusculas_sin_tildes":
        return sin_tildes(minusculas(texto))
    if variante.startswith("errores_") and variante in VARIANTES_TEXTO:
        fraccion = int(variante.removeprefix("errores_")) / 100
        return con_errores(texto, fraccion, random.Random(f"{semilla}:{clave}"))
    raise ValueError(f"Variante desconocida: {variante!r}. Opciones: {list(VARIANTES_TEXTO)}")


def _tiene_errores(variante: str) -> bool:
    return variante.startswith("errores_")


# --- DisTEMIST -----------------------------------------------------------------------


def es_sigla(texto: str) -> bool:
    """Una mención que es solo una sigla: de 2 a 6 mayúsculas o dígitos, con alguna letra."""
    texto = texto.strip()
    return bool(_SIGLA.fullmatch(texto)) and any(c.isalpha() for c in texto)


def recall_siglas(
    emparejamientos: Sequence[Emparejamiento], modo: Modo, replicas: int, semilla: int
) -> dict[str, Any]:
    """Recall de las menciones gold que son siglas, con IC por bootstrap sobre documentos.

    Las siglas del gold se emparejan aparte contra todas las predicciones del documento
    (las de los pares verdaderos y los falsos positivos). Con el emparejamiento uno a uno
    del documento entero, una sigla anotada dentro de un span más largo ("insuficiencia
    renal crónica (IRC)") perdería la predicción frente al span largo y contaría como no
    encontrada aunque el modelo la viera.
    """
    conteos = []
    for e in emparejamientos:
        gold = [g for g, _ in e.verdaderos] + list(e.falsos_negativos)
        predichas = [p for _, p in e.verdaderos] + list(e.falsos_positivos)
        siglas = [g for g in gold if es_sigla(g.texto)]
        conteos.append(emparejar(siglas, predichas, modo).conteos._replace(fp=0))
    encontradas = sum(c.vp for c in conteos)
    total = encontradas + sum(c.fn for c in conteos)
    ic = bootstrap_ic(conteos, n=replicas, semilla=semilla).recall
    return {
        "encontradas": encontradas,
        "total": total,
        "recall": round(encontradas / total, 4) if total else None,
        "ic95": [round(ic.inferior, 4), round(ic.superior, 4)],
    }


def variar_documentos(
    documentos: Sequence[Documento], variante: str, semilla: int = SEMILLA_DISTEMIST
) -> tuple[list[Documento], list[str]]:
    """Los documentos con el texto en `variante` y los ids de los que cambiaron de largo."""
    variados, excluidos = [], []
    for documento in documentos:
        texto = variar(documento.texto, variante, semilla, documento.id)
        if len(texto) != len(documento.texto):
            excluidos.append(documento.id)
        else:
            variados.append(replace(documento, texto=texto))
    return variados, excluidos


def medir_distemist(
    documentos: Sequence[Documento],
    extractor: Extractor,
    variantes: Iterable[str] = VARIANTES_TEXTO,
    semilla: int = SEMILLA_DISTEMIST,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
) -> dict[str, Any]:
    """P/R/F1 estrictos y laxos del modelo crudo y del pipeline en cada variante."""
    resultados: dict[str, Any] = {}
    for variante in variantes:
        variados, excluidos = variar_documentos(documentos, variante, semilla)
        corrida = distemist.correr(variados, extractor, FILAS_DISTEMIST)
        resultados[variante] = {
            "documentos": len(variados),
            "excluidos": excluidos,
            "segundos": round(corrida.segundos, 2),
            **{
                fila: {
                    **{
                        modo: distemist.metricas_de(corrida.emparejamientos[fila][modo], replicas)
                        for modo in distemist.MODOS
                    },
                    "siglas": {
                        modo: recall_siglas(
                            corrida.emparejamientos[fila][modo],
                            modo,
                            replicas,
                            distemist.SEMILLA_BOOTSTRAP,
                        )
                        for modo in distemist.MODOS
                    },
                }
                for fila in FILAS_DISTEMIST
            },
        }
    return resultados


def correcciones_en(documentos: Sequence[Documento], extractor: Extractor) -> list[dict[str, Any]]:
    """Las entidades que reciben categoría de Charlson solo por corregir la ortografía.

    Corre el pipeline sobre el texto original de cada documento y devuelve, por entidad
    con `correcciones`, el documento, el texto, el estado, la categoría, el CIE-10, las
    correcciones y el contexto (60 caracteres a cada lado de la primera mención). Sirve
    para revisar a mano los falsos positivos del fallback (#115, #117). **Contiene texto
    del corpus**: se guarda fuera del repo, en `data/evaluacion/`.
    """
    filas = []
    for documento in documentos:
        for entidad in analizar(documento.texto, extractor=extractor).entidades:
            if not entidad.correcciones:
                continue
            inicio, fin = entidad.inicio or 0, entidad.fin or 0
            filas.append(
                {
                    "documento": documento.id,
                    "texto": entidad.texto,
                    "estado": entidad.estado.value,
                    "categoria": entidad.categoria_charlson and entidad.categoria_charlson.value,
                    "cie10": entidad.cie10,
                    "correcciones": [list(c) for c in entidad.correcciones],
                    "contexto": documento.texto[max(0, inicio - 60) : fin + 60],
                }
            )
    return filas


# --- Set sintético -------------------------------------------------------------------


def _categorias_confirmadas(resultado: Resultado) -> set[str]:
    return {
        e.categoria_charlson.value
        for e in resultado.entidades
        if e.estado is Estado.CONFIRMADA and e.categoria_charlson is not None
    }


def _categorias_gold(nota: dict[str, Any]) -> set[str]:
    return {
        e["categoria"]
        for e in nota["entidades"]
        if e["estado"] == Estado.CONFIRMADA.value and e.get("categoria")
    }


def medir_notas(
    notas: Sequence[dict[str, Any]], extractor: Extractor, variante: str, semilla: int
) -> dict[str, Any]:
    """Índice y categorías de Charlson del set sintético con el texto en `variante`."""
    inicio = time.perf_counter()
    exactos = cercanos = con_indice = encontradas = esperadas = sobrantes = 0
    errores: list[dict[str, Any]] = []
    for nota in notas:
        texto = variar(nota["texto"], variante, semilla, nota["id"])
        resultado = analizar(
            texto, diagnosticos_registrados=nota.get("registrados"), extractor=extractor
        )
        esperado = nota.get("indice")
        if esperado is not None:
            con_indice += 1
            exactos += resultado.indice_charlson == esperado
            cercanos += abs(resultado.indice_charlson - esperado) <= 1
        gold, obtenidas = _categorias_gold(nota), _categorias_confirmadas(resultado)
        encontradas += len(gold & obtenidas)
        esperadas += len(gold)
        sobrantes += len(obtenidas - gold)
        if (esperado is not None and resultado.indice_charlson != esperado) or gold != obtenidas:
            errores.append(
                {
                    "nota": nota["id"],
                    "indice": [esperado, resultado.indice_charlson],
                    "faltan": sorted(gold - obtenidas),
                    "sobran": sorted(obtenidas - gold),
                }
            )
    return {
        "notas_con_indice": con_indice,
        "indice_exacto": exactos,
        "indice_a_1": cercanos,
        "categorias": {
            "encontradas": encontradas,
            "esperadas": esperadas,
            "recall": round(encontradas / esperadas, 4) if esperadas else None,
            "sobrantes": sobrantes,
        },
        "errores": errores,
        "segundos": round(time.perf_counter() - inicio, 2),
    }


def medir_sintetico(
    notas: Sequence[dict[str, Any]],
    extractor: Extractor,
    variantes: Iterable[str] = VARIANTES_TEXTO,
    semillas: Sequence[int] = (SEMILLA_DESARROLLO, *SEMILLAS_REPORTE),
) -> dict[str, Any]:
    """`medir_notas` por variante; las de errores, una vez por semilla (`"semilla_<n>"`)."""
    resultados: dict[str, Any] = {}
    for variante in variantes:
        if _tiene_errores(variante):
            resultados[variante] = {
                f"semilla_{s}": medir_notas(notas, extractor, variante, s) for s in semillas
            }
        else:
            resultados[variante] = medir_notas(notas, extractor, variante, SEMILLA_DESARROLLO)
    return resultados


# --- Evaluación completa -------------------------------------------------------------


def evaluar(
    extractor: Extractor,
    documentos: Sequence[Documento] | None = None,
    notas: Sequence[dict[str, Any]] | None = None,
    ruta_notas: Path | None = None,
    semillas: Sequence[int] = (SEMILLA_DESARROLLO, *SEMILLAS_REPORTE),
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    limite: int | None = None,
    cache: ExtractorConCache | None = None,
) -> dict[str, Any]:
    """El JSON de resultados (sin texto del corpus). Mide DisTEMIST si se pasan
    `documentos` y el set sintético si se pasan `notas`."""
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": distemist.entorno(),
        "limite": limite,
        "variantes": list(VARIANTES_TEXTO),
        "distemist": (
            None
            if documentos is None
            else {
                "particion": "test",
                "semilla_errores": SEMILLA_DISTEMIST,
                "bootstrap": {
                    "replicas": replicas,
                    "semilla": distemist.SEMILLA_BOOTSTRAP,
                    "nivel": 0.95,
                },
                "variantes": medir_distemist(documentos, extractor, replicas=replicas),
            }
        ),
        "sintetico": (
            None
            if notas is None
            else {
                "notas": len(notas),
                "ruta": str(ruta_notas) if ruta_notas else None,
                "sha256": contexto_charlson.huella(ruta_notas) if ruta_notas else None,
                "semilla_desarrollo": SEMILLA_DESARROLLO,
                "semillas": list(semillas),
                "variantes": medir_sintetico(notas, extractor, semillas=semillas),
            }
        ),
        "cache": (
            None
            if cache is None
            else {"aciertos": cache.aciertos, "fallos": cache.fallos, "llamadas": cache.llamadas}
        ),
    }


# --- Reporte -------------------------------------------------------------------------


def _resumen_semillas(por_semilla: dict[str, Any], semillas: Sequence[int]) -> str:
    valores = [por_semilla[f"semilla_{s}"]["indice_exacto"] for s in semillas]
    if not valores:
        return "—"
    media = sum(valores) / len(valores)
    return f"{media:.1f} ({min(valores)}–{max(valores)})"


def tabla(resultados: dict[str, Any]) -> str:
    """Resumen en Markdown: F1 de DisTEMIST y el índice exacto del set sintético."""
    lineas: list[str] = []
    if resultados["distemist"] is not None:
        lineas += [
            "| Variante | Crudo laxo | Crudo estricto | Pipeline laxo | Pipeline estricto "
            "| Recall siglas (pipeline, laxo) |",
            "|---|---|---|---|---|---|",
        ]
        for variante, r in resultados["distemist"]["variantes"].items():
            celdas = [f"{r[f][m]['f1']:.3f}" for f in FILAS_DISTEMIST for m in ("laxo", "estricto")]
            siglas = r["pipeline"]["siglas"]["laxo"]
            celdas.append(
                "—"
                if siglas["recall"] is None
                else f"{siglas['recall']:.3f} ({siglas['encontradas']}/{siglas['total']})"
            )
            lineas.append(f"| `{variante}` | " + " | ".join(celdas) + " |")
        lineas.append("")
    if resultados["sintetico"] is not None:
        lineas += [
            "| Variante | Índice exacto | Recall de categorías | Categorías sobrantes |",
            "|---|---|---|---|",
        ]
        for variante, r in resultados["sintetico"]["variantes"].items():
            if _tiene_errores(variante):
                desarrollo = r.get(f"semilla_{SEMILLA_DESARROLLO}")
                if desarrollo is not None:
                    lineas.append(_fila_sintetico(f"`{variante}` semilla 0", desarrollo))
                reporte = [s for s in SEMILLAS_REPORTE if f"semilla_{s}" in r]
                if reporte:
                    lineas.append(
                        f"| `{variante}` semillas {reporte[0]}–{reporte[-1]} "
                        f"| {_resumen_semillas(r, reporte)} | | |"
                    )
            else:
                lineas.append(_fila_sintetico(f"`{variante}`", r))
    return "\n".join(lineas)


def _fila_sintetico(nombre: str, r: dict[str, Any]) -> str:
    categorias = r["categorias"]
    recall = "—" if categorias["recall"] is None else f"{categorias['recall']:.3f}"
    return (
        f"| {nombre} | {r['indice_exacto']} | {recall} "
        f"({categorias['encontradas']}/{categorias['esperadas']}) | {categorias['sobrantes']} |"
    )
