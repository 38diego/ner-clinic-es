"""Medición final de la normalización en CodiEsp **prueba** (PRD `normalizacion`, FR7, #134).

Es la única vez que se usa la partición de prueba de CodiEsp. Todo lo que se mide aquí
quedó fijo antes, en entrenamiento y desarrollo:

- **Métodos**: las reglas + corrector (#130), SapBERT con los sinónimos de CodiEsp
  entrenamiento (#132), el híbrido con las exclusiones de las reglas (#133) y, como
  alternativa medida pero no adoptada, el híbrido con respaldo de las reglas.
- **Sinónimos**: solo de entrenamiento, como en #132 (`sinonimos_codiesp.sinonimos_de`).
  Se verifica que ninguno sale de un documento de prueba.
- **Umbral**: `UMBRAL`, el que eligió la validación cruzada en entrenamiento en #132. No se
  vuelve a elegir.

Para cada método: P/R/F1 micro y por categoría con IC del 95 % (bootstrap por documento),
las variantes de robustez, el F1 en las menciones no vistas, las siglas y, para SapBERT,
el código CIE-10 a 3 caracteres (top-1 y top-5). También los casos de los issues (#120,
#123): qué categoría da cada método a "accidente cerebrovascular", "ictus" y "linfoma de
Hodgkin sin metástasis".

La salida no se sobrescribe: si el JSON ya existe, la corrida no arranca. La guarda es
sobre la ruta: con otra `--salida` se podría volver a correr, y no se debe. Se corre desde
un árbol limpio (`git status` sin cambios), con:

    uv run python -m evaluacion prueba --modelo sapbert
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist, normalizacion_codiesp, robustez
from evaluacion.cache import version_de
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.normalizador import (
    CANDIDATOS,
    Codificador,
    CodificadorConMemoria,
    Indice,
    Normalizador,
    con_exclusiones,
    con_respaldo,
)
from evaluacion.sinonimos_codiesp import (
    _medir,
    metricas_siglas,
    resumen_sinonimos,
    siglas_en_sinonimos,
    sinonimos_de,
    verificar_origen,
)
from ner_clinic.tipos import CategoriaCharlson

RUTA_RESULTADOS = Path("evaluacion/resultados/codiesp_prueba.json")
UMBRAL = 0.69  # elegido por validación cruzada en entrenamiento (#132)
METODOS = ("reglas_corrector", "sapbert_sinonimos", "hibrido_exclusiones", "hibrido_respaldo")
CASOS_ISSUES: dict[str, CategoriaCharlson] = {
    "accidente cerebrovascular": CategoriaCharlson.CEREBROVASCULAR,
    "ictus": CategoriaCharlson.CEREBROVASCULAR,
    "linfoma de Hodgkin sin metástasis": CategoriaCharlson.TUMOR_MALIGNO,
}


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    prueba: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    variantes: Sequence[str] = robustez.VARIANTES_TEXTO,
    umbral: float = UMBRAL,
) -> dict[str, Any]:
    """El JSON de resultados en prueba, sin texto del corpus."""
    con_memoria = CodificadorConMemoria(codificador)
    sinonimos = sinonimos_de(entrenamiento)
    verificacion = verificar_origen(sinonimos, {"test": [d.id for d in prueba]})
    vistas = {s.texto.casefold() for s in sinonimos}

    normalizador = Normalizador(indice.con_sinonimos(sinonimos, con_memoria), con_memoria)
    normalizador.precalcular(normalizacion_codiesp.textos_de(prueba, variantes))
    normalizador.precalcular(CASOS_ISSUES)
    modelo = normalizador.mapeador(umbral)
    reglas = charlson_codiesp.METODOS["reglas_corrector"]
    mapeadores = {
        "reglas_corrector": reglas,
        "sapbert_sinonimos": modelo,
        "hibrido_exclusiones": con_exclusiones(modelo),
        "hibrido_respaldo": con_respaldo(con_exclusiones(modelo), reglas),
    }

    metodos: dict[str, Any] = {}
    originales_reglas: list[charlson_codiesp.Caso] = []
    for nombre, mapeador in mapeadores.items():
        medicion, originales = _medir(mapeador, prueba, variantes, replicas, vistas)
        metodos[nombre] = {
            **medicion,
            "siglas": metricas_siglas(
                originales, normalizador if nombre == "sapbert_sinonimos" else None
            ),
            "casos_issues": {texto: _categoria(mapeador(texto)) for texto in CASOS_ISSUES},
        }
        if nombre == "reglas_corrector":
            originales_reglas = originales
    metodos["sapbert_sinonimos"]["codigo"] = normalizacion_codiesp.metricas_codigo(
        normalizador, prueba
    )

    modelo_embeddings = CANDIDATOS[nombre_modelo]
    return {
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
        "particiones": {"sinonimos": "train", "medicion": "test"},
        "umbral": {"valor": umbral, "elegido_en": "train, validación cruzada (#132)"},
        "semilla_errores": charlson_codiesp.SEMILLA_ERRORES,
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": {"train": len(entrenamiento), "test": len(prueba)},
        "sinonimos": resumen_sinonimos(entrenamiento),
        "verificacion_origen": verificacion,
        "siglas": {"menciones_en_sinonimos": siglas_en_sinonimos(originales_reglas, sinonimos)},
        "casos_issues_esperados": {t: c.value for t, c in CASOS_ISSUES.items()},
        "metodos": metodos,
    }


def _categoria(resultado: tuple[CategoriaCharlson, str] | None) -> str | None:
    return None if resultado is None else resultado[0].value


def _f1(m: dict[str, Any]) -> str:
    f1 = m["micro"]
    return f"{f1['f1']:.3f} [{f1['ic95']['f1'][0]:.3f}; {f1['ic95']['f1'][1]:.3f}]"


def tabla(resultados: dict[str, Any]) -> str:
    """CodiEsp prueba: categoría de Charlson por método, con robustez y siglas."""
    lineas = [
        "| Método | P | R | F1 [IC 95 %] | F1 minúsculas | F1 errores 10 % | F1 no vistas | "
        "Siglas de Charlson | Casos de los issues |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    esperados = resultados["casos_issues_esperados"]
    for nombre, m in resultados["metodos"].items():
        o = m["categoria"]["original"]
        s = m["siglas"]["todas"]
        casos = sum(m["casos_issues"][t] == c for t, c in esperados.items())
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {o['micro']['precision']:.3f} | "
            f"{o['micro']['recall']:.3f} | {_f1(o)} | "
            f"{m['categoria']['minusculas']['micro']['f1']:.3f} | "
            f"{m['categoria']['errores_10']['micro']['f1']:.3f} | "
            f"{m['no_vistas']['micro']['f1']:.3f} | "
            f"{s['aciertos_charlson']}/{s['gold_charlson']} | {casos}/{len(esperados)} |"
        )
    return "\n".join(lineas)
