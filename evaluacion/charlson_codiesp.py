"""Línea base de las reglas de Charlson en CodiEsp (PRD `normalizacion`, FR2 y FR3, #130).

**Gold.** Cada mención de diagnóstico de CodiEsp-X lleva un código CIE-10-ES puesto a mano.
Su categoría de Charlson es la que da la tabla de Quan (`charlson.quan.categoria_de`), o
ninguna. Las menciones de procedimientos no entran.

**Qué se mide.** Las reglas se corren sobre el **texto de cada mención del gold**, no sobre
la nota: se mide la normalización (texto → categoría), aislada del NER y del contexto. Es
clasificación por mención:

- Verdadero positivo de la categoría `c`: el gold es `c` y las reglas dan `c`.
- Falso positivo de `c`: las reglas dan `c` y el gold es otra categoría o ninguna.
- Falso negativo de `c`: el gold es `c` y las reglas dan otra o ninguna.

P/R/F1 micro suman las 17 categorías; "ninguna" no es una clase. El IC del 95 % es por
bootstrap sobre documentos (`metricas.bootstrap_ic`).

**Qué codifica CodiEsp** (guía de anotación v1.3, Zenodo 4121205, y una muestra de
desarrollo): anota toda mención de un diagnóstico, **también las negadas** ("sin tos",
"no presentaba hematuria") y **las de familiares** (regla P4: "anotar independientemente
del sujeto"). Para esta medición da igual, porque las reglas no miran el contexto. Para
el índice por documento (#134) no: CodiEsp-D incluye diagnósticos negados. Por eso el
resultado trae, aparte, el estado que nuestro módulo de contexto le da a cada mención
gold (`estado_segun_contexto`): es una estimación, no un gold.

**Métodos** (`METODOS`):

- `reglas`: las 17 reglas exactas (`charlson.reglas.mapear`).
- `reglas_corrector`: las reglas con el corrector de ortografía de #115
  (`charlson.correccion.mapear`). Es lo que usa el pipeline.

**Variantes**: las de la prueba de estrés (`robustez.VARIANTES_TEXTO`). Se aplican a la nota
completa y la mención se vuelve a leer de sus tramos, que siguen valiendo porque las
variantes conservan el largo. Los errores usan la semilla `SEMILLA_ERRORES` y el id del
documento.

**Código CIE-10.** Además de la categoría, `cie10_3` es la fracción de menciones gold con
categoría de Charlson en las que el código de las reglas coincide con el gold en sus 3
primeros caracteres. Las reglas dan códigos de la CIE-10 de la OMS, y algunos no existen en
la CIE-10-ES (`docs/datos.md`): por eso la comparación principal es por categoría.

**Particiones.** Solo entrenamiento y desarrollo. La de prueba queda para la medición final
(#134) y aquí se rechaza.

Uso:

    uv run python -m evaluacion codiesp [--raiz data/codiesp] [--particiones train dev]
        [--salida evaluacion/resultados/codiesp_reglas_desarrollo.json]
        [--detalle data/evaluacion/codiesp_reglas_errores.json]
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import distemist, robustez
from evaluacion.codiesp import DocumentoCodificado, MencionCodificada
from evaluacion.metricas import bootstrap_ic, micro_prf
from ner_clinic.charlson import correccion, quan, reglas
from ner_clinic.contexto import clasificar_mencion, intervalos
from ner_clinic.tipos import CategoriaCharlson, Estado, Mencion

RUTA_RESULTADOS = Path("evaluacion/resultados/codiesp_reglas_desarrollo.json")
RUTA_DETALLE = Path("data/evaluacion/codiesp_reglas_errores.json")
PARTICIONES_DESARROLLO = ("train", "dev")
SEMILLA_ERRORES = 1  # como DisTEMIST; la 0 se usó para ajustar el corrector (#115)

Mapeador = Callable[[str], tuple[CategoriaCharlson, str] | None]


def _con_corrector(texto: str) -> tuple[CategoriaCharlson, str] | None:
    mapeo = correccion.mapear(texto)
    return None if mapeo is None else (mapeo.categoria, mapeo.cie10)


METODOS: dict[str, Mapeador] = {"reglas": reglas.mapear, "reglas_corrector": _con_corrector}


@dataclass(frozen=True)
class Caso:
    """Una mención gold y lo que dio un método sobre su texto."""

    documento: str
    texto: str
    codigo_gold: str
    gold: CategoriaCharlson | None
    codigo: str | None
    prediccion: CategoriaCharlson | None


def texto_en(texto_nota: str, mencion: MencionCodificada) -> str:
    """El texto de la mención leído de sus tramos en `texto_nota` (que puede ser una variante)."""
    return " ".join(texto_nota[a:b] for a, b in mencion.tramos)


def _validar_particiones(particiones: Sequence[str]) -> None:
    fuera = [p for p in particiones if p not in PARTICIONES_DESARROLLO]
    if fuera:
        raise ValueError(
            f"Particiones no permitidas: {fuera}. La línea base solo usa "
            f"{list(PARTICIONES_DESARROLLO)}; la de prueba queda para #134."
        )


def casos(
    documentos: Sequence[DocumentoCodificado],
    mapeador: Mapeador,
    variante: str = "original",
    semilla: int = SEMILLA_ERRORES,
) -> list[Caso]:
    """Cada mención de diagnóstico, con su categoría gold y la del `mapeador` en la `variante`."""
    salida = []
    for documento in documentos:
        nota = robustez.variar(documento.texto, variante, semilla, documento.id)
        if len(nota) != len(documento.texto):  # las variantes conservan el largo
            raise ValueError(f"La variante {variante!r} cambió el largo de {documento.id}")
        for mencion in documento.menciones_diagnostico:
            texto = texto_en(nota, mencion)
            resultado = mapeador(texto)
            categoria, codigo = resultado if resultado is not None else (None, None)
            salida.append(
                Caso(
                    documento.id,
                    texto,
                    mencion.codigo,
                    quan.categoria_de(mencion.codigo),
                    codigo,
                    categoria,
                )
            )
    return salida


# --- Métricas ------------------------------------------------------------------------


def _conteos(caso: Caso, categoria: CategoriaCharlson | None = None) -> tuple[int, int, int]:
    """(vp, fp, fn) de un caso, de todas las categorías o solo de `categoria`."""
    gold = caso.gold if categoria is None or caso.gold is categoria else None
    prediccion = caso.prediccion if categoria is None or caso.prediccion is categoria else None
    acierto = gold is not None and gold is prediccion
    return (
        int(acierto),
        int(prediccion is not None and not acierto),
        int(gold is not None and not acierto),
    )


def _por_documento(
    lista: Sequence[Caso], documentos: Sequence[str], categoria: CategoriaCharlson | None = None
) -> list[tuple[int, int, int]]:
    """Conteos sumados por documento, en el orden de `documentos` (también los que no tienen
    menciones, para que el bootstrap sortee sobre todos)."""
    suma = {id_: [0, 0, 0] for id_ in documentos}
    for caso in lista:
        for i, valor in enumerate(_conteos(caso, categoria)):
            suma[caso.documento][i] += valor
    return [tuple(suma[id_]) for id_ in documentos]


def _prf(conteos: list[tuple[int, int, int]], replicas: int) -> dict[str, Any]:
    puntual = micro_prf(conteos)
    ic = bootstrap_ic(conteos, n=replicas, semilla=distemist.SEMILLA_BOOTSTRAP)
    vp, fp, fn = (sum(c[i] for c in conteos) for i in range(3))
    return {
        "precision": round(puntual.precision, 4),
        "recall": round(puntual.recall, 4),
        "f1": round(puntual.f1, 4),
        "ic95": {
            "precision": [round(x, 4) for x in ic.precision],
            "recall": [round(x, 4) for x in ic.recall],
            "f1": [round(x, 4) for x in ic.f1],
        },
        "vp": vp,
        "fp": fp,
        "fn": fn,
    }


def metricas(
    lista: Sequence[Caso],
    documentos: Sequence[str],
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    por_categoria: bool = True,
) -> dict[str, Any]:
    """P/R/F1 micro (y por categoría) con IC, cobertura y acierto del código a 3 caracteres."""
    con_gold = [c for c in lista if c.gold is not None]
    con_prediccion = [c for c in lista if c.prediccion is not None]
    codigo_3 = sum(c.codigo is not None and c.codigo[:3] == c.codigo_gold[:3] for c in con_gold)
    salida: dict[str, Any] = {
        "micro": _prf(_por_documento(lista, documentos), replicas),
        "menciones": len(lista),
        "menciones_gold_charlson": len(con_gold),
        "menciones_con_categoria": len(con_prediccion),
        "con_categoria_sin_gold": sum(c.gold is None for c in con_prediccion),
        "cie10_3": round(codigo_3 / len(con_gold), 4) if con_gold else 0.0,
    }
    if por_categoria:
        salida["por_categoria"] = {
            categoria.value: _prf(_por_documento(lista, documentos, categoria), replicas)
            for categoria in CategoriaCharlson
        }
    return salida


# --- Estado de las menciones gold según nuestro contexto -----------------------------


def estados(documentos: Sequence[DocumentoCodificado]) -> dict[tuple[str, int], Estado]:
    """Estado (confirmada, posible, negada, familiar) que `contexto` le da a cada mención de
    diagnóstico, por `(documento, índice en menciones_diagnostico)`."""
    salida = {}
    for documento in documentos:
        tramos = intervalos(documento.texto)
        for i, m in enumerate(documento.menciones_diagnostico):
            salida[(documento.id, i)] = clasificar_mencion(
                Mencion(m.texto, m.inicio, m.fin), tramos
            )
    return salida


def resumen_estados(
    documentos: Sequence[DocumentoCodificado], lista: Sequence[Caso]
) -> dict[str, Any]:
    """Cuántas menciones gold con categoría caen en cada estado, y el recall de `lista` en
    cada uno. `lista` tiene que venir de `casos(documentos, ...)` en ese orden."""
    por_mencion = list(estados(documentos).items())
    if len(por_mencion) != len(lista) or any(
        clave[0] != caso.documento for (clave, _), caso in zip(por_mencion, lista, strict=False)
    ):
        raise ValueError("`lista` no corresponde a `documentos`")
    total: Counter[str] = Counter()
    aciertos: Counter[str] = Counter()
    for (_, estado), caso in zip(por_mencion, lista, strict=True):
        if caso.gold is None:
            continue
        total[estado.value] += 1
        aciertos[estado.value] += caso.prediccion is caso.gold
    return {
        estado.value: {
            "menciones_gold_charlson": total[estado.value],
            "recall": round(aciertos[estado.value] / total[estado.value], 4)
            if total[estado.value]
            else None,
        }
        for estado in Estado
    }


# --- Evaluación completa -------------------------------------------------------------


def evaluar(
    por_particion: dict[str, Sequence[DocumentoCodificado]],
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    variantes: Sequence[str] = robustez.VARIANTES_TEXTO,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Corre los métodos y variantes en cada partición y en todas juntas.

    Devuelve `(resultados, detalle)`: `resultados` es el JSON versionado, sin texto del
    corpus; `detalle`, los falsos positivos y negativos de la variante original con su
    texto, para el análisis de errores (va a `data/`, que no se versiona).
    Las métricas por categoría solo se calculan en el conjunto de todas las particiones.
    """
    _validar_particiones(list(por_particion))
    grupos = dict(por_particion)
    if len(por_particion) > 1:
        grupos["+".join(por_particion)] = [d for docs in por_particion.values() for d in docs]
    conjunto = list(grupos)[-1]

    resultados: dict[str, Any] = {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": distemist.entorno(),
        "particiones": list(por_particion),
        "conjunto_principal": conjunto,
        "semilla_errores": SEMILLA_ERRORES,
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": {nombre: len(docs) for nombre, docs in grupos.items()},
        "metodos": {},
        "estado_segun_contexto": {},
    }
    detalle: dict[str, Any] = {"conjunto": conjunto, "metodos": {}}

    for nombre_metodo, mapeador in METODOS.items():
        por_variante: dict[str, Any] = {}
        for variante in variantes:
            por_grupo = {}
            for nombre, docs in grupos.items():
                lista = casos(docs, mapeador, variante)
                principal = nombre == conjunto
                por_grupo[nombre] = metricas(
                    lista, [d.id for d in docs], replicas, por_categoria=principal
                )
                if principal and variante == "original":
                    resultados["estado_segun_contexto"][nombre_metodo] = resumen_estados(
                        docs, lista
                    )
                    detalle["metodos"][nombre_metodo] = _errores(lista)
            por_variante[variante] = por_grupo
        resultados["metodos"][nombre_metodo] = por_variante
    return resultados, detalle


def _errores(lista: Sequence[Caso]) -> list[dict[str, Any]]:
    return [
        {
            "documento": c.documento,
            "texto": c.texto,
            "codigo_gold": c.codigo_gold,
            "gold": c.gold and c.gold.value,
            "codigo": c.codigo,
            "prediccion": c.prediccion and c.prediccion.value,
        }
        for c in lista
        if c.gold is not c.prediccion
    ]


# --- Reporte -------------------------------------------------------------------------


def _celda(m: dict[str, Any], metrica: str) -> str:
    inferior, superior = m["ic95"][metrica]
    return f"{m[metrica]:.3f} [{inferior:.3f}, {superior:.3f}]"


def tabla(resultados: dict[str, Any]) -> str:
    """Resumen en Markdown del conjunto principal: micro por método y variante."""
    conjunto = resultados["conjunto_principal"]
    lineas = [
        f"Conjunto: {conjunto} ({resultados['documentos'][conjunto]} documentos)",
        "",
        "| Método | Variante | P | R | F1 | Con categoría (sin gold) | CIE-10 a 3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for metodo, por_variante in resultados["metodos"].items():
        for variante, por_grupo in por_variante.items():
            r = por_grupo[conjunto]
            m = r["micro"]
            lineas.append(
                f"| {metodo} | {variante} | {_celda(m, 'precision')} | {_celda(m, 'recall')} | "
                f"{_celda(m, 'f1')} | {r['menciones_con_categoria']} "
                f"({r['con_categoria_sin_gold']}) | {r['cie10_3']:.3f} |"
            )
    return "\n".join(lineas)


def tabla_categorias(resultados: dict[str, Any], metodo: str = "reglas_corrector") -> str:
    """P/R/F1 por categoría del método en la variante original, conjunto principal."""
    r = resultados["metodos"][metodo]["original"][resultados["conjunto_principal"]]
    lineas = [
        f"Por categoría ({metodo}, original)",
        "",
        "| Categoría | Gold | P | R | F1 |",
        "|---|---|---|---|---|",
    ]
    for categoria, m in r["por_categoria"].items():
        lineas.append(
            f"| {categoria} | {m['vp'] + m['fn']} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} |"
        )
    return "\n".join(lineas)
