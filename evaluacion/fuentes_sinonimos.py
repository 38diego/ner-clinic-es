"""Medición de las fuentes de sinónimos públicos (#153) en la validación cruzada de #149.

Cada configuración suma unas fuentes externas (`sinonimos_publicos`) a los sinónimos de
CodiEsp entrenamiento de cada pliegue, y mide el híbrido de hoy (umbral 0,69 fijo) y el
clasificador de #151 en validación cruzada anidada, con los mismos pliegues. Se compara
cada una contra `base` (solo CodiEsp), pareado por documento, en general y en tumor
maligno y metastásico. No se mide en desarrollo ni en prueba. De prueba solo se leen
los ids, para excluir sus documentos de DisTEMIST. De desarrollo, además, los textos, para
buscar casos de CANTEMIST compartidos con CodiEsp.

    uv run python -m evaluacion fuentes [--modelo sapbert]
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist
from evaluacion.clasificador import ReglasDe, en_pliegue
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.dos_etapas import (
    UMBRAL_HIBRIDO,
    K,
    _prf,
    comparar,
    pliegues_internos,
    pliegues_topk,
    primer_candidato,
)
from evaluacion.dos_etapas import casos as casos_de
from evaluacion.normalizador import CANDIDATOS, Codificador, Indice
from evaluacion.sinonimos_codiesp import PLIEGUES, sinonimos_de
from evaluacion.sinonimos_publicos import (
    MencionExterna,
    lexicas,
    sin_documentos,
    sinonimos_externos,
)
from ner_clinic.charlson import quan
from ner_clinic.tipos import CategoriaCharlson

RUTA_RESULTADOS = Path("evaluacion/resultados/dos_etapas_fuentes.json")
# Declaradas antes de medir; `ambas` usa la variante léxica de CANTEMIST
CONFIGURACIONES = {
    "base": (),
    "distemist": ("distemist",),
    "cantemist": ("cantemist",),
    "cantemist_lexico": ("cantemist_lexico",),
    "ambas": ("distemist", "cantemist_lexico"),
}
CATEGORIAS = (CategoriaCharlson.TUMOR_MALIGNO, CategoriaCharlson.TUMOR_METASTASICO)
METODOS = ("hibrido_exclusiones", "clasificador")


def _tamano(
    entrenamiento: Sequence[DocumentoCodificado], externas: Sequence[MencionExterna], filas: int
) -> dict[str, int]:
    """Filas del índice con los 5 pliegues (el que se usaría en desarrollo)."""
    propios = sinonimos_de(entrenamiento)
    externos = sinonimos_externos(externas, (), (s.texto for s in propios))
    return {
        "catalogo": filas,
        "sinonimos_codiesp": len(propios),
        "sinonimos_externos": len(externos),
        "externos_con_categoria": sum(quan.categoria_de(s.codigo) is not None for s in externos),
        "total": filas + len(propios) + len(externos),
    }


def medir(
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    externas: Sequence[MencionExterna],
    reglas: ReglasDe,
    pliegues: int = PLIEGUES,
    k: int = K,
) -> tuple[list[str], list[list[str]], dict[str, list], list[dict[str, Any]]]:
    """Ids, ids por pliegue, casos de cada método y lo elegido en cada pliegue externo."""
    externos = pliegues_topk(indice, codificador, entrenamiento, pliegues, k, externas)
    por_pliegue = [[d.id for d in p.documentos] for p in externos]
    casos = {"hibrido_exclusiones": casos_de(externos, primer_candidato(UMBRAL_HIBRIDO))}
    casos["clasificador"] = []
    elegidas = []
    for i, externo in enumerate(externos):
        internos = pliegues_internos(indice, codificador, entrenamiento, i, pliegues, k, externas)
        lista, eleccion = en_pliegue(internos, externo, reglas)
        casos["clasificador"] += lista
        elegidas.append(eleccion)
    return [i for ids in por_pliegue for i in ids], por_pliegue, casos, elegidas


def _metricas(lista: Sequence[charlson_codiesp.Caso], ids: Sequence[str], replicas: int) -> dict:
    todas = charlson_codiesp.metricas(lista, ids, replicas)
    return {
        "micro": todas["micro"],
        **{c.value: todas["por_categoria"][c.value] for c in CATEGORIAS},
    }


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    fuentes: dict[str, Sequence[MencionExterna]],
    reporte_fuentes: dict[str, Any],
    pliegues: int = PLIEGUES,
    k: int = K,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    configuraciones: dict[str, tuple[str, ...]] = CONFIGURACIONES,
) -> dict[str, Any]:
    """El JSON de #153: solo entrenamiento y conteos, sin texto de ningún corpus."""
    reglas: ReglasDe = {}
    filas = len(indice.codigos)
    salida: dict[str, Any] = {}
    referencia: dict[str, list] = {}
    for nombre, usadas in configuraciones.items():
        externas = [m for fuente in usadas for m in fuentes[fuente]]
        ids, por_pliegue, casos, elegidas = medir(
            indice, codificador, entrenamiento, externas, reglas, pliegues, k
        )
        if not referencia:
            referencia = casos
        salida[nombre] = {
            "fuentes": list(usadas),
            "indice": _tamano(entrenamiento, externas, filas),
            "metodos": {m: _metricas(casos[m], ids, replicas) for m in METODOS},
            "f1_por_pliegue": {
                m: [
                    _prf([c for c in casos[m] if c.documento in set(propios)], propios)["f1"]
                    for propios in por_pliegue
                ]
                for m in METODOS
            },
            "elegidas": elegidas,
            "contra_base": {
                m: {
                    "micro": comparar(referencia[m], casos[m], ids, replicas),
                    **{
                        c.value: comparar(referencia[m], casos[m], ids, replicas, c)
                        for c in CATEGORIAS
                    },
                }
                for m in METODOS
            }
            if casos is not referencia
            else {},
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
        },
        "indice": meta_indice,
        "particion": "train (validación cruzada anidada); dev solo para excluir y buscar "
        "casos compartidos; test solo por sus ids",
        "pliegues": pliegues,
        "k_guardado": k,
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": len(entrenamiento),
        "fuentes": reporte_fuentes,
        "configuraciones": salida,
    }


def tabla(resultados: dict[str, Any]) -> str:
    """F1 general y de tumor maligno por configuración y método, con la diferencia
    pareada contra `base`."""
    lineas = [
        "| Fuentes | Método | Filas del índice | F1 [IC 95 %] | Δ F1 vs base [IC] | "
        "F1 tumor maligno | Δ tumor maligno [IC] |",
        "|---|---|---|---|---|---|---|",
    ]
    maligno = CategoriaCharlson.TUMOR_MALIGNO.value
    for nombre, config in resultados["configuraciones"].items():
        for metodo in METODOS:
            m = config["metodos"][metodo]
            micro, tumor = m["micro"], m[maligno]
            delta = delta_tumor = "—"
            if config["contra_base"]:
                d = config["contra_base"][metodo]
                delta = _delta(d["micro"]["f1"])
                delta_tumor = _delta(d[maligno]["f1"])
            lineas.append(
                f"| {nombre} | {metodo.replace('_', ' ')} | {config['indice']['total']:,} | "
                f"{micro['f1']:.3f} [{micro['ic95']['f1'][0]:.3f}; {micro['ic95']['f1'][1]:.3f}]"
                f" | {delta} | {tumor['f1']:.3f} | {delta_tumor} |"
            )
    return "\n".join(lineas)


def _delta(d: dict[str, Any]) -> str:
    return f"{d['diferencia']:+.3f} [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]"


def preparar(
    cantemist: Sequence[MencionExterna],
    distemist_: Sequence[MencionExterna],
    descartes: dict[str, Any],
    ids_codiesp: dict[str, Sequence[str]],
    compartidos_cantemist: dict[str, Any],
) -> tuple[dict[str, list[MencionExterna]], dict[str, Any]]:
    """Las fuentes sin los documentos de CodiEsp desarrollo y prueba, y el reporte del
    cruce (solo conteos). Los casos de CANTEMIST compartidos con CodiEsp, si los hubiera,
    también se quitan."""
    prohibidos = {p: ids_codiesp[p] for p in ("dev", "test")}
    compartidos = set(compartidos_cantemist["compartidos"])
    cantemist = [m for m in cantemist if m.documento not in compartidos]
    reporte: dict[str, Any] = {}
    fuentes: dict[str, list[MencionExterna]] = {}
    for nombre, menciones in (("cantemist", cantemist), ("distemist", distemist_)):
        limpias, cruce = sin_documentos(menciones, prohibidos)
        _, en_train = sin_documentos(limpias, {"train": ids_codiesp["train"]})
        fuentes[nombre] = limpias
        reporte[nombre] = {
            "menciones_leidas": len(menciones) + sum(descartes[nombre].values()),
            "descartes": dict(descartes[nombre]),
            "documentos": len({m.documento for m in menciones}),
            "excluidas_por_codiesp": cruce,
            "en_codiesp_train": en_train["train"],
            "menciones_usables": len(limpias),
            "textos_distintos": len({m.texto.casefold() for m in limpias}),
            "con_categoria": sum(quan.categoria_de(m.codigo) is not None for m in limpias),
        }
    fuentes["cantemist_lexico"] = lexicas(fuentes["cantemist"])
    reporte["cantemist_lexico"] = {
        "menciones_usables": len(fuentes["cantemist_lexico"]),
        "textos_distintos": len({m.texto.casefold() for m in fuentes["cantemist_lexico"]}),
    }
    reporte["cantemist"]["casos_compartidos_con_codiesp"] = {
        "compartidos": len(compartidos),
        "fraccion_maxima_8gramas": compartidos_cantemist["fraccion_maxima"],
        "comparado_con": "train y dev de CodiEsp (prueba no se lee)",
    }
    return fuentes, reporte
