"""Construye los artefactos del normalizador que usa la librería (#164, epic #162).

1. **Índice final**: el catálogo CIE-10-ES 2018 más los sinónimos de CodiEsp
   entrenamiento, sin DisTEMIST ni CANTEMIST (decisión de Diego en #154). Se verifica que
   ningún sinónimo salga de desarrollo ni de prueba. Va a `artefactos.RUTA_INDICE` (no se
   versiona).
2. **Clasificador final**: el de #151 sin fuentes, entrenado con los 5 pliegues de
   entrenamiento, con la penalización y el umbral elegidos por validación cruzada sobre
   ellos (`medicion_final.entrenar_final`, lo mismo que se midió en #154). Va al JSON
   versionado del paquete (`artefactos.RUTA_CLASIFICADOR`), sin texto de ningún corpus.
3. **Comprobación, no medición**: con lo guardado y por la ruta de la librería
   (`artefactos.cargar` → `SegundaEtapa`), el F1 de categoría en las menciones gold de
   desarrollo tiene que ser el de #154 sin fuentes, con los mismos aciertos y errores
   (vp, fp, fn). Desarrollo ya está gastado; esto solo verifica que el artefacto es el
   método medido. Usa los vectores en caché de #154; la misma comprobación con SapBERT en
   vivo (lo que corre la librería) queda en `en_vivo`, como dato.

    uv run python -m evaluacion artefactos
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.dos_etapas import K, _prf
from evaluacion.medicion_final import entrenar_final
from evaluacion.normalizacion_codiesp import textos_de
from evaluacion.sinonimos_codiesp import PLIEGUES, sinonimos_de, verificar_origen
from ner_clinic.normalizacion import artefactos
from ner_clinic.normalizacion.decision import modelo_a_json
from ner_clinic.normalizacion.indice import SAPBERT, Codificador, Indice, huella_configuracion

RUTA_MEDICION_154 = Path("evaluacion/resultados/normalizacion_final_desarrollo.json")
METODO_154 = "clasificador_sin_fuentes"


def construir(
    indice_catalogo: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    ids_prohibidos: dict[str, Sequence[str]],
    pliegues: int = PLIEGUES,
    k: int = K,
) -> tuple[Indice, dict[str, Any], dict[str, Any]]:
    """El índice final, su meta y el JSON del clasificador (sin guardar nada)."""
    sinonimos = sinonimos_de(entrenamiento)
    verificacion = verificar_origen(sinonimos, ids_prohibidos)
    final = indice_catalogo.con_sinonimos(sinonimos, codificador)
    huella = artefactos.huella_indice(final)
    meta_indice = {
        "huella_indice": huella,
        "modelo": SAPBERT.nombre,
        "revision": SAPBERT.revision,
        "configuracion_embeddings": huella_configuracion(SAPBERT),
        "filas": len(final.codigos),
        "filas_catalogo": len(indice_catalogo.codigos),
        "sinonimos_codiesp_entrenamiento": len(sinonimos),
        "verificacion_origen": verificacion,
    }
    modelo, eleccion = entrenar_final(
        indice_catalogo, codificador, entrenamiento, [], {}, pliegues, k
    )
    clasificador = modelo_a_json(modelo, eleccion["umbral"])
    clasificador["meta"] = {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": SAPBERT.nombre},
        "metodo": "regresión logística L2 sobre el top-20 (#151), sin fuentes externas (#154)",
        "entrenamiento": f"CodiEsp entrenamiento, {len(entrenamiento)} documentos, "
        f"{pliegues} pliegues",
        "huella_indice": huella,
        "configuracion_embeddings": huella_configuracion(SAPBERT),
        "k": k,
        "eleccion": eleccion,
    }
    return final, meta_indice, clasificador


def comprobar(
    segunda: artefactos.SegundaEtapa,
    desarrollo: Sequence[DocumentoCodificado],
    esperado: dict[str, Any] | None,
) -> dict[str, Any]:
    """F1 y conteos (vp, fp, fn) de categoría en las menciones gold de desarrollo por la
    ruta de la librería, contra los de #154 sin fuentes (`esperado`, su `micro`). Coincide
    si los tres conteos son exactamente los mismos."""
    decisiones = segunda.decidir(textos_de(desarrollo, ["original"]))

    def mapear(texto: str):
        decision = decisiones[texto]
        return None if decision is None else (decision.categoria, decision.codigo)

    casos = charlson_codiesp.casos(desarrollo, mapear)
    ids = [d.id for d in desarrollo]
    por_documento = charlson_codiesp._por_documento(casos, ids)
    conteos = dict(zip(("vp", "fp", "fn"), map(sum, zip(*por_documento, strict=True)), strict=True))
    conteos_154 = None if esperado is None else {k: esperado[k] for k in ("vp", "fp", "fn")}
    return {
        "f1_desarrollo": _prf(casos, ids)["f1"],
        "conteos": conteos,
        "f1_154": None if esperado is None else esperado["f1"],
        "conteos_154": conteos_154,
        "coincide": conteos == conteos_154,
    }


def de_154(ruta: Path = RUTA_MEDICION_154) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """El micro de categoría (F1 y conteos) y la elección del clasificador sin fuentes en
    #154."""
    import json

    if not ruta.is_file():
        return None, None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    metodo = datos["desarrollo"]["metodos"][METODO_154]
    return metodo["categoria"]["original"]["micro"], datos["clasificador_final"][METODO_154]
