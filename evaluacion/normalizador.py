"""Normalización por embeddings: mención → CIE-10 → Charlson (PRD `normalizacion`, FR4, #131).

Lo que se usa en ejecución (índice, codificador, `Normalizador`, exclusiones) vive en
`ner_clinic.normalizacion` (#163) y se reexporta aquí; este módulo agrega los modelos
candidatos del experimento y la construcción del índice del catálogo.

Es un experimento: no lo usa el pipeline. Cada descripción del catálogo CIE-10-ES de
CodiEsp (98.288 códigos) se convierte en un vector con un modelo de embeddings; una
mención se convierte igual y se buscan los códigos con mayor similitud de coseno. La
categoría de Charlson sale del código con la tabla de Quan. Si la similitud del primero no
llega al `umbral`, la mención queda sin código, como cuando ninguna regla aplica.

**Modelos candidatos** (`CANDIDATOS`), cada uno con su revisión fija y su licencia:

- `sapbert`: SapBERT multilingüe (Liu et al., 2021), XLM-R base afinado para que los
  sinónimos de un mismo concepto de UMLS queden cerca. *Pooling* CLS y texto en minúscula,
  como en su página.
  Su página no declara licencia y se entrenó con UMLS, que tiene licencia propia: no se
  puede redistribuir sin resolverlo.
- `e5_base` y `e5_small`: multilingual-e5 (Wang et al., 2024), modelos de similitud de
  propósito general, licencia MIT. *Mean pooling* y el prefijo "query: " en los dos lados,
  que es lo que recomiendan para tareas simétricas. La tesis usaba `e5_base` como primer
  filtro (`Emmbeding.ipynb`).
- `bsc_bio_ehr`: el RoBERTa biomédico del BSC sin afinar para similitud, con *mean pooling*
  (Apache 2.0). Es el que la tesis usaba para reordenar; se mide para ver si sirve así.

Los vectores se normalizan (norma 1), así que el coseno es el producto punto. El índice de
cada modelo se guarda en `data/normalizacion/` (no se versiona), con la huella del catálogo
y la revisión del modelo; si alguna cambia, se reconstruye.

**Sinónimos** (#132): `Indice.con_sinonimos` agrega filas que no son descripciones del
catálogo sino otras formas de nombrar un código (en el experimento, las menciones de
CodiEsp entrenamiento con su código gold). Cada fila guarda su origen, así que un
candidato puede explicarse: "parecido a 'ACV isquémico', mención de CodiEsp".
"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from ner_clinic.normalizacion.decision import (  # noqa: F401  (se reexportan)
    Mapeador,
    _cedida_sin_destino,
    con_exclusiones,
    excluida,
)
from ner_clinic.normalizacion.indice import (  # noqa: F401  (se reexportan)
    LARGO_MAXIMO,
    LOTE,
    ORIGEN_CATALOGO,
    SAPBERT,
    TOP_K,
    Candidato,
    Codificador,
    CodificadorHF,
    Indice,
    ModeloEmbeddings,
    Normalizador,
    Pooling,
    Sinonimo,
    huella_configuracion,
)
from ner_clinic.tipos import CategoriaCharlson

RAIZ_INDICES = Path("data/normalizacion")

CANDIDATOS: dict[str, ModeloEmbeddings] = {
    "sapbert": SAPBERT,
    "e5_base": ModeloEmbeddings(
        "intfloat/multilingual-e5-base",
        "d128750597153bb5987e10b1c3493a34e5a4502a",
        "media",
        "MIT",
        "query: ",
    ),
    "e5_small": ModeloEmbeddings(
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "media",
        "MIT",
        "query: ",
    ),
    "bsc_bio_ehr": ModeloEmbeddings(
        "PlanTL-GOB-ES/bsc-bio-ehr-es",
        "a3e805738e28ade482d74b57d8a1ad4ac18441da",
        "media",
        "Apache-2.0",
    ),
}


class CodificadorConMemoria:
    """Un `Codificador` que guarda el vector de cada texto y no lo vuelve a calcular.

    Sirve cuando varios índices comparten las mismas consultas o los mismos sinónimos
    (los pliegues de la validación cruzada de #132).
    """

    def __init__(self, codificador: Codificador):
        self.codificador = codificador
        self._vectores: dict[str, torch.Tensor] = {}

    def __call__(self, textos: Sequence[str]) -> torch.Tensor:
        nuevos = list(dict.fromkeys(t for t in textos if t not in self._vectores))
        if nuevos:
            self._vectores.update(zip(nuevos, self.codificador(nuevos), strict=True))
        if not textos:
            return torch.empty(0, 0)
        return torch.stack([self._vectores[t] for t in textos])


def huella_catalogo(catalogo: dict[str, str]) -> str:
    """SHA-256 del catálogo (código y descripción, en orden)."""
    contenido = json.dumps(sorted(catalogo.items()), ensure_ascii=False)
    return hashlib.sha256(contenido.encode()).hexdigest()


def construir_indice(catalogo: dict[str, str], codificador: Codificador) -> Indice:
    codigos = sorted(catalogo)
    descripciones = [catalogo[c] for c in codigos]
    return Indice(codigos, descripciones, codificador(descripciones))


def indice_en_disco(
    catalogo: dict[str, str],
    modelo: ModeloEmbeddings,
    codificador: Callable[[], Codificador],
    raiz: Path = RAIZ_INDICES,
) -> tuple[Indice, dict[str, Any]]:
    """El índice de `modelo` desde `raiz`, o construido (y guardado) si no está o no cuadra.

    `codificador` se llama solo si hay que construirlo. Devuelve también los datos del
    índice: huella del catálogo, revisión, tamaño y segundos de construcción. Se
    reconstruye si cambia el catálogo o la configuración del modelo (`huella_configuracion`).
    """
    huella = huella_catalogo(catalogo)
    configuracion = huella_configuracion(modelo)
    ruta = Path(raiz) / f"{modelo.nombre.replace('/', '__')}_{modelo.revision[:12]}.pt"
    if ruta.is_file():
        guardado = torch.load(ruta, weights_only=True)
        if (
            guardado["meta"]["catalogo"] == huella
            and guardado["meta"].get("configuracion") == configuracion
        ):
            indice = Indice(
                guardado["codigos"],
                [catalogo[c] for c in guardado["codigos"]],
                guardado["vectores"],
            )
            return indice, {**guardado["meta"], "bytes": ruta.stat().st_size}
    inicio = time.perf_counter()
    indice = construir_indice(catalogo, codificador())
    meta = {
        "modelo": modelo.nombre,
        "revision": modelo.revision,
        "catalogo": huella,
        "configuracion": configuracion,
        "codigos": len(indice.codigos),
        "dimension": indice.vectores.shape[1],
        "segundos_construccion": round(time.perf_counter() - inicio, 1),
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"codigos": indice.codigos, "vectores": indice.vectores, "meta": meta}, ruta)
    meta["bytes"] = ruta.stat().st_size
    return indice, meta


# --- Híbrido (#133) ------------------------------------------------------------------


def con_respaldo(mapeador: Mapeador, respaldo: Mapeador) -> Mapeador:
    """El `mapeador`, y `respaldo` cuando no da categoría."""

    def mapear(texto: str) -> tuple[CategoriaCharlson, str] | None:
        return mapeador(texto) or respaldo(texto)

    return mapear
