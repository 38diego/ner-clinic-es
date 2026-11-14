"""Los dos artefactos del normalizador y su carga: el índice y el clasificador (#164).

- **Índice** (`RUTA_INDICE`, no se versiona, unos 300 MB): las filas del catálogo
  CIE-10-ES 2018 más los sinónimos de CodiEsp entrenamiento, con sus vectores de SapBERT.
  Lo construye `python -m evaluacion artefactos`.
- **Clasificador** (`RUTA_CLASIFICADOR`, versionado dentro del paquete): la regresión
  logística de #151 sin fuentes externas (decisión de #154), entrenada con los 5 pliegues
  de CodiEsp entrenamiento. Son números (pesos, media, desviación, sesgo y umbral) y
  metadatos, sin texto de ningún corpus.

El clasificador lleva la huella del índice (códigos, textos, origen y configuración de
SapBERT; los vectores no, porque los fija esa configuración). `cargar` la recalcula con las
filas cargadas: si no coincide, falla en vez de decidir con un clasificador entrenado sobre
otro índice. `RUTA_INDICE` es relativa a la raíz del repo; la API la cambia con
`NER_INDICE_NORMALIZADOR`. La imagen Docker completa trae SapBERT y el índice (#189).
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ner_clinic.normalizacion.decision import SegundaEtapa, modelo_de_json
from ner_clinic.normalizacion.indice import (
    SAPBERT,
    Codificador,
    CodificadorHF,
    Indice,
    ModeloEmbeddings,
    Normalizador,
    huella_configuracion,
)

RUTA_INDICE = Path("data/normalizacion/indice_final.pt")
RUTA_CLASIFICADOR = Path(__file__).resolve().parent / "clasificador.json"
COMO_CONSTRUIR = (
    "Constrúyelos con `uv run python -m evaluacion artefactos` desde la raíz del repo (ver "
    "docs/normalizacion.md). En la API, `NER_NORMALIZADOR=reglas` corre sin ellos y "
    "`hibrido` (el de por defecto) corre con las reglas si falta el índice."
)


def huella_indice(indice: Indice, modelo: ModeloEmbeddings = SAPBERT) -> str:
    """SHA-256 de las filas del índice (código, texto y origen, en orden) y de la
    configuración del modelo que hizo los vectores."""
    filas = [
        [codigo, texto, indice.origen(i)]
        for i, (codigo, texto) in enumerate(zip(indice.codigos, indice.descripciones, strict=True))
    ]
    contenido = json.dumps([filas, huella_configuracion(modelo)], ensure_ascii=False)
    return hashlib.sha256(contenido.encode()).hexdigest()


def guardar_indice(indice: Indice, ruta: Path, meta: dict[str, Any]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "codigos": indice.codigos,
            "descripciones": indice.descripciones,
            "origenes": [indice.origen(i) for i in range(len(indice.codigos))],
            "vectores": indice.vectores,
            "meta": meta,
        },
        ruta,
    )


def cargar_indice(ruta: Path = RUTA_INDICE) -> tuple[Indice, dict[str, Any]]:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No está el índice del normalizador en {ruta}. {COMO_CONSTRUIR}")
    guardado = torch.load(ruta, weights_only=True)
    indice = Indice(
        guardado["codigos"], guardado["descripciones"], guardado["vectores"], guardado["origenes"]
    )
    return indice, guardado["meta"]


def leer_clasificador(ruta: Path = RUTA_CLASIFICADOR) -> dict[str, Any]:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No está el clasificador del normalizador en {ruta}. {COMO_CONSTRUIR}"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar(
    ruta_indice: Path = RUTA_INDICE,
    ruta_clasificador: Path = RUTA_CLASIFICADOR,
    codificador: Callable[[], Codificador] | None = None,
) -> SegundaEtapa:
    """La segunda etapa lista para usar: índice, clasificador y codificador de SapBERT.

    Falla si falta un artefacto o si el clasificador se entrenó sobre otro índice o con
    otro modelo de embeddings. `codificador` (una función que lo crea) es para los tests;
    por defecto carga SapBERT con su revisión fija.
    """
    datos = leer_clasificador(ruta_clasificador)
    indice, _ = cargar_indice(ruta_indice)
    esperada = datos["meta"]["huella_indice"]
    # Se recalcula con las filas cargadas: no basta con la huella guardada en el archivo
    obtenida = huella_indice(indice)
    if obtenida != esperada:
        raise ValueError(
            f"El índice de {ruta_indice} no es aquel con el que se entrenó el clasificador "
            f"(huella {obtenida[:12]} y no {esperada[:12]}). " + COMO_CONSTRUIR
        )
    if datos["meta"]["configuracion_embeddings"] != huella_configuracion(SAPBERT):
        raise ValueError(
            f"El clasificador se entrenó con otra configuración de SapBERT. {COMO_CONSTRUIR}"
        )
    modelo, umbral = modelo_de_json(datos)
    crear = codificador or (lambda: CodificadorHF(SAPBERT))
    return SegundaEtapa(Normalizador(indice, crear()), modelo, umbral)
