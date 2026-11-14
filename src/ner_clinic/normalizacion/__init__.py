"""Normalizador por embeddings: mención → top-k de códigos CIE-10 → categoría de Charlson.

Es la opción elegida en el epic #148: SapBERT con el catálogo CIE-10-ES y los sinónimos
de CodiEsp entrenamiento (`indice`), y el clasificador de #151 sobre el top-20
(`decision`). Importar `ner_clinic` no carga este paquete ni `torch`.
"""

from ner_clinic.normalizacion.decision import (
    Decision,
    SegundaEtapa,
    modelo_a_json,
    modelo_de_json,
)
from ner_clinic.normalizacion.indice import SAPBERT, CodificadorHF, Indice, Normalizador

__all__ = [
    "SAPBERT",
    "CodificadorHF",
    "Decision",
    "Indice",
    "Normalizador",
    "SegundaEtapa",
    "modelo_a_json",
    "modelo_de_json",
]
