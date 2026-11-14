"""Extracción de comorbilidades de notas clínicas en español.

Uso:

    from ner_clinic import analizar

    resultado = analizar("Paciente con HTA y diabetes mellitus tipo 2. Niega EPOC.")
    resultado.to_dict()
"""

from ner_clinic.pipeline import analizar
from ner_clinic.tipos import (
    CategoriaCharlson,
    Descarte,
    Entidad,
    Estado,
    Mencion,
    MotivoDescarte,
    Resultado,
)

__all__ = [
    "CategoriaCharlson",
    "Descarte",
    "Entidad",
    "Estado",
    "Mencion",
    "MotivoDescarte",
    "Resultado",
    "analizar",
]
