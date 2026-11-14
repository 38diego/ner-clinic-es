"""Caché en disco de la salida cruda del modelo NER, por fragmento.

La evaluación sobre DisTEMIST corre el modelo crudo, el pipeline completo y cada fila de
la ablación sobre los mismos documentos. Todas esas variantes segmentan igual y le pasan
al modelo los mismos fragmentos, así que basta con una inferencia por fragmento: la
primera vez se guarda la salida del modelo y las siguientes se lee del disco, también
entre corridas.

`ExtractorConCache` envuelve cualquier `Extractor` (el `ExtractorHF` real o el
`ExtractorFalso` de los tests) y cumple la misma interfaz. Guarda un JSON por fragmento:

    data/cache/<sha256>.json    {"entidades": [{"entity_group": ..., "start": ...}, ...]}

La clave es el sha256 de (nombre del modelo, versión de `transformers`, `STRIDE_TOKENS`,
texto del fragmento): si cambia cualquiera de los cuatro, la entrada vieja no se usa. El
archivo no guarda el texto del fragmento, pero sí las palabras que el modelo marcó
(`word`), así que la caché vive en `data/`, que está en `.gitignore`.
"""

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ner_clinic.extraccion import NOMBRE_MODELO, STRIDE_TOKENS, Extractor

RAIZ_CACHE = Path("data/cache")


def version_de(paquete: str) -> str | None:
    """Versión instalada de `paquete`, sin importarlo; `None` si no está instalado."""
    try:
        return version(paquete)
    except PackageNotFoundError:
        return None


def clave_de(texto: str, nombre_modelo: str, version_transformers: str | None, stride: int) -> str:
    """sha256 en hexadecimal de lo que determina la salida del modelo para `texto`."""
    contenido = json.dumps(
        [nombre_modelo, version_transformers, stride, texto], ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(contenido).hexdigest()


def _a_json(valor: Any) -> Any:
    """Convierte los escalares de numpy (`score` es `float32`) a tipos de Python."""
    if hasattr(valor, "item"):
        return valor.item()
    return valor


def _normalizar(entidades: list[dict]) -> list[dict]:
    return [{k: _a_json(v) for k, v in e.items()} for e in entidades]


class ExtractorConCache:
    """Extractor envoltorio que guarda en disco la salida de `extractor` por fragmento.

    `extraer` busca cada fragmento en `carpeta` y solo le pasa a `extractor` los que no
    están (sin repetidos), en una sola llamada. Las entidades salen siempre ya pasadas
    por JSON, vengan del modelo o del disco, para que la primera corrida y las siguientes
    vean exactamente lo mismo.

    Cuenta los fragmentos leídos del disco (`aciertos`), los que tuvo que inferir
    (`fallos`) y las llamadas al extractor envuelto (`llamadas`).
    """

    def __init__(
        self,
        extractor: Extractor,
        carpeta: Path = RAIZ_CACHE,
        nombre_modelo: str = NOMBRE_MODELO,
        version_transformers: str | None = None,
        stride: int = STRIDE_TOKENS,
    ):
        self.extractor = extractor
        self.carpeta = Path(carpeta)
        self.nombre_modelo = nombre_modelo
        self.version_transformers = version_transformers or version_de("transformers")
        self.stride = stride
        self.aciertos = 0
        self.fallos = 0
        self.llamadas = 0

    def ruta(self, fragmento: str) -> Path:
        """Archivo de la caché que corresponde a `fragmento`."""
        clave = clave_de(fragmento, self.nombre_modelo, self.version_transformers, self.stride)
        return self.carpeta / f"{clave}.json"

    def _leer(self, fragmento: str) -> list[dict] | None:
        """Entidades guardadas de `fragmento`; `None` si no están o el archivo está dañado."""
        try:
            with self.ruta(fragmento).open(encoding="utf-8") as f:
                return json.load(f)["entidades"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _guardar(self, fragmento: str, entidades: list[dict]) -> None:
        """Escribe en un temporal y lo renombra, para no dejar archivos a medias."""
        self.carpeta.mkdir(parents=True, exist_ok=True)
        ruta = self.ruta(fragmento)
        temporal = ruta.with_suffix(f".{os.getpid()}.tmp")
        with temporal.open("w", encoding="utf-8") as f:
            json.dump({"entidades": entidades}, f, ensure_ascii=False)
        temporal.replace(ruta)

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        memoria: dict[str, list[dict]] = {}
        nuevos = []
        for fragmento in dict.fromkeys(fragmentos):
            guardadas = self._leer(fragmento)
            if guardadas is None:
                nuevos.append(fragmento)
            else:
                memoria[fragmento] = guardadas
                self.aciertos += 1

        if nuevos:
            self.llamadas += 1
            self.fallos += len(nuevos)
            salidas = self.extractor.extraer(nuevos)
            for fragmento, entidades in zip(nuevos, salidas, strict=True):
                normalizadas = json.loads(json.dumps(_normalizar(entidades), ensure_ascii=False))
                self._guardar(fragmento, normalizadas)
                memoria[fragmento] = normalizadas

        return [[dict(e) for e in memoria[f]] for f in fragmentos]
