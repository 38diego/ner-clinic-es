"""Límites de la API, leídos de variables de entorno.

| Variable | Por defecto | Qué controla |
|---|---|---|
| `MAX_CARACTERES` | 5000 | Tamaño máximo de la nota; si se pasa, 413. |
| `TIMEOUT_S` | 30 | Segundos máximos de espera por el turno de inferencia; si se pasa, 503. |
| `TOPE_IP_DIA` | 20 | Análisis al día por IP; si se pasa, 429. |
| `TOPE_GLOBAL_DIA` | 200 | Análisis al día en total; si se pasa, 429. |
| `SAL_IP` | aleatoria | Sal de la clave por IP. Es un secreto y nunca va en el repo. |
| `PROXIES_CONFIABLES` | 1 | Proxies delante de la app que agregan su valor a `X-Forwarded-For`. |
| `CONTADOR` | `memoria` | Dónde se cuentan los análisis: `memoria` o `firestore`. |
| `FIRESTORE_PROYECTO` | el del entorno | Proyecto de GCP de Firestore (solo con `firestore`). |
| `FIRESTORE_BASE` | `(default)` | Base de Firestore (solo con `firestore`). |
| `NER_NORMALIZADOR` | `hibrido` | Categoría de Charlson: `hibrido`, `sapbert` o `reglas` (#182). |
| `NER_INDICE_NORMALIZADOR` | el de `artefactos` | Índice del normalizador (`hibrido`, `sapbert`). |

Sin `SAL_IP` se usa una sal aleatoria por proceso: sirve en local, pero en producción las
claves cambiarían en cada arranque y el tope por IP se reiniciaría.

`PROXIES_CONFIABLES`: la IP del visitante es el valor de `X-Forwarded-For` en la
posición `-PROXIES_CONFIABLES`, contando desde el final, porque los primeros los puede
escribir el cliente. Con 1 es el último valor, el que agrega el frontend de Google en
Cloud Run. Si se pone un balanceador delante, sería 2. El valor exacto se verifica en
producción en #102.

Los tres mapeos (epic #179):

- `hibrido` (por defecto, #182): primero las reglas y, para lo que no mapean, el
  normalizador (índice CIE-10, clasificador y SapBERT, que se cargan al arrancar). Si no
  está el índice, la app arranca con las reglas, lo registra y `/api/salud` dice
  `mapeo: "reglas"`; si está y algo más falla, no arranca. Así la misma configuración
  sirve con o sin SapBERT en la imagen.
- `sapbert`: solo el normalizador (#168). Si falta un artefacto, no arranca.
- `reglas`: solo las reglas; no carga nada más.

La imagen Docker completa trae SapBERT y el índice (#189); la etapa `base`, no, y con
`hibrido` corre con las reglas. Licencia de los pesos: `docs/datos.md`.

`CONTADOR=memoria` sirve en local y en los tests, pero cada instancia y cada reinicio
empiezan de cero; en Cloud Run va `firestore` (ver `ContadorFirestore`).
"""

import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["Configuracion"]

CONTADORES = ("memoria", "firestore")
NORMALIZADORES = ("hibrido", "sapbert", "reglas")


def _entero(entorno: Mapping[str, str], nombre: str, defecto: int) -> int:
    valor = entorno.get(nombre)
    if valor is None or not valor.strip():
        return defecto
    try:
        numero = int(valor)
    except ValueError as error:
        raise ValueError(f"{nombre} debe ser un entero positivo") from error
    if numero <= 0:
        raise ValueError(f"{nombre} debe ser un entero positivo")
    return numero


def _contador(entorno: Mapping[str, str]) -> Literal["memoria", "firestore"]:
    valor = (entorno.get("CONTADOR") or "memoria").strip().lower()
    if valor not in CONTADORES:
        raise ValueError("CONTADOR debe ser `memoria` o `firestore`")
    return valor  # type: ignore[return-value]


def _normalizador(entorno: Mapping[str, str]) -> Literal["hibrido", "sapbert", "reglas"]:
    valor = (entorno.get("NER_NORMALIZADOR") or "hibrido").strip().lower()
    if valor not in NORMALIZADORES:
        raise ValueError("NER_NORMALIZADOR debe ser `hibrido`, `sapbert` o `reglas`")
    return valor  # type: ignore[return-value]


@dataclass(frozen=True)
class Configuracion:
    """Límites de la API. `desde_entorno()` los lee de las variables de entorno."""

    max_caracteres: int = 5000
    timeout_s: float = 30
    tope_ip_dia: int = 20
    tope_global_dia: int = 200
    proxies_confiables: int = 1
    sal_ip: str = field(default_factory=lambda: secrets.token_hex(16), repr=False)
    contador: Literal["memoria", "firestore"] = "memoria"
    firestore_proyecto: str | None = None
    firestore_base: str | None = None
    normalizador: Literal["hibrido", "sapbert", "reglas"] = "hibrido"
    indice_normalizador: str | None = None

    @classmethod
    def desde_entorno(cls, entorno: Mapping[str, str] | None = None) -> "Configuracion":
        entorno = os.environ if entorno is None else entorno
        sal = entorno.get("SAL_IP") or secrets.token_hex(16)
        return cls(
            max_caracteres=_entero(entorno, "MAX_CARACTERES", 5000),
            timeout_s=_entero(entorno, "TIMEOUT_S", 30),
            tope_ip_dia=_entero(entorno, "TOPE_IP_DIA", 20),
            tope_global_dia=_entero(entorno, "TOPE_GLOBAL_DIA", 200),
            proxies_confiables=_entero(entorno, "PROXIES_CONFIABLES", 1),
            sal_ip=sal,
            contador=_contador(entorno),
            firestore_proyecto=entorno.get("FIRESTORE_PROYECTO") or None,
            firestore_base=entorno.get("FIRESTORE_BASE") or None,
            normalizador=_normalizador(entorno),
            indice_normalizador=entorno.get("NER_INDICE_NORMALIZADOR") or None,
        )
