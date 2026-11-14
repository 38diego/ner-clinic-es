"""Contador de análisis por día: el tope por IP y el tope global.

`Contador` es la interfaz. `ContadorEnMemoria` sirve para tests y desarrollo local;
`ContadorFirestore`, para producción. Las dos cuentan con una sola operación atómica
(`consumir`), que mira los dos topes y suma solo si cabe. Se elige con la variable
`CONTADOR` (ver `configuracion.py`).

La IP nunca se guarda: la clave es `sha256(sal + ip + fecha)` (`clave_ip`), con una sal
secreta, así que no se puede volver a la IP ni cruzar días.
"""

import hashlib
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any, Literal, Protocol

__all__ = [
    "Consumo",
    "Contador",
    "ContadorEnMemoria",
    "ContadorFirestore",
    "clave_ip",
    "hoy_en_colombia",
]

# Colombia no tiene horario de verano: UTC-5 todo el año. El día se reinicia a
# medianoche de Colombia, no de UTC.
_COLOMBIA = timezone(timedelta(hours=-5))


def hoy_en_colombia() -> date:
    return datetime.now(UTC).astimezone(_COLOMBIA).date()


def clave_ip(sal: str, ip: str, dia: date) -> str:
    """`sha256(sal + ip + fecha)` en hexadecimal: la clave del contador por IP."""
    return hashlib.sha256(f"{sal}{ip}{dia.isoformat()}".encode()).hexdigest()


@dataclass(frozen=True)
class Consumo:
    """Resultado de `Contador.consumir`.

    `permitido` dice si se contó el análisis. Si no, `tope` dice cuál se alcanzó.
    `restantes_ip` es lo que le queda a esa IP hoy, contando este análisis si se permitió.
    """

    permitido: bool
    restantes_ip: int
    tope: Literal["ip", "global"] | None = None


class Contador(Protocol):
    """Interfaz del contador de uso diario."""

    def consumir(self, clave: str, dia: date, tope_ip: int, tope_global: int) -> Consumo:
        """Cuenta un análisis para `clave` en `dia` si no se pasa ningún tope.

        Debe ser atómico: dos peticiones a la vez no pueden pasar el tope. Si falla (la
        base de datos no responde), lanza una excepción y la API falla cerrada (503).
        """
        ...


class ContadorEnMemoria:
    """Contador en memoria del proceso, con un candado. Solo guarda el día en curso."""

    def __init__(self):
        self._candado = threading.Lock()
        self._dia: date | None = None
        self._por_clave: Counter[str] = Counter()
        self._global = 0

    def consumir(self, clave: str, dia: date, tope_ip: int, tope_global: int) -> Consumo:
        with self._candado:
            if dia != self._dia:
                self._dia, self._por_clave, self._global = dia, Counter(), 0
            usados = self._por_clave[clave]
            if usados >= tope_ip:
                return Consumo(permitido=False, restantes_ip=0, tope="ip")
            if self._global >= tope_global:
                return Consumo(permitido=False, restantes_ip=tope_ip - usados, tope="global")
            self._por_clave[clave] = usados + 1
            self._global += 1
            return Consumo(permitido=True, restantes_ip=tope_ip - usados - 1)


class ContadorFirestore:
    """Contador en Firestore, para producción (Cloud Run con varias instancias o reinicios).

    Modelo de datos (solo el hash y números, nunca la IP ni el texto):

    - `uso/{fecha}`: el contador global del día, p. ej. `uso/2026-11-03`.
    - `uso_ip/{clave}`: el contador de una IP en un día; `clave` es `clave_ip(...)`, que ya
      incluye la fecha.

    Los dos documentos tienen los campos `contador` (entero), `fecha` (`AAAA-MM-DD`, día
    de Colombia) y `expira` (medianoche de Colombia de `fecha + 2 días`), que usa la TTL.

    Atomicidad: `consumir` corre en una transacción de Firestore que lee los dos
    documentos, verifica los dos topes y solo entonces escribe los dos incrementos. Si
    otra petición cambió alguno entre la lectura y el commit, Firestore reintenta la
    transacción completa, así que dos peticiones a la vez no pueden pasar el tope.

    Falla cerrado: cualquier excepción de Firestore sale de `consumir` y la API responde
    503 sin analizar.

    TTL (se crea una vez, en #102): Firestore borra los documentos cuando pasa `expira`
    (con hasta 24 h de retraso, sin costo de escritura)::

        gcloud firestore fields ttls update expira --collection-group=uso \\
            --enable-ttl --database=BASE
        gcloud firestore fields ttls update expira --collection-group=uso_ip \\
            --enable-ttl --database=BASE

    (`--database` se omite si es la base `(default)`.)

    La librería `google-cloud-firestore` se importa aquí y no al cargar el módulo, para
    que el modo en memoria no la necesite.
    """

    COLECCION_GLOBAL = "uso"
    COLECCION_IP = "uso_ip"
    DIAS_TTL = 2

    def __init__(
        self,
        proyecto: str | None = None,
        base: str | None = None,
        cliente: Any = None,
        transaccional: Any = None,
        timeout_s: float = 10,
    ):
        """Crea el cliente de Firestore con las credenciales del entorno.

        Args:
            proyecto: el proyecto de GCP; por defecto, el del entorno.
            base: la base de Firestore; por defecto, `(default)`.
            cliente: un cliente ya creado (los tests inyectan uno falso).
            transaccional: el decorador de transacciones; por defecto
                `google.cloud.firestore.transactional`.
            timeout_s: segundos máximos de cada lectura antes de fallar.
        """
        if cliente is None or transaccional is None:
            from google.cloud import firestore

            if cliente is None:
                cliente = firestore.Client(project=proyecto, database=base)
            if transaccional is None:
                transaccional = firestore.transactional
        self._cliente = cliente
        self._timeout_s = timeout_s
        self._consumir_en = transaccional(self._consumir_en_transaccion)

    def consumir(self, clave: str, dia: date, tope_ip: int, tope_global: int) -> Consumo:
        ref_global = self._cliente.collection(self.COLECCION_GLOBAL).document(dia.isoformat())
        ref_ip = self._cliente.collection(self.COLECCION_IP).document(clave)
        return self._consumir_en(
            self._cliente.transaction(), ref_global, ref_ip, dia, tope_ip, tope_global
        )

    def _consumir_en_transaccion(
        self, transaccion, ref_global, ref_ip, dia: date, tope_ip: int, tope_global: int
    ) -> Consumo:
        # En una transacción de Firestore todas las lecturas van antes de las escrituras.
        usados = self._leer(ref_ip, transaccion)
        global_ = self._leer(ref_global, transaccion)
        if usados >= tope_ip:
            return Consumo(permitido=False, restantes_ip=0, tope="ip")
        if global_ >= tope_global:
            return Consumo(permitido=False, restantes_ip=tope_ip - usados, tope="global")
        transaccion.set(ref_ip, self._documento(usados + 1, dia))
        transaccion.set(ref_global, self._documento(global_ + 1, dia))
        return Consumo(permitido=True, restantes_ip=tope_ip - usados - 1)

    def _leer(self, ref, transaccion) -> int:
        documento = ref.get(transaction=transaccion, timeout=self._timeout_s)
        if not documento.exists:
            return 0
        return int((documento.to_dict() or {}).get("contador", 0))

    @classmethod
    def _documento(cls, contador: int, dia: date) -> dict:
        medianoche = datetime.combine(dia, datetime.min.time(), tzinfo=_COLOMBIA)
        return {
            "contador": contador,
            "fecha": dia.isoformat(),
            "expira": medianoche + timedelta(days=cls.DIAS_TTL),
        }
