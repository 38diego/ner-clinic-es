"""Tests de `ContadorFirestore` sin red: un Firestore falso en memoria.

`FirestoreFalso` imita lo que usa el contador: colecciones, documentos, lecturas dentro de
una transacción y el decorador `transactional`, con el control optimista de Firestore (si
un documento leído cambió antes del commit, la transacción se reintenta completa).

El test marcado `firestore` usa el emulador real y se excluye por defecto; se corre con
`FIRESTORE_EMULATOR_HOST=localhost:8080 pytest -m firestore`.
"""

import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from ner_clinic_api import crear_app
from ner_clinic_api.app import _contador_configurado
from ner_clinic_api.configuracion import Configuracion
from ner_clinic_api.contador import Consumo, ContadorEnMemoria, ContadorFirestore, clave_ip
from tests.test_api import analizar, configuracion, extractor_falso

DIA = date(2026, 11, 3)
COLOMBIA = timezone(timedelta(hours=-5))


class Conflicto(Exception):
    """Un documento leído cambió antes del commit."""


class Instantanea:
    def __init__(self, datos: dict | None):
        self.exists = datos is not None
        self._datos = datos

    def to_dict(self) -> dict | None:
        return None if self._datos is None else dict(self._datos)


class Referencia:
    def __init__(self, base: "FirestoreFalso", ruta: tuple[str, str]):
        self._base = base
        self.ruta = ruta

    def get(self, transaction=None, timeout=None) -> Instantanea:
        self._base.lecturas.append((self.ruta, timeout))
        if self._base.error is not None:
            raise self._base.error
        with self._base.candado:
            datos, version = self._base.documentos.get(self.ruta, (None, 0))
        if transaction is not None:
            transaction.leidos.setdefault(self.ruta, version)
        return Instantanea(datos)


class Coleccion:
    def __init__(self, base: "FirestoreFalso", nombre: str):
        self._base = base
        self._nombre = nombre

    def document(self, id_: str) -> Referencia:
        return Referencia(self._base, (self._nombre, id_))


class Transaccion:
    def __init__(self, base: "FirestoreFalso"):
        self._base = base
        self.leidos: dict[tuple[str, str], int] = {}
        self.escrituras: list[tuple[tuple[str, str], dict]] = []

    def set(self, referencia: Referencia, datos: dict) -> None:
        self.escrituras.append((referencia.ruta, dict(datos)))

    def commit(self) -> None:
        with self._base.candado:
            for ruta, version in self.leidos.items():
                if self._base.documentos.get(ruta, (None, 0))[1] != version:
                    raise Conflicto
            for ruta, datos in self.escrituras:
                version = self._base.documentos.get(ruta, (None, 0))[1]
                self._base.documentos[ruta] = (datos, version + 1)


class FirestoreFalso:
    """Cliente falso de Firestore en memoria, seguro entre hilos."""

    def __init__(self, error: Exception | None = None):
        self.candado = threading.Lock()
        self.documentos: dict[tuple[str, str], tuple[dict, int]] = {}
        self.lecturas: list = []
        self.reintentos = 0
        self.error = error

    def collection(self, nombre: str) -> Coleccion:
        return Coleccion(self, nombre)

    def transaction(self) -> Transaccion:
        return Transaccion(self)

    def transactional(self, funcion):
        """Como `firestore.transactional`: reintenta la función completa si hay conflicto."""

        def envuelta(transaccion: Transaccion, *args):
            while True:
                resultado = funcion(transaccion, *args)
                try:
                    transaccion.commit()
                except Conflicto:
                    self.reintentos += 1
                    transaccion = self.transaction()
                    continue
                return resultado

        return envuelta

    def datos(self, coleccion: str, id_: str) -> dict | None:
        return self.documentos.get((coleccion, id_), (None, 0))[0]


def contador_falso(error: Exception | None = None) -> tuple[ContadorFirestore, FirestoreFalso]:
    base = FirestoreFalso(error)
    return ContadorFirestore(cliente=base, transaccional=base.transactional), base


def test_misma_semantica_que_el_contador_en_memoria():
    contador, _ = contador_falso()
    memoria = ContadorEnMemoria()
    for clave in ["a", "a", "a", "b", "c"]:
        assert contador.consumir(clave, DIA, 2, 3) == memoria.consumir(clave, DIA, 2, 3)
    assert contador.consumir("c", DIA, 2, 3) == Consumo(False, 2, "global")
    # Otro día es otro documento global: vuelve a haber cupo.
    assert contador.consumir("c2", date(2026, 11, 4), 2, 3) == Consumo(True, 1)


def test_modelo_de_datos_solo_hash_contador_fecha_y_expira():
    contador, base = contador_falso()
    clave = clave_ip("sal", "1.2.3.4", DIA)
    contador.consumir(clave, DIA, 20, 200)
    contador.consumir(clave, DIA, 20, 200)

    expira = datetime(2026, 11, 5, tzinfo=COLOMBIA)
    assert base.datos("uso_ip", clave) == {"contador": 2, "fecha": "2026-11-03", "expira": expira}
    assert base.datos("uso", "2026-11-03") == {
        "contador": 2,
        "fecha": "2026-11-03",
        "expira": expira,
    }
    assert set(base.documentos) == {("uso_ip", clave), ("uso", "2026-11-03")}
    assert "1.2.3.4" not in repr(base.documentos)


def test_si_no_cabe_no_escribe_nada():
    contador, base = contador_falso()
    contador.consumir("a", DIA, 1, 5)
    antes = dict(base.documentos)
    assert contador.consumir("a", DIA, 1, 5) == Consumo(False, 0, "ip")
    assert base.documentos == antes

    contador, base = contador_falso()
    contador.consumir("a", DIA, 5, 1)
    antes = dict(base.documentos)
    assert contador.consumir("b", DIA, 5, 1) == Consumo(False, 5, "global")
    assert base.documentos == antes
    assert base.datos("uso_ip", "b") is None


def test_las_lecturas_llevan_timeout():
    contador, base = contador_falso()
    contador.consumir("a", DIA, 2, 3)
    assert [timeout for _, timeout in base.lecturas] == [10, 10]


def test_es_atomico_entre_hilos_con_reintentos():
    contador, base = contador_falso()
    permitidos = []

    def consumir(i: int):
        for _ in range(30):
            permitidos.append(contador.consumir(f"ip{i % 3}", DIA, 1000, 100).permitido)
            time.sleep(0)

    hilos = [threading.Thread(target=consumir, args=(i,)) for i in range(8)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    assert sum(permitidos) == 100
    assert base.datos("uso", "2026-11-03")["contador"] == 100
    assert sum(base.datos("uso_ip", f"ip{i}")["contador"] for i in range(3)) == 100


def test_conflicto_reintenta_y_no_pasa_el_tope():
    """Otra petición escribe entre la lectura y el commit: se reintenta con el valor nuevo."""
    contador, base = contador_falso()
    original = Transaccion.commit
    intrusa = {"pendiente": True}

    def commit_con_intrusa(transaccion):
        if intrusa["pendiente"]:
            intrusa["pendiente"] = False
            otra = base.transaction()
            base.transactional(contador._consumir_en_transaccion)(
                otra,
                base.collection("uso").document("2026-11-03"),
                base.collection("uso_ip").document("b"),
                DIA,
                5,
                1,
            )
        original(transaccion)

    Transaccion.commit = commit_con_intrusa
    try:
        assert contador.consumir("a", DIA, 5, 1) == Consumo(False, 5, "global")
    finally:
        Transaccion.commit = original
    assert base.reintentos == 1
    assert base.datos("uso", "2026-11-03")["contador"] == 1
    assert base.datos("uso_ip", "a") is None


def test_si_firestore_falla_la_api_responde_503_y_no_analiza():
    contador, _ = contador_falso(ConnectionError("Firestore no responde"))
    with pytest.raises(ConnectionError):
        contador.consumir("a", DIA, 2, 3)

    extractor = extractor_falso()
    app = crear_app(extractor=extractor, contador=contador, configuracion=configuracion())
    with TestClient(app) as cliente:
        llamadas = len(extractor.llamadas)
        respuesta = analizar(cliente)
    assert respuesta.status_code == 503
    assert respuesta.json() == {"detalle": "El servicio no está disponible. Intenta más tarde."}
    assert len(extractor.llamadas) == llamadas


def test_api_con_firestore_cuenta_y_responde_429():
    contador, base = contador_falso()
    app = crear_app(
        extractor=extractor_falso(), contador=contador, configuracion=configuracion(tope_ip_dia=2)
    )
    with TestClient(app) as cliente:
        assert analizar(cliente, ip="1.1.1.1").json()["meta"]["restantes_hoy"] == 1
        assert analizar(cliente, ip="1.1.1.1").json()["meta"]["restantes_hoy"] == 0
        assert analizar(cliente, ip="1.1.1.1").status_code == 429
    assert sum(d["contador"] for (col, _), (d, _) in base.documentos.items() if col == "uso") == 2


# --- Configuración ---


def test_contador_por_defecto_es_memoria():
    assert Configuracion.desde_entorno({}).contador == "memoria"
    assert isinstance(_contador_configurado(Configuracion()), ContadorEnMemoria)


def test_configuracion_de_firestore():
    config = Configuracion.desde_entorno(
        {"CONTADOR": " Firestore ", "FIRESTORE_PROYECTO": "proyecto", "FIRESTORE_BASE": "base"}
    )
    assert (config.contador, config.firestore_proyecto, config.firestore_base) == (
        "firestore",
        "proyecto",
        "base",
    )
    config = Configuracion.desde_entorno({"CONTADOR": "firestore"})
    assert (config.firestore_proyecto, config.firestore_base) == (None, None)


def test_configuracion_rechaza_otro_contador():
    with pytest.raises(ValueError, match="CONTADOR debe ser `memoria` o `firestore`"):
        Configuracion.desde_entorno({"CONTADOR": "redis"})


def test_con_firestore_crea_el_cliente_con_proyecto_y_base(monkeypatch):
    from google.cloud import firestore

    creados = []

    class ClienteFalso:
        def __init__(self, project=None, database=None):
            creados.append((project, database))

    monkeypatch.setattr(firestore, "Client", ClienteFalso)
    config = Configuracion(contador="firestore", firestore_proyecto="p", firestore_base="b")
    contador = _contador_configurado(config)
    assert isinstance(contador, ContadorFirestore)
    assert isinstance(contador._cliente, ClienteFalso)
    assert creados == [("p", "b")]


def test_el_modo_memoria_no_importa_firestore():
    codigo = (
        "import sys\n"
        "from ner_clinic_api.app import _contador_configurado\n"
        "from ner_clinic_api.configuracion import Configuracion\n"
        "_contador_configurado(Configuracion.desde_entorno({}))\n"
        "print('google.cloud.firestore' in sys.modules)\n"
    )
    entorno = {k: v for k, v in os.environ.items() if k != "CONTADOR"}
    salida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True, env=entorno
    )
    assert salida.stdout.strip().splitlines()[-1] == "False"


# --- Emulador ---


@pytest.mark.firestore
def test_con_el_emulador_de_firestore():
    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):
        pytest.skip("Falta FIRESTORE_EMULATOR_HOST")
    contador = ContadorFirestore(proyecto="demo-ner-clinic")
    dia = date(2000, 1, int(time.time()) % 28 + 1)
    clave = f"prueba-{time.time_ns()}"
    tope_global = 10_000
    assert contador.consumir(clave, dia, 2, tope_global).restantes_ip == 1
    assert contador.consumir(clave, dia, 2, tope_global) == Consumo(True, 0)
    assert contador.consumir(clave, dia, 2, tope_global) == Consumo(False, 0, "ip")
    documento = contador._cliente.collection("uso_ip").document(clave).get().to_dict()
    assert documento["contador"] == 2
    assert documento["fecha"] == dia.isoformat()
