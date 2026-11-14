"""Tests de la API (`ner_clinic_api`), sin red ni modelo.

Usan `ExtractorFalso` y `ContadorEnMemoria`; solo el test marcado `modelo` arranca la app
con el modelo real. Todas las notas son sintéticas.
"""

import hashlib
import logging
import threading
import time
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ner_clinic.extraccion import ExtractorFalso
from ner_clinic.limpieza import limpiar
from ner_clinic.segmentacion import segmentar
from ner_clinic_api import __main__ as principal
from ner_clinic_api import crear_app
from ner_clinic_api.app import NOTA_CALENTAMIENTO, VERSION
from ner_clinic_api.configuracion import Configuracion
from ner_clinic_api.contador import (
    Consumo,
    ContadorEnMemoria,
    clave_ip,
    hoy_en_colombia,
)

NOTA = (
    "Paciente femenina de 67 años con antecedente de HTA y diabetes mellitus tipo 2 en "
    "manejo con metformina. Niega EPOC. Madre con cáncer de mama."
)
PALABRAS = ("HTA", "diabetes mellitus tipo 2", "EPOC", "cáncer de mama")

# Cadena única que no debe aparecer en ningún log, salida ni mensaje de error.
CENTINELA = "Zq7CENTINELAxk94"
NOTA_CENTINELA = f"Paciente {CENTINELA} con HTA. Niega EPOC."


def salidas_para(nota: str, *palabras: str) -> dict:
    """Salidas de `ExtractorFalso` que marcan cada aparición de `palabras` en `nota`."""
    salidas = {}
    for fragmento in segmentar(limpiar(nota).texto):
        entidades = []
        for palabra in palabras:
            desde = 0
            while (inicio := fragmento.texto.find(palabra, desde)) != -1:
                fin = inicio + len(palabra)
                entidades.append(
                    {
                        "entity_group": "ENFERMEDAD",
                        "score": 0.99,
                        "word": palabra,
                        "start": inicio,
                        "end": fin,
                    }
                )
                desde = fin
        salidas[fragmento.texto] = sorted(entidades, key=lambda e: e["start"])
    return salidas


def extractor_falso() -> ExtractorFalso:
    salidas = {}
    for nota in (NOTA, NOTA_CENTINELA, NOTA_CALENTAMIENTO):
        salidas |= salidas_para(nota, *PALABRAS)
    return ExtractorFalso(salidas)


class Dia:
    """Fecha controlable para los contadores."""

    def __init__(self, dia: date = date(2026, 11, 3)):
        self.dia = dia

    def __call__(self) -> date:
        return self.dia


def configuracion(**cambios) -> Configuracion:
    valores = {"sal_ip": "sal-de-prueba", "timeout_s": 5, "normalizador": "reglas"} | cambios
    return Configuracion(**valores)


@pytest.fixture
def cliente():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion())
    with TestClient(app) as cliente:
        yield cliente


def analizar(cliente: TestClient, texto: str = NOTA, ip: str | None = None, **extra):
    cabeceras = {"x-forwarded-for": ip} if ip else {}
    return cliente.post("/api/analizar", json={"texto": texto, **extra}, headers=cabeceras)


# --- Rutas ---


def test_salud_con_el_modelo_cargado_y_caliente():
    extractor = extractor_falso()
    app = crear_app(extractor=extractor, configuracion=configuracion())
    with TestClient(app) as cliente:
        respuesta = cliente.get("/api/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "estado": "ok",
        "modelo_cargado": True,
        "mapeo": "reglas",
        "version": VERSION,
    }
    # El calentamiento corrió el extractor al arrancar, antes de cualquier petición.
    assert extractor.llamadas


def test_analizar_devuelve_el_resultado_y_meta(cliente):
    respuesta = analizar(cliente)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    estados = {e["texto"]: e["estado"] for e in datos["entidades"]}
    assert estados == {
        "HTA": "confirmada",
        "diabetes mellitus tipo 2": "confirmada",
        "EPOC": "negada",
        "cáncer de mama": "familiar",
    }
    assert datos["indice_charlson"] == 1
    assert set(datos) == {
        "entidades",
        "descartes",
        "indice_charlson",
        "indice_charlson_si_se_confirman",
        "meta",
    }
    assert datos["meta"]["version"] == VERSION
    assert datos["meta"]["segundos"] >= 0
    assert datos["meta"]["restantes_hoy"] == 19
    assert analizar(cliente).json()["meta"]["restantes_hoy"] == 18


def test_analizar_con_diagnosticos_registrados(cliente):
    respuesta = analizar(
        cliente, diagnosticos_registrados=["Diabetes mellitus no insulinodependiente"]
    )
    entidades = {e["texto"]: e for e in respuesta.json()["entidades"]}
    assert entidades["diabetes mellitus tipo 2"]["registrado"] is True
    assert entidades["HTA"]["registrado"] is False


def test_inicio_sirve_el_estatico_con_cabeceras_de_seguridad(cliente):
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/html")
    assert "No pegues datos reales" in respuesta.text
    assert "default-src 'self'" in respuesta.headers["content-security-policy"]
    assert "unsafe-inline" not in respuesta.headers["content-security-policy"]
    assert respuesta.headers["x-content-type-options"] == "nosniff"
    assert respuesta.headers["referrer-policy"] == "no-referrer"
    assert "cache-control" not in respuesta.headers


def test_api_no_se_guarda_en_cache_y_lleva_cabeceras(cliente):
    respuesta = analizar(cliente)
    assert respuesta.headers["cache-control"] == "no-store"
    assert respuesta.headers["x-content-type-options"] == "nosniff"


def test_sin_documentacion_interactiva_ni_cors(cliente):
    assert cliente.get("/docs").status_code == 404
    respuesta = analizar(cliente)
    assert "access-control-allow-origin" not in respuesta.headers


def test_ruta_desconocida_en_espanol(cliente):
    respuesta = cliente.get("/no-existe")
    assert respuesta.status_code == 404
    assert respuesta.json() == {"detalle": "No encontrado."}
    assert cliente.get("/api/analizar").json() == {"detalle": "Método no permitido."}


# --- Límites ---


def test_nota_muy_larga_responde_413_y_no_cuenta():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion(max_caracteres=100))
    with TestClient(app) as cliente:
        respuesta = analizar(cliente, texto="x" * 101)
        assert respuesta.status_code == 413
        assert respuesta.json() == {
            "detalle": "La nota es muy larga: el máximo es de 100 caracteres."
        }
        assert analizar(cliente, texto="Paciente con HTA.").json()["meta"]["restantes_hoy"] == 19


def test_cuerpo_enorme_responde_413_sin_leerlo():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion(max_caracteres=10))
    with TestClient(app) as cliente:
        cuerpo = b'{"texto": "' + b"x" * 300_000 + b'"}'
        respuesta = cliente.post(
            "/api/analizar", content=cuerpo, headers={"content-type": "application/json"}
        )
    assert respuesta.status_code == 413
    assert "máximo es de 10 caracteres" in respuesta.json()["detalle"]


def test_cuerpo_enorme_sin_content_length_responde_413():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion(max_caracteres=10))

    def trozos():
        yield b'{"texto": "'
        for _ in range(20):
            yield b"x" * 10_000
        yield b'"}'

    with TestClient(app) as cliente:
        respuesta = cliente.post(
            "/api/analizar", content=trozos(), headers={"content-type": "application/json"}
        )
    assert respuesta.status_code == 413


def test_content_length_invalido_no_rompe(cliente):
    respuesta = cliente.get("/api/salud", headers={"content-length": "abc"})
    assert respuesta.status_code == 200


@pytest.mark.parametrize(
    "cuerpo",
    [
        {},
        {"texto": 5},
        {"texto": NOTA, "otro": 1},
        {"texto": NOTA, "diagnosticos_registrados": "no es lista"},
        {"texto": NOTA, "diagnosticos_registrados": ["x"] * 51},
    ],
)
def test_peticion_invalida_responde_422_en_espanol_y_no_cuenta(cliente, cuerpo):
    respuesta = cliente.post("/api/analizar", json=cuerpo)
    assert respuesta.status_code == 422
    datos = respuesta.json()
    assert datos["detalle"].startswith("La petición no es válida")
    assert all(set(e) == {"campo", "tipo"} for e in datos["errores"])
    assert analizar(cliente).json()["meta"]["restantes_hoy"] == 19


def test_422_no_repite_claves_extra_del_usuario(cliente):
    respuesta = cliente.post("/api/analizar", json={"texto": "x", CENTINELA: 1})
    assert respuesta.status_code == 422
    assert CENTINELA not in respuesta.text
    assert respuesta.json()["errores"] == [{"campo": ["body", "?"], "tipo": "extra_forbidden"}]


def test_json_invalido_responde_422(cliente):
    respuesta = cliente.post(
        "/api/analizar", content=b"{no es json", headers={"content-type": "application/json"}
    )
    assert respuesta.status_code == 422


def test_tope_por_ip_responde_429():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion(tope_ip_dia=2))
    with TestClient(app) as cliente:
        assert analizar(cliente, ip="1.1.1.1").json()["meta"]["restantes_hoy"] == 1
        # Cuenta el último valor de X-Forwarded-For (el que agrega Cloud Run).
        assert analizar(cliente, ip="9.9.9.9, 1.1.1.1").json()["meta"]["restantes_hoy"] == 0
        respuesta = analizar(cliente, ip="1.1.1.1")
        assert respuesta.status_code == 429
        assert respuesta.json() == {
            "detalle": "Llegaste al máximo de 2 análisis por día. Vuelve mañana."
        }
        assert analizar(cliente, ip="2.2.2.2").status_code == 200


def test_tope_global_responde_429():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion(tope_global_dia=2))
    with TestClient(app) as cliente:
        assert analizar(cliente, ip="1.1.1.1").status_code == 200
        assert analizar(cliente, ip="2.2.2.2").status_code == 200
        respuesta = analizar(cliente, ip="3.3.3.3")
    assert respuesta.status_code == 429
    assert "La demo llegó al máximo" in respuesta.json()["detalle"]


def test_el_tope_se_reinicia_al_otro_dia():
    dia = Dia()
    app = crear_app(
        extractor=extractor_falso(), configuracion=configuracion(tope_ip_dia=1), hoy=dia
    )
    with TestClient(app) as cliente:
        assert analizar(cliente).status_code == 200
        assert analizar(cliente).status_code == 429
        dia.dia = date(2026, 11, 4)
        assert analizar(cliente).status_code == 200


def test_sin_x_forwarded_for_usa_la_ip_de_la_conexion():
    contador = ContadorEnMemoria()
    claves = []
    consumir = contador.consumir

    def espia(clave, *args):
        claves.append(clave)
        return consumir(clave, *args)

    contador.consumir = espia
    dia = Dia()
    app = crear_app(
        extractor=extractor_falso(), contador=contador, configuracion=configuracion(), hoy=dia
    )
    with TestClient(app) as cliente:
        analizar(cliente)
        analizar(cliente, ip=" 5.5.5.5 , 6.6.6.6 ")
    assert claves == [
        clave_ip("sal-de-prueba", "testclient", dia.dia),
        clave_ip("sal-de-prueba", "6.6.6.6", dia.dia),
    ]


def _app_que_registra_ips(proxies_confiables: int = 1, **cambios):
    """App cuyo contador guarda la clave de cada análisis, y la lista de claves."""
    contador = ContadorEnMemoria()
    claves = []
    consumir = contador.consumir

    def espia(clave, *args):
        claves.append(clave)
        return consumir(clave, *args)

    contador.consumir = espia
    config = configuracion(proxies_confiables=proxies_confiables, **cambios)
    return crear_app(extractor=extractor_falso(), contador=contador, configuracion=config), claves


def clave(ip: str) -> str:
    return clave_ip("sal-de-prueba", ip, hoy_en_colombia())


def test_x_forwarded_for_falsificado_cuenta_como_la_ip_real():
    app, claves = _app_que_registra_ips()
    with TestClient(app) as cliente:
        analizar(cliente, ip="falsa1, real")
    assert claves == [clave("real")]


def test_falsificar_x_forwarded_for_no_salta_el_tope_por_ip():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion())
    with TestClient(app) as cliente:
        respuestas = [analizar(cliente, ip=f"falsa{i}, 8.8.8.8") for i in range(21)]
    assert [r.status_code for r in respuestas[:20]] == [200] * 20
    assert respuestas[19].json()["meta"]["restantes_hoy"] == 0
    assert respuestas[20].status_code == 429


@pytest.mark.parametrize(
    ("cabecera", "esperada"),
    [
        ("falsa, real, balanceador", "real"),
        ("real, balanceador", "real"),
        # Menos valores de los esperados: la IP de la conexión.
        ("solo-uno", "testclient"),
        # Valor vacío en la posición: la IP de la conexión.
        (", balanceador", "testclient"),
    ],
)
def test_con_dos_proxies_confiables_toma_el_penultimo(cabecera, esperada):
    app, claves = _app_que_registra_ips(proxies_confiables=2)
    with TestClient(app) as cliente:
        analizar(cliente, ip=cabecera)
    assert claves == [clave(esperada)]


def test_si_el_contador_falla_responde_503_y_no_analiza():
    class ContadorRoto:
        def consumir(self, *args):
            raise ConnectionError("Firestore no responde")

    extractor = extractor_falso()
    app = crear_app(extractor=extractor, contador=ContadorRoto(), configuracion=configuracion())
    with TestClient(app) as cliente:
        llamadas = len(extractor.llamadas)
        respuesta = analizar(cliente)
    assert respuesta.status_code == 503
    assert respuesta.json() == {"detalle": "El servicio no está disponible. Intenta más tarde."}
    assert len(extractor.llamadas) == llamadas


class ExtractorLento(ExtractorFalso):
    """Se bloquea hasta `soltar` después del calentamiento."""

    def __init__(self):
        super().__init__(extractor_falso().salidas)
        self.calentado = False
        self.dentro = threading.Event()
        self.soltar = threading.Event()

    def extraer(self, fragmentos):
        if self.calentado:
            self.dentro.set()
            self.soltar.wait(10)
        return super().extraer(fragmentos)


def test_una_inferencia_a_la_vez_y_503_si_la_espera_pasa_el_timeout():
    extractor = ExtractorLento()
    app = crear_app(extractor=extractor, configuracion=configuracion(timeout_s=0.2))
    with TestClient(app) as cliente:
        extractor.calentado = True
        respuestas = {}
        hilo = threading.Thread(target=lambda: respuestas.update(primera=analizar(cliente)))
        hilo.start()
        assert extractor.dentro.wait(5)
        segunda = analizar(cliente, ip="7.7.7.7")
        extractor.soltar.set()
        hilo.join(10)
        tercera = analizar(cliente, ip="7.7.7.7")

    assert segunda.status_code == 503
    assert segunda.json() == {
        "detalle": "Hay mucha demanda en este momento. Intenta en unos segundos."
    }
    assert respuestas["primera"].status_code == 200
    # La 503 no contó: a esa IP le quedan los mismos análisis que a una nueva.
    assert tercera.json()["meta"]["restantes_hoy"] == 19


class ExtractorQueFalla(ExtractorFalso):
    """Falla después del calentamiento, con un mensaje que repite el texto recibido."""

    def __init__(self):
        super().__init__(extractor_falso().salidas)
        self.calentado = False

    def extraer(self, fragmentos):
        if self.calentado:
            raise ValueError(f"fallo procesando {fragmentos}")
        return super().extraer(fragmentos)


def test_error_interno_responde_500_sin_detalles(caplog):
    extractor = ExtractorQueFalla()
    app = crear_app(extractor=extractor, configuracion=configuracion())
    with TestClient(app) as cliente:
        extractor.calentado = True
        with caplog.at_level(logging.DEBUG):
            respuesta = analizar(cliente)
        # El turno de inferencia se libera aunque falle.
        extractor.calentado = False
        assert analizar(cliente).status_code == 200
    assert respuesta.status_code == 500
    assert respuesta.json() == {"detalle": "Error interno. Intenta de nuevo más tarde."}
    assert "Error no controlado: ValueError" in caplog.text
    assert "fallo procesando" not in caplog.text


# --- Logs y privacidad ---


def test_log_de_una_peticion_solo_lleva_metodo_ruta_codigo_duracion_y_longitud(cliente, caplog):
    with caplog.at_level(logging.INFO, logger="ner_clinic_api"):
        analizar(cliente)
        cliente.get("/api/salud")
        cliente.get(f"/{CENTINELA}")
        cliente.get(f"/estatico/{CENTINELA}.js")
    lineas = [r.getMessage() for r in caplog.records if r.name == "ner_clinic_api"]
    assert lineas[0].startswith("POST /api/analizar 200 ")
    assert lineas[0].endswith(f" ms longitud={len(NOTA)}")
    assert lineas[1].startswith("GET /api/salud 200 ")
    assert lineas[1].endswith("longitud=-")
    # Las rutas desconocidas las escribe el usuario: no se registran.
    assert lineas[2].startswith("GET (otra) 404 ")
    assert lineas[3].startswith("GET /estatico/* 404 ")
    assert CENTINELA not in caplog.text


def test_el_access_log_de_uvicorn_queda_apagado(cliente):
    assert logging.getLogger("uvicorn.access").disabled


def test_main_corre_uvicorn_sin_access_log(monkeypatch):
    argumentos = {}
    monkeypatch.setattr(principal.uvicorn, "run", lambda app, **kw: argumentos.update(kw, app=app))
    monkeypatch.setenv("PORT", "9000")
    principal.main()
    assert argumentos == {
        "app": "ner_clinic_api.app:app",
        "host": "127.0.0.1",
        "port": 9000,
        "access_log": False,
    }


def _escenarios_centinela():
    """(nombre, configuracion, extractor, peticion) para cada código de respuesta."""
    nota = {"texto": NOTA_CENTINELA}
    return {
        200: (configuracion(), extractor_falso(), nota),
        413: (configuracion(max_caracteres=10), extractor_falso(), nota),
        # El manejador por defecto de FastAPI devolvería la lista (con la nota) en `input`.
        422: (configuracion(), extractor_falso(), {"texto": [NOTA_CENTINELA]}),
        429: (configuracion(tope_ip_dia=1), extractor_falso(), nota),
        500: (configuracion(), ExtractorQueFalla(), nota),
    }


@pytest.mark.parametrize("codigo", [200, 413, 422, 429, 500])
def test_centinela_el_texto_no_aparece_en_logs_ni_salidas(codigo, caplog, capsys):
    """Analiza una nota con una cadena única y la busca en todos los logs y salidas."""
    config, extractor, peticion = _escenarios_centinela()[codigo]
    caplog.set_level(logging.DEBUG)
    app = crear_app(extractor=extractor, configuracion=config)
    with TestClient(app) as cliente:
        extractor.calentado = True
        if codigo == 429:
            cliente.post("/api/analizar", json=peticion)
        respuesta = cliente.post("/api/analizar", json=peticion)
        # El cuerpo crudo como texto inválido también da 422 y tampoco debe filtrarse.
        cliente.post(
            "/api/analizar",
            content=f'{{"texto": "{CENTINELA}"'.encode(),
            headers={"content-type": "application/json"},
        )

    assert respuesta.status_code == codigo
    if codigo != 200:
        assert CENTINELA not in respuesta.text
    salida = capsys.readouterr()
    assert CENTINELA not in caplog.text
    assert all(CENTINELA not in str(r.args) for r in caplog.records)
    assert CENTINELA not in salida.out
    assert CENTINELA not in salida.err


# --- Configuración y contador ---


def test_configuracion_por_defecto_desde_el_entorno():
    config = Configuracion.desde_entorno({})
    assert (config.max_caracteres, config.timeout_s, config.tope_ip_dia) == (5000, 30, 20)
    assert (config.tope_global_dia, config.proxies_confiables) == (200, 1)
    assert len(config.sal_ip) == 32
    assert config.sal_ip != Configuracion.desde_entorno({}).sal_ip
    assert "sal_ip" not in repr(config)


def test_configuracion_desde_variables_de_entorno():
    config = Configuracion.desde_entorno(
        {
            "MAX_CARACTERES": "3000",
            "TIMEOUT_S": "10",
            "TOPE_IP_DIA": "5",
            "TOPE_GLOBAL_DIA": "50",
            "PROXIES_CONFIABLES": "2",
            "SAL_IP": "secreta",
            "OTRA": "x",
        }
    )
    assert config == Configuracion(3000, 10, 5, 50, 2, "secreta")


@pytest.mark.parametrize("variable", ["TOPE_IP_DIA", "PROXIES_CONFIABLES"])
@pytest.mark.parametrize("valor", ["abc", "0", "-3"])
def test_configuracion_rechaza_valores_invalidos(variable, valor):
    with pytest.raises(ValueError, match=f"{variable} debe ser un entero positivo"):
        Configuracion.desde_entorno({variable: valor})


def test_configuracion_lee_os_environ(monkeypatch):
    monkeypatch.setenv("TOPE_GLOBAL_DIA", "7")
    assert Configuracion.desde_entorno().tope_global_dia == 7


def test_clave_ip_es_sha256_de_sal_ip_y_fecha():
    dia = date(2026, 11, 3)
    esperada = hashlib.sha256(b"sal1.2.3.42026-11-03").hexdigest()
    assert clave_ip("sal", "1.2.3.4", dia) == esperada
    assert clave_ip("sal", "1.2.3.4", date(2026, 11, 4)) != esperada
    assert clave_ip("otra", "1.2.3.4", dia) != esperada


def test_contador_en_memoria():
    contador = ContadorEnMemoria()
    dia = date(2026, 11, 3)
    assert contador.consumir("a", dia, 2, 3) == Consumo(True, 1)
    assert contador.consumir("a", dia, 2, 3) == Consumo(True, 0)
    assert contador.consumir("a", dia, 2, 3) == Consumo(False, 0, "ip")
    assert contador.consumir("b", dia, 2, 3) == Consumo(True, 1)
    assert contador.consumir("c", dia, 2, 3) == Consumo(False, 2, "global")
    # Al cambiar de día se olvida el anterior.
    assert contador.consumir("a", date(2026, 11, 4), 2, 3) == Consumo(True, 1)


def test_contador_en_memoria_es_atomico_entre_hilos():
    contador = ContadorEnMemoria()
    dia = date(2026, 11, 3)
    permitidos = []

    def consumir():
        for _ in range(50):
            permitidos.append(contador.consumir("a", dia, 1000, 120).permitido)
            time.sleep(0)

    hilos = [threading.Thread(target=consumir) for _ in range(8)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    assert sum(permitidos) == 120


def test_hoy_en_colombia_es_una_fecha():
    assert isinstance(hoy_en_colombia(), date)


def test_la_libreria_no_importa_fastapi():
    import subprocess
    import sys

    codigo = "import sys, ner_clinic; print('fastapi' in sys.modules)"
    salida = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
    )
    assert salida.stdout.strip() == "False"


# --- Modelo real ---


@pytest.mark.modelo
def test_api_con_el_modelo_real_y_la_nota_del_readme():
    app = crear_app(configuracion=configuracion())
    with TestClient(app) as cliente:
        assert cliente.get("/api/salud").json()["modelo_cargado"] is True
        respuesta = analizar(
            cliente, diagnosticos_registrados=["Diabetes mellitus no insulinodependiente"]
        )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    estados = {e["texto"]: e["estado"] for e in datos["entidades"]}
    assert estados["diabetes mellitus tipo 2"] == "confirmada"
    assert estados["EPOC"] == "negada"
    assert estados["cáncer de mama"] == "familiar"
    assert datos["indice_charlson"] == 1
    assert datos["meta"]["restantes_hoy"] == 19


def test_analizar_devuelve_la_categoria_por_correccion():
    """#116: la API trae las correcciones de ortografía que dieron la categoría (#115)."""
    nota = "Paciente con diabetis mellitus tipo 2 en manejo."
    extractor = ExtractorFalso(
        salidas_para(nota, "diabetis mellitus tipo 2") | salidas_para(NOTA_CALENTAMIENTO, *PALABRAS)
    )
    app = crear_app(extractor=extractor, configuracion=configuracion())
    with TestClient(app) as cliente:
        (entidad,) = analizar(cliente, nota).json()["entidades"]
    assert entidad["categoria_charlson"] == "diabetes"
    assert entidad["categoria_por_correccion"] is True
    assert entidad["correcciones"] == [{"escrita": "diabetis", "corregida": "diabetes"}]


# --- Carga en segundo plano (la app de producción) ---


class ExtractorQueEspera(ExtractorFalso):
    """`ExtractorFalso` cuya primera llamada (el calentamiento) espera una señal."""

    def __init__(self, salidas: dict, falla: bool = False):
        super().__init__(salidas)
        self.soltar = threading.Event()
        self.falla = falla

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        if not self.llamadas:
            assert self.soltar.wait(5), "el test no soltó la carga"
            if self.falla:
                self.llamadas.append(list(fragmentos))
                raise FileNotFoundError("falta el modelo")
        return super().extraer(fragmentos)


def app_que_carga_detras(falla: bool = False, contador=None):
    extractor = ExtractorQueEspera(extractor_falso().salidas, falla=falla)
    app = crear_app(
        extractor=extractor,
        configuracion=configuracion(),
        contador=contador,
        cargar_en_segundo_plano=True,
    )
    return app, extractor


def test_mientras_carga_la_pagina_responde_y_analizar_da_503_sin_contar():
    contador = ContadorEnMemoria()
    app, extractor = app_que_carga_detras(contador=contador)
    with TestClient(app) as cliente:
        # El puerto ya está abierto: la página y la salud responden sin esperar al modelo.
        assert cliente.get("/").status_code == 200
        salud = cliente.get("/api/salud")
        assert salud.status_code == 200
        assert salud.json()["estado"] == "cargando"
        assert salud.json()["modelo_cargado"] is False

        respuesta = analizar(cliente)
        assert respuesta.status_code == 503
        assert "despertando" in respuesta.json()["detalle"]
        assert respuesta.json()["estado_modelo"] == "cargando"

        extractor.soltar.set()
        salud = cliente.get("/api/salud", params={"esperar": 5}).json()
        assert salud["estado"] == "ok" and salud["modelo_cargado"] is True

        respuesta = analizar(cliente)
        assert respuesta.status_code == 200
        # El 503 de la carga no gastó del tope: este es el primer análisis del día.
        assert respuesta.json()["meta"]["restantes_hoy"] == 19


def test_salud_con_esperar_vuelve_aunque_no_termine_la_carga():
    app, extractor = app_que_carga_detras()
    with TestClient(app) as cliente:
        inicio = time.perf_counter()
        salud = cliente.get("/api/salud", params={"esperar": 0.2}).json()
        assert time.perf_counter() - inicio < 2
        assert salud["estado"] == "cargando"
        extractor.soltar.set()


def test_si_la_carga_falla_salud_da_503_y_analizar_tambien(caplog):
    app, extractor = app_que_carga_detras(falla=True)
    extractor.soltar.set()
    with caplog.at_level(logging.ERROR), TestClient(app) as cliente:
        salud = cliente.get("/api/salud", params={"esperar": 5})
        assert salud.status_code == 503
        assert salud.json()["estado"] == "error"
        respuesta = analizar(cliente)
        assert respuesta.status_code == 503
        assert "no está disponible" in respuesta.json()["detalle"]
        assert respuesta.json()["estado_modelo"] == "error"
    assert "FileNotFoundError" in caplog.text


def test_la_app_de_produccion_carga_en_segundo_plano():
    """`app` (la que corre uvicorn) no puede esperar al modelo para abrir el puerto."""
    from ner_clinic_api import app as modulo

    fuente = Path(modulo.__file__).read_text(encoding="utf-8")
    assert "\napp = crear_app(cargar_en_segundo_plano=True)\n" in fuente
