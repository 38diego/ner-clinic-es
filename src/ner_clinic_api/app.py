"""La app FastAPI: `crear_app()` arma la API y `app` es la instancia que corre uvicorn.

Rutas:
- `POST /api/analizar`: `{texto, diagnosticos_registrados?}` → `Resultado.to_dict()` más
  `meta` (`version`, `segundos`, `restantes_hoy`, `mapeo`). El esquema está en el OpenAPI
  (`esquema.py`).
- `GET /api/salud`: estado del servicio (`cargando`, `ok` o `error`), si el modelo está
  cargado y el `mapeo` activo (`reglas`, `normalizador` o `hibrido`). Con `?esperar=N`
  espera hasta N segundos (máx. 25) a que el modelo termine de cargar.
- `GET /`: el front estático (`estatico/index.html`).
- `GET /estatico/*`: el JS, el CSS, las fuentes y el ícono del front.
- `GET /favicon.ico`: el ícono (`estatico/icono.svg`), que los navegadores piden solos.

Orden de un análisis:
1. Pydantic valida el cuerpo (422) y se mira el tamaño de la nota (413). Nada de eso
   cuenta como uso.
2. Se espera el turno de inferencia (`asyncio.Semaphore(1)`), como mucho `timeout_s`
   segundos; si no llega, 503. La inferencia es CPU pura: dos a la vez solo la hacen
   más lenta.
3. Ya con el turno, el `Contador` mira los topes y cuenta el análisis (429 si no cabe;
   503 si el contador falla, para no gastar sin control).
4. `analizar()` corre en un hilo, para no bloquear el bucle de eventos.

Arranque: la app de producción (`app`) abre el puerto de una vez y carga el modelo en
segundo plano, para que la página salga en un par de segundos y avise mientras el modelo
despierta (en frío tarda unos 40 s; antes Cloud Run se rendía a los ~20 s y respondía 500).
Mientras carga, `/api/analizar` responde 503 sin contar el uso. El front espera con
`/api/salud?esperar=25`: esa petición abierta hace que Cloud Run no le quite la CPU a la
carga (el servicio corre con `--cpu-throttling`). Si la carga falla, el estado queda en
`error` y `/api/salud` responde 503. `crear_app()` sin `cargar_en_segundo_plano` carga
antes de abrir el puerto, como lo esperan los tests.

Por defecto (`NER_NORMALIZADOR=hibrido`, #182) el normalizador (índice CIE-10,
clasificador y SapBERT) se carga al arrancar, junto con el modelo NER, y decide lo que las
reglas no mapean. Si no está su índice, la app arranca con las reglas y lo registra.
Con `sapbert` decide solo el normalizador y, si falta un artefacto, la app no arranca; con
`reglas` no lo carga. No guarda las notas entre análisis.

El timeout solo limita la espera del turno: una inferencia que ya empezó no se puede
abortar (torch no se interrumpe desde otro hilo) y termina aunque tarde más. Por eso el
tamaño máximo de la nota es lo que acota su duración.

Privacidad: `MiddlewarePrivado` registra solo método, ruta, código, duración y longitud
del texto; atrapa toda excepción y la registra por su tipo, sin mensaje ni traza (el
mensaje podría traer el texto). Los errores 422 se responden sin el `input` que FastAPI
devuelve por defecto, y el access log de uvicorn se apaga al arrancar.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ner_clinic import analizar
from ner_clinic.extraccion import Extractor, ExtractorHF
from ner_clinic_api.configuracion import Configuracion
from ner_clinic_api.contador import (
    Contador,
    ContadorEnMemoria,
    ContadorFirestore,
    clave_ip,
    hoy_en_colombia,
)
from ner_clinic_api.esquema import RespuestaAnalisis, Salud

__all__ = ["app", "crear_app"]

logger = logging.getLogger("ner_clinic_api")

VERSION = version("ner-clinic")
ESTATICO = Path(__file__).parent / "estatico"

# Nota sintética para calentar el modelo al arrancar.
NOTA_CALENTAMIENTO = "Paciente con HTA y diabetes mellitus tipo 2. Niega EPOC."

# Lo más que `/api/salud?esperar=N` deja abierta una petición (el timeout del servicio es 60 s).
MAX_ESPERA_S = 25.0

MENSAJE_CARGANDO = (
    "El modelo está despertando: la primera vez tarda menos de un minuto. Intenta en unos segundos."
)
MENSAJE_NO_DISPONIBLE = "El servicio no está disponible. Intenta más tarde."

MAX_DIAGNOSTICOS = 50
MAX_CARACTERES_DIAGNOSTICO = 300
# Holgura del cuerpo HTTP sobre la nota: escapes JSON (hasta 6 bytes por carácter) y los
# diagnósticos registrados. Corta antes de leer cuerpos enormes.
HOLGURA_BYTES = 128 * 1024

CABECERAS_SEGURIDAD = {
    "content-security-policy": (
        "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
}

# Rutas que se registran tal cual; cualquier otra la escribe el usuario y va anonimizada.
RUTAS_CONOCIDAS = frozenset(
    {"/", "/favicon.ico", "/api/analizar", "/api/salud", "/api/openapi.json"}
)

MENSAJES_HTTP = {
    404: "No encontrado.",
    405: "Método no permitido.",
}


class PeticionAnalisis(BaseModel):
    """Cuerpo de `POST /api/analizar`."""

    model_config = ConfigDict(extra="forbid")

    texto: str
    diagnosticos_registrados: (
        list[Annotated[str, StringConstraints(max_length=MAX_CARACTERES_DIAGNOSTICO)]] | None
    ) = Field(default=None, max_length=MAX_DIAGNOSTICOS)


class CuerpoDemasiadoGrande(StarletteHTTPException):
    """El cuerpo HTTP pasa el máximo. Es `HTTPException` para que FastAPI no lo vuelva 400."""

    def __init__(self, max_caracteres: int):
        super().__init__(413, _mensaje_413(max_caracteres))


def _mensaje_413(max_caracteres: int) -> str:
    return f"La nota es muy larga: el máximo es de {max_caracteres:,} caracteres.".replace(",", ".")


def _respuesta_error(codigo: int, detalle: str) -> JSONResponse:
    return JSONResponse({"detalle": detalle}, status_code=codigo)


class MiddlewarePrivado:
    """Middleware ASGI: logs sin texto, cabeceras de seguridad y red para toda excepción.

    - Registra una línea por petición con método, ruta (sin query string y solo si es
      una ruta conocida), código,
      duración y longitud del texto (si el endpoint la dejó en `request.state`).
    - Corta con 413 los cuerpos de más de `max_bytes`, por `Content-Length` o contando
      lo que llega, sin leer el resto.
    - Atrapa cualquier excepción, registra solo su tipo y responde 500. No la vuelve a
      lanzar: así ni Starlette ni uvicorn imprimen la traza, que podría traer el texto.
    - Pone las cabeceras de seguridad en todas las respuestas y `no-store` en `/api/`.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, max_caracteres: int):
        self.app = app
        self.max_bytes = max_bytes
        self.max_caracteres = max_caracteres

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inicio = time.perf_counter()
        estado: dict[str, Any] = {"codigo": 500, "iniciada": False}
        estado_peticion = scope.setdefault("state", {})
        es_api = scope["path"].startswith("/api/")

        async def enviar(mensaje: Message) -> None:
            if mensaje["type"] == "http.response.start":
                estado["iniciada"] = True
                estado["codigo"] = mensaje["status"]
                cabeceras = MutableHeaders(scope=mensaje)
                for nombre, valor in CABECERAS_SEGURIDAD.items():
                    cabeceras.setdefault(nombre, valor)
                if es_api:
                    cabeceras["cache-control"] = "no-store"
            await send(mensaje)

        recibidos = 0

        async def recibir() -> Message:
            nonlocal recibidos
            mensaje = await receive()
            if mensaje["type"] == "http.request":
                recibidos += len(mensaje.get("body", b""))
                if recibidos > self.max_bytes:
                    raise CuerpoDemasiadoGrande(self.max_caracteres)
            return mensaje

        try:
            if self._content_length(scope) > self.max_bytes:
                raise CuerpoDemasiadoGrande(self.max_caracteres)
            await self.app(scope, recibir, enviar)
        except CuerpoDemasiadoGrande as error:
            if not estado["iniciada"]:
                await _respuesta_error(413, error.detail)(scope, receive, enviar)
        except Exception as error:
            # Solo el tipo: el mensaje o la traza podrían repetir el texto de la nota.
            logger.error("Error no controlado: %s", type(error).__name__)
            if not estado["iniciada"]:
                respuesta = _respuesta_error(500, "Error interno. Intenta de nuevo más tarde.")
                await respuesta(scope, receive, enviar)
        finally:
            longitud = estado_peticion.get("longitud")
            logger.info(
                "%s %s %s %.0f ms longitud=%s",
                scope["method"],
                _ruta_para_log(scope["path"]),
                estado["codigo"],
                (time.perf_counter() - inicio) * 1000,
                "-" if longitud is None else longitud,
            )

    @staticmethod
    def _content_length(scope: Scope) -> int:
        for nombre, valor in scope.get("headers", []):
            if nombre == b"content-length":
                try:
                    return int(valor)
                except ValueError:
                    return 0
        return 0


def _ruta_para_log(ruta: str) -> str:
    """La ruta si es conocida; si no, un marcador (la ruta la escribe el usuario)."""
    if ruta in RUTAS_CONOCIDAS:
        return ruta
    if ruta.startswith("/estatico/"):
        return "/estatico/*"
    return "(otra)"


def _ip_cliente(request: Request, proxies_confiables: int) -> str:
    """La IP del visitante, contando `proxies_confiables` valores desde el final.

    El cliente puede mandar su propio `X-Forwarded-For` y cada proxy **agrega** la IP de
    quien le habló al final, sin reemplazar lo anterior. Los primeros valores no son de
    fiar (con uno distinto en cada petición se saltaría el tope por IP); el que agregó
    el último proxy confiable sí: el valor en la posición `-proxies_confiables`. Si no
    hay cabecera o tiene menos valores, se usa la IP de la conexión.
    """
    valores = [v.strip() for v in request.headers.get("x-forwarded-for", "").split(",")]
    if len(valores) >= proxies_confiables and valores[-proxies_confiables]:
        return valores[-proxies_confiables]
    return request.client.host if request.client else "desconocida"


def _campo_seguro(parte: Any) -> str | int:
    """Una parte de la ubicación de un error de validación, sin datos del usuario.

    Solo se devuelven índices y los nombres de los campos conocidos: una clave extra del
    JSON la escribe el usuario y podría ser texto de la nota.
    """
    if isinstance(parte, int):
        return parte
    if parte in ("body", *PeticionAnalisis.model_fields):
        return parte
    return "?"


def _configurar_logs() -> None:
    """Logs de la app a INFO y el access log de uvicorn apagado (lleva la IP)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").disabled = True


def _contador_configurado(configuracion: Configuracion) -> Contador:
    """El contador que pide `CONTADOR`: en memoria o en Firestore."""
    if configuracion.contador == "firestore":
        return ContadorFirestore(
            proyecto=configuracion.firestore_proyecto, base=configuracion.firestore_base
        )
    return ContadorEnMemoria()


def _mapeo(app: FastAPI) -> str:
    """El mapeo efectivo: `reglas` si no hay normalizador cargado (también en `hibrido` sin
    artefactos), `normalizador` con `sapbert` y `hibrido` si no."""
    if app.state.normalizador is None:
        return "reglas"
    return "normalizador" if app.state.modo_mapeo == "normalizador" else "hibrido"


def _cargar_si_hay(configuracion: Configuracion) -> Any | None:
    """En `hibrido`, el normalizador si su índice existe; si no existe, `None` (reglas).

    Solo la ausencia del índice degrada a reglas (la imagen sin SapBERT). Si el índice
    está, cualquier otro problema (corrupto, huella distinta, SapBERT ausente sin red,
    permisos) tumba el arranque: es un error de build que no debe quedar escondido en un
    warning.
    """
    from ner_clinic.normalizacion import artefactos

    ruta = Path(configuracion.indice_normalizador or artefactos.RUTA_INDICE)
    if not ruta.is_file():
        # Sin la ruta: puede ser local y no hace falta para operar.
        logger.warning("No está el índice del normalizador: el mapeo híbrido queda en reglas")
        return None
    return _cargar_normalizador(configuracion)


def _cargar_normalizador(configuracion: Configuracion) -> Any:
    """La segunda etapa del normalizador desde sus artefactos (importa torch y SapBERT)."""
    from ner_clinic.normalizacion import artefactos

    ruta = configuracion.indice_normalizador or artefactos.RUTA_INDICE
    return artefactos.cargar(Path(ruta))


def crear_app(
    extractor: Extractor | None = None,
    contador: Contador | None = None,
    configuracion: Configuracion | None = None,
    hoy: Callable[[], date] = hoy_en_colombia,
    normalizador: Any | None = None,
    cargar_en_segundo_plano: bool = False,
) -> FastAPI:
    """Arma la app. Sin argumentos usa el modelo real y el contador y los límites del entorno.

    Args:
        extractor: el modelo NER; por defecto `ExtractorHF`. Los tests inyectan un
            `ExtractorFalso`.
        contador: el contador de uso; por defecto el que pide `CONTADOR` en la
            configuración (`ContadorEnMemoria` si no se dice nada).
        configuracion: los límites; por defecto `Configuracion.desde_entorno()`.
        hoy: la fecha del día para los contadores (inyectable en los tests).
        normalizador: la segunda etapa del normalizador (`SegundaEtapa`). Por defecto se
            carga al arrancar si la configuración pide `hibrido` o `sapbert`; los tests
            inyectan una falsa. Con `sapbert` decide sola; si no, como respaldo de las
            reglas.
        cargar_en_segundo_plano: si es `True`, el puerto se abre de una vez y el modelo
            carga detrás (`estado: cargando` hasta que termina). Si es `False`, el
            arranque espera la carga y un error la tumba.
    """
    extractor = ExtractorHF() if extractor is None else extractor
    configuracion = Configuracion.desde_entorno() if configuracion is None else configuracion
    contador = _contador_configurado(configuracion) if contador is None else contador

    async def cargar(app: FastAPI) -> None:
        """Carga el normalizador y calienta el modelo una sola vez: las peticiones nunca
        lo cargan."""
        if normalizador is None and configuracion.normalizador == "sapbert":
            app.state.normalizador = await run_in_threadpool(_cargar_normalizador, configuracion)
        elif normalizador is None and configuracion.normalizador == "hibrido":
            app.state.normalizador = await run_in_threadpool(_cargar_si_hay, configuracion)
        await run_in_threadpool(
            analizar,
            NOTA_CALENTAMIENTO,
            extractor=extractor,
            normalizador=app.state.normalizador,
            modo_mapeo=app.state.modo_mapeo,
        )
        app.state.modelo_cargado = True
        app.state.estado = "ok"
        logger.info("Modelo cargado y caliente (mapeo: %s)", _mapeo(app))

    async def cargar_detras(app: FastAPI) -> None:
        try:
            await cargar(app)
        except Exception as error:
            # La carga no ve notas de usuarios (el calentamiento es una nota sintética):
            # el mensaje sirve para saber qué artefacto falló.
            app.state.estado = "error"
            logger.error("No se pudo cargar el modelo: %s: %s", type(error).__name__, error)
        finally:
            app.state.cargado.set()

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        _configurar_logs()
        app.state.semaforo = asyncio.Semaphore(1)
        app.state.modelo_cargado = False
        app.state.estado = "cargando"
        app.state.cargado = asyncio.Event()
        # Con `reglas`, ni el inyectado: la configuración manda.
        app.state.normalizador = None if configuracion.normalizador == "reglas" else normalizador
        app.state.modo_mapeo = (
            "normalizador" if configuracion.normalizador == "sapbert" else "hibrido"
        )
        if cargar_en_segundo_plano:
            # Se guarda la tarea: el bucle solo tiene una referencia débil.
            app.state.tarea_carga = asyncio.create_task(cargar_detras(app))
        else:
            await cargar(app)
            app.state.cargado.set()
        yield

    app = FastAPI(
        title="NER clínico en español",
        version=VERSION,
        lifespan=ciclo_de_vida,
        # La documentación interactiva carga scripts de un CDN y rompe la CSP.
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        MiddlewarePrivado,
        max_bytes=configuracion.max_caracteres * 6 + HOLGURA_BYTES,
        max_caracteres=configuracion.max_caracteres,
    )

    @app.exception_handler(RequestValidationError)
    async def error_de_validacion(request: Request, error: RequestValidationError) -> JSONResponse:
        # El manejador de FastAPI devuelve el `input` (el texto de la nota): aquí solo
        # va dónde y de qué tipo es cada error.
        errores = [
            {"campo": [_campo_seguro(p) for p in e.get("loc", ())], "tipo": e.get("type")}
            for e in error.errors()
        ]
        return JSONResponse(
            {
                "detalle": (
                    "La petición no es válida. Envía un JSON con `texto` (obligatorio) y "
                    "`diagnosticos_registrados` (opcional, lista de textos)."
                ),
                "errores": errores,
            },
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def error_http(request: Request, error: StarletteHTTPException) -> JSONResponse:
        detalle = MENSAJES_HTTP.get(error.status_code, error.detail)
        return JSONResponse(
            {"detalle": detalle}, status_code=error.status_code, headers=error.headers
        )

    @app.post("/api/analizar", response_model=None, responses={200: {"model": RespuestaAnalisis}})
    async def analizar_nota(peticion: PeticionAnalisis, request: Request) -> dict | JSONResponse:
        request.state.longitud = len(peticion.texto)
        if len(peticion.texto) > configuracion.max_caracteres:
            raise HTTPException(413, _mensaje_413(configuracion.max_caracteres))
        # Antes del contador: un análisis que no corre no gasta del tope.
        estado = request.app.state.estado
        if estado != "ok":
            # `estado_modelo` le dice al front que espere y analice solo cuando cargue.
            mensaje = MENSAJE_CARGANDO if estado == "cargando" else MENSAJE_NO_DISPONIBLE
            return JSONResponse({"detalle": mensaje, "estado_modelo": estado}, status_code=503)

        semaforo: asyncio.Semaphore = request.app.state.semaforo
        try:
            await asyncio.wait_for(semaforo.acquire(), timeout=configuracion.timeout_s)
        except TimeoutError:
            raise HTTPException(
                503, "Hay mucha demanda en este momento. Intenta en unos segundos."
            ) from None

        try:
            dia = hoy()
            clave = clave_ip(
                configuracion.sal_ip, _ip_cliente(request, configuracion.proxies_confiables), dia
            )
            try:
                consumo = await run_in_threadpool(
                    contador.consumir,
                    clave,
                    dia,
                    configuracion.tope_ip_dia,
                    configuracion.tope_global_dia,
                )
            except Exception as error:
                # Falla cerrado: sin contador no se analiza, para no gastar sin control.
                logger.error("El contador falló: %s", type(error).__name__)
                raise HTTPException(503, MENSAJE_NO_DISPONIBLE) from None
            if not consumo.permitido:
                if consumo.tope == "ip":
                    mensaje = (
                        f"Llegaste al máximo de {configuracion.tope_ip_dia} análisis por día. "
                        "Vuelve mañana."
                    )
                else:
                    mensaje = "La demo llegó al máximo de análisis de hoy. Vuelve mañana."
                raise HTTPException(429, mensaje)

            inicio = time.perf_counter()
            # En un hilo: no se cancela a medias, así que el semáforo se suelta cuando
            # la inferencia de verdad terminó.
            resultado = await run_in_threadpool(
                analizar,
                peticion.texto,
                peticion.diagnosticos_registrados,
                extractor=extractor,
                normalizador=request.app.state.normalizador,
                modo_mapeo=request.app.state.modo_mapeo,
            )
            segundos = time.perf_counter() - inicio
        finally:
            semaforo.release()

        return {
            **resultado.to_dict(),
            "meta": {
                "version": VERSION,
                "segundos": round(segundos, 3),
                "restantes_hoy": consumo.restantes_ip,
                "mapeo": _mapeo(request.app),
            },
        }

    @app.get("/api/salud", response_model=None, responses={200: {"model": Salud}})
    async def salud(request: Request, esperar: float = 0) -> dict | JSONResponse:
        estado = request.app.state
        espera = min(max(esperar, 0.0), MAX_ESPERA_S)
        if espera and not estado.cargado.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(estado.cargado.wait(), timeout=espera)
        cuerpo = {
            "estado": estado.estado,
            "modelo_cargado": estado.modelo_cargado,
            "mapeo": _mapeo(request.app),
            "version": VERSION,
        }
        if estado.estado == "error":
            return JSONResponse(cuerpo, status_code=503)
        return cuerpo

    @app.get("/", include_in_schema=False)
    async def inicio() -> FileResponse:
        return FileResponse(ESTATICO / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        # Navegadores y lectores de enlaces lo piden por su cuenta (#226).
        return FileResponse(ESTATICO / "icono.svg", media_type="image/svg+xml")

    app.mount("/estatico", StaticFiles(directory=ESTATICO), name="estatico")
    return app


app = crear_app(cargar_en_segundo_plano=True)
