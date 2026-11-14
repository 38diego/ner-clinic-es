"""API web de `ner_clinic`: `POST /api/analizar`, `GET /api/salud` y el front en `GET /`.

Va en un paquete aparte para que la librería no dependa de FastAPI. Se instala con el
extra `api` (`pip install ner-clinic[api]`); con `uv sync` ya viene en el grupo `dev`.

Correr en local (carga el modelo al arrancar; la primera vez lo descarga):

    uv run python -m ner_clinic_api

o, con uvicorn directamente:

    uv run uvicorn ner_clinic_api.app:app --no-access-log

`python -m ner_clinic_api` arranca uvicorn con el access log desactivado. La app también
lo apaga al arrancar (`uvicorn.access`), así que `--no-access-log` es una segunda
barrera: el access log registra la IP del visitante y no hace falta.

Límites (variables de entorno, ver `configuracion`): `MAX_CARACTERES`, `TIMEOUT_S`,
`TOPE_IP_DIA`, `TOPE_GLOBAL_DIA`, `PROXIES_CONFIABLES` y la sal de la clave por IP,
`SAL_IP`.

Privacidad: el texto recibido nunca se guarda, ni se registra en logs, ni se repite en
los errores. Los logs solo llevan método, ruta, código, duración y longitud del texto.
"""

from ner_clinic_api.app import crear_app

__all__ = ["crear_app"]
