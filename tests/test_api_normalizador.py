"""La API con el normalizador (#166), sin red ni modelos: una segunda etapa falsa."""

import logging

import pytest
from fastapi.testclient import TestClient

from ner_clinic.normalizacion.decision import Decision
from ner_clinic.normalizacion.indice import ORIGEN_CATALOGO, Candidato
from ner_clinic.tipos import CategoriaCharlson as C
from ner_clinic_api import app as modulo_app
from ner_clinic_api import crear_app
from ner_clinic_api.configuracion import Configuracion
from ner_clinic_api.esquema import RespuestaAnalisis, Salud
from tests.test_api import CENTINELA, NOTA_CENTINELA, analizar, configuracion, extractor_falso
from tests.test_pipeline import extractor_para
from tests.test_pipeline_normalizador import SegundaFalsa

DIABETES = Decision(
    C.DIABETES, "E11.9", 0.97, Candidato("E11.9", "DM tipo 2", 0.81, ORIGEN_CATALOGO)
)


def _segunda():
    return SegundaFalsa({"diabetes mellitus tipo 2": DIABETES})


def test_con_normalizador_la_respuesta_trae_mapeo_y_candidato():
    segunda = _segunda()
    app = crear_app(
        extractor=extractor_falso(),
        configuracion=configuracion(normalizador="sapbert"),
        normalizador=segunda,
    )
    with TestClient(app) as cliente:
        datos = analizar(cliente).json()
        salud = cliente.get("/api/salud").json()
    entidades = {e["texto"]: e for e in datos["entidades"]}
    diabetes = entidades["diabetes mellitus tipo 2"]
    assert diabetes["mapeo"] == "normalizador" and diabetes["cie10"] == "E11.9"
    assert diabetes["candidato"] == {
        "texto": "DM tipo 2",
        "codigo": "E11.9",
        "origen": ORIGEN_CATALOGO,
        "similitud": 0.81,
    }
    assert entidades["HTA"]["mapeo"] is None  # la segunda etapa falsa no la conoce
    assert datos["meta"]["mapeo"] == "normalizador"
    assert salud["mapeo"] == "normalizador"
    RespuestaAnalisis.model_validate(datos)
    Salud.model_validate(salud)
    # La nota de calentamiento también pasó por el normalizador, antes de la petición
    assert len(segunda.lotes) == 2


def test_sin_normalizador_la_respuesta_cumple_el_esquema(cliente_reglas):
    datos = analizar(cliente_reglas).json()
    RespuestaAnalisis.model_validate(datos)
    assert datos["meta"]["mapeo"] == "reglas"
    assert {e["mapeo"] for e in datos["entidades"]} <= {"reglas", None}


@pytest.fixture
def cliente_reglas():
    app = crear_app(extractor=extractor_falso(), configuracion=configuracion())
    with TestClient(app) as cliente:
        yield cliente


def test_sapbert_carga_el_normalizador_al_arrancar(monkeypatch):
    llamadas = []

    def cargar(config):
        llamadas.append(config.indice_normalizador)
        return _segunda()

    monkeypatch.setattr(modulo_app, "_cargar_normalizador", cargar)
    config = configuracion(normalizador="sapbert", indice_normalizador="/opt/indice.pt")
    with TestClient(crear_app(extractor=extractor_falso(), configuracion=config)) as cliente:
        assert cliente.get("/api/salud").json()["mapeo"] == "normalizador"
    assert llamadas == ["/opt/indice.pt"]


def test_sapbert_sin_artefactos_no_arranca(tmp_path):
    config = configuracion(normalizador="sapbert", indice_normalizador=str(tmp_path / "no.pt"))
    app = crear_app(extractor=extractor_falso(), configuracion=config)
    with pytest.raises(FileNotFoundError, match="evaluacion artefactos"), TestClient(app):
        pass


def test_los_logs_no_llevan_el_texto_con_normalizador(caplog):
    app = crear_app(
        extractor=extractor_falso(),
        configuracion=configuracion(normalizador="sapbert"),
        normalizador=_segunda(),
    )
    with caplog.at_level(logging.DEBUG), TestClient(app) as cliente:
        assert analizar(cliente, NOTA_CENTINELA).status_code == 200
    assert CENTINELA not in caplog.text
    assert "mapeo: normalizador" in caplog.text


def test_el_openapi_documenta_mapeo_y_candidato(cliente_reglas):
    esquema = cliente_reglas.get("/api/openapi.json").json()
    entidad = esquema["components"]["schemas"]["EntidadSalida"]["properties"]
    assert {"mapeo", "candidato"} <= set(entidad)
    assert "Salud" in esquema["components"]["schemas"]


def test_configuracion_del_normalizador():
    # Por defecto, el híbrido (#182; antes, el normalizador, #168)
    assert Configuracion.desde_entorno({}).normalizador == "hibrido"
    assert Configuracion().normalizador == "hibrido"
    assert Configuracion.desde_entorno({"NER_NORMALIZADOR": "Hibrido"}).normalizador == "hibrido"
    assert Configuracion.desde_entorno({"NER_NORMALIZADOR": "reglas"}).normalizador == "reglas"
    config = Configuracion.desde_entorno(
        {"NER_NORMALIZADOR": " SAPBERT ", "NER_INDICE_NORMALIZADOR": "/opt/i.pt"}
    )
    assert (config.normalizador, config.indice_normalizador) == ("sapbert", "/opt/i.pt")
    with pytest.raises(ValueError, match="NER_NORMALIZADOR"):
        Configuracion.desde_entorno({"NER_NORMALIZADOR": "otro"})


def test_los_valores_del_esquema_son_los_de_la_libreria():
    from typing import get_args

    from ner_clinic.tipos import MotivoDescarte
    from ner_clinic_api.esquema import CategoriaSalida, DescarteSalida

    assert set(get_args(CategoriaSalida)) == {c.value for c in C}
    motivo = DescarteSalida.model_fields["motivo"].annotation
    assert set(get_args(motivo)) == {m.value for m in MotivoDescarte}


def test_la_imagen_docker_no_fija_el_mapeo():
    """#182: la imagen usa el híbrido por defecto, que sin SapBERT corre con las reglas."""
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    # Las líneas de instrucciones, no los comentarios
    lineas = [ln for ln in dockerfile.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("NER_NORMALIZADOR" in ln for ln in lineas)


# --- Un solo camino (#182): híbrido por defecto, con reglas si faltan los artefactos ---


def test_hibrido_reglas_primero_y_normalizador_de_respaldo():
    """La diabetes la mapean las reglas; "falla renal crónica" solo el normalizador."""
    nota = "Paciente con diabetes mellitus tipo 2 y falla renal crónica."
    renal = Decision(C.RENAL_CRONICA, "N18.9", 0.9, Candidato("N18.9", "IRC", 0.9, ORIGEN_CATALOGO))
    segunda = SegundaFalsa({"diabetes mellitus tipo 2": DIABETES, "falla renal crónica": renal})
    app = crear_app(
        extractor=extractor_para(nota, "diabetes mellitus tipo 2", "falla renal crónica"),
        configuracion=configuracion(normalizador="hibrido"),
        normalizador=segunda,
    )
    with TestClient(app) as cliente:
        datos = analizar(cliente, nota).json()
        salud = cliente.get("/api/salud").json()
    entidades = {e["texto"]: e for e in datos["entidades"]}
    assert entidades["diabetes mellitus tipo 2"]["mapeo"] == "reglas"
    assert entidades["falla renal crónica"]["mapeo"] == "normalizador"
    assert datos["meta"]["mapeo"] == "hibrido" and salud["mapeo"] == "hibrido"
    RespuestaAnalisis.model_validate(datos)
    Salud.model_validate(salud)


def test_hibrido_sin_artefactos_arranca_con_reglas(tmp_path, caplog):
    ruta = tmp_path / "no-existe" / "indice.pt"
    config = configuracion(normalizador="hibrido", indice_normalizador=str(ruta))
    app = crear_app(extractor=extractor_falso(), configuracion=config)
    with caplog.at_level(logging.DEBUG), TestClient(app) as cliente:
        salud = cliente.get("/api/salud").json()
        datos = analizar(cliente, NOTA_CENTINELA).json()
    assert salud["mapeo"] == "reglas" and salud["modelo_cargado"]
    assert datos["meta"]["mapeo"] == "reglas"
    assert "híbrido queda en reglas" in caplog.text
    assert "no-existe" not in caplog.text and CENTINELA not in caplog.text


def test_hibrido_carga_el_normalizador_si_estan_los_artefactos(tmp_path, monkeypatch):
    indice = tmp_path / "indice.pt"
    indice.write_bytes(b"x")  # basta con que exista: la carga es falsa
    monkeypatch.setattr(modulo_app, "_cargar_normalizador", lambda config: _segunda())
    config = configuracion(normalizador="hibrido", indice_normalizador=str(indice))
    with TestClient(crear_app(extractor=extractor_falso(), configuracion=config)) as cliente:
        assert cliente.get("/api/salud").json()["mapeo"] == "hibrido"


@pytest.mark.parametrize("error", [OSError("sin SapBERT"), ValueError("huella"), RuntimeError])
def test_hibrido_con_el_indice_presente_un_error_tumba_el_arranque(tmp_path, monkeypatch, error):
    indice = tmp_path / "indice.pt"
    indice.write_bytes(b"x")

    def falla(config):
        raise error

    monkeypatch.setattr(modulo_app, "_cargar_normalizador", falla)
    config = configuracion(normalizador="hibrido", indice_normalizador=str(indice))
    app = crear_app(extractor=extractor_falso(), configuracion=config)
    with pytest.raises((OSError, ValueError, RuntimeError)), TestClient(app):
        pass


def test_sapbert_con_un_oserror_no_arranca(monkeypatch):
    def falla(config):
        raise OSError("sin SapBERT")

    monkeypatch.setattr(modulo_app, "_cargar_normalizador", falla)
    app = crear_app(
        extractor=extractor_falso(), configuracion=configuracion(normalizador="sapbert")
    )
    with pytest.raises(OSError), TestClient(app):
        pass


def test_con_reglas_no_se_carga_ni_se_usa_el_normalizador(monkeypatch):
    def no_llamar(config):
        raise AssertionError("no se debe cargar")

    monkeypatch.setattr(modulo_app, "_cargar_normalizador", no_llamar)
    app = crear_app(
        extractor=extractor_falso(), configuracion=configuracion(), normalizador=_segunda()
    )
    with TestClient(app) as cliente:
        assert cliente.get("/api/salud").json()["mapeo"] == "reglas"
        assert analizar(cliente).json()["meta"]["mapeo"] == "reglas"


def test_el_openapi_lista_el_hibrido(cliente_reglas):
    esquemas = cliente_reglas.get("/api/openapi.json").json()["components"]["schemas"]
    for nombre in ("Meta", "Salud"):
        assert "hibrido" in esquemas[nombre]["properties"]["mapeo"]["enum"]


# --- SapBERT en la imagen (#189) -------------------------------------------------


def test_la_imagen_completa_es_la_de_por_defecto_y_fija_las_revisiones():
    from pathlib import Path

    from ner_clinic.normalizacion.indice import SAPBERT

    raiz = Path(__file__).resolve().parents[1]
    dockerfile = (raiz / "Dockerfile").read_text(encoding="utf-8")
    etapas = [ln for ln in dockerfile.splitlines() if ln.startswith("FROM ")]
    assert etapas[-1] == "FROM base AS completa"
    tokenizador, revision_tokenizador = SAPBERT.tokenizador
    for linea in (
        f"ARG SAPBERT={SAPBERT.nombre}",
        f"ARG SAPBERT_REVISION={SAPBERT.revision}",
        f"ARG TOKENIZADOR={tokenizador}",
        f"ARG TOKENIZADOR_REVISION={revision_tokenizador}",
    ):
        assert linea in dockerfile.splitlines()
    assert "FROM dependencias AS sapbert" in etapas  # sin el código: no se descarga de nuevo
    assert "target: base" in (raiz / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    costo = (raiz / "evaluacion/costo_integrado.py").read_text(encoding="utf-8")
    assert '"--target", "base"' in costo
    assert "COPY data/normalizacion/indice_final.pt" in dockerfile
    ignorados = (raiz / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ignorados.index("!data/normalizacion/indice_final.pt") > ignorados.index("data/")
