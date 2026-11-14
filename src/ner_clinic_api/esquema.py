"""Esquemas de las respuestas, para documentarlas en el OpenAPI (`/api/openapi.json`).

Solo documentan: la ruta devuelve `Resultado.to_dict()` tal cual (no se valida ni se
filtra la salida con ellos). Un test comprueba que una respuesta real cumple el esquema.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CategoriaSalida = Literal[
    "infarto_miocardio",
    "insuficiencia_cardiaca",
    "vascular_periferica",
    "cerebrovascular",
    "demencia",
    "pulmonar_cronica",
    "reumatologica",
    "ulcera_peptica",
    "hepatica_leve",
    "diabetes",
    "diabetes_complicada",
    "hemiplejia",
    "renal_cronica",
    "tumor_maligno",
    "hepatica_severa",
    "tumor_metastasico",
    "vih_sida",
]


class Estricto(BaseModel):
    """Sin campos de más: el test de esquema falla si `to_dict()` agrega uno sin
    documentarlo aquí."""

    model_config = ConfigDict(extra="forbid")


EstadoSalida = Literal["confirmada", "negada", "posible", "familiar"]


class MencionSalida(Estricto):
    texto: str
    inicio: int
    fin: int
    # El de esta mención por su contexto; el de la entidad sale de todas (#208).
    estado: EstadoSalida


class Correccion(Estricto):
    escrita: str
    corregida: str


class CandidatoSalida(Estricto):
    """El texto del índice CIE-10 que explica la categoría dada por el normalizador."""

    texto: str = Field(
        description="Descripción del catálogo CIE-10-ES o sinónimo de CodiEsp; nunca la nota."
    )
    codigo: str
    origen: str = Field(description="`catalogo` o el corpus del sinónimo.")
    similitud: float


class AlternativaSalida(Estricto):
    """Un código CIE-10-ES entre los más parecidos a la entidad (#230)."""

    codigo: str
    descripcion: str = Field(description="Descripción del catálogo CIE-10-ES; nunca la nota.")
    similitud: float


class EntidadSalida(Estricto):
    texto: str
    inicio: int | None
    fin: int | None
    menciones: list[MencionSalida]
    estado: EstadoSalida
    # Estados de las otras menciones, distintos del de la entidad, en orden (#208).
    otros_estados: list[EstadoSalida]
    categoria_charlson: CategoriaSalida | None
    cie10: str | None
    registrado: bool | None
    coincide_con: str | None
    coincide_por: Literal["texto", "categoria"] | None = Field(
        description="Cómo coincidió con el diagnóstico registrado: por texto o por categoría."
    )
    categoria_por_correccion: bool
    correcciones: list[Correccion]
    mapeo: Literal["reglas", "normalizador"] | None = Field(
        description="De dónde salió la categoría de Charlson; `null` si no tiene."
    )
    candidato: CandidatoSalida | None = Field(
        description="Con el normalizador, el texto del índice que explica la categoría."
    )
    alternativas_cie10: list[AlternativaSalida] = Field(
        description="Hasta 5 códigos distintos más parecidos, para que el experto elija; "
        "vacío sin normalizador o si la entidad no es confirmada ni posible."
    )


class DescarteSalida(Estricto):
    texto: str
    motivo: Literal["basura", "duplicado_interno", "diagnostico_registrado"]
    detalle: str


class Meta(Estricto):
    version: str
    segundos: float
    restantes_hoy: int
    mapeo: Literal["reglas", "normalizador", "hibrido"] = Field(
        description="Cómo se dio la categoría de Charlson en este análisis: solo reglas, "
        "solo normalizador, o reglas y normalizador para lo que no mapean (cada entidad "
        "dice el suyo en `mapeo`)."
    )


class RespuestaAnalisis(Estricto):
    entidades: list[EntidadSalida]
    descartes: list[DescarteSalida]
    indice_charlson: int
    indice_charlson_si_se_confirman: int
    meta: Meta


class Salud(Estricto):
    estado: Literal["cargando", "ok", "error"]
    modelo_cargado: bool
    mapeo: Literal["reglas", "normalizador", "hibrido"]
    version: str
