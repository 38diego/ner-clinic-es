"""Tipos compartidos entre las fases del pipeline.

Es el contrato entre módulos: `Resultado.to_dict()` es lo que consumen la API y la demo.

Flujo de tipos: la extracción produce `Candidato` (sin estado), la depuración interna
fusiona candidatos, el contexto los convierte en `Entidad` y el mapeo les asigna
categoría de Charlson y CIE-10. El contexto clasifica cada mención por su posición: si
las menciones de un candidato quedan en estados distintos, el candidato da una `Entidad`
por estado.

Posiciones: dentro del pipeline, las `Mencion` son relativas al texto limpio; el
orquestador las traduce al texto original antes de armar el `Resultado`.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class Estado(StrEnum):
    """Estado de una comorbilidad según el contexto en que aparece en la nota.

    - `CONFIRMADA`: el paciente la tiene. Es la única que suma a `indice_charlson`.
    - `NEGADA`: la nota dice que no la tiene ("niega EPOC").
    - `POSIBLE`: la nota la plantea sin afirmarla ni negarla (una sospecha). Recibe
      categoría de Charlson y CIE-10 para mostrarla, pero solo suma a
      `indice_charlson_si_se_confirman`.
    - `FAMILIAR`: es un antecedente de un familiar, no del paciente.
    """

    CONFIRMADA = "confirmada"
    NEGADA = "negada"
    POSIBLE = "posible"
    FAMILIAR = "familiar"


class CategoriaCharlson(StrEnum):
    """Las 17 categorías del índice de Charlson, en el orden de las reglas de la tesis."""

    INFARTO_MIOCARDIO = "infarto_miocardio"
    INSUFICIENCIA_CARDIACA = "insuficiencia_cardiaca"
    VASCULAR_PERIFERICA = "vascular_periferica"
    CEREBROVASCULAR = "cerebrovascular"
    DEMENCIA = "demencia"
    PULMONAR_CRONICA = "pulmonar_cronica"
    REUMATOLOGICA = "reumatologica"
    ULCERA_PEPTICA = "ulcera_peptica"
    HEPATICA_LEVE = "hepatica_leve"
    DIABETES = "diabetes"
    DIABETES_COMPLICADA = "diabetes_complicada"
    HEMIPLEJIA = "hemiplejia"
    RENAL_CRONICA = "renal_cronica"
    TUMOR_MALIGNO = "tumor_maligno"
    HEPATICA_SEVERA = "hepatica_severa"
    TUMOR_METASTASICO = "tumor_metastasico"
    VIH_SIDA = "vih_sida"


class MotivoDescarte(StrEnum):
    """Por qué una entidad detectada no quedó en el resultado final."""

    BASURA = "basura"
    DUPLICADO_INTERNO = "duplicado_interno"
    DIAGNOSTICO_REGISTRADO = "diagnostico_registrado"


@dataclass(frozen=True)
class Mencion:
    """Una aparición en la nota: lo que está escrito en `texto_fuente[inicio:fin]`.

    Guarda su propio `texto` porque una entidad fusionada (p. ej. "hipertension" con
    "hipertensión") reúne menciones escritas de forma distinta.
    """

    texto: str
    inicio: int
    fin: int

    def __post_init__(self):
        if self.inicio < 0:
            raise ValueError(f"inicio ({self.inicio}) es negativo")
        if self.fin < self.inicio:
            raise ValueError(f"fin ({self.fin}) es menor que inicio ({self.inicio})")

    def to_dict(self) -> dict:
        return {"texto": self.texto, "inicio": self.inicio, "fin": self.fin}


@dataclass
class Candidato:
    """Entidad detectada por el modelo, antes de clasificarla por contexto."""

    texto: str
    menciones: list[Mencion]

    def a_entidad(self, estado: Estado, menciones: list[Mencion] | None = None) -> "Entidad":
        """La entidad con `estado` y `menciones` (por defecto, todas las del candidato).

        Si las menciones elegidas no incluyen el texto del candidato (la deduplicación
        fusionó "diabetes" dentro de "diabetes mellitus tipo 2" y la mención corta quedó
        en otro estado), la entidad toma el texto de su mención más larga: así "Padre con
        diabetes" sale como familiar "diabetes", no con el nombre de la del paciente.
        """
        elegidas = list(self.menciones if menciones is None else menciones)
        texto = self.texto
        clave = " ".join(texto.lower().split())
        if elegidas and all(" ".join(m.texto.lower().split()) != clave for m in elegidas):
            texto = max(elegidas, key=lambda m: (len(m.texto), -m.inicio)).texto
        return Entidad(texto=texto, menciones=elegidas, estado=estado)


@dataclass(frozen=True)
class CandidatoIndice:
    """El texto del índice CIE-10 que explica una categoría dada por el normalizador.

    `texto` es la descripción del catálogo CIE-10-ES o un sinónimo de CodiEsp
    entrenamiento (`origen`), nunca texto de la nota; `similitud`, el coseno con la mención.
    """

    texto: str
    codigo: str
    origen: str
    similitud: float

    def to_dict(self) -> dict:
        return {
            "texto": self.texto,
            "codigo": self.codigo,
            "origen": self.origen,
            "similitud": self.similitud,
        }


@dataclass(frozen=True)
class AlternativaCie10:
    """Un código CIE-10-ES entre los más parecidos a la entidad, para que el experto elija
    (#230). `descripcion` es la del catálogo (nunca texto de la nota); `similitud`, el
    coseno de la mejor fila del índice con ese código."""

    codigo: str
    descripcion: str
    similitud: float

    def to_dict(self) -> dict:
        return {"codigo": self.codigo, "descripcion": self.descripcion, "similitud": self.similitud}


@dataclass
class Entidad:
    """Una comorbilidad detectada en un estado, con todas sus apariciones en ese estado.

    Si la enfermedad del paciente se menciona varias veces con estados distintos, es una
    sola entidad (#208): confirmada si alguna mención lo es; si no, con el estado de la
    última ("Sospecha de migraña … se descarta la migraña" da negada). `otros_estados`
    son los demás, en orden. Lo de un familiar va en otra entidad. El cruce con los registrados y
    `indice_charlson` solo miran la confirmada. La posible también lleva categoría de
    Charlson y CIE-10, y cuenta solo en `indice_charlson_si_se_confirman`.
    `estados_menciones` es el estado de cada mención, alineado con `menciones` (vacío si
    todas tienen el de la entidad); `estado_de_mencion(i)` lo lee.

    `registrado` dice si la comorbilidad ya está entre los diagnósticos registrados del
    egreso (`None` si no se enviaron diagnósticos o la entidad no es confirmada) y
    `coincide_con` es el diagnóstico registrado con el que coincidió y `coincide_por`, cómo:
    `"texto"` (comparten una palabra) o `"categoria"` (misma categoría de Charlson, #176).
    Una confirmada con `registrado=False` es un posible subregistro.

    `correcciones` son las palabras mal escritas que hubo que corregir para darle la
    categoría de Charlson (`(escrita, corregida)`, en minúscula y sin tildes; #115):
    "diabetis mellitus" da diabetes con `(("diabetis", "diabetes"),)`. Vacío si la
    categoría salió sin corregir nada. `categoria_por_correccion` lo resume.

    `mapeo` dice de dónde salió la categoría: `"reglas"` (las regex de Charlson con el
    corrector) o `"normalizador"` (SapBERT y el clasificador sobre el top-20, #165);
    `None` si no tiene categoría. Con el normalizador, `candidato` es el texto del índice
    CIE-10 que la explica y `cie10` es su código CIE-10-ES, leído para Charlson (un código
    padre sin categoría se lee como su hijo `.9`: el candidato "E11" da `cie10` "E11.9").
    Si la categoría solo la dieron las reglas dentro del clasificador, `candidato` es
    `None` y `cie10` es el de las reglas.

    `alternativas_cie10` son los códigos más parecidos según el normalizador (hasta 5,
    distintos, de mayor a menor similitud), para que el experto elija (#230). Solo las
    confirmadas y posibles, y solo con normalizador; vacío si no.
    """

    texto: str
    menciones: list[Mencion]
    estado: Estado
    categoria_charlson: CategoriaCharlson | None = None
    cie10: str | None = None
    registrado: bool | None = None
    coincide_con: str | None = None
    coincide_por: str | None = None
    correcciones: tuple[tuple[str, str], ...] = ()
    mapeo: str | None = None
    candidato: CandidatoIndice | None = None
    otros_estados: tuple[Estado, ...] = ()
    estados_menciones: tuple[Estado, ...] = ()
    alternativas_cie10: tuple[AlternativaCie10, ...] = ()

    def estado_de_mencion(self, i: int) -> Estado:
        """El estado de la mención `i` según su propio contexto (#208)."""
        return self.estados_menciones[i] if self.estados_menciones else self.estado

    @property
    def categoria_por_correccion(self) -> bool:
        """Si la categoría de Charlson salió de corregir la ortografía (#115)."""
        return bool(self.correcciones)

    @property
    def primera_mencion(self) -> Mencion | None:
        """La mención más temprana en el texto, sin importar el orden de la lista."""
        return min(self.menciones, key=lambda m: m.inicio, default=None)

    @property
    def inicio(self) -> int | None:
        primera = self.primera_mencion
        return primera.inicio if primera else None

    @property
    def fin(self) -> int | None:
        primera = self.primera_mencion
        return primera.fin if primera else None

    def to_dict(self) -> dict:
        return {
            "texto": self.texto,
            "inicio": self.inicio,
            "fin": self.fin,
            "menciones": [
                {**m.to_dict(), "estado": self.estado_de_mencion(i).value}
                for i, m in enumerate(self.menciones)
            ],
            "estado": self.estado.value,
            "otros_estados": [e.value for e in self.otros_estados],
            "categoria_charlson": (
                self.categoria_charlson.value if self.categoria_charlson else None
            ),
            "cie10": self.cie10,
            "registrado": self.registrado,
            "coincide_con": self.coincide_con,
            "coincide_por": self.coincide_por,
            "categoria_por_correccion": self.categoria_por_correccion,
            "correcciones": [
                {"escrita": escrita, "corregida": corregida}
                for escrita, corregida in self.correcciones
            ],
            "mapeo": self.mapeo,
            "candidato": self.candidato.to_dict() if self.candidato else None,
            "alternativas_cie10": [a.to_dict() for a in self.alternativas_cie10],
        }


@dataclass
class Descarte:
    """Una entidad descartada, con el motivo y el detalle para el auditor."""

    texto: str
    motivo: MotivoDescarte
    detalle: str

    def to_dict(self) -> dict:
        return {"texto": self.texto, "motivo": self.motivo.value, "detalle": self.detalle}


@dataclass
class Resultado:
    """Salida de `analizar()`.

    `entidades` va en orden de aparición. Una enfermedad puede aparecer dos veces, con
    estados distintos y sus propias menciones (ver `Entidad`).

    `indice_charlson` suma solo las confirmadas. `indice_charlson_si_se_confirman` suma
    las confirmadas y las posibles, con la misma jerarquía leve/severa: es el índice si
    se confirman las sospechas. Sin posibles, los dos son iguales.
    """

    entidades: list[Entidad] = field(default_factory=list)
    descartes: list[Descarte] = field(default_factory=list)
    indice_charlson: int = 0
    indice_charlson_si_se_confirman: int = 0

    def to_dict(self) -> dict:
        return {
            "entidades": [e.to_dict() for e in self.entidades],
            "descartes": [d.to_dict() for d in self.descartes],
            "indice_charlson": self.indice_charlson,
            "indice_charlson_si_se_confirman": self.indice_charlson_si_se_confirman,
        }
