"""Extracción NER: del texto limpio a candidatos con posición.

Porta la lógica de `extraer_enfermedades_historia`, `fusionar_entidades_cercanas`,
`expandir_palabra_cortada` y `limpiar_basura_ner` de la tesis:

1. Segmenta el texto limpio y corre el modelo una sola vez, por lotes.
2. Completa las palabras que el modelo cortó. La fusión de entidades cercanas de la
   tesis (a 18 caracteres o menos y sin barrera) ya no corre por defecto (#55); queda en
   `ConfiguracionExtraccion` para la ablación.
3. Recorta puntuación de los bordes y viñetas numéricas iniciales.
4. Descarta basura (longitud, terminaciones truncadas, lista negra) y lo registra como
   `Descarte` con motivo `BASURA`.

El modelo está detrás de la interfaz `Extractor`: `ExtractorHF` usa el modelo real y
`ExtractorFalso` devuelve salidas fijas para los tests.

Diferencias con la tesis:
- Sin fusión de entidades cercanas por defecto (#55): unía enfermedades distintas
  ("demencia mixta + HTA") y quitarla mejora DisTEMIST y el set sintético.
- Las conjunciones aisladas " y ", " e " y " o " son barrera de fusión (desviación
  intencional). La tesis solo cortaba en salto de línea, coma, punto, guion aislado y
  numeral, así que unía "HTA y diabetes mellitus tipo 2" en una sola entidad aunque el
  modelo las devolviera separadas, y se perdían comorbilidades y puntos de Charlson.
- Ventana deslizante (`stride`) en el pipeline (desviación intencional). La tesis
  segmentaba por caracteres (2.000) y en notas densas (siglas, cifras, laboratorios) el
  modelo truncaba en silencio en el token 512, desde unos 1.200 caracteres.
- Filtro de longitud mínimo (#56): de lo corto solo se descarta lo que no tiene letras y
  las palabras en minúscula de 2 letras o menos. La tesis descartaba todo lo de 4
  caracteres o menos; hasta #56 pasaban solo las siglas de una lista blanca
  (`SIGLAS_CLINICAS`, `PALABRAS_CORTAS_CLINICAS`), que hoy cubre las de 2 letras en
  minúscula ("dm"). Las siglas que se escriben igual que una palabra común
  (`SIGLAS_SENSIBLES_A_MAYUSCULAS`, como "LES" y el pronombre "les") solo pasan en
  mayúsculas (#24).
- La viñeta se quita antes del filtro de longitud ("1. tos" se descarta como "tos").
- La lista negra se compara sin tildes ("Diagnóstico" cae como "diagnostico").
- Las terminaciones truncadas se comparan sin distinguir mayúsculas.
- No deduplica subcadenas: eso es de `depuracion.deduplicar_internas`.
- El modelo ve las siglas conocidas en mayúscula (`siglas_para_el_modelo`, #114): distingue
  mayúsculas y en "pte con epoc" o "hta, dm, epoc" no veía nada. Se cambia una copia del
  fragmento que solo ve el modelo; las menciones se recortan del texto real.

Cada decisión heredada de la tesis (fusión, expansión, filtro de longitud, lista negra y
sufijos truncados) se puede apagar o cambiar con `ConfiguracionExtraccion`, para la
ablación de la evaluación (#42). El valor por defecto es el del pipeline.
`menciones_crudas` devuelve la salida del modelo sin ninguno de esos pasos.
"""

import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, get_args

from ner_clinic.segmentacion import segmentar
from ner_clinic.siglas import _SOLO_MAYUSCULAS, SIGLAS
from ner_clinic.tipos import Candidato, Descarte, Mencion, MotivoDescarte

NOMBRE_MODELO = "BSC-NLP4BIA/bsc-bio-ehr-es-distemist"
# Variable de entorno con la revisión (commit) del modelo en Hugging Face. La imagen
# Docker la fija para cargar exactamente el snapshot que descargó en el build.
VARIABLE_REVISION = "NER_MODELO_REVISION"
# Solapamiento (en tokens) entre ventanas de 512 cuando un fragmento no cabe en una.
STRIDE_TOKENS = 128


class Extractor(Protocol):
    """Interfaz del modelo NER.

    `extraer` recibe los textos de los fragmentos y devuelve, para cada uno, la lista de
    entidades con el formato de `pipeline("ner", aggregation_strategy="simple")`:
    diccionarios con `entity_group`, `score`, `word`, `start` y `end` (posiciones
    locales al fragmento).
    """

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]: ...


class ExtractorFalso:
    """Extractor para tests: devuelve salidas fijas según el texto del fragmento.

    Los fragmentos que no están en `salidas` no tienen entidades. Guarda en `llamadas`
    las listas de fragmentos que recibió. Las claves de `salidas` son el texto del
    fragmento tal como está en la nota; se indexan como las verá el modelo, con las siglas
    en mayúscula (`siglas_para_el_modelo`).
    """

    def __init__(self, salidas: Mapping[str, list[dict]]):
        self.salidas = {siglas_para_el_modelo(k): v for k, v in salidas.items()}
        if len(self.salidas) != len(salidas):
            raise ValueError(
                "Dos fragmentos de `salidas` solo difieren en las mayúsculas de una sigla: "
                "el modelo los ve iguales."
            )
        self.llamadas: list[list[str]] = []

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        self.llamadas.append(list(fragmentos))
        return [[dict(e) for e in self.salidas.get(f, [])] for f in fragmentos]


def _dispositivo() -> int:
    """0 (primera GPU) si hay CUDA; -1 (CPU) si no."""
    import torch

    return 0 if torch.cuda.is_available() else -1


def _argumentos_pipeline(
    nombre_modelo: str, dispositivo: int, revision: str | None = None
) -> dict[str, Any]:
    """Argumentos del pipeline NER, con ventana deslizante para más de 512 tokens.

    Sin `stride`, el pipeline trunca en silencio en el token 512 (unos 1.200 caracteres
    en notas densas) y todo lo que sigue se pierde. Con `stride`, el tokenizador rápido
    parte el texto en ventanas solapadas y el pipeline une las entidades; `start` y `end`
    siguen siendo relativos al texto completo del fragmento.

    `revision` (un commit, rama o etiqueta del repo de Hugging Face) solo se pasa si se
    da; si no, transformers usa `main`.
    """
    argumentos: dict[str, Any] = {
        "task": "ner",
        "model": nombre_modelo,
        "aggregation_strategy": "simple",
        "stride": STRIDE_TOKENS,
        "device": dispositivo,
    }
    if revision:
        argumentos["revision"] = revision
    return argumentos


def _crear_pipeline(nombre_modelo: str, dispositivo: int, revision: str | None = None) -> Any:
    from transformers import pipeline

    return pipeline(**_argumentos_pipeline(nombre_modelo, dispositivo, revision))


class ExtractorHF:
    """Extractor con el modelo de Hugging Face.

    El modelo se carga en la primera llamada a `extraer` (no al construir el objeto),
    en GPU si hay y si no en CPU. Se descarga a la caché de Hugging Face del usuario.

    `revision` fija el commit del repo del modelo. Si no se da, se lee de la variable de
    entorno `NER_MODELO_REVISION`, y si tampoco está se usa `main` (lo de siempre).
    """

    def __init__(
        self,
        nombre_modelo: str = NOMBRE_MODELO,
        batch_size: int = 8,
        revision: str | None = None,
    ):
        self.nombre_modelo = nombre_modelo
        self.batch_size = batch_size
        self.revision = revision or os.environ.get(VARIABLE_REVISION) or None
        self._pipeline: Any = None

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        if not fragmentos:
            return []
        if self._pipeline is None:
            self._pipeline = _crear_pipeline(self.nombre_modelo, _dispositivo(), self.revision)
        # Con una lista de textos (aunque tenga uno solo), transformers 5.x devuelve una
        # lista de listas; la tesis (4.x) se cubría de una lista plana, aquí no hace falta.
        salidas = self._pipeline(list(fragmentos), batch_size=self.batch_size)
        return [list(entidades) for entidades in salidas]


MAX_DISTANCIA_FUSION = 18
LONGITUD_MAXIMA_DESCARTE = 4

# Variantes del filtro de longitud (experimento de siglas, #8; #56):
# - "minimo" (por defecto desde #56): de las menciones de 4 caracteres o menos descarta
#   solo las que no tienen letras ("1", ",") y las palabras en minúscula de 2 letras o
#   menos ("de", "en"), salvo las de la lista blanca ("dm", "fa"). Las
#   siglas en mayúscula y las palabras de 3 o 4 letras ("lúes", "coma") pasan sin lista.
#   Las siglas ambiguas (`SIGLAS_SENSIBLES_A_MAYUSCULAS`: "les", "tía", con o sin tilde)
#   siguen pasando solo en mayúsculas.
# - "lista_blanca": descarta 4 caracteres o menos salvo `SIGLAS_CLINICAS` y
#   `PALABRAS_CORTAS_CLINICAS` (el de defecto hasta #56).
# - "tesis": descarta 4 caracteres o menos sin excepciones, como la tesis.
# - "ninguno": no filtra por longitud (los demás filtros de basura siguen activos).
# En DisTEMIST entrenamiento, la lista blanca descartaba 160 aciertos (siglas que no
# estaban en la lista, "lúes", "coma") por 20 falsos positivos; ver `docs/normalizacion.md`.
FiltroLongitud = Literal["minimo", "lista_blanca", "tesis", "ninguno"]
FILTROS_LONGITUD: tuple[str, ...] = get_args(FiltroLongitud)

# Barreras de la fusión de entidades cercanas (#55): `tesis` son las de la tesis;
# `ampliadas` suma los conectores que en #46 unían dos enfermedades distintas (" ni ",
# " con ", ": ", " y/o ", paréntesis) y "+" ("demencia mixta + HTA", #167).
BarrerasFusion = Literal["tesis", "ampliadas"]
BARRERAS_FUSION: tuple[str, ...] = get_args(BarrerasFusion)

# Siglas clínicas que sobreviven al filtro de longitud (se comparan en mayúsculas, salvo
# `SIGLAS_SENSIBLES_A_MAYUSCULAS`). Desviación intencional frente a la tesis (FR3): el
# filtro original las eliminaba. El experimento de siglas (#8,
# `docs/experimentos/siglas.md`) mostró que el modelo las detecta y que el filtro era lo
# que las perdía; de ahí salen los agregados.
SIGLAS_CLINICAS: frozenset[str] = frozenset(
    {
        "HTA",  # hipertensión arterial
        "EPOC",  # enfermedad pulmonar obstructiva crónica
        "IAM",  # infarto agudo de miocardio
        "IMA",  # infarto de miocardio agudo (#8)
        "SCA",  # síndrome coronario agudo (#8)
        "ACV",  # accidente cerebrovascular
        "ECV",  # enfermedad o evento cerebrovascular (#8)
        "AIT",  # ataque isquémico transitorio
        "TIA",  # ataque isquémico transitorio, sigla en inglés (#8)
        "ICC",  # insuficiencia cardiaca congestiva
        "ERC",  # enfermedad renal crónica
        "IRC",  # insuficiencia renal crónica
        "DM",  # diabetes mellitus
        "DM1",
        "DM2",
        "VIH",
        "HIV",  # VIH, sigla en inglés (#8)
        "SIDA",
        "FA",  # fibrilación auricular
        "TEP",  # tromboembolismo pulmonar
        "TVP",  # trombosis venosa profunda (#8)
        "EAP",  # enfermedad arterial periférica
        "EVP",  # enfermedad vascular periférica (#8)
        "LES",  # lupus eritematoso sistémico (#8)
        "TBC",  # tuberculosis (#8)
        "IVU",  # infección de vías urinarias (#8)
        "ERGE",  # enfermedad por reflujo gastroesofágico (#8)
        "HPB",  # hiperplasia prostática benigna (#8)
    }
)

# Siglas que se escriben igual que una palabra común: solo se conservan escritas en
# mayúsculas, como sigla ("LES activo" sí; "se les explica" o "su tia materna" no).
# Las demás siglas se comparan sin distinguir mayúsculas ("hta", "Epoc").
SIGLAS_SENSIBLES_A_MAYUSCULAS: frozenset[str] = frozenset(
    {
        "LES",  # el pronombre "les"
        "TIA",  # "tía", que en las notas suele escribirse sin tilde
    }
)

# Palabras clínicas completas de 4 letras o menos que sobreviven al filtro de longitud
# (se comparan sin distinguir mayúsculas). No son siglas: hasta #24 estaban mezcladas en
# `SIGLAS_CLINICAS`. Vienen del experimento de siglas (#8).
PALABRAS_CORTAS_CLINICAS: frozenset[str] = frozenset({"ASMA", "GOTA"})


# Siglas que el modelo ve en mayúscula aunque la nota las traiga en minúscula (#114): las
# de `siglas.SIGLAS` (el cruce con registrados) y las de `SIGLAS_CLINICAS` (el filtro de
# longitud), menos `_NO_SUBIR`. La línea base de #112 (set sintético en minúscula) mostró
# que sin esto el índice exacto cae de 92 a 80; en DisTEMIST prueba, el recall de siglas
# del modelo crudo cae de 85 a 51 de 90. La lista no se ajustó con los errores de
# DisTEMIST prueba. Las que entran, por qué son seguras en minúscula:
# - Las de enfermedad de Charlson y comorbilidades frecuentes (HTA, DM, DM1, DM2, DMID,
#   DMNID, EPOC, IAM, IMA, SCA, ACV, ECV, AIT, ICC, IC, ERC, IRC, VIH, HIV, SIDA, EVP,
#   TBC, TEP, TVP, FA, IVU, ERGE, HBP, HPB, SAHOS): en minúscula no son palabras del
#   español clínico. "ic" (interconsulta), "fa" (fosfatasa alcalina), "ima" e "ic 95 %"
#   se probaron en la revisión del PR #122 sin falsos positivos con el modelo real.
#   Falsos positivos conocidos y aceptados: "fa" como nota musical ("canta en fa mayor")
#   sale como entidad, sin categoría ni efecto en el índice; "dm" como decímetro ("talla
#   15 dm") daría diabetes, pero en las notas la talla va en cm.
_NO_SUBIR: frozenset[str] = frozenset(
    {
        "LES",  # el pronombre "les"
        "AR",  # sílaba y terminación
        "TIA",  # "tía", que en las notas suele escribirse sin tilde
        # En las notas colombianas "EAP" suele ser edema agudo de pulmón, no enfermedad
        # arterial periférica: en mayúscula el modelo la vería como sigla y la regla de
        # Charlson la mapearía a vascular periférica (revisión del PR #122). Escrita en
        # mayúscula en la nota se sigue tratando como hasta ahora.
        "EAP",
    }
)
SIGLAS_PARA_EL_MODELO: frozenset[str] = frozenset(
    (set(SIGLAS) | SIGLAS_CLINICAS) - _NO_SUBIR - _SOLO_MAYUSCULAS - SIGLAS_SENSIBLES_A_MAYUSCULAS
)
_PATRON_SIGLAS_PARA_EL_MODELO = re.compile(
    r"\b(?:" + "|".join(sorted(SIGLAS_PARA_EL_MODELO, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def siglas_para_el_modelo(texto: str) -> str:
    """`texto` con las siglas conocidas en mayúscula, del mismo largo (#114).

    "pte con epoc" da "pte con EPOC". Solo cambian palabras completas de
    `SIGLAS_PARA_EL_MODELO`; el resto del texto queda igual, así que las posiciones que
    devuelve el modelo sirven para el texto original.
    """
    return _PATRON_SIGLAS_PARA_EL_MODELO.sub(lambda m: m.group().upper(), texto)


@dataclass(frozen=True)
class ConfiguracionExtraccion:
    """Interruptores de las decisiones de la tesis, para la ablación (#42).

    Los valores por defecto reproducen el pipeline tal como está; fuera de la evaluación
    no se cambian. No mueven umbrales ni reglas, solo permiten apagarlos:

    - `max_distancia_fusion`: distancia máxima (en caracteres) para fusionar entidades
      cercanas; `None` (por defecto desde #55) no fusiona nada. La fusión de la tesis
      (`MAX_DISTANCIA_FUSION`) unía enfermedades distintas ("demencia mixta + HTA") y en
      DisTEMIST entrenamiento y el set sintético quitarla mejora todo.
    - `expandir_palabras`: completa las palabras que el modelo cortó.
    - `filtro_longitud`: variante del filtro de 4 caracteres o menos (`FiltroLongitud`).
    - `usar_lista_negra`: descarta los subtítulos y el texto administrativo de
      `LISTA_NEGRA`.
    - `descartar_sufijos_truncados`: descarta las entidades que terminan en
      `SUFIJOS_TRUNCADOS`.
    - `barreras_fusion`: qué separa dos entidades que no se fusionan (`BarrerasFusion`).
    - `cortar_parentesis`: corta la entidad en un paréntesis sin pareja (#195).
    """

    max_distancia_fusion: int | None = None  # sin fusión desde #55
    expandir_palabras: bool = True
    filtro_longitud: FiltroLongitud = "minimo"
    usar_lista_negra: bool = True
    descartar_sufijos_truncados: bool = True
    barreras_fusion: BarrerasFusion = "tesis"
    cortar_parentesis: bool = True

    def __post_init__(self):
        if self.barreras_fusion not in BARRERAS_FUSION:
            raise ValueError(
                f"barreras_fusion desconocidas: {self.barreras_fusion!r}; "
                f"opciones: {BARRERAS_FUSION}"
            )
        if self.filtro_longitud not in FILTROS_LONGITUD:
            raise ValueError(
                f"filtro_longitud desconocido: {self.filtro_longitud!r}; "
                f"opciones: {FILTROS_LONGITUD}"
            )
        if self.max_distancia_fusion is not None and self.max_distancia_fusion < 0:
            raise ValueError(
                f"max_distancia_fusion ({self.max_distancia_fusion}) es negativa; "
                "use None para no fusionar"
            )


# Configuración del pipeline normal: la que usa `analizar()`.
CONFIGURACION_POR_DEFECTO = ConfiguracionExtraccion()

# Terminaciones de entidades truncadas por el modelo.
SUFIJOS_TRUNCADOS = (" per", " end", " inf")

# Subtítulos y texto administrativo que el modelo confunde con enfermedades
# (lista de `limpiar_basura_ner` de la tesis, con sus errores de digitación).
LISTA_NEGRA: frozenset[str] = frozenset(
    {
        "diagnostico",
        "diagnosticos",
        "daignosticos",
        "diagnosticos de",
        "daignosticos de",
        "antecedentes",
        "paciente",
        "historia clinica",
        "enfermedad",
        "tratamiento",
        "impresion diagnostica",
        "plan",
        "de 1",
        "herida quriurgica",
        "lesión",
    }
)

# Barreras fuertes: salto de línea, coma, punto, guion aislado (" - ") o numeral de lista,
# como en la tesis, más las conjunciones aisladas "y", "e", "o" (desviación intencional:
# ver el docstring del módulo). La conjunción debe ser palabra completa entre espacios o
# bordes del intermedio, así "y/o" o "de origen" no cortan la fusión.
_BARRERA = re.compile(r"[\n,\.]|\s-\s|\d+[\.\)]|(?:^|\s)[yeo](?:\s|$)", re.IGNORECASE)
_BARRERA_AMPLIADA = re.compile(
    _BARRERA.pattern + r"|[+:()]|(?:^|\s)(?:ni|con|y/o)(?:\s|$)", re.IGNORECASE
)
_BARRERAS = {"tesis": _BARRERA, "ampliadas": _BARRERA_AMPLIADA}
_FIN_DE_PALABRA = frozenset(" \n\t,.;:()[]{}-*_#")
# Bordes que se recortan: los de la tesis (".,:;()- ") más cualquier espacio en blanco y
# las marcas de markdown (`*`, `_`, `#`) de un texto pegado (#211).
_BORDES = frozenset(".,:;()- \n\t\r*_#")
# Numeral de lista: "1.", "2)", "3-", "1.-"; no si sigue un dígito ("12.5 mg").
_VINETA = re.compile(r"\d+(?:[.)]-?|-)(?!\d)\s*")


def expandir_palabra_cortada(inicio: int, fin: int, texto: str) -> tuple[int, int]:
    """Extiende `[inicio, fin)` hasta los bordes de palabra si el modelo la cortó.

    Avanza a la derecha y retrocede a la izquierda hasta un espacio o un signo de
    puntuación (`_FIN_DE_PALABRA`), como `expandir_palabra_cortada` de la tesis.
    """
    while fin < len(texto) and texto[fin] not in _FIN_DE_PALABRA:
        fin += 1
    while inicio > 0 and texto[inicio - 1] not in _FIN_DE_PALABRA:
        inicio -= 1
    return inicio, fin


def fusionar_entidades_cercanas(
    entidades: list[dict],
    texto: str,
    max_distancia: int | None = None,
    expandir: bool = True,
    barreras: BarrerasFusion = "tesis",
) -> list[tuple[int, int]]:
    """Une entidades partidas por el modelo y completa las palabras cortadas.

    Dos entidades consecutivas se unen si las separan entre 0 y `max_distancia`
    caracteres y entre ellas no hay una barrera (`_BARRERA`), contando los bordes de las
    dos entidades (#54). Porta
    `fusionar_entidades_cercanas` de la tesis; devuelve rangos `(inicio, fin)` locales
    al `texto` del fragmento.

    Para la ablación: con `max_distancia=None` no une nada (cada entidad queda sola) y
    con `expandir=False` no completa las palabras cortadas. `barreras="ampliadas"` no
    une a través de los conectores de `_BARRERA_AMPLIADA` (#55).
    """
    if not entidades:
        return []

    def cerrar(inicio: int, fin: int) -> tuple[int, int]:
        return expandir_palabra_cortada(inicio, fin, texto) if expandir else (inicio, fin)

    ordenadas = sorted(entidades, key=lambda e: e["start"])
    rangos: list[tuple[int, int]] = []
    inicio, fin = ordenadas[0]["start"], ordenadas[0]["end"]
    for siguiente in ordenadas[1:]:
        distancia = siguiente["start"] - fin
        # La barrera se busca también en los bordes de las dos entidades: el modelo a veces
        # incluye el salto de línea en la entidad ("TB\n"), y entonces no está entre ellas
        # (#54).
        desde, hasta = fin, siguiente["start"]
        while desde > inicio and texto[desde - 1] in _BORDES:
            desde -= 1
        while hasta < siguiente["end"] and texto[hasta] in _BORDES:
            hasta += 1
        intermedio = texto[desde:hasta]
        if (
            max_distancia is not None
            and 0 <= distancia <= max_distancia
            and not _BARRERAS[barreras].search(intermedio)
        ):
            fin = siguiente["end"]
        else:
            rangos.append(cerrar(inicio, fin))
            inicio, fin = siguiente["start"], siguiente["end"]
    rangos.append(cerrar(inicio, fin))
    return rangos


def _normalizar(texto: str) -> str:
    """Colapsa los espacios internos (incluidos saltos de línea) en uno solo."""
    return re.sub(r"\s+", " ", texto)


def _sin_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


_LISTA_NEGRA_SIN_TILDES = frozenset(_sin_tildes(p) for p in LISTA_NEGRA)


# Viñeta de letra o romano antes de ")": "a)", "B)", "iv)".
_VINETA_DE_LETRA = re.compile(r"[ \t]*[A-Za-z]{1,3}[ \t]*")


def cortar_parentesis(inicio: int, fin: int, texto: str) -> tuple[int, int]:
    """Corta `[inicio, fin)` en el primer paréntesis sin pareja (#195).

    El modelo a veces une una sigla entre paréntesis con lo que sigue: en "enfermedad
    pulmonar obstructiva crónica (EPOC) secundaria a tabaquismo" devuelve "EPOC)
    secundaria", que ya no es la sigla y queda fuera del contexto de "sospecha". Un ")"
    sin "(" antes, o un "(" que no se cierra, corta la entidad: queda lo de antes. Si el
    paréntesis es el primer carácter, `_recortar` lo quita. Los paréntesis con pareja
    no se tocan: "insuficiencia cardiaca (IC) secundaria a HTA" es una sola entidad. Una
    viñeta de letra ("a) asma bronquial", "iv) diabetes") no es un paréntesis: se quita y
    queda la enfermedad.
    """
    abiertos: list[int] = []
    for posicion in range(inicio, fin):
        caracter = texto[posicion]
        if caracter == "(":
            abiertos.append(posicion)
        elif caracter == ")":
            if abiertos:
                abiertos.pop()
            elif _VINETA_DE_LETRA.fullmatch(texto, inicio, posicion):
                inicio = posicion + 1  # "a) asma", "ii) diabetes": es viñeta, se quita
            elif posicion > inicio:  # al inicio lo quita `_recortar`
                return inicio, posicion
    if abiertos and abiertos[0] > inicio:
        return inicio, abiertos[0]
    return inicio, fin


def _recortar(inicio: int, fin: int, texto: str) -> tuple[int, int]:
    """Quita puntuación y espacios de los bordes y la viñeta numérica inicial.

    Repite hasta que no cambie nada, porque al quitar la viñeta pueden quedar bordes
    nuevos ("1. (HTA)" → "(HTA)" → "HTA").
    """
    while True:
        antes = (inicio, fin)
        while inicio < fin and texto[inicio] in _BORDES:
            inicio += 1
        while fin > inicio and texto[fin - 1] in _BORDES:
            fin -= 1
        vineta = _VINETA.match(texto, inicio, fin)
        if vineta:
            inicio = vineta.end()
        if (inicio, fin) == antes:
            return inicio, fin


def _en_lista_blanca(visible: str) -> bool:
    """Si `visible` es una sigla clínica o una palabra clínica corta conocida.

    Las siglas ambiguas (`SIGLAS_SENSIBLES_A_MAYUSCULAS`) solo cuentan escritas tal cual,
    en mayúsculas; las demás siglas y las palabras cortas, sin distinguir mayúsculas.
    """
    mayusculas = visible.upper()
    if mayusculas in SIGLAS_SENSIBLES_A_MAYUSCULAS:
        return visible == mayusculas
    return mayusculas in SIGLAS_CLINICAS or mayusculas in PALABRAS_CORTAS_CLINICAS


def _es_corta_descartable(visible: str, filtro_longitud: FiltroLongitud) -> bool:
    if filtro_longitud == "ninguno" or len(visible) > LONGITUD_MAXIMA_DESCARTE:
        return False
    if filtro_longitud == "tesis":
        return True
    if _en_lista_blanca(visible):
        return False
    if filtro_longitud == "lista_blanca":
        return True
    if _sin_tildes(visible).upper() in SIGLAS_SENSIBLES_A_MAYUSCULAS:
        return True  # "les", "tía": la palabra común, no la sigla (#24)
    letras = sum(c.isalpha() for c in visible)
    # Solo en minúscula: "Mx" (metástasis) es un acierto en DisTEMIST entrenamiento y
    # "En" o "De" al inicio de frase no aparecieron como entidad.
    return letras == 0 or (letras <= 2 and visible == visible.lower())


def _regla_de_basura(
    visible: str, configuracion: ConfiguracionExtraccion = CONFIGURACION_POR_DEFECTO
) -> str | None:
    """Devuelve el detalle de la regla que descarta `visible`, o `None` si pasa.

    Solo aplica las reglas que `configuracion` deja encendidas.
    """
    minusculas = visible.lower()
    if _es_corta_descartable(visible, configuracion.filtro_longitud):
        return f"longitud de {LONGITUD_MAXIMA_DESCARTE} caracteres o menos"
    if configuracion.descartar_sufijos_truncados:
        for sufijo in SUFIJOS_TRUNCADOS:
            if minusculas.endswith(sufijo):
                return f"termina en '{sufijo}'"
    if configuracion.usar_lista_negra and _sin_tildes(minusculas) in _LISTA_NEGRA_SIN_TILDES:
        return "lista negra administrativa"
    return None


def extraer_candidatos(
    texto_limpio: str,
    extractor: Extractor,
    max_caracteres: int = 2000,
    filtro_longitud: FiltroLongitud | None = None,
    configuracion: ConfiguracionExtraccion = CONFIGURACION_POR_DEFECTO,
) -> tuple[list[Candidato], list[Descarte]]:
    """Corre el modelo sobre el texto limpio y devuelve candidatos y descartes.

    Pasos: segmenta el texto, llama al extractor una sola vez con todos los fragmentos,
    fusiona entidades cercanas, completa palabras cortadas, corta en los paréntesis sin
    pareja, recorta bordes y viñetas, y
    aplica los filtros de basura. Las menciones con el mismo texto (sin distinguir
    mayúsculas ni espacios repetidos) se agrupan en un candidato; los candidatos salen
    en orden de primera aparición. No deduplica subcadenas ni variantes ortográficas:
    eso es de `depuracion.deduplicar_internas`.

    `configuracion` apaga o cambia las decisiones de la tesis (ver
    `ConfiguracionExtraccion`); fuera de la evaluación se deja el valor por defecto.
    `filtro_longitud`, si se pasa, reemplaza el de `configuracion` (es el atajo que usa
    el experimento de siglas).

    Las posiciones de las menciones son del `texto_limpio`.
    """
    if filtro_longitud is not None:
        # `replace` vuelve a validar el valor en `ConfiguracionExtraccion.__post_init__`.
        configuracion = replace(configuracion, filtro_longitud=filtro_longitud)
    fragmentos = segmentar(texto_limpio, max_caracteres=max_caracteres)
    if not fragmentos:
        return [], []

    salidas = extractor.extraer([siglas_para_el_modelo(f.texto) for f in fragmentos])

    candidatos: dict[str, Candidato] = {}
    descartes: dict[tuple[str, str], Descarte] = {}
    for fragmento, entidades in zip(fragmentos, salidas, strict=True):
        rangos = fusionar_entidades_cercanas(
            entidades,
            fragmento.texto,
            max_distancia=configuracion.max_distancia_fusion,
            expandir=configuracion.expandir_palabras,
            barreras=configuracion.barreras_fusion,
        )
        for inicio_local, fin_local in rangos:
            inicio, fin = _recortar(
                fragmento.desplazamiento + inicio_local,
                fragmento.desplazamiento + fin_local,
                texto_limpio,
            )
            # Después de quitar la viñeta: el ")" de "3) asma" no es un paréntesis.
            if configuracion.cortar_parentesis:
                inicio, fin = _recortar(*cortar_parentesis(inicio, fin, texto_limpio), texto_limpio)
            if inicio == fin:  # solo puntuación: no hay nada que reportar
                continue
            mencion = Mencion(texto_limpio[inicio:fin], inicio, fin)
            # Los espacios se colapsan solo para comparar y agrupar: el texto del
            # candidato es el de la nota, tal como se escribió.
            normalizado = _normalizar(mencion.texto)
            clave = normalizado.lower()

            regla = _regla_de_basura(normalizado, configuracion)
            if regla is not None:
                descartes.setdefault(
                    (clave, regla), Descarte(mencion.texto, MotivoDescarte.BASURA, regla)
                )
            elif clave in candidatos:
                candidatos[clave].menciones.append(mencion)
            else:
                candidatos[clave] = Candidato(texto=mencion.texto, menciones=[mencion])
    return list(candidatos.values()), list(descartes.values())


def menciones_crudas(
    texto_limpio: str, extractor: Extractor, max_caracteres: int = 2000
) -> list[Mencion]:
    """Lo que devuelve el modelo, sin fusión, expansión, recorte ni filtros.

    Segmenta el texto limpio igual que `extraer_candidatos` y llama al extractor una sola
    vez. Cada entidad del modelo es una mención con el `start` y el `end` que él dio, en
    posiciones del `texto_limpio` y en orden de aparición. Es la fila "modelo crudo" de
    la evaluación (#42). Se omiten las entidades vacías (`start == end`). El modelo ve las
    siglas en mayúscula, igual que en `extraer_candidatos`.
    """
    fragmentos = segmentar(texto_limpio, max_caracteres=max_caracteres)
    if not fragmentos:
        return []

    salidas = extractor.extraer([siglas_para_el_modelo(f.texto) for f in fragmentos])

    menciones = []
    for fragmento, entidades in zip(fragmentos, salidas, strict=True):
        for entidad in entidades:
            inicio = fragmento.desplazamiento + entidad["start"]
            fin = fragmento.desplazamiento + entidad["end"]
            if inicio < fin:
                menciones.append(Mencion(texto_limpio[inicio:fin], inicio, fin))
    menciones.sort(key=lambda m: (m.inicio, m.fin))
    return menciones
