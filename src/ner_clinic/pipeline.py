"""Orquestador del pipeline: `analizar()` une todas las fases.

Orden (el de la tesis, más el cruce con los diagnósticos registrados y el índice):

1. `limpieza.limpiar`: quita encabezados de folio, guiones y saltos repetidos, con el
   mapa de posiciones al original.
2. `extraccion.extraer_candidatos`: segmenta el texto limpio, corre el modelo y descarta
   la basura.
3. `depuracion.deduplicar_internas`: fusiona variantes y entidades contenidas en otras.
4. `contexto`: confirmada, negada, posible o familiar para cada mención, según su
   posición en el texto limpio (donde corren las regex). Si las menciones de un
   candidato quedan en estados distintos, salen una entidad por estado.
5. `depuracion.cruzar_con_registrados`: marca (o descarta) las confirmadas que ya están
   entre los diagnósticos registrados del egreso.
6. `charlson.reglas.mapear`: categoría de Charlson y CIE-10, para confirmadas y
   posibles. Si el nombre de la entidad no mapea, se prueba con sus menciones, salvo
   que la regla de la mención excluya el nombre (#113). Si nada mapea, se repite
   corrigiendo la ortografía (`charlson.correccion`, #115). Con `normalizador`, lo que
   las reglas dejan sin categoría lo decide la segunda etapa de
   `ner_clinic.normalizacion` (modo `hibrido`, #180); en modo `normalizador` decide ella
   sola (#165), en el mismo orden.
7. `charlson.indice.calcular`: índice de Charlson de la nota (solo confirmadas) y el
   índice si se confirman las posibles.
8. Traduce las posiciones de las menciones al texto original.

El modelo NER se pasa como `extractor`. Si no se pasa, se usa un `ExtractorHF`
compartido por todo el proceso, que se crea en la primera llamada y carga el modelo
solo cuando hay texto que analizar. Importar este módulo no carga `torch`.

Para la evaluación (#42), `_analizar` es el mismo orquestador con los interruptores de la
ablación (`ConfiguracionExtraccion` y la deduplicación interna), y `menciones_del_modelo`
devuelve la salida cruda del modelo en posiciones del texto original. `analizar()` llama
a `_analizar` con los valores por defecto.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from ner_clinic.charlson import correccion, indice, reglas
from ner_clinic.contexto import (
    _DISPARADORES_NEGACION,
    _PALABRA,
    CLAVES_POSIBLE,
    Intervalo,
    clasificar_mencion,
    intervalos,
)
from ner_clinic.depuracion import (
    CATEGORIAS_CON_CRUCE,
    contenida_por_palabras,
    cruzar_con_registrados,
    cruzar_por_categoria,
    deduplicar_internas,
)
from ner_clinic.extraccion import (
    CONFIGURACION_POR_DEFECTO,
    ConfiguracionExtraccion,
    Extractor,
    ExtractorHF,
    extraer_candidatos,
    menciones_crudas,
)
from ner_clinic.limpieza import TextoLimpio, limpiar
from ner_clinic.tipos import (
    Candidato,
    CandidatoIndice,
    CategoriaCharlson,
    Entidad,
    Estado,
    Mencion,
    Resultado,
)

if TYPE_CHECKING:  # importar `normalizacion` carga torch: solo para los tipos
    from ner_clinic.normalizacion.decision import Decision, SegundaEtapa

__all__ = ["MODOS_MAPEO", "ModoMapeo", "analizar"]

ModoMapeo = Literal["hibrido", "normalizador"]
MODOS_MAPEO: tuple[ModoMapeo, ...] = ("hibrido", "normalizador")

_EXTRACTOR: ExtractorHF | None = None


def _extractor_compartido() -> ExtractorHF:
    """El `ExtractorHF` del proceso; se crea una sola vez (el modelo carga al usarlo)."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = ExtractorHF()
    return _EXTRACTOR


def analizar(
    texto: str | None,
    diagnosticos_registrados: Iterable[str] | None = None,
    descartar_registrados: bool = False,
    extractor: Extractor | None = None,
    normalizador: "SegundaEtapa | None" = None,
    modo_mapeo: ModoMapeo = "hibrido",
) -> Resultado:
    """Extrae las comorbilidades de una nota clínica y calcula su índice de Charlson.

    Args:
        texto: la nota clínica tal como la escribió el médico.
        diagnosticos_registrados: diagnósticos codificados del egreso (principal y
            secundarios), como textos. Si se envían, cada confirmada sale con
            `registrado=True` (con `coincide_con` y `coincide_por`: por texto o, en la
            diabetes, por categoría de Charlson, #176) o `registrado=False` (posible
            subregistro).
        descartar_registrados: si es `True`, las confirmadas que coinciden con un
            diagnóstico registrado se descartan, como hacía la tesis, en vez de marcarse.
        extractor: el modelo NER; por defecto, el `ExtractorHF` compartido.
        normalizador: la segunda etapa del normalizador por embeddings
            (`ner_clinic.normalizacion.artefactos.cargar()`). Con `None` (por defecto),
            la categoría de Charlson sale de las reglas, como siempre. Con él, depende
            de `modo_mapeo`. Las entidades que mapea el normalizador dicen
            `mapeo="normalizador"` y traen el `candidato` del índice que la explica.
        modo_mapeo: solo cuenta con `normalizador` (sin él se ignora: solo reglas).
            `"hibrido"` (por defecto, #180):
            primero las reglas y, para lo que dejan sin categoría, SapBERT y el
            clasificador sobre el top-20; `"normalizador"`: solo SapBERT y el
            clasificador (#165). Los diagnósticos registrados se mapean igual.

    Returns:
        Un `Resultado` con las entidades en orden de aparición, los descartes de todas
        las fases con su motivo, el índice de Charlson (solo confirmadas) y el índice si
        se confirman las posibles (`indice_charlson_si_se_confirman`). Las posibles
        llevan categoría y CIE-10, pero no `registrado`. Si la categoría salió de corregir
        una palabra mal escrita, la entidad trae `correcciones`. Una enfermedad puede salir dos
        veces con estados distintos (negada en los antecedentes y confirmada en el
        ingreso), cada una con sus menciones. Las posiciones de las menciones son del
        `texto` recibido: `texto[m.inicio:m.fin] == m.texto`. Texto vacío o `None`
        devuelve un `Resultado` vacío sin cargar el modelo.
    """
    return _analizar(
        texto,
        diagnosticos_registrados=diagnosticos_registrados,
        descartar_registrados=descartar_registrados,
        extractor=extractor,
        normalizador=normalizador,
        modo_mapeo=modo_mapeo,
    )


def _analizar(
    texto: str | None,
    diagnosticos_registrados: Iterable[str] | None = None,
    descartar_registrados: bool = False,
    extractor: Extractor | None = None,
    configuracion: ConfiguracionExtraccion = CONFIGURACION_POR_DEFECTO,
    deduplicar: bool = True,
    normalizador: "SegundaEtapa | None" = None,
    modo_mapeo: ModoMapeo = "hibrido",
    filtros_respaldo: bool = True,
) -> Resultado:
    """`analizar()` con los interruptores de la ablación de la evaluación (#42).

    `configuracion` apaga o cambia las decisiones de la extracción (ver
    `ConfiguracionExtraccion`) y `deduplicar=False` se salta
    `depuracion.deduplicar_internas`; `filtros_respaldo=False` quita los filtros del
    respaldo del híbrido (#185). Con los valores por defecto es exactamente `analizar()`;
    no es API pública.
    """
    if modo_mapeo not in MODOS_MAPEO:
        raise ValueError(f"modo_mapeo debe ser uno de {MODOS_MAPEO}, no {modo_mapeo!r}.")
    limpio = limpiar(texto)
    if not limpio.texto:
        return Resultado()
    if extractor is None:
        extractor = _extractor_compartido()

    candidatos, descartes = extraer_candidatos(limpio.texto, extractor, configuracion=configuracion)
    if deduplicar:
        candidatos, duplicados = deduplicar_internas(candidatos)
        descartes += duplicados

    # Se clasifica en coordenadas del texto limpio; `_a_original` traduce al final.
    entidades = _clasificar(candidatos, limpio.texto)

    # Lista: se recorre dos veces (por texto aquí y por categoría después del mapeo)
    diagnosticos = list(diagnosticos_registrados or [])
    entidades, registrados = cruzar_con_registrados(
        entidades, diagnosticos, descartar=descartar_registrados
    )
    descartes += registrados

    # Las posibles también se mapean: se muestran con su categoría y su código, aunque
    # solo sumen al índice si se confirman.
    a_mapear = [e for e in entidades if e.estado in (Estado.CONFIRMADA, Estado.POSIBLE)]
    if normalizador is not None and modo_mapeo == "normalizador":
        _normalizar(a_mapear, normalizador)
    else:
        sin_categoria = [e for e in a_mapear if not _con_reglas(e)]
        if normalizador is not None and sin_categoria:
            _normalizar(sin_categoria, normalizador, respaldo=filtros_respaldo)
    # Los códigos más parecidos, para que el experto elija (#230): también los de las
    # entidades que mapearon las reglas.
    if normalizador is not None and a_mapear:
        alternativas = normalizador.alternativas([e.texto for e in a_mapear])
        for entidad in a_mapear:
            entidad.alternativas_cie10 = tuple(alternativas.get(entidad.texto, ()))

    # Segunda pasada del cruce, ya con las categorías (#176)
    registrados_ = [d.strip() for d in diagnosticos if isinstance(d, str) and d.strip()]
    if registrados_ and _hay_para_cruzar(entidades, descartar_registrados):
        entidades, por_categoria = cruzar_por_categoria(
            entidades,
            list(
                zip(
                    registrados_,
                    _categorias(registrados_, normalizador, modo_mapeo, filtros_respaldo),
                    strict=True,
                )
            ),
            descartar=descartar_registrados,
        )
        descartes += por_categoria

    entidades = [_a_original(entidad, limpio) for entidad in entidades]
    entidades.sort(key=lambda e: e.inicio)
    return Resultado(
        entidades=entidades,
        descartes=descartes,
        indice_charlson=indice.calcular(entidades),
        indice_charlson_si_se_confirman=indice.calcular(entidades, con_posibles=True),
    )


def menciones_del_modelo(texto: str | None, extractor: Extractor | None = None) -> list[Mencion]:
    """Menciones crudas del modelo en posiciones del `texto` original (fila "modelo crudo").

    Limpia el texto como `analizar()` (el modelo ve el mismo texto limpio y los mismos
    fragmentos) y devuelve cada entidad del modelo tal cual, sin fusión, expansión,
    recorte, filtros ni deduplicación, en orden de aparición:
    `texto[m.inicio:m.fin] == m.texto`. Texto vacío o `None` devuelve `[]` sin cargar el
    modelo.

    La traducción usa el mapa de `limpiar`: si una mención cruza un tramo eliminado (un
    encabezado de folio, una línea de guiones), su rango en el original lo abarca, igual
    que en `analizar()`.
    """
    limpio = limpiar(texto)
    if not limpio.texto:
        return []
    if extractor is None:
        extractor = _extractor_compartido()
    return [limpio.mencion_a_original(m) for m in menciones_crudas(limpio.texto, extractor)]


def _normalizar(
    entidades: Sequence[Entidad], normalizador: "SegundaEtapa", respaldo: bool = False
) -> None:
    """Categoría, CIE-10 y candidato de cada entidad con la segunda etapa del normalizador.

    Los nombres y las menciones de todas las entidades se deciden en un solo lote. Como
    `_mapear` (#113), primero el nombre y, si no da categoría, una mención. Hay dos
    salvaguardas más que con las reglas, porque el normalizador da categoría a textos que
    las reglas no conocen:

    - Solo cuentan las menciones que son una variante del nombre ("EPOC" para "eppoc"), no
      las que la deduplicación fusionó por estar contenidas en él: "insuficiencia renal"
      no le da renal crónica a "insuficiencia renal aguda".
    - La mención no vale si la exclusión de su categoría quita el nombre, tal como está
      escrito o con la ortografía corregida ("pre diabetis", #124).

    `correcciones` sale de las reglas solo si dan la misma categoría corrigiendo la
    ortografía.

    Con `respaldo=True` (el modo híbrido, #185), cada decisión pasa además por
    `_respaldo_valido`.
    """
    from ner_clinic.charlson.base import normalizar
    from ner_clinic.normalizacion.decision import excluida

    textos = [t for e in entidades for t in (e.texto, *(m.texto for m in e.menciones))]
    decisiones = normalizador.decidir(textos)

    def de(texto: str) -> "Decision | None":
        d = decisiones[texto]
        return d if d is not None and (not respaldo or _respaldo_valido(texto, d)) else None

    for entidad in entidades:
        nombre = normalizar(entidad.texto)
        nombres = {entidad.texto, correccion.corregir(entidad.texto)[0]}
        decision = de(entidad.texto) or next(
            (
                d
                for m in entidad.menciones
                if not _contenida(normalizar(m.texto), nombre)
                and (d := de(m.texto)) is not None
                and not any(excluida(n, d.categoria) for n in nombres)
            ),
            None,
        )
        # La segunda etapa real ya excluye sobre el texto tal como viene; aquí se suma el
        # nombre corregido ("pre diabetis" → "pre diabetes"), como en las menciones.
        if decision is None or any(excluida(n, decision.categoria) for n in nombres):
            continue
        entidad.categoria_charlson, entidad.cie10 = decision.categoria, decision.codigo
        entidad.mapeo = "normalizador"
        entidad.candidato = _candidato(decision)
        reglas_ = _mapear(entidad)
        if reglas_ is not None and reglas_.categoria is decision.categoria:
            entidad.correcciones = reglas_.correcciones


def _con_reglas(entidad: Entidad) -> bool:
    """Categoría, CIE-10 y correcciones de las reglas (`_mapear`); si no mapean, `False`."""
    mapeo = _mapear(entidad)
    if mapeo is None:
        return False
    entidad.categoria_charlson, entidad.cie10 = mapeo.categoria, mapeo.cie10
    entidad.correcciones = mapeo.correcciones
    entidad.mapeo = "reglas"
    return True


# Señales de metástasis en el texto normalizado (#185): sin ellas, el respaldo no da
# tumor metastásico ("proliferación neoplásica", "lesión tumoral" pesaban 6 puntos).
_SENAL_METASTASIS = re.compile(
    r"me?s?tasta|\bmts\b|\bmetas\b|secundari|disemina|infiltra|carcinomatos|siembra|implantes?\b|"
    r"(?<![a-z])[cp]?m1[a-c]?\b|[tn]\d[a-c]?m1|estadio\s+(iv[a-c]?|4)\b"
)
# Una lesión focal no es una hepatopatía crónica ("lesiones hepáticas", "LOE en hígado").
_LESION_HEPATICA = re.compile(
    r"lesion(es)?\s+(ocupantes?\s+de\s+espacio\s+)?(en\s+(el\s+)?)?(hepatic|higado)|\bloes?\b"
)
_VOCALES = re.compile(r"[aeiou]")
_TNM = re.compile(r"[cp]?t[\dxis]+[a-c]?n[\dx]+[a-c]?m[\dx]+[a-c]?", re.IGNORECASE)


def _es_sigla(texto: str) -> bool:
    """Sigla para el respaldo: hasta 3 letras en cualquier caja ("SB", "sb", "S.B."), o en
    mayúsculas con hasta 4 caracteres o sin vocales. Las notas colombianas suelen ir en
    mayúsculas: "GLIOMA" o "TUMOR" no son siglas."""
    from ner_clinic.charlson.base import normalizar
    from ner_clinic.normalizacion.decision import PATRON_SIGLA

    limpio = texto.strip().replace(".", "")
    if _TNM.fullmatch(limpio):  # una estadificación, no una sigla
        return False
    if limpio.isalpha() and len(limpio) <= 3:
        return True
    return bool(PATRON_SIGLA.fullmatch(limpio)) and (
        len(limpio) <= 4 or not _VOCALES.search(normalizar(limpio))
    )


def _respaldo_valido(texto: str, decision: "Decision") -> bool:
    """Si la decisión del normalizador vale como respaldo de las reglas (#185).

    - Una sigla ("SB", "NM", "SA") solo vale si el candidato es la misma sigla: SapBERT
      acerca siglas distintas por su forma ("SB" → "LB", linfoma; "NM" → "DM"). Cuesta
      recall: una sigla correcta cuyo candidato es el nombre completo también se pierde.
    - Tumor metastásico solo con una señal de metástasis en la mención: los términos
      genéricos ("proliferación neoplásica", "lesión tumoral") salían metastásicos.
    - Hepatopatía leve no para una lesión focal. Va aquí y no en la regla: "cirrosis con
      LOE" la siguen mapeando las reglas.

    Elegido con CodiEsp entrenamiento.
    """
    from ner_clinic.charlson.base import normalizar

    normalizado = normalizar(texto)
    if _es_sigla(texto):
        candidato = decision.candidato
        if candidato is None or normalizar(candidato.descripcion) != normalizado:
            return False
    if decision.categoria is CategoriaCharlson.TUMOR_METASTASICO:
        return bool(_SENAL_METASTASIS.search(normalizado))
    if decision.categoria is CategoriaCharlson.HEPATICA_LEVE:
        return not _LESION_HEPATICA.search(normalizado)
    return True


def _contenida(mencion: str, nombre: str) -> bool:
    """Si la mención (normalizada) está dentro del nombre y es más corta: una parte, no
    una variante."""
    return mencion != nombre and contenida_por_palabras(mencion, nombre)


def _hay_para_cruzar(entidades: Sequence[Entidad], descartar: bool) -> bool:
    """Si alguna confirmada sin coincidencia por texto es de una categoría que cruza: si no,
    no hace falta mapear los registrados (con el normalizador, una pasada de SapBERT)."""
    return any(
        e.estado is Estado.CONFIRMADA
        and (e.registrado is False or (descartar and e.registrado is None))
        and e.categoria_charlson in CATEGORIAS_CON_CRUCE
        for e in entidades
    )


def _categorias(
    textos: Sequence[str],
    normalizador: "SegundaEtapa | None",
    modo_mapeo: ModoMapeo = "hibrido",
    filtros_respaldo: bool = True,
) -> list[CategoriaCharlson | None]:
    """La categoría de Charlson de cada diagnóstico registrado, con el mismo mapeo que las
    entidades: las reglas con el corrector y, si se pasa el normalizador, él para lo que
    las reglas no mapean (`hibrido`) o para todo (`normalizador`)."""
    if normalizador is not None and modo_mapeo == "normalizador":
        categorias: list[CategoriaCharlson | None] = [None] * len(textos)
    else:
        categorias = [m.categoria if (m := correccion.mapear(t)) else None for t in textos]
    faltan = [t for t, c in zip(textos, categorias, strict=True) if c is None]
    if normalizador is None or not faltan:
        return categorias
    decisiones = normalizador.decidir(faltan)
    respaldo = modo_mapeo == "hibrido" and filtros_respaldo
    return [
        c
        if c is not None
        else (
            d.categoria
            if (d := decisiones[t]) and (not respaldo or _respaldo_valido(t, d))
            else None
        )
        for t, c in zip(textos, categorias, strict=True)
    ]


def _candidato(decision: "Decision") -> CandidatoIndice | None:
    c = decision.candidato
    return None if c is None else CandidatoIndice(c.descripcion, c.codigo, c.origen, c.similitud)


def _mapear(entidad: Entidad) -> correccion.Mapeo | None:
    """Categoría y CIE-10 de la entidad: primero sin corregir la ortografía y después
    corrigiéndola (#115), en cada caso con el nombre y luego con las menciones (#113).

    La fusión puede nombrar la entidad con una variante mal escrita ("EPOC, epoc y eppoc"
    queda "eppoc"); entonces gana la primera mención, en orden de aparición, que mapea, y
    una mención bien escrita gana a corregir el nombre. Una mención no vale si la regla
    que la mapea conoce el nombre y lo excluye ("diabetes gestacional", fusionada por
    contención con "diabetes"): la exclusión manda.
    """
    for mapear, nombre in (
        (_mapear_exacto, entidad.texto),
        (correccion.mapear, correccion.corregir(entidad.texto)[0]),
    ):
        mapeo = mapear(entidad.texto)
        if mapeo is not None:
            return mapeo
        # En la pasada corregida, la exclusión se busca en el nombre corregido: "pre
        # diabetis" no la muestra, "pre diabetes" sí.
        for mencion in entidad.menciones:
            mapeo = mapear(mencion.texto)
            if mapeo is not None and not reglas.reconoce(nombre, mapeo.categoria):
                return mapeo
    return None


def _mapear_exacto(texto: str) -> correccion.Mapeo | None:
    mapeo = reglas.mapear(texto)
    return None if mapeo is None else correccion.Mapeo(*mapeo)


def _clasificar(candidatos: Iterable[Candidato], texto_limpio: str) -> list[Entidad]:
    """Una entidad del paciente por candidato y, aparte, una familiar si la hay.

    Cada mención se clasifica por su posición en `texto_limpio` (`contexto`) y guarda su
    estado (`estados_menciones`). La entidad del paciente es confirmada si alguna mención
    lo es y, si no, toma el de la última (`_estado_final`, #208); los demás van en
    `otros_estados`. Las entidades salen en el orden de su primera mención.

    La sigla o el código entre paréntesis pegado a una mención ("artritis reumatoide
    (M05)", "enfermedad pulmonar obstructiva crónica (EPOC)") es otra mención de esa
    entidad, no una entidad aparte (#203): pasa a su candidato si es la misma enfermedad
    (`_misma_entidad`).
    """
    candidatos = list(candidatos)
    tramos = intervalos(texto_limpio)
    estados = {
        (m.inicio, m.fin): clasificar_mencion(m, tramos) for c in candidatos for m in c.menciones
    }
    aclaraciones = _heredar_parentesis(estados, texto_limpio)
    _propagar_enumeraciones(estados, texto_limpio, aclaraciones, tramos)
    _heredar_parentesis(estados, texto_limpio)  # la sigla toma el estado ya propagado
    de_quien = {(m.inicio, m.fin): i for i, c in enumerate(candidatos) for m in c.menciones}
    menciones_de: list[list[Mencion]] = [[] for _ in candidatos]
    for i, candidato in enumerate(candidatos):
        for mencion in candidato.menciones:
            exterior = aclaraciones.get((mencion.inicio, mencion.fin))
            if exterior is not None and _misma_entidad(mencion, exterior, texto_limpio):
                i_destino = de_quien[exterior]
            else:
                i_destino = i
            menciones_de[i_destino].append(mencion)
    entidades = []
    for candidato, menciones_candidato in zip(candidatos, menciones_de, strict=True):
        del_paciente: list[Mencion] = []
        familiares: list[Mencion] = []
        for mencion in sorted(menciones_candidato, key=lambda m: m.inicio):
            es_familiar = estados[(mencion.inicio, mencion.fin)] is Estado.FAMILIAR
            (familiares if es_familiar else del_paciente).append(mencion)
        if del_paciente:
            secuencia = [estados[(m.inicio, m.fin)] for m in del_paciente]
            # La mención que solo presenta el tema no decide si en la misma oración hay
            # otra de la misma enfermedad (#212).
            decisivas = [
                e
                for i, (e, m) in enumerate(zip(secuencia, del_paciente, strict=True))
                if not _presenta_el_tema(m, del_paciente[i + 1 :], texto_limpio)
            ]
            final = _estado_final(decisivas or secuencia)
            entidad = candidato.a_entidad(final, del_paciente)
            entidad.otros_estados = _otros_estados(secuencia, final)
            entidad.estados_menciones = tuple(secuencia)
            entidades.append(entidad)
        if familiares:
            entidades.append(candidato.a_entidad(Estado.FAMILIAR, familiares))
    entidades.sort(key=lambda e: e.menciones[0].inicio)
    return entidades


# "Respecto a la insuficiencia cardíaca, …" presenta el tema; no afirma nada (#212).
_INTRODUCTOR_DE_TEMA = re.compile(
    r"\b(?:(?:con\s+)?respecto\s+(?:a|al)|en\s+cuanto\s+(?:a|al)|(?:en|con)\s+relaci[oó]n\s+"
    r"(?:a|al|con)|acerca\s+de(?:l)?)(?:\s+(?:la|el|los|las|su|sus))?\s+$",
    re.IGNORECASE,
)


def _presenta_el_tema(mencion: Mencion, siguientes: list[Mencion], texto: str) -> bool:
    """Si la mención solo introduce el tema y la enfermedad sigue en la misma oración.

    "Respecto a la insuficiencia cardíaca, la historia anterior mencionaba una sospecha de
    insuficiencia cardíaca…": la primera no afirma nada. En "Respecto a la HTA, en manejo
    con losartán. Se descarta HTA secundaria" la oración siguiente habla de otra cosa y la
    primera sigue contando (revisión de #212).
    """
    if not _INTRODUCTOR_DE_TEMA.search(texto, max(0, mencion.inicio - 40), mencion.inicio):
        return False
    return any(not _FIN_DE_ORACION.search(texto, mencion.fin, otra.inicio) for otra in siguientes)


_FIN_DE_ORACION = re.compile(r"[.;\n]")


def _estado_final(secuencia: list[Estado]) -> Estado:
    """El estado de una enfermedad del paciente con varias menciones (#208).

    Entre sospecha y negación gana la última mención (Diego, 2026-11-10): "sospecha de
    migraña … se descarta la migraña" queda negada; "se descartó … ahora se sospecha",
    posible. Una confirmada no se baja: "Diabetes mellitus tipo 2 de larga data. Niega
    diabetes gestacional" o "EPOC en manejo. Niega EPOC agudizada" niegan otra cosa (otro
    tipo, un episodio agudo) y el paciente sigue teniendo la enfermedad.
    """
    return Estado.CONFIRMADA if Estado.CONFIRMADA in secuencia else secuencia[-1]


def _otros_estados(secuencia: list[Estado], final: Estado) -> tuple[Estado, ...]:
    """Los estados de las menciones distintos del final, en orden y sin repetir (#208)."""
    return tuple(dict.fromkeys(e for e in secuencia if e is not final))


# Lo que puede haber entre dos menciones de una enumeración (#196, #207): la sigla o
# aclaración entre paréntesis de la anterior; otros elementos que el modelo no marcó
# ("síncope, palpitaciones sostenidas o …"), de hasta 3 palabras y separados por comas;
# el separador final (coma, conjunción o las dos), y un encabezado de antecedente delante
# de la siguiente ("… o historia previa de infarto"). Las palabras no son barreras de
# `contexto` ("presenta", "pero", "refiere"…) ni conjunciones, y nada cruza un corte.
# El encabezado es una lista cerrada: con cualquier palabra terminada en "de", "el
# paciente tiene historia de diabetes" o "es portador de VIH" seguían la enumeración.
_TROZO = rf"{_PALABRA}(?:[ \t]+{_PALABRA}){{0,2}}"
_ENCABEZADO_DE_ANTECEDENTE = (
    r"(?:historia|antecedentes?|episodios?)(?:[ \t]+(?:previ[oa]s?|personal(?:es)?))?"
    r"[ \t]+del?[ \t]+"
)
_SEPARADOR_ENUMERACION = re.compile(
    r"(?:[ \t]*\([^()\n]*\))?"
    rf"(?:[ \t]*,[ \t]*{_TROZO})*"
    # El encabezado de antecedente va solo tras la conjunción ("… o historia previa de
    # IAM"); tras una coma sola abre otra cláusula: "Sin signos de infección, antecedente
    # de EPOC y diabetes" (revisión de #213).
    r"(?:[ \t]*,[ \t]*(?:(?P<conjuncion>(?:y/o|y|e|o|u|ni)[ \t]+)"
    rf"(?:{_ENCABEZADO_DE_ANTECEDENTE})?)?"
    r"|[ \t]+(?P<conjuncion_sola>(?:y/o|y|e|o|u|ni)[ \t]+)"
    rf"(?:{_ENCABEZADO_DE_ANTECEDENTE})?)",
    re.IGNORECASE,
)
# "… disnea paroxística nocturna o palpitaciones," cierra la enumeración aunque el
# último elemento no sea una mención (#213).
_CIERRE_TRAS_LA_ULTIMA = re.compile(
    rf"[ \t]*(?:,[ \t]*)?(?:y/o|y|e|o|u|ni)[ \t]+(?P<trozo>{_TROZO})[ \t]*(?=[,.;:\n]|$)",
    re.IGNORECASE,
)
_DISPARADOR_ADELANTE = re.compile(
    rf"(?:{_DISPARADORES_NEGACION}|{'|'.join(CLAVES_POSIBLE.values())})\b", re.IGNORECASE
)
_ESTADOS_QUE_SE_PROPAGAN = (Estado.NEGADA, Estado.POSIBLE, Estado.FAMILIAR)


def _propagar_enumeraciones(
    estados: dict[tuple[int, int], Estado],
    texto: str,
    aclaraciones: Mapping[tuple[int, int], tuple[int, int]] | None = None,
    tramos: Iterable[Intervalo] = (),
) -> None:
    """En "niega A, B o C" los tres comparten el estado de A (#196).

    La gramática de listas de `contexto` cuenta palabras por elemento (hasta 3) y se
    quedaba en el primero con nombres largos: "niega antecedente de asma bronquial,
    tosferina o tuberculosis pulmonar". Aquí los elementos son las menciones del modelo:
    una enumeración son menciones seguidas, separadas solo por comas y con una conjunción
    antes de la última ("y", "e", "o", "u", "ni", "y/o"). Si la primera es negada, posible
    o familiar ("madre con diabetes, cardiopatía isquémica y cáncer de mama"), las
    confirmadas de la enumeración toman su estado. Sin conjunción final no hay
    enumeración: en "sin secuelas motoras, demencia vascular leve" la coma separa dos
    cláusulas (#84) y la demencia sigue confirmada. Las `aclaraciones` (siglas entre
    paréntesis, `_heredar_parentesis`) no son elementos: van con el anterior.

    #213: el primer elemento puede no ser una mención. Cada tramo negado o posible de
    `contexto` (`tramos`) entra como un elemento más con su estado: en "niega episodios
    de dolor precordial opresivo, disnea paroxística nocturna o palpitaciones" el modelo
    no marcó el primero y el tramo de "niega" corta en la coma. Y la enumeración también
    se cierra si tras la última mención viene la conjunción y un elemento corto no marcado
    hasta un corte ("… disnea paroxística nocturna o palpitaciones,").
    """
    virtuales: dict[tuple[int, int], Estado] = {}
    for tramo in tramos:
        # Solo los tramos con el disparador delante ("niega …", "sospecha de …"): en "TEP
        # descartado, neumonía y derrame" el descarte es del TEP (revisión de #213).
        if (
            tramo.estado in (Estado.NEGADA, Estado.POSIBLE)
            and (tramo.inicio, tramo.fin) not in estados
            and _DISPARADOR_ADELANTE.match(texto, tramo.inicio)
        ):
            clave = (tramo.inicio, tramo.fin)
            if virtuales.get(clave) is not Estado.POSIBLE:  # posible gana, como en contexto
                virtuales[clave] = tramo.estado
    reales = [m for m in estados if m not in (aclaraciones or {})]
    menciones = sorted([*reales, *virtuales])
    k = 0
    while k < len(menciones):
        lista = [menciones[k]]
        cerrada = False
        while k + len(lista) < len(menciones):
            anterior, siguiente = lista[-1], menciones[k + len(lista)]
            if siguiente[0] < anterior[1]:  # se solapan: no es otro elemento
                break
            if siguiente in virtuales:  # un tramo solo abre la enumeración, no la sigue
                break
            hueco = _SEPARADOR_ENUMERACION.fullmatch(texto, anterior[1], siguiente[0])
            if hueco is None:
                break
            lista.append(siguiente)
            if hueco.group("conjuncion") or hueco.group("conjuncion_sola"):
                cerrada = True
                break
        if not cerrada and len(lista) > 1:
            cierre = _CIERRE_TRAS_LA_ULTIMA.match(texto, lista[-1][1])
            # "… tbq leve y sin tratamiento" o "… y posible intoxicación" abren otra
            # cláusula con su propio disparador: no cierran la enumeración (NUBes).
            cerrada = cierre is not None and not _DISPARADOR_ADELANTE.match(
                texto, cierre.start("trozo")
            )
        origen = estados.get(lista[0]) or virtuales[lista[0]]
        if cerrada and origen in _ESTADOS_QUE_SE_PROPAGAN:
            for mencion in lista[1:]:
                if mencion in estados and estados[mencion] is Estado.CONFIRMADA:
                    estados[mencion] = origen
        k += len(lista) if cerrada else 1


def _misma_entidad(aclaracion: Mencion, exterior: tuple[int, int], texto: str) -> bool:
    """Si la mención entre paréntesis nombra la misma enfermedad que la de antes (#203).

    Lo es si no tiene categoría propia (un código, "M05") o si tiene la misma ("EPOC" de
    "enfermedad pulmonar obstructiva crónica"). Si tiene otra, el paréntesis dice la causa
    o el contexto ("nefropatía (DM2)", "demencia (ACV)", "anemia (ERC)") y es otra
    entidad: unirlas haría perder su categoría.
    """
    propia = correccion.mapear(aclaracion.texto)
    if propia is None:
        return True
    de_fuera = correccion.mapear(texto[exterior[0] : exterior[1]])
    return de_fuera is not None and de_fuera.categoria is propia.categoria


_ABRE_PARENTESIS = re.compile(r"\([ \t]*$")
_CIERRA_PARENTESIS = re.compile(r"[ \t]*\)")


def _heredar_parentesis(
    estados: dict[tuple[int, int], Estado], texto: str
) -> dict[tuple[int, int], tuple[int, int]]:
    """La mención entre paréntesis justo después de otra toma su estado (#195).

    En "sospecha de enfermedad pulmonar obstructiva crónica (EPOC)" la sigla es la misma
    enfermedad, pero el contexto corta en el paréntesis y quedaba confirmada. La mención
    hereda si es lo único que hay dentro del paréntesis (una sigla o un sinónimo) y antes
    del "(" solo hay espacios desde el fin de otra mención. Un paréntesis con más texto es
    una explicación, no la misma enfermedad: en NUBes desarrollo, "metrorragias
    (menopausia hace 4 años)" negaba la menopausia. En "A (B) (C)" solo B hereda: antes
    de "(C)" hay un ")" y no el fin de una mención. Devuelve, para cada mención que
    heredó, la mención de la que heredó.
    """
    heredadas: dict[tuple[int, int], tuple[int, int]] = {}
    por_fin = {fin: (inicio, fin) for inicio, fin in estados}
    for inicio, fin in sorted(estados):
        abre = _ABRE_PARENTESIS.search(texto, max(0, inicio - 8), inicio)
        if abre is None or not _CIERRA_PARENTESIS.match(texto, fin):
            continue
        antes = abre.start()
        while antes > 0 and texto[antes - 1] in " \t":
            antes -= 1
        exterior = por_fin.get(antes)
        if exterior is not None and exterior != (inicio, fin):
            estados[(inicio, fin)] = estados[exterior]
            heredadas[(inicio, fin)] = exterior
    return heredadas


def _a_original(entidad: Entidad, limpio: TextoLimpio) -> Entidad:
    """Traduce las menciones de la entidad al texto original, ordenadas por posición."""
    pares = sorted(
        (
            (limpio.mencion_a_original(m), entidad.estado_de_mencion(i))
            for i, m in enumerate(entidad.menciones)
        ),
        key=lambda par: par[0].inicio,
    )
    entidad.menciones = [m for m, _ in pares]
    if entidad.estados_menciones:
        entidad.estados_menciones = tuple(e for _, e in pares)
    return entidad
