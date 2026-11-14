"""Depuración de entidades: duplicados internos y cruce con los diagnósticos registrados.

- `deduplicar_internas` trabaja sobre `Candidato` (antes del contexto): fusiona variantes
  ortográficas y entidades contenidas en otras más largas, sumando sus menciones.
- `cruzar_con_registrados` trabaja sobre `Entidad` confirmadas: marca las que ya están
  entre los diagnósticos registrados del egreso (principal y secundarios), o las descarta
  como hacía la tesis. `filtrar_por_diagnostico` es el atajo para el principal.
- `cruzar_por_categoria` (#176) es la segunda pasada, ya con la categoría de Charlson, solo
  para la diabetes: una confirmada que no coincidió por texto coincide si un diagnóstico
  registrado tiene su categoría (o la versión con complicaciones). "retinopatía
  diabética" no comparte palabras con "Diabetes mellitus … con complicaciones
  oftálmicas", pero las dos son diabetes con complicaciones.

Los umbrales por defecto son los de la tesis. Para todos los diagnósticos se usa el del
principal (0,88); la tesis usaba 0,85 para los secundarios.

Corregido respecto a la tesis (#24):

- Las variantes ortográficas se comparan palabra por palabra, no sobre la frase
  completa: Jaro-Winkler premia el prefijo común y fusionaba "insuficiencia cardiaca"
  con "insuficiencia renal crónica" o "enfermedad de Alzheimer" con "enfermedad renal
  crónica".
- Entre variantes de igual longitud queda la que tiene tildes.
- En el cruce, "insuficiencia", "enfermedad", "trastorno", "síndrome", "agudo" y
  "crónico" no cuentan: nombran el tipo de afección, no cuál es, y hacían coincidir
  enfermedades distintas.
- Las siglas ("HTA", "EPOC", ...) se expanden con `siglas.expandir` antes del cruce;
  antes nunca coincidían porque solo cuentan palabras de más de 3 letras.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import replace

from rapidfuzz.distance import JaroWinkler, Levenshtein

from ner_clinic.charlson.indice import JERARQUIAS
from ner_clinic.siglas import expandir
from ner_clinic.tipos import (
    Candidato,
    CategoriaCharlson,
    Descarte,
    Entidad,
    Estado,
    MotivoDescarte,
)

# Categorías que cruzan por categoría (#176): la diabetes, porque la CIE-10 registra sus
# complicaciones como "Diabetes mellitus … con complicaciones X". En las demás, una
# categoría agrupa enfermedades distintas (un cáncer de mama no está registrado porque se
# registró uno de próstata; una hepatitis B no es la cirrosis registrada) o hereda los
# errores del mapeo; en el set sintético, cruzarlas marcó registradas dos que no lo eran.
CATEGORIAS_CON_CRUCE = frozenset(
    {CategoriaCharlson.DIABETES, CategoriaCharlson.DIABETES_COMPLICADA}
)

UMBRAL_DUPLICADO_INTERNO = 0.85
UMBRAL_DIAGNOSTICO = 0.88

# Términos genéricos de las descripciones CIE-10 que no aportan a la comparación.
_GENERICOS_CIE10 = re.compile(
    r"\b(?:no especificad[oa]s?|otr[oa]s?|tipos?|debido a|clasificad[oa]s? en otra parte)\b|,"
)
# Palabras que nombran el tipo de afección y no cuál es (#24): "insuficiencia cardiaca"
# y "insuficiencia renal" son enfermedades distintas aunque compartan la primera palabra.
_GENERICOS_AFECCION = frozenset(
    {
        "insuficiencia",
        "insuficiencias",
        "enfermedad",
        "enfermedades",
        "trastorno",
        "trastornos",
        "sindrome",
        "síndrome",
        "agudo",
        "aguda",
        "agudos",
        "agudas",
        "cronico",
        "cronica",
        "cronicos",
        "cronicas",
        "crónico",
        "crónica",
        "crónicos",
        "crónicas",
    }
)
_STOPWORDS = frozenset({"para", "como", "pero", "sino", "tales", "sobre", "desde", "hasta"})
_LONGITUD_MINIMA_PALABRA = 4  # solo cuentan palabras de más de 3 letras


def deduplicar_internas(
    candidatos: Iterable[Candidato], umbral_similitud: float = UMBRAL_DUPLICADO_INTERNO
) -> tuple[list[Candidato], list[Descarte]]:
    """Fusiona candidatos redundantes dentro de una misma nota.

    Recorre los candidatos de mayor a menor longitud y, entre los de igual longitud,
    primero los que tienen tildes. Uno se fusiona con un candidato ya guardado si está
    contenido en él como palabras completas (`contenida_por_palabras`, #194) o si es una
    variante ortográfica: cada una de sus palabras tiene
    similitud Jaro-Winkler de `umbral_similitud` o más con la palabra en la misma
    posición del guardado ("diabetis melitus" ~ "diabetes mellitus", "hipertension" ~
    "hipertensión arterial"). Sus menciones pasan al que queda, salvo las que caen dentro
    de una que ya tiene (#223), y se registra un `Descarte` con motivo
    `DUPLICADO_INTERNO`.

    Devuelve los candidatos que quedan, en el orden de entrada, y los descartes. No
    modifica los candidatos recibidos.
    """
    entrada = list(candidatos)
    por_longitud = sorted(
        range(len(entrada)),
        key=lambda i: (len(entrada[i].texto), _tildes(entrada[i].texto)),
        reverse=True,
    )

    guardados: dict[int, Candidato] = {}
    descartes: list[Descarte] = []

    for i in por_longitud:
        actual = entrada[i]
        texto_actual = actual.texto.lower()
        for guardado in guardados.values():
            texto_guardado = guardado.texto.lower()
            if contenida_por_palabras(texto_actual, texto_guardado):
                detalle = f"absorbida por '{guardado.texto}'"
            elif _variante_por_palabras(texto_actual, texto_guardado, umbral_similitud):
                detalle = f"variante ortográfica de '{guardado.texto}'"
            else:
                continue
            # Una mención dentro de otra que ya tiene es el mismo lugar del texto
            # ("Fumador" dentro de "Fumador pesado"): no se cuenta dos veces (#223).
            guardado.menciones.extend(
                m
                for m in actual.menciones
                if not any(g.inicio <= m.inicio and m.fin <= g.fin for g in guardado.menciones)
            )
            descartes.append(Descarte(actual.texto, MotivoDescarte.DUPLICADO_INTERNO, detalle))
            break
        else:
            guardados[i] = Candidato(texto=actual.texto, menciones=list(actual.menciones))

    return [guardados[i] for i in sorted(guardados)], descartes


def contenida_por_palabras(corta: str, larga: str) -> bool:
    """Si `corta` aparece en `larga` como palabras completas (#194).

    "insuficiencia renal" está en "insuficiencia renal crónica" y "EPOC" en "enfermedad
    pulmonar obstructiva crónica (EPOC)", pero "tos" no está en "depósitos": antes se
    comparaban letras y la tos quedaba absorbida por una artropatía por depósitos.
    """
    return bool(corta) and re.search(rf"(?<!\w){re.escape(corta)}(?!\w)", larga) is not None


def cruzar_con_registrados(
    entidades: Iterable[Entidad],
    diagnosticos_registrados: Iterable[object] | None,
    umbral_similitud: float = UMBRAL_DIAGNOSTICO,
    descartar: bool = False,
) -> tuple[list[Entidad], list[Descarte]]:
    """Cruza las entidades confirmadas con los diagnósticos registrados del egreso.

    Limpia cada diagnóstico de términos genéricos CIE-10, palabras que solo nombran el
    tipo de afección ("insuficiencia", "enfermedad", "crónica", ...) y stopwords. Una
    entidad confirmada coincide con un diagnóstico si alguna de sus palabras de más de
    3 letras coincide con una del diagnóstico, de forma exacta o con similitud
    Jaro-Winkler o Levenshtein de `umbral_similitud` o más. Si la entidad tiene siglas
    ("HTA"), también se prueba con su nombre completo ("hipertensión arterial"). Si
    coincide con varios, cuenta el primero de la lista. Las entidades con otro estado
    (negadas, posibles y familiares) pasan sin tocar, con `registrado=None`: una sospecha
    no es un subregistro. Las confirmadas sin coincidencia pasan después, ya mapeadas, por
    `cruzar_por_categoria` (#176).

    - `descartar=False` (por defecto): devuelve todas las entidades; las confirmadas
      llevan `registrado=True` y `coincide_con`, o `registrado=False` si no coinciden
      con ningún diagnóstico (posible subregistro).
    - `descartar=True`: comportamiento de la tesis; las confirmadas que coinciden se
      descartan con motivo `DIAGNOSTICO_REGISTRADO` y las demás pasan sin tocar.

    Los elementos que no son texto o están vacíos se ignoran; si no queda ningún
    diagnóstico, no toca nada. No modifica las entidades recibidas.
    """
    entrada = list(entidades)
    diagnosticos = [
        (d.strip(), _palabras_significativas(_GENERICOS_CIE10.sub(" ", d.lower())))
        for d in diagnosticos_registrados or []
        if isinstance(d, str) and d.strip()
    ]
    if not diagnosticos:
        return entrada, []

    quedan: list[Entidad] = []
    descartes: list[Descarte] = []
    for entidad in entrada:
        if entidad.estado is not Estado.CONFIRMADA:
            quedan.append(entidad)
            continue
        # También con las menciones: "EPOC" de "enfermedad pulmonar obstructiva crónica
        # (EPOC)" es otra mención de la entidad (#203) y puede ser lo que está registrado.
        coincidencia = next(
            (
                c
                for texto in dict.fromkeys([entidad.texto, *(m.texto for m in entidad.menciones)])
                if (c := _primer_registrado(texto, diagnosticos, umbral_similitud)) is not None
            ),
            None,
        )
        if descartar:
            if coincidencia is None:
                quedan.append(entidad)
            else:
                descartes.append(
                    Descarte(entidad.texto, MotivoDescarte.DIAGNOSTICO_REGISTRADO, coincidencia[1])
                )
        else:
            quedan.append(
                replace(
                    entidad,
                    registrado=coincidencia is not None,
                    coincide_con=coincidencia[0] if coincidencia else None,
                    coincide_por="texto" if coincidencia else None,
                )
            )
    return quedan, descartes


def cruzar_por_categoria(
    entidades: Iterable[Entidad],
    registrados: Sequence[tuple[str, CategoriaCharlson | None]],
    descartar: bool = False,
) -> tuple[list[Entidad], list[Descarte]]:
    """Segunda pasada del cruce (#176), con las entidades ya mapeadas a Charlson.

    `registrados` son los diagnósticos registrados con la categoría que les da el mismo
    mapeo que a las entidades. Una confirmada que no coincidió por texto (`registrado` es
    `False`, o `None` con `descartar=True`) coincide con el primer registrado de su misma
    categoría, o de la severa de su jerarquía: "DM2" coincide con "Diabetes … con
    complicaciones", pero "retinopatía diabética" no coincide con "Diabetes … sin
    complicación" (la complicación no está registrada). Solo aplica a
    `CATEGORIAS_CON_CRUCE`. La coincidencia queda con `coincide_por="categoria"`; con
    `descartar=True`, la entidad se descarta como en `cruzar_con_registrados`.

    Límite: no distingue la complicación. "nefropatía diabética" coincide con "Diabetes …
    con complicaciones oftálmicas" aunque la nefropatía no esté registrada; los códigos no
    sirven para separarlas, porque las reglas dan a la descripción registrada un código
    genérico. Para Charlson son la misma categoría; para la codificación, no.
    """
    quedan: list[Entidad] = []
    descartes: list[Descarte] = []
    for entidad in entidades:
        categoria = entidad.categoria_charlson
        sin_coincidir = entidad.registrado is False or (descartar and entidad.registrado is None)
        if (
            entidad.estado is not Estado.CONFIRMADA
            or not sin_coincidir
            or categoria is None
            or categoria not in CATEGORIAS_CON_CRUCE
        ):
            quedan.append(entidad)
            continue
        aceptadas = {categoria, JERARQUIAS.get(categoria)}
        diagnostico = next((d for d, c in registrados if c is not None and c in aceptadas), None)
        if diagnostico is None:
            quedan.append(entidad)
        elif descartar:
            detalle = (
                f"misma categoría de Charlson ({categoria.value}) que el diagnóstico "
                f"registrado: '{diagnostico}'"
            )
            descartes.append(
                Descarte(entidad.texto, MotivoDescarte.DIAGNOSTICO_REGISTRADO, detalle)
            )
        else:
            quedan.append(
                replace(
                    entidad, registrado=True, coincide_con=diagnostico, coincide_por="categoria"
                )
            )
    return quedan, descartes


def filtrar_por_diagnostico(
    confirmadas: Iterable[Entidad],
    diagnostico_principal: str | None,
    umbral_similitud: float = UMBRAL_DIAGNOSTICO,
) -> tuple[list[Entidad], list[Descarte]]:
    """Descarta las entidades confirmadas que repiten el diagnóstico principal.

    Atajo de `cruzar_con_registrados([diagnostico_principal], descartar=True)`, el filtro
    de la tesis. Sin diagnóstico principal (vacío, `None` o no texto) no filtra nada.
    """
    return cruzar_con_registrados(
        confirmadas, [diagnostico_principal], umbral_similitud, descartar=True
    )


def _primer_registrado(
    texto: str, diagnosticos: list[tuple[str, list[str]]], umbral: float
) -> tuple[str, str] | None:
    """El primer diagnóstico registrado que coincide con `texto` y el detalle, o `None`.

    Prueba `texto` tal cual y, si tiene siglas, cada una de sus expansiones.
    """
    variantes = [(texto, "")] + [
        (expansion, f" (sigla expandida: '{expansion}')") for expansion in expandir(texto)
    ]
    for diagnostico, palabras in diagnosticos:
        for variante, nota in variantes:
            detalle = _coincidencia(variante, palabras, umbral)
            if detalle is not None:
                return diagnostico, f"{detalle}{nota} (diagnóstico registrado: '{diagnostico}')"
    return None


def _tildes(texto: str) -> int:
    """Cuántas letras con tilde o diéresis tiene `texto` (para preferir la versión con tilde)."""
    return sum(letra in "áéíóúüñÁÉÍÓÚÜÑ" for letra in texto)


def _variante_por_palabras(corta: str, larga: str, umbral: float) -> bool:
    """Si cada palabra de `corta` se parece (Jaro-Winkler) a la de `larga` en su posición.

    `corta` puede tener menos palabras que `larga` ("hipertension" frente a "hipertensión
    arterial"), pero no más.
    """
    palabras_corta = corta.split()
    palabras_larga = larga.split()
    if not palabras_corta or len(palabras_corta) > len(palabras_larga):
        return False
    return all(
        JaroWinkler.normalized_similarity(a, b) >= umbral
        for a, b in zip(palabras_corta, palabras_larga, strict=False)
    )


def _palabras_significativas(texto: str) -> list[str]:
    return [
        p
        for p in texto.lower().split()
        if len(p) >= _LONGITUD_MINIMA_PALABRA
        and p not in _STOPWORDS
        and p not in _GENERICOS_AFECCION
    ]


def _coincidencia(texto: str, palabras_diagnostico: list[str], umbral: float) -> str | None:
    """Detalle de la primera palabra que coincide con el diagnóstico, o `None`."""
    for palabra in _palabras_significativas(texto):
        for palabra_diag in palabras_diagnostico:
            if palabra == palabra_diag:
                return f"coincidencia exacta con el diagnóstico: '{palabra}'"
            if (
                JaroWinkler.normalized_similarity(palabra, palabra_diag) >= umbral
                or Levenshtein.normalized_similarity(palabra, palabra_diag) >= umbral
            ):
                return (
                    f"coincidencia ortográfica con el diagnóstico: '{palabra}' ~ '{palabra_diag}'"
                )
    return None
