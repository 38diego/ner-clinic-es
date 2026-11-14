"""Segunda etapa del normalizador: el clasificador sobre el top-k (#151) y las exclusiones.

La primera etapa (`indice.Normalizador`) trae los k candidatos más parecidos a la mención.
Aquí se decide la categoría de Charlson:

1. Por cada categoría que aparece en el top-k, más la que dan las reglas + el corrector
   aunque no esté, se arma un par (mención, categoría) con 39 atributos numéricos
   (`ATRIBUTOS`): similitud, votos del top-k, jerarquía CIE-10, origen de los candidatos,
   forma de la mención, reglas y la categoría misma. Ninguno es texto.
2. Una regresión logística (`Modelo`, cargada de un JSON con `modelo_de_json`) da la
   probabilidad de cada par. Gana el más probable si pasa el umbral; si no, la mención
   queda sin categoría.
3. Después va la exclusión de la regla de la categoría elegida (`con_exclusiones`): quita
   "mioma" de tumor maligno aunque el modelo lo acerque a "mieloma".

`SegundaEtapa` junta todo para usarlo en ejecución. El entrenamiento y la validación
cruzada están en `evaluacion.clasificador`, que importa de aquí.
"""

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ner_clinic.charlson import correccion, quan
from ner_clinic.charlson.base import Regla, normalizar
from ner_clinic.charlson.reglas import REGLAS
from ner_clinic.normalizacion.indice import ORIGEN_CATALOGO, Candidato, Normalizador
from ner_clinic.normalizacion.jerarquia import bloque, capitulo
from ner_clinic.tipos import AlternativaCie10, CategoriaCharlson

K = 20  # candidatos que mira la segunda etapa (#149)
# Debajo de esta similitud una alternativa de código es ruido ("EPOC" daba "E. coli" con
# 0,59): no se muestra (#230).
SIMILITUD_MINIMA_ALTERNATIVA = 0.6
PATRON_SIGLA = re.compile(r"[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9]{1,5}")
CATEGORIAS = tuple(CategoriaCharlson)

Resultado = tuple[CategoriaCharlson, str] | None
Mapeador = Callable[[str], Resultado]
ReglasDe = dict[str, Resultado]
Extra = Callable[[str, Sequence["Propuesta"], CategoriaCharlson], tuple[float, ...]]


# --- Exclusiones de las reglas (#133, #144) -------------------------------------------

_REGLA_DE = {regla.categoria: regla for regla in REGLAS}


def con_exclusiones(mapeador: Mapeador) -> Mapeador:
    """El `mapeador` y, después, la exclusión de la regla de la categoría que dio (#133).

    El modelo confunde a veces un tumor benigno con uno maligno porque se escriben
    parecido ("miomas" queda junto a "mieloma"). Las exclusiones de las reglas ya separan
    esos casos ("mioma", "adenoma", "hemangioma"): si la de la categoría elegida coincide,
    la mención queda sin categoría. Solo quita categorías, nunca las agrega.

    Algunas exclusiones no dicen "no es esta enfermedad" sino "le toca a otra regla": la de
    tumor maligno excluye "metástasis" para ceder la mención a tumor metastásico. Si esa
    otra regla tampoco acepta el texto (está negado o en duda), no hay a quién cederla y
    la exclusión no se aplica (#144): "linfoma de Hodgkin sin metástasis" (#123) sigue
    como tumor maligno. En "metástasis hepáticas" la otra regla sí la acepta, así que el
    tumor maligno se excluye. Límite conocido: una exclusión de "no es esta enfermedad"
    que otra regla incluye y excluye a la vez se toma como precedencia ("aterosclerosis
    carotídea" dada como reumatológica se conserva); no apareció en CodiEsp.
    """

    def mapear(texto: str) -> tuple[CategoriaCharlson, str] | None:
        resultado = mapeador(texto)
        if resultado is None or excluida(texto, resultado[0]):
            return None
        return resultado

    return mapear


def excluida(texto: str, categoria: CategoriaCharlson) -> bool:
    """Si la exclusión de la regla de `categoria` quita esa categoría a `texto` (ver
    `con_exclusiones`)."""
    normalizado = normalizar(texto)
    regla = _REGLA_DE[categoria]
    return bool(regla.exclusion.search(normalizado)) and not _cedida_sin_destino(normalizado, regla)


def _cedida_sin_destino(texto_normalizado: str, regla: Regla) -> bool:
    """Si cada coincidencia de la exclusión de `regla` cae dentro de la inclusión de otra
    regla que también excluye el texto: la exclusión cedía la mención, pero nadie la toma.
    """
    tramos = [
        (m.start(), m.end())
        for otra in REGLAS
        if otra.categoria is not regla.categoria and otra.exclusion.search(texto_normalizado)
        for m in otra.inclusion.finditer(texto_normalizado)
    ]
    return all(
        any(a <= m.start() and m.end() <= b for a, b in tramos)
        for m in regla.exclusion.finditer(texto_normalizado)
    )


# --- Propuestas: el top-k con su categoría -----------------------------------------------


@dataclass(frozen=True)
class Propuesta:
    """Un candidato del top-k, con lo que una segunda etapa necesita para decidir.

    `codigo_categoria` es el código que se lee para Charlson (un padre sin categoría se
    lee como su hijo `.9`, `Normalizador.codigo_para_categoria`); `categoria`, la de Quan.
    """

    candidato: Candidato
    codigo_categoria: str
    categoria: CategoriaCharlson | None

    @property
    def similitud(self) -> float:
        return self.candidato.similitud

    @property
    def codigo_3(self) -> str:
        return self.candidato.codigo[:3]

    @property
    def bloque(self) -> str | None:
        return bloque(self.candidato.codigo)

    @property
    def capitulo(self) -> str | None:
        return capitulo(self.candidato.codigo)


Decisor = Callable[[str, Sequence[Propuesta]], Resultado]


def propuestas(normalizador: Normalizador, texto: str) -> list[Propuesta]:
    """El top-k de `texto` (con la memoria del `normalizador`) como propuestas."""
    return propuestas_de(normalizador, normalizador.candidatos(texto))


def propuestas_de(normalizador: Normalizador, candidatos: Iterable[Candidato]) -> list[Propuesta]:
    salida = []
    for candidato in candidatos:
        codigo = normalizador.codigo_para_categoria(candidato.codigo)
        salida.append(Propuesta(candidato, codigo, quan.categoria_de(codigo)))
    return salida


def pesos_de(top: Sequence[Propuesta], temperatura: float | None) -> list[float]:
    if temperatura is None:
        return [p.similitud for p in top]
    maximo = max(p.similitud for p in top)
    return [math.exp((p.similitud - maximo) / temperatura) for p in top]


def fraccion(claves: Sequence[Any], pesos: Sequence[float], clave: Any) -> float:
    total = sum(pesos)
    if not total:
        return 0.0
    return sum(p for c, p in zip(claves, pesos, strict=True) if c == clave) / total


def reglas_de(texto: str, memoria: ReglasDe | None = None) -> Resultado:
    """Categoría y código de las reglas + el corrector de ortografía (#115), lo que usa el
    pipeline sin normalizador. Con `memoria`, cada texto se calcula una vez."""
    if memoria is not None and texto in memoria:
        return memoria[texto]
    mapeo = correccion.mapear(texto)
    resultado = None if mapeo is None else (mapeo.categoria, mapeo.cie10)
    if memoria is not None:
        memoria[texto] = resultado
    return resultado


# --- Atributos de cada par ---------------------------------------------------------------

ATRIBUTOS = (
    "similitud",
    "inverso_rango",
    "margen_primero",
    "margen_otra",
    "voto_t005",
    "voto_t01",
    "voto_suma",
    "ninguna_t005",
    "fraccion_top5",
    "fraccion_top20",
    "voto_capitulo",
    "voto_bloque",
    "mejor_es_sinonimo",
    "fraccion_sinonimos",
    "exacta",
    "log_largo",
    "palabras",
    "sigla",
    "reglas_esta",
    "reglas_otra",
    "excluida",
    "en_topk",
    *(f"categoria_{c.value}" for c in CATEGORIAS),
)


@dataclass(frozen=True)
class Par:
    categoria: CategoriaCharlson
    codigo: str
    atributos: tuple[float, ...]
    # El mejor candidato de la categoría en el top-k (`None` si solo la dan las reglas)
    candidato: Candidato | None = None


def pares(
    texto: str, top: Sequence[Propuesta], reglas: Resultado, extra: Extra | None = None
) -> list[Par]:
    """Los pares (mención, categoría candidata) con sus atributos (ver el módulo).

    `extra` agrega más atributos a cada par (los del reordenador de #152).
    """
    categorias = list(dict.fromkeys(p.categoria for p in top if p.categoria is not None))
    if reglas is not None and reglas[0] not in categorias:
        categorias.append(reglas[0])
    if not categorias:
        return []
    claves = [p.categoria for p in top]
    pesos = {t: pesos_de(top, t) for t in (0.05, 0.1, None)}
    primero = top[0].similitud
    largo = len(texto.strip())
    salida = []
    for categoria in categorias:
        propias = [p for p in top if p.categoria is categoria]
        mejor = max(propias, key=lambda p: p.similitud) if propias else None
        similitud = mejor.similitud if mejor else 0.0
        otras = [p.similitud for p in top if p.categoria is not categoria]
        rango = next((i for i, p in enumerate(top, 1) if p.categoria is categoria), None)
        atributos = [
            similitud,
            1 / rango if rango else 0.0,
            similitud - primero,
            similitud - (max(otras) if otras else 0.0),
            fraccion(claves, pesos[0.05], categoria),
            fraccion(claves, pesos[0.1], categoria),
            fraccion(claves, pesos[None], categoria),
            fraccion(claves, pesos[0.05], None),
            sum(p.categoria is categoria for p in top[:5]) / 5,
            sum(p.categoria is categoria for p in top[:20]) / 20,
            fraccion([p.capitulo for p in top], pesos[0.05], mejor.capitulo) if mejor else 0.0,
            fraccion([p.bloque for p in top], pesos[0.05], mejor.bloque) if mejor else 0.0,
            float(mejor is not None and mejor.candidato.origen != ORIGEN_CATALOGO),
            sum(p.candidato.origen != ORIGEN_CATALOGO for p in propias) / len(propias)
            if propias
            else 0.0,
            float(similitud >= 0.999),
            math.log1p(largo),
            float(len(texto.split())),
            float(PATRON_SIGLA.fullmatch(texto.strip()) is not None),
            float(reglas is not None and reglas[0] is categoria),
            float(reglas is not None and reglas[0] is not categoria),
            float(excluida(texto, categoria)),
            float(mejor is not None),
            *(float(c is categoria) for c in CATEGORIAS),
            *(extra(texto, top, categoria) if extra else ()),
        ]
        codigo = mejor.codigo_categoria if mejor else reglas[1]
        salida.append(Par(categoria, codigo, tuple(atributos), mejor.candidato if mejor else None))
    return salida


# --- Modelo ----------------------------------------------------------------------------


@dataclass
class Modelo:
    """Regresión logística sobre atributos estandarizados."""

    media: torch.Tensor
    desviacion: torch.Tensor
    pesos: torch.Tensor
    sesgo: float

    def probabilidades(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(((x - self.media) / self.desviacion) @ self.pesos + self.sesgo)

    def pesos_por_atributo(self, nombres: Sequence[str] = ATRIBUTOS) -> dict[str, float]:
        """Peso de cada atributo estandarizado: cuánto sube el logit con una desviación más."""
        return {n: round(float(w), 4) for n, w in zip(nombres, self.pesos, strict=True)}


def mejor_par(
    modelo: Modelo,
    texto: str,
    top: Sequence[Propuesta],
    reglas: Resultado,
    extra: Extra | None = None,
) -> tuple[Par, float] | None:
    """El par más probable de la mención y su probabilidad, sin umbral (`None` si no hay
    ninguna categoría candidata)."""
    candidatos = pares(texto, top, reglas, extra)
    if not candidatos:
        return None
    x = torch.tensor([p.atributos for p in candidatos], dtype=torch.float32)
    probabilidades = modelo.probabilidades(x).tolist()
    j = max(range(len(candidatos)), key=probabilidades.__getitem__)
    return candidatos[j], probabilidades[j]


def decisor(
    modelo: Modelo, umbral: float, reglas: ReglasDe | None = None, extra: Extra | None = None
) -> Decisor:
    """El clasificador como segunda etapa (las exclusiones van después, en `con_exclusiones`)."""
    memoria = {} if reglas is None else reglas

    def decidir(texto: str, top: Sequence[Propuesta]) -> Resultado:
        mejor = mejor_par(modelo, texto, top, reglas_de(texto, memoria), extra)
        if mejor is None or mejor[1] < umbral:
            return None
        return mejor[0].categoria, mejor[0].codigo

    return decidir


def modelo_a_json(modelo: Modelo, umbral: float) -> dict[str, Any]:
    """Todo lo que hace falta para usar el clasificador: un número por atributo en cada
    vector, sin texto."""
    return {
        "atributos": list(ATRIBUTOS),
        "media": [float(x) for x in modelo.media],
        "desviacion": [float(x) for x in modelo.desviacion],
        "pesos": [float(x) for x in modelo.pesos],
        "sesgo": modelo.sesgo,
        "umbral": umbral,
    }


def modelo_de_json(datos: dict[str, Any]) -> tuple[Modelo, float]:
    """El clasificador y su umbral desde `modelo_a_json`. Falla si los atributos guardados
    no son los de este módulo (un modelo viejo con otros atributos daría basura)."""
    if datos["atributos"] != list(ATRIBUTOS):
        raise ValueError("Los atributos del modelo guardado no son los de `decision.ATRIBUTOS`.")
    return (
        Modelo(
            torch.tensor(datos["media"]),
            torch.tensor(datos["desviacion"]),
            torch.tensor(datos["pesos"]),
            float(datos["sesgo"]),
        ),
        float(datos["umbral"]),
    )


# --- En ejecución ------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """La categoría que dio la segunda etapa, con el código y el candidato que la explica.

    `candidato` es el mejor de la categoría en el top-k (texto del índice: catálogo o
    sinónimo, nunca de la nota); `None` si la categoría solo la dieron las reglas.
    """

    categoria: CategoriaCharlson
    codigo: str
    probabilidad: float
    candidato: Candidato | None


class SegundaEtapa:
    """Mención → `Decision` (o `None`): top-k, clasificador, umbral y exclusiones.

    No guarda nada entre llamadas: ni los textos ni sus vectores. Así, en la API, una nota
    no deja rastro después de analizarla. Ojo: `Normalizador.candidatos`, `.precalcular` y
    `propuestas` sí guardan cada texto en la memoria del normalizador; en ejecución, el
    texto de una nota solo debe pasar por `decidir`.
    """

    def __init__(self, normalizador: Normalizador, modelo: Modelo, umbral: float):
        self.normalizador = normalizador
        self.modelo = modelo
        self.umbral = umbral

    def decidir(self, textos: Sequence[str]) -> dict[str, Decision | None]:
        """La decisión de cada texto distinto, codificando todos en un lote."""
        unicos = list(dict.fromkeys(textos))
        if not unicos:
            return {}
        indice = self.normalizador.indice
        encontrados = indice.buscar(self.normalizador.codificador(unicos), K)
        return {
            texto: self._decidir(texto, propuestas_de(self.normalizador, top))
            for texto, top in zip(unicos, encontrados, strict=True)
        }

    def alternativas(
        self, textos: Sequence[str], k: int = 5, minimo: float = SIMILITUD_MINIMA_ALTERNATIVA
    ) -> dict[str, list[AlternativaCie10]]:
        """Los `k` códigos distintos más parecidos a cada texto, para que el experto elija
        (#230). Del top-`K` del índice se queda la primera fila de cada código (la de más
        similitud) y la descripción del catálogo; si el código solo está como sinónimo, la
        del sinónimo (texto público de CodiEsp, nunca de la nota). Las de similitud menor
        que `minimo` no salen. Como `decidir`, no guarda nada."""
        unicos = list(dict.fromkeys(textos))
        if not unicos:
            return {}
        indice = self.normalizador.indice
        encontrados = indice.buscar(self.normalizador.codificador(unicos), K)
        salida = {}
        for texto, top in zip(unicos, encontrados, strict=True):
            vistos: dict[str, AlternativaCie10] = {}
            for c in top:
                if c.similitud < minimo:
                    break  # el top viene de mayor a menor
                if c.codigo not in vistos:
                    descripcion = indice.descripcion_catalogo(c.codigo) or c.descripcion
                    vistos[c.codigo] = AlternativaCie10(c.codigo, descripcion, c.similitud)
                if len(vistos) == k:
                    break
            salida[texto] = list(vistos.values())
        return salida

    def _decidir(self, texto: str, top: Sequence[Propuesta]) -> Decision | None:
        mejor = mejor_par(self.modelo, texto, top, reglas_de(texto))
        if mejor is None or mejor[1] < self.umbral or excluida(texto, mejor[0].categoria):
            return None
        par, probabilidad = mejor
        return Decision(par.categoria, par.codigo, round(probabilidad, 4), par.candidato)
