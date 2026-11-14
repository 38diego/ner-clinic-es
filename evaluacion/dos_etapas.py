"""Arnés de la decisión en dos etapas (PRD `normalizacion-f1`, FR1, #149).

Hoy el primer candidato del índice decide solo: si su similitud pasa el umbral, su
categoría de Charlson es la respuesta. El epic #148 separa las dos cosas:

1. **Primera etapa, recuperar.** El índice (catálogo + sinónimos) trae un top-k de
   candidatos por mención (`Propuesta`: el candidato, el código que se lee para Charlson,
   su categoría de Quan, su bloque y su capítulo CIE-10).
2. **Segunda etapa, decidir.** Un `Decisor` mira el top-k completo y devuelve la categoría
   y el código que la sostiene, o nada. Este arnés trae solo la línea base
   (`primer_candidato`, el híbrido de #133 cuando se le suman las exclusiones); las demás
   segundas etapas llegan en #150 a #152.

**Top-k una vez por índice.** `pliegues_topk` arma el índice de cada pliegue, busca el
top-k de sus menciones y se queda solo con esas tablas (`Pliegue.topk`): ninguna segunda
etapa vuelve a codificar ni a buscar. Los vectores de catálogo, sinónimos y menciones se
guardan por texto en disco (`CodificadorEnDisco`, bajo `data/`), así que rehacer un índice
cuesta solo el producto de matrices.

**Validación cruzada** de `PLIEGUES` pliegues por documento en CodiEsp entrenamiento, como
en #132: cada pliegue se busca en un índice con los sinónimos de los otros.
**Anidada** (`pliegues_internos`): lo que se entrene o elija dentro del pliegue externo `i`
se ajusta en los otros pliegues `j`, cada uno buscado en un índice sin los sinónimos de `i`
ni de `j`. El pliegue externo no entra ni en el índice ni en el ajuste.

**Comparación pareada** (`comparar`): la diferencia de P/R/F1 micro entre dos métodos
medidos en los mismos documentos, con IC por bootstrap pareado por documento
(`metricas.bootstrap_diferencia`).

**Errores** (`causa_perdida`): por qué una mención gold con categoría no la recibe, en este
orden (gana la primera que aplica):

- `excluida`: el primer candidato tenía la categoría correcta y pasaba el umbral, pero una
  exclusión de las reglas la quitó.
- `bajo_umbral`: el primer candidato tenía la categoría correcta, bajo el umbral.
- `top5`: la categoría correcta está entre el segundo y el quinto candidato.
- `topk`: está más abajo, hasta el candidato `k`.
- `solo_reglas`: no está en el top-k, pero las reglas + corrector la dan.
- `fuera`: ni el top-k ni las reglas.

El análisis hecho a mano después de #134 tenía además "ruido de anotación"; eso no se
puede decidir solo, y aquí esas menciones caen en alguna de las causas de arriba.

**Techo** (`techo`): el F1 de un oráculo que acierta siempre que la categoría correcta esté
en el top-k y se abstiene si no (nunca da un falso positivo). Es el máximo al que puede
llegar una segunda etapa que solo mire esos `k` candidatos.

**Desarrollo** se usa solo para reproducir la cifra de #133 (0,885) con el arnés y para
versionar las menciones que el híbrido pierde (documento, posición, código gold y causa;
sin texto), para que #154 mida cuántas recupera. Nada se elige con desarrollo, y prueba no
se lee (solo los nombres de sus archivos, para verificar el origen de los sinónimos).

    uv run python -m evaluacion dos-etapas [--modelo sapbert] [--k 20]
"""

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from evaluacion import charlson_codiesp, distemist
from evaluacion.cache import version_de
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.metricas import bootstrap_diferencia, micro_prf
from evaluacion.normalizacion_codiesp import textos_de
from evaluacion.normalizador import (
    CANDIDATOS,
    RAIZ_INDICES,
    Codificador,
    Indice,
    Mapeador,
    ModeloEmbeddings,
    Normalizador,
    con_exclusiones,
    huella_configuracion,
)
from evaluacion.sinonimos_codiesp import (
    PLIEGUES,
    pliegues_de,
    sinonimos_de,
    verificar_origen,
)
from evaluacion.sinonimos_publicos import MencionExterna, sinonimos_externos
from ner_clinic.charlson import quan
from ner_clinic.normalizacion.decision import (  # noqa: F401  (se reexportan)
    Decisor,
    K,
    Propuesta,
    Resultado,
    propuestas,
)
from ner_clinic.normalizacion.jerarquia import bloque, capitulo  # noqa: F401
from ner_clinic.tipos import CategoriaCharlson

RUTA_RESULTADOS = Path("evaluacion/resultados/dos_etapas_arnes.json")
KS_TECHO = (1, 5, 10, 20)
UMBRAL_HIBRIDO = 0.69  # el de #132, elegido por validación cruzada en entrenamiento
CAUSAS = ("excluida", "bajo_umbral", "top5", "topk", "solo_reglas", "fuera")


def primer_candidato(umbral: float = UMBRAL_HIBRIDO) -> Decisor:
    """La decisión de hoy: el primer candidato, si pasa el umbral y tiene categoría."""

    def decidir(texto: str, top: Sequence[Propuesta]) -> Resultado:
        primero = top[0]
        if primero.similitud < umbral or primero.categoria is None:
            return None
        return primero.categoria, primero.codigo_categoria

    return decidir


def mapeador_de(
    decisor: Decisor, topk: dict[str, list[Propuesta]], exclusiones: bool = True
) -> Mapeador:
    """El `decisor` sobre las tablas `topk` (texto → propuestas), como un `Mapeador`.

    Con `exclusiones`, después se aplica la exclusión de la regla de la categoría elegida
    (`normalizador.con_exclusiones`, #133 y #144), igual que en el híbrido.
    """

    def mapear(texto: str) -> Resultado:
        return decisor(texto, topk[texto])

    return con_exclusiones(mapear) if exclusiones else mapear


# --- Vectores en disco ---------------------------------------------------------------


class CodificadorEnDisco:
    """Un `Codificador` que guarda el vector de cada texto en `ruta` (bajo `data/`).

    El codificador real se crea solo si hace falta un texto nuevo (`crear`), así que una
    corrida con todo en disco no carga el modelo. `guardar` escribe los vectores nuevos.
    El archivo trae textos del corpus público, por eso vive en `data/` y no se versiona.
    """

    def __init__(self, crear: Callable[[], Codificador], ruta: Path):
        self.crear = crear
        self.ruta = Path(ruta)
        self._codificador: Codificador | None = None
        self._vectores: dict[str, torch.Tensor] = {}
        self.nuevos = 0
        if self.ruta.is_file():
            guardado = torch.load(self.ruta, weights_only=True)
            self._vectores = dict(zip(guardado["textos"], guardado["vectores"], strict=True))

    def __call__(self, textos: Sequence[str]) -> torch.Tensor:
        faltan = list(dict.fromkeys(t for t in textos if t not in self._vectores))
        if faltan:
            if self._codificador is None:
                self._codificador = self.crear()
            self._vectores.update(zip(faltan, self._codificador(faltan), strict=True))
            self.nuevos += len(faltan)
        if not textos:
            return torch.empty(0, 0)
        return torch.stack([self._vectores[t] for t in textos])

    def guardar(self) -> None:
        if not self.nuevos:
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        textos = list(self._vectores)
        temporal = self.ruta.with_suffix(".tmp")
        torch.save(
            {"textos": textos, "vectores": torch.stack(list(self._vectores.values()))}, temporal
        )
        temporal.replace(self.ruta)
        self.nuevos = 0


def ruta_vectores(modelo: ModeloEmbeddings, raiz: Path = RAIZ_INDICES) -> Path:
    return Path(raiz) / f"vectores_{huella_configuracion(modelo)[:16]}.pt"


# --- Pliegues ------------------------------------------------------------------------


@dataclass
class Pliegue:
    """Unos documentos y el top-k de sus menciones, buscado en un índice sin sus sinónimos."""

    documentos: list[DocumentoCodificado]
    topk: dict[str, list[Propuesta]]


def _buscar(
    indice: Indice,
    codificador: Codificador,
    sinonimos_de_docs: Sequence[DocumentoCodificado],
    evaluados: Sequence[DocumentoCodificado],
    k: int,
    externo: Sequence[DocumentoCodificado] = (),
    externas: Sequence[MencionExterna] = (),
) -> Pliegue:
    sinonimos = sinonimos_de(sinonimos_de_docs)
    fuera = [d.id for d in evaluados] + [d.id for d in externo]
    sinonimos += sinonimos_externos(externas, fuera, (s.texto for s in sinonimos))
    verificar_origen(
        sinonimos,
        {"evaluados": [d.id for d in evaluados], "externo": [d.id for d in externo]},
    )
    normalizador = Normalizador(indice.con_sinonimos(sinonimos, codificador), codificador, k)
    textos = textos_de(evaluados, ["original"])
    normalizador.precalcular(textos)
    return Pliegue(list(evaluados), {t: propuestas(normalizador, t) for t in textos})


def pliegues_topk(
    indice: Indice,
    codificador: Codificador,
    documentos: Sequence[DocumentoCodificado],
    n: int = PLIEGUES,
    k: int = K,
    externas: Sequence[MencionExterna] = (),
) -> list[Pliegue]:
    """Validación cruzada: cada pliegue con el top-k de un índice con los sinónimos de los
    otros. Solo queda en memoria un índice a la vez. `externas` son menciones de otros
    corpus (#153): entran salvo las de documentos del pliegue que se evalúa."""
    pliegues = pliegues_de(documentos, n)
    return [
        _buscar(
            indice,
            codificador,
            [d for j, p in enumerate(pliegues) if j != i for d in p],
            p,
            k,
            externas=externas,
        )
        for i, p in enumerate(pliegues)
    ]


def pliegues_internos(
    indice: Indice,
    codificador: Codificador,
    documentos: Sequence[DocumentoCodificado],
    externo: int,
    n: int = PLIEGUES,
    k: int = K,
    externas: Sequence[MencionExterna] = (),
) -> list[Pliegue]:
    """Para ajustar algo sin ver el pliegue `externo`: los otros pliegues, cada uno con el
    top-k de un índice sin los sinónimos del externo ni los suyos."""
    pliegues = pliegues_de(documentos, n)
    if not 0 <= externo < n:
        raise ValueError(f"El pliegue externo tiene que estar entre 0 y {n - 1}, no {externo}.")
    if n < 3:
        raise ValueError("La validación anidada necesita al menos 3 pliegues.")
    return [
        _buscar(
            indice,
            codificador,
            [d for m, q in enumerate(pliegues) if m not in (externo, j) for d in q],
            p,
            k,
            pliegues[externo],
            externas,
        )
        for j, p in enumerate(pliegues)
        if j != externo
    ]


# --- Medición ------------------------------------------------------------------------


def casos(
    pliegues: Sequence[Pliegue], decisor: Decisor, exclusiones: bool = True
) -> list[charlson_codiesp.Caso]:
    """Los casos de todos los pliegues, cada uno decidido sobre el top-k de su pliegue."""
    salida = []
    for pliegue in pliegues:
        mapeador = mapeador_de(decisor, pliegue.topk, exclusiones)
        salida.extend(charlson_codiesp.casos(pliegue.documentos, mapeador))
    return salida


def _ids(pliegues: Sequence[Pliegue]) -> list[str]:
    return [d.id for p in pliegues for d in p.documentos]


def _prf(casos_: Sequence[charlson_codiesp.Caso], ids: Sequence[str]) -> dict[str, float]:
    prf = micro_prf(charlson_codiesp._por_documento(casos_, ids))
    return {k: round(v, 4) for k, v in prf._asdict().items()}


def por_pliegue(
    pliegues: Sequence[Pliegue], decisor: Decisor, exclusiones: bool = True
) -> list[dict[str, float]]:
    """P/R/F1 micro de cada pliegue, sin IC."""
    return [_prf(casos([p], decisor, exclusiones), [d.id for d in p.documentos]) for p in pliegues]


def comparar(
    a: Sequence[charlson_codiesp.Caso],
    b: Sequence[charlson_codiesp.Caso],
    ids: Sequence[str],
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    categoria: CategoriaCharlson | None = None,
) -> dict[str, Any]:
    """`b` menos `a` en P/R/F1 micro (o de una `categoria`), con el IC del 95 % por
    bootstrap pareado."""
    conteos_a = charlson_codiesp._por_documento(a, ids, categoria)
    conteos_b = charlson_codiesp._por_documento(b, ids, categoria)
    puntual_a, puntual_b = micro_prf(conteos_a), micro_prf(conteos_b)
    ic = bootstrap_diferencia(conteos_a, conteos_b, n=replicas, semilla=distemist.SEMILLA_BOOTSTRAP)
    return {
        metrica: {
            "diferencia": round(getattr(puntual_b, metrica) - getattr(puntual_a, metrica), 4),
            "ic95": [round(x, 4) for x in getattr(ic, metrica)],
        }
        for metrica in ("precision", "recall", "f1")
    }


def causa_perdida(
    caso: charlson_codiesp.Caso,
    top: Sequence[Propuesta],
    umbral: float,
    reglas: Resultado,
) -> str | None:
    """Por qué la mención no recibió su categoría gold (ver `CAUSAS`), o `None` si la
    recibió o no tiene categoría gold."""
    if caso.gold is None or caso.prediccion is caso.gold:
        return None
    primero = top[0]
    if primero.categoria is caso.gold:
        return "excluida" if primero.similitud >= umbral else "bajo_umbral"
    if any(p.categoria is caso.gold for p in top[1:5]):
        return "top5"
    if any(p.categoria is caso.gold for p in top[5:]):
        return "topk"
    if reglas is not None and reglas[0] is caso.gold:
        return "solo_reglas"
    return "fuera"


def _menciones(pliegues: Sequence[Pliegue]) -> list[tuple[str, int, int, dict]]:
    """(documento, inicio, fin, top-k de su pliegue) de cada mención, en el orden de
    `charlson_codiesp.casos`. El mismo texto tiene otro top-k en otro pliegue."""
    return [
        (d.id, m.inicio, m.fin, p.topk)
        for p in pliegues
        for d in p.documentos
        for m in d.menciones_diagnostico
    ]


def errores(
    pliegues: Sequence[Pliegue],
    casos_: Sequence[charlson_codiesp.Caso],
    umbral: float = UMBRAL_HIBRIDO,
    reglas: Mapeador = charlson_codiesp.METODOS["reglas_corrector"],
) -> dict[str, Any]:
    """Causas de las menciones perdidas y tipo de los falsos positivos, con el detalle de
    cada perdida (sin texto)."""
    perdidas: Counter[str] = Counter({c: 0 for c in CAUSAS})
    por_categoria: Counter[str] = Counter()
    falsos_positivos = Counter({"gold_sin_categoria": 0, "gold_otra_categoria": 0})
    detalle = []
    for caso, (documento, inicio, fin, topk) in zip(casos_, _menciones(pliegues), strict=True):
        if caso.documento != documento:
            raise ValueError(f"Casos y menciones desalineados: {caso.documento} y {documento}")
        if caso.prediccion is not None and caso.prediccion is not caso.gold:
            falsos_positivos[
                "gold_sin_categoria" if caso.gold is None else "gold_otra_categoria"
            ] += 1
        causa = causa_perdida(caso, topk[caso.texto], umbral, reglas(caso.texto))
        if causa is None:
            continue
        perdidas[causa] += 1
        por_categoria[caso.gold.value] += 1
        detalle.append(
            {
                "documento": documento,
                "inicio": inicio,
                "fin": fin,
                "codigo_gold": caso.codigo_gold,
                "gold": caso.gold.value,
                "prediccion": None if caso.prediccion is None else caso.prediccion.value,
                "causa": causa,
            }
        )
    return {
        "perdidas": sum(perdidas.values()),
        "causas": dict(perdidas),
        "por_categoria": dict(por_categoria.most_common()),
        "falsos_positivos": dict(falsos_positivos),
        "detalle": detalle,
    }


def techo(pliegues: Sequence[Pliegue], ks: Sequence[int] = KS_TECHO) -> dict[str, Any]:
    """P/R/F1 del oráculo que acierta si la categoría gold está en el top-k (ver arriba).

    Un `k` mayor que el de las tablas daría lo mismo que el de las tablas, así que falla.
    """
    disponible = max((len(top) for p in pliegues for top in p.topk.values()), default=0)
    if any(k > disponible for k in ks):
        raise ValueError(f"El top-k guardado tiene {disponible} candidatos; se pidió {max(ks)}.")
    ids = _ids(pliegues)
    salida = {}
    for k in ks:
        lista = []
        for pliegue in pliegues:
            for documento in pliegue.documentos:
                for m in documento.menciones_diagnostico:
                    texto = charlson_codiesp.texto_en(documento.texto, m)
                    gold = quan.categoria_de(m.codigo)
                    alcanza = gold is not None and any(
                        p.categoria is gold for p in pliegue.topk[texto][:k]
                    )
                    lista.append(
                        charlson_codiesp.Caso(
                            documento.id, texto, m.codigo, gold, None, gold if alcanza else None
                        )
                    )
        salida[str(k)] = _prf(lista, ids)
    return salida


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    desarrollo: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    ids_prueba: Sequence[str],
    k: int = K,
    pliegues: int = PLIEGUES,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    umbral: float = UMBRAL_HIBRIDO,
) -> dict[str, Any]:
    """El JSON del arnés, sin texto del corpus (ver el docstring del módulo)."""
    verificacion = verificar_origen(
        sinonimos_de(entrenamiento),
        {"dev": [d.id for d in desarrollo], "test": ids_prueba},
    )
    inicio = time.perf_counter()
    internos_cv = pliegues_topk(indice, codificador, entrenamiento, pliegues, k)
    segundos_topk = round(time.perf_counter() - inicio, 1)
    ids = _ids(internos_cv)
    decisor = primer_candidato(umbral)
    metodos = {
        "reglas_corrector": [
            c
            for p in internos_cv
            for c in charlson_codiesp.casos(
                p.documentos, charlson_codiesp.METODOS["reglas_corrector"]
            )
        ],
        "modelo_umbral": casos(internos_cv, decisor, exclusiones=False),
        "hibrido_exclusiones": casos(internos_cv, decisor),
    }
    validacion = {
        nombre: {
            "total": charlson_codiesp.metricas(lista, ids, replicas, por_categoria=False)["micro"],
            "por_pliegue": (
                por_pliegue(internos_cv, decisor, exclusiones=nombre == "hibrido_exclusiones")
                if nombre != "reglas_corrector"
                else None
            ),
        }
        for nombre, lista in metodos.items()
    }
    ks = [x for x in KS_TECHO if x <= k]
    errores_cv = errores(internos_cv, metodos["hibrido_exclusiones"], umbral)
    del errores_cv["detalle"]

    final = _buscar(indice, codificador, entrenamiento, desarrollo, k)
    casos_dev = casos([final], decisor)
    ids_dev = [d.id for d in desarrollo]

    modelo = CANDIDATOS[nombre_modelo]
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": modelo.nombre},
        "modelo": {
            "clave": nombre_modelo,
            "nombre": modelo.nombre,
            "revision": modelo.revision,
            "licencia": modelo.licencia,
            "transformers": version_de("transformers"),
        },
        "indice": meta_indice,
        "k": k,
        "pliegues": pliegues,
        "umbral": {"valor": umbral, "elegido_en": "train, validación cruzada (#132)"},
        "particiones": {
            "validacion_cruzada": "train",
            "reproduccion_y_perdidas": "dev",
            "prueba": "no se lee; solo los nombres de archivo",
        },
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": {"train": len(entrenamiento), "dev": len(desarrollo)},
        "verificacion_origen": verificacion,
        "validacion_cruzada": validacion,
        "comparaciones": {
            "hibrido_menos_modelo": comparar(
                metodos["modelo_umbral"], metodos["hibrido_exclusiones"], ids, replicas
            ),
            "hibrido_menos_reglas": comparar(
                metodos["reglas_corrector"], metodos["hibrido_exclusiones"], ids, replicas
            ),
        },
        "errores_validacion_cruzada": errores_cv,
        "techo_validacion_cruzada": techo(internos_cv, ks),
        "desarrollo": {
            "hibrido_exclusiones": charlson_codiesp.metricas(
                casos_dev, ids_dev, replicas, por_categoria=False
            )["micro"],
            "techo": techo([final], ks),
            "errores": errores([final], casos_dev, umbral),
        },
        "costo": {
            "segundos_topk_validacion_cruzada": segundos_topk,
            # Con 0, los vectores ya estaban en disco: los segundos no incluyen codificar
            "vectores_codificados": getattr(codificador, "nuevos", None),
        },
    }


def tabla(resultados: dict[str, Any]) -> str:
    """Validación cruzada en entrenamiento, causas de las perdidas y techo por k."""
    lineas = [
        "| Método (validación cruzada, train) | P | R | F1 [IC 95 %] | F1 por pliegue |",
        "|---|---|---|---|---|",
    ]
    for nombre, m in resultados["validacion_cruzada"].items():
        t = m["total"]
        pliegues_ = " · ".join(f"{p['f1']:.3f}" for p in m["por_pliegue"] or [])
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {t['precision']:.3f} | {t['recall']:.3f} | "
            f"{t['f1']:.3f} [{t['ic95']['f1'][0]:.3f}; {t['ic95']['f1'][1]:.3f}] | "
            f"{pliegues_ or '—'} |"
        )
    lineas += ["", "| Causa de la perdida | train (VC) | dev |", "|---|---|---|"]
    cv = resultados["errores_validacion_cruzada"]
    dev = resultados["desarrollo"]["errores"]
    for causa in CAUSAS:
        lineas.append(f"| {causa} | {cv['causas'][causa]} | {dev['causas'][causa]} |")
    lineas.append(f"| **total** | {cv['perdidas']} | {dev['perdidas']} |")
    lineas += ["", "| k | F1 del techo (train, VC) | F1 del techo (dev) |", "|---|---|---|"]
    for k, m in resultados["techo_validacion_cruzada"].items():
        lineas.append(f"| {k} | {m['f1']:.3f} | {resultados['desarrollo']['techo'][k]['f1']:.3f} |")
    return "\n".join(lineas)
