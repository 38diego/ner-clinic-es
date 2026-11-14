"""Segunda etapa con un reordenador local (PRD `normalizacion-f1`, FR4, #152).

Un *cross-encoder* lee la mención junto a la descripción de cada candidato y les da una
puntuación. A diferencia del índice, que compara dos vectores calculados por separado, el
reordenador mira los dos textos a la vez. Es más preciso en teoría y mucho más caro: una
pasada del modelo por cada par (mención, candidato).

**Candidatos** (`CANDIDATOS`), con revisión fija. Ninguno ejecuta código remoto:

- `bge_m3`: `BAAI/bge-reranker-v2-m3`, Apache 2.0, multilingüe, 568M de parámetros.
- `bge_base`: `BAAI/bge-reranker-base`, MIT, 278M. Solo declara inglés y chino.
- `mmarco`: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, Apache 2.0, 118M, con español.
  Se entrenó con mMARCO, derivado de MS MARCO, cuyos términos son solo para investigación
  no comercial: la misma clase de duda que SapBERT.

**Dos usos**, con los primeros `k` candidatos del top (5 o 10):

1. **Solo** (`reordenar`): las puntuaciones reemplazan a la similitud y el top se ordena por
   ellas. Después se usa la búsqueda de votos de #150; con `k=1` decide el mejor puntuado.
2. **Como atributos del clasificador de #151** (`atributos`): la mejor puntuación de la
   categoría, el margen contra la mejor de otra y su fracción de la suma.

**Costo** (`latencia`): segundos que suma a una nota de unas 8 entidades, con pares reales
del top-k y 2 hilos de CPU en el portátil. Es una aproximación a los 2 vCPU de Cloud Run,
no una medición en Docker.

**Sin fuga.** La elección es la validación cruzada anidada de #150 y #151. Las puntuaciones
no se entrenan: dependen solo del par de textos, así que se guardan en disco por par
(`PuntuacionesEnDisco`, bajo `data/`) y sirven para todos los pliegues.

    uv run python -m evaluacion reordenador --reordenador mmarco [--modelo sapbert]
"""

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from evaluacion import charlson_codiesp, clasificador, distemist
from evaluacion.cache import version_de
from evaluacion.clasificador import Datos, Extra, ReglasDe, _juntar, elegir, en_pliegue, entrenar
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.dos_etapas import (
    UMBRAL_HIBRIDO,
    K,
    Pliegue,
    Propuesta,
    _prf,
    comparar,
    pliegues_internos,
    pliegues_topk,
    primer_candidato,
)
from evaluacion.dos_etapas import casos as casos_de
from evaluacion.normalizador import CANDIDATOS as MODELOS_EMBEDDINGS
from evaluacion.normalizador import RAIZ_INDICES, Codificador, Indice
from evaluacion.sinonimos_codiesp import PLIEGUES
from evaluacion.votos import _Buscador, _menciones, recuperadas, votar
from ner_clinic.tipos import CategoriaCharlson

CARPETA_RESULTADOS = Path("evaluacion/resultados")
KS_REORDENADOR = (5, 10)
LARGO_MAXIMO = 64  # tokens: mención y descripción son cortas
LOTE = 64
ENTIDADES_POR_NOTA = 8  # 1.084 entidades en 138 notas de 2.000 caracteres (#134)
HILOS_LATENCIA = 2  # los 2 vCPU de Cloud Run
NOTAS_LATENCIA = 10
UMBRALES = tuple(round(0.05 * i, 2) for i in range(1, 20))  # puntuaciones entre 0 y 1
ATRIBUTOS = ("reordenador", "reordenador_margen", "reordenador_fraccion")

Par = tuple[str, str]
Puntuador = Callable[[Sequence[Par]], list[float]]


@dataclass(frozen=True)
class ModeloReordenador:
    nombre: str
    revision: str
    licencia: str
    datos: str


CANDIDATOS = {
    "bge_m3": ModeloReordenador(
        "BAAI/bge-reranker-v2-m3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "Apache-2.0",
        "su tarjeta no declara los datos de entrenamiento",
    ),
    "bge_base": ModeloReordenador(
        "BAAI/bge-reranker-base",
        "2cfc18c9415c912f9d8155881c133215df768a70",
        "MIT",
        "su tarjeta no declara los datos de entrenamiento; solo inglés y chino",
    ),
    "mmarco": ModeloReordenador(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "1427fd652930e4ba29e8149678df786c240d8825",
        "Apache-2.0",
        "mMARCO, derivado de MS MARCO (términos: investigación no comercial)",
    ),
}


def ruta_resultados(clave: str) -> Path:
    return CARPETA_RESULTADOS / f"dos_etapas_reordenador_{clave}.json"


def ruta_puntuaciones(modelo: ModeloReordenador, raiz: Path = RAIZ_INDICES) -> Path:
    nombre = modelo.nombre.replace("/", "__")
    return Path(raiz) / f"reordenador_{nombre}_{modelo.revision[:12]}.pt"


class ReordenadorHF:  # pragma: no cover - usa el modelo real
    """Un *cross-encoder* de Hugging Face en CPU. Devuelve la puntuación entre 0 y 1."""

    def __init__(self, modelo: ModeloReordenador, lote: int = LOTE):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.lote = lote
        self.tokenizador = AutoTokenizer.from_pretrained(modelo.nombre, revision=modelo.revision)
        self.red = AutoModelForSequenceClassification.from_pretrained(
            modelo.nombre, revision=modelo.revision
        ).eval()

    def __call__(self, pares: Sequence[Par]) -> list[float]:
        # Por largo, para que cada lote rellene lo menos posible
        orden = sorted(range(len(pares)), key=lambda i: len(pares[i][0]) + len(pares[i][1]))
        salida = [0.0] * len(pares)
        for i in range(0, len(orden), self.lote):
            lote = orden[i : i + self.lote]
            entrada = self.tokenizador(
                [pares[j][0] for j in lote],
                [pares[j][1] for j in lote],
                padding=True,
                truncation=True,
                max_length=LARGO_MAXIMO,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.red(**entrada).logits.squeeze(-1)
            for j, valor in zip(lote, torch.sigmoid(logits).tolist(), strict=True):
                salida[j] = valor
        return salida


class PuntuacionesEnDisco:
    """Un `Puntuador` que guarda la puntuación de cada par en `ruta` (bajo `data/`).

    Como `dos_etapas.CodificadorEnDisco`: el modelo se crea solo si falta un par, y el
    archivo trae textos del corpus público, así que no se versiona.
    """

    def __init__(self, crear: Callable[[], Puntuador], ruta: Path):
        self.crear = crear
        self.ruta = Path(ruta)
        self._puntuador: Puntuador | None = None
        self._memoria: dict[Par, float] = {}
        self.nuevos = 0
        if self.ruta.is_file():
            guardado = torch.load(self.ruta, weights_only=True)
            self._memoria = dict(
                zip(map(tuple, guardado["pares"]), guardado["puntuaciones"], strict=True)
            )

    def __call__(self, pares: Sequence[Par]) -> list[float]:
        faltan = list(dict.fromkeys(p for p in pares if p not in self._memoria))
        if faltan:
            if self._puntuador is None:
                self._puntuador = self.crear()
            self._memoria.update(zip(faltan, self._puntuador(faltan), strict=True))
            self.nuevos += len(faltan)
        return [self._memoria[p] for p in pares]

    def guardar(self) -> None:
        if not self.nuevos:
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = self.ruta.with_suffix(".tmp")
        torch.save(
            {"pares": list(self._memoria), "puntuaciones": list(self._memoria.values())}, temporal
        )
        temporal.replace(self.ruta)
        self.nuevos = 0


def memoria_residente_mib() -> float:
    """Memoria residente del proceso, en MiB (Linux)."""
    for linea in Path("/proc/self/status").read_text().splitlines():
        if linea.startswith("VmRSS:"):
            return int(linea.split()[1]) / 1024
    return 0.0


def pares_de(pliegues: Iterable[Pliegue], k: int) -> list[Par]:
    """Los pares (mención, descripción) de los `k` primeros candidatos de cada mención."""
    return [
        (texto, p.candidato.descripcion)
        for pliegue in pliegues
        for texto, top in pliegue.topk.items()
        for p in top[:k]
    ]


def reordenar(pliegue: Pliegue, puntuar: Puntuador, k: int) -> Pliegue:
    """El mismo pliegue con los `k` primeros candidatos, ordenados por el reordenador y con
    su puntuación en el lugar de la similitud."""
    topk = {}
    for texto, top in pliegue.topk.items():
        primeros = top[:k]
        puntos = puntuar([(texto, p.candidato.descripcion) for p in primeros])
        nuevos = [
            replace(p, candidato=replace(p.candidato, similitud=round(s, 4)))
            for p, s in zip(primeros, puntos, strict=True)
        ]
        topk[texto] = sorted(nuevos, key=lambda p: -p.similitud)
    return Pliegue(pliegue.documentos, topk)


def atributos(puntuar: Puntuador, k: int) -> Extra:
    """Los atributos del reordenador para el clasificador, con los `k` primeros candidatos.

    Una categoría sin candidatos en esos `k` (la dan solo las reglas, o aparece más abajo
    en el top) tiene puntuación 0.
    """

    def extra(
        texto: str, top: Sequence[Propuesta], categoria: CategoriaCharlson
    ) -> tuple[float, ...]:
        primeros = top[:k]
        puntos = puntuar([(texto, p.candidato.descripcion) for p in primeros])
        propios = [s for p, s in zip(primeros, puntos, strict=True) if p.categoria is categoria]
        otros = [s for p, s in zip(primeros, puntos, strict=True) if p.categoria is not categoria]
        mejor = max(propios, default=0.0)
        total = sum(puntos)
        return (
            mejor,
            mejor - max(otros, default=0.0),
            sum(propios) / total if total else 0.0,
        )

    return extra


def latencia(
    puntuar: Puntuador,
    pliegue: Pliegue,
    ks: Sequence[int] = KS_REORDENADOR,
    notas: int = NOTAS_LATENCIA,
    entidades: int = ENTIDADES_POR_NOTA,
    hilos: int = HILOS_LATENCIA,
) -> dict[str, dict[str, float]]:
    """Segundos que suma el reordenador a una nota, con `hilos` hilos de CPU.

    Cada nota son `entidades` menciones reales de `pliegue` con sus `k` primeros candidatos
    (`puntuar` tiene que ser el modelo, sin la memoria en disco). La primera pasada calienta
    y no cuenta. `hilos` limita a torch, no al tokenizador, que tiene hilos propios.
    """
    textos = list(pliegue.topk)
    hilos_antes = torch.get_num_threads()
    torch.set_num_threads(hilos)
    salida = {}
    try:
        for k in ks:
            tiempos = []
            for nota in range(notas + 1):
                inicio = nota * entidades
                grupo = [textos[(inicio + i) % len(textos)] for i in range(entidades)]
                pares = [
                    (texto, p.candidato.descripcion)
                    for texto in grupo
                    for p in pliegue.topk[texto][:k]
                ]
                antes = time.perf_counter()
                puntuar(pares)
                tiempos.append(time.perf_counter() - antes)
            medidos = sorted(tiempos[1:])
            salida[f"k{k}"] = {
                "pares_por_nota": entidades * k,
                "mediana_s": round(medidos[len(medidos) // 2], 3),
                "maximo_s": round(medidos[-1], 3),
            }
    finally:
        torch.set_num_threads(hilos_antes)
    return salida


def evaluar(
    clave: str,
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    puntuar: Puntuador,
    entrenamiento: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    pliegues: int = PLIEGUES,
    k: int = K,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    ks: Sequence[int] = KS_REORDENADOR,
    modelo_real: Puntuador | None = None,
) -> dict[str, Any]:
    """El JSON de #152 para un reordenador: solo entrenamiento, sin texto del corpus.

    Con `modelo_real` (el reordenador sin la memoria en disco) mide además la latencia.
    """
    reglas: ReglasDe = {}
    externos = pliegues_topk(indice, codificador, entrenamiento, pliegues, k)
    ids = [d.id for p in externos for d in p.documentos]
    base = casos_de(externos, primer_candidato(UMBRAL_HIBRIDO))

    inicio = time.perf_counter()
    casos_: dict[str, list[charlson_codiesp.Caso]] = {"clasificador": []}
    casos_ |= {f"solo_k{x}": [] for x in ks} | {f"clasificador_k{x}": [] for x in ks}
    elegidas: dict[str, list[dict[str, Any]]] = {nombre: [] for nombre in casos_}
    for i, externo in enumerate(externos):
        internos = pliegues_internos(indice, codificador, entrenamiento, i, pliegues, k)
        puntuar(pares_de([*internos, externo], max(ks)))

        lista, eleccion = en_pliegue(internos, externo, reglas, None)
        casos_["clasificador"] += lista
        elegidas["clasificador"].append(eleccion)
        for x in ks:
            reordenados = [reordenar(p, puntuar, x) for p in internos]
            buscador = _Buscador(_menciones(reordenados))
            voto, f1 = buscador.mejor(ks=[n for n in (1, 3, 5, 10) if n <= x], umbrales=UMBRALES)
            casos_[f"solo_k{x}"] += casos_de([reordenar(externo, puntuar, x)], votar(voto))
            elegidas[f"solo_k{x}"].append({**voto.a_json(), "f1_internos": round(f1, 4)})

            lista, eleccion = en_pliegue(internos, externo, reglas, atributos(puntuar, x))
            casos_[f"clasificador_k{x}"] += lista
            elegidas[f"clasificador_k{x}"].append(eleccion)
    segundos = time.perf_counter() - inicio

    def metricas(lista: Sequence[charlson_codiesp.Caso]) -> dict[str, Any]:
        return charlson_codiesp.metricas(lista, ids, replicas, por_categoria=False)["micro"]

    # Pesos del clasificador con el reordenador, con los 5 pliegues (solo para leerlos)
    pesos = {}
    for x in ks:
        datos = [Datos.de([p], reglas, atributos(puntuar, x)) for p in externos]
        penalizacion, _, _, _ = elegir(datos)
        modelo = entrenar(*_juntar(datos).matriz(), penalizacion)
        todos = modelo.pesos_por_atributo((*clasificador.ATRIBUTOS, *ATRIBUTOS))
        pesos[f"k{x}"] = dict(sorted(todos.items(), key=lambda p: -abs(p[1]))[:12])

    reordenador = CANDIDATOS[clave]
    embeddings = MODELOS_EMBEDDINGS[nombre_modelo]
    referencia = casos_["clasificador"]
    ids_por_pliegue = [[d.id for d in p.documentos] for p in externos]
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": embeddings.nombre},
        "reordenador": {
            "clave": clave,
            "nombre": reordenador.nombre,
            "revision": reordenador.revision,
            "licencia": reordenador.licencia,
            "datos": reordenador.datos,
            "transformers": version_de("transformers"),
        },
        "modelo": {"clave": nombre_modelo, "revision": embeddings.revision},
        "indice": meta_indice,
        "particion": "train (validación cruzada anidada); desarrollo y prueba no se leen",
        "pliegues": pliegues,
        "ks": list(ks),
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": len(entrenamiento),
        "metodos": {
            "hibrido_exclusiones": metricas(base),
            **{nombre: metricas(lista) for nombre, lista in casos_.items()},
        },
        "f1_por_pliegue": {
            nombre: [
                _prf([c for c in lista if c.documento in set(propios)], propios)["f1"]
                for propios in ids_por_pliegue
            ]
            for nombre, lista in {"hibrido_exclusiones": base, **casos_}.items()
        },
        "contra_clasificador": {
            nombre: comparar(referencia, lista, ids, replicas)
            for nombre, lista in casos_.items()
            if nombre != "clasificador"
        },
        "recuperadas": {
            nombre: recuperadas(externos, base, lista) for nombre, lista in casos_.items()
        },
        "elegidas": elegidas,
        "pesos_principales": pesos,
        "costo": {
            "segundos_evaluacion": round(segundos, 1),
            "pares_nuevos": getattr(puntuar, "nuevos", None),
            "latencia": {
                "entidades_por_nota": ENTIDADES_POR_NOTA,
                "hilos": HILOS_LATENCIA,
                "notas": NOTAS_LATENCIA,
                "por_k": latencia(modelo_real, externos[0], ks) if modelo_real else None,
            },
        },
    }


def tabla(resultados: dict[str, Any]) -> str:
    """F1 en validación cruzada anidada y diferencia pareada contra el clasificador."""
    lineas = [
        f"Reordenador: {resultados['reordenador']['nombre']}",
        "",
        "| Segunda etapa (VC anidada, train) | P | R | F1 [IC 95 %] | Δ F1 contra el "
        "clasificador [IC 95 %] |",
        "|---|---|---|---|---|",
    ]
    for nombre, m in resultados["metodos"].items():
        c = resultados["contra_clasificador"].get(nombre)
        delta = (
            f"{c['f1']['diferencia']:+.3f} [{c['f1']['ic95'][0]:+.3f}; {c['f1']['ic95'][1]:+.3f}]"
            if c
            else "—"
        )
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} [{m['ic95']['f1'][0]:.3f}; {m['ic95']['f1'][1]:.3f}] | {delta} |"
        )
    return "\n".join(lineas)
