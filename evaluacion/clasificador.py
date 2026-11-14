"""Segunda etapa entrenada: un clasificador sobre el top-k (PRD `normalizacion-f1`, FR3, #151).

Los votos de #150 deciden solo con la similitud. Aquí un modelo pequeño aprende a decidir
con más señales. Por cada mención se arman **pares (mención, categoría candidata)**: cada
categoría de Charlson que aparece en el top-k, más la que dan las reglas aunque no esté
(así se pueden recuperar las menciones que solo las reglas reconocen). El modelo da la
probabilidad de que el par sea correcto. Gana el par más probable si pasa el `umbral`; si
no, la mención queda sin categoría. Después van las exclusiones de las reglas, como en
todas las segundas etapas.

**Atributos** de cada par (`ATRIBUTOS`), todos sin texto:

- Similitud: la del mejor candidato de la categoría, su rango y los márgenes contra el
  primero del top y contra la mejor de otra categoría (o "ninguna"). La del primero no va
  aparte: es la similitud menos el margen, y repetirla haría los pesos menos legibles.
- Votos: la fracción del peso de la categoría y de "ninguna" con el softmax de #150
  (temperaturas 0,05 y 0,1) y con la suma de similitudes, y cuántos de los 5 y de los 20
  primeros son de la categoría.
- Jerarquía: la fracción del peso del capítulo y del bloque de su mejor candidato.
- Origen: si su mejor candidato es un sinónimo, qué parte de sus candidatos lo son y si la
  similitud es de 0,999 o más (en la práctica, el texto está tal cual en el índice).
- Mención: largo, número de palabras y si es una sigla.
- Reglas: si las reglas + corrector dan esta categoría, si dan otra y si la exclusión de la
  regla de esta categoría aplica.
- La categoría misma (una columna por categoría), para que cada una tenga su sesgo.

**Modelo**: regresión logística con penalización L2, en torch (ya es dependencia; no hace
falta scikit-learn). Los atributos se estandarizan con la media y la desviación del
entrenamiento. Es explicable: cada peso dice cuánto empuja un atributo, y se reportan.

**Sin fuga** (validación cruzada anidada, como en #150). Para cada pliegue externo `i`:

1. Los pares de los pliegues internos salen de índices sin los sinónimos de `i` ni del
   interno (`dos_etapas.pliegues_internos`), así que se parecen a lo que se verá en `i`.
2. La penalización y el umbral se eligen dejando fuera un interno a la vez: se entrena con
   los otros tres y se predice el que quedó fuera.
3. Con lo elegido se entrena con los cuatro internos y se decide el pliegue `i`.

En la misma vuelta se eligen los votos de #150 con los mismos internos, para compararlos
pareados. `final` entrena con los 5 pliegues: es lo que se medirá en desarrollo (#154).

    uv run python -m evaluacion clasificador [--modelo sapbert]
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from evaluacion import charlson_codiesp, distemist
from evaluacion.cache import version_de
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.dos_etapas import (
    UMBRAL_HIBRIDO,
    K,
    Pliegue,
    comparar,
    pliegues_internos,
    pliegues_topk,
    primer_candidato,
)
from evaluacion.dos_etapas import casos as casos_de
from evaluacion.normalizador import CANDIDATOS, Codificador, Indice, excluida
from evaluacion.sinonimos_codiesp import PLIEGUES
from evaluacion.votos import _Buscador, _menciones, recuperadas, votar
from ner_clinic.normalizacion.decision import (  # noqa: F401  (se reexportan)
    ATRIBUTOS,
    CATEGORIAS,
    Extra,
    Modelo,
    Par,
    ReglasDe,
    decisor,
    pares,
)
from ner_clinic.normalizacion.decision import fraccion as _fraccion  # noqa: F401
from ner_clinic.normalizacion.decision import reglas_de as _reglas
from ner_clinic.tipos import CategoriaCharlson

RUTA_RESULTADOS = Path("evaluacion/resultados/dos_etapas_clasificador.json")
PENALIZACIONES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
UMBRALES = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0,05 a 0,95


def entrenar(x: torch.Tensor, y: torch.Tensor, penalizacion: float) -> Modelo:
    """Regresión logística con L2 (sin penalizar el sesgo), por L-BFGS. Determinista."""
    media = x.mean(0)
    desviacion = x.std(0).clamp(min=1e-6)
    z = (x - media) / desviacion
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    z, y = z.double(), y.double()
    optimizador = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def perdida():
        optimizador.zero_grad()
        valor = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y)
        valor = valor + penalizacion * (w @ w)
        valor.backward()
        return valor

    optimizador.step(perdida)
    return Modelo(media, desviacion, w.detach().float(), float(b.detach()))


# --- Datos por menciones -------------------------------------------------------------


@dataclass
class Datos:
    """Los pares de unas menciones, en una matriz, con el índice de su mención."""

    textos: list[str]
    gold: list[CategoriaCharlson | None]
    pares: list[list[Par]]

    @classmethod
    def de(
        cls, pliegues: Sequence[Pliegue], reglas: ReglasDe, extra: Extra | None = None
    ) -> "Datos":
        menciones = _menciones(pliegues)
        return cls(
            [m.texto for m in menciones],
            [m.gold for m in menciones],
            [pares(m.texto, m.top, _reglas(m.texto, reglas), extra) for m in menciones],
        )

    def matriz(self) -> tuple[torch.Tensor, torch.Tensor]:
        filas = [p.atributos for ps in self.pares for p in ps]
        etiquetas = [
            float(p.categoria is g) for ps, g in zip(self.pares, self.gold, strict=True) for p in ps
        ]
        return torch.tensor(filas, dtype=torch.float32), torch.tensor(etiquetas)

    def unir(self, otros: "Datos") -> "Datos":
        return Datos(self.textos + otros.textos, self.gold + otros.gold, self.pares + otros.pares)

    def mejores(self, modelo: Modelo) -> list[tuple[Par, float] | None]:
        """Por mención, el par más probable y su probabilidad (sin umbral)."""
        x, _ = self.matriz()
        if not len(x):
            return [None] * len(self.pares)
        probabilidades = modelo.probabilidades(x).tolist()
        salida, i = [], 0
        for ps in self.pares:
            if not ps:
                salida.append(None)
                continue
            propias = probabilidades[i : i + len(ps)]
            j = max(range(len(ps)), key=propias.__getitem__)
            salida.append((ps[j], propias[j]))
            i += len(ps)
        return salida

    def f1(self, mejores: Sequence[tuple[Par, float] | None], umbral: float) -> float:
        """F1 micro con `umbral`, con las exclusiones de las reglas después."""
        vp = fp = fn = 0
        for texto, gold, mejor in zip(self.textos, self.gold, mejores, strict=True):
            prediccion = None
            if mejor is not None and mejor[1] >= umbral and not excluida(texto, mejor[0].categoria):
                prediccion = mejor[0].categoria
            acierto = gold is not None and prediccion is gold
            vp += acierto
            fp += prediccion is not None and not acierto
            fn += gold is not None and not acierto
        return 2 * vp / (2 * vp + fp + fn) if vp else 0.0


def elegir(internos: Sequence[Datos]) -> tuple[float, float, float, dict[float, float]]:
    """Penalización y umbral por validación cruzada sobre `internos` (dejando uno fuera).

    A igual F1, más penalización y umbral más alto. Devuelve también el F1 elegido y el
    mejor F1 de cada penalización (con su mejor umbral), para ver si la rejilla alcanza.
    """
    mejor = (PENALIZACIONES[0], UMBRALES[0], -1.0)
    curva: dict[float, float] = {}
    for penalizacion in sorted(PENALIZACIONES, reverse=True):
        predichos: list[tuple[Datos, list]] = []
        for j, fuera in enumerate(internos):
            entrenamiento = _juntar([d for m, d in enumerate(internos) if m != j])
            modelo = entrenar(*entrenamiento.matriz(), penalizacion)
            predichos.append((fuera, fuera.mejores(modelo)))
        todos = _juntar([d for d, _ in predichos])
        mejores = [m for _, ms in predichos for m in ms]
        for umbral in sorted(UMBRALES, reverse=True):
            f1 = todos.f1(mejores, umbral)
            curva[penalizacion] = max(curva.get(penalizacion, 0.0), f1)
            if f1 > mejor[2] + 1e-12:
                mejor = (penalizacion, umbral, f1)
    return (*mejor, dict(sorted(curva.items())))


def _juntar(datos: Sequence[Datos]) -> Datos:
    salida = Datos([], [], [])
    for d in datos:
        salida = salida.unir(d)
    return salida


# --- Evaluación ----------------------------------------------------------------------


def en_pliegue(
    internos: Sequence[Pliegue], externo: Pliegue, reglas: ReglasDe, extra: Extra | None = None
) -> tuple[list[charlson_codiesp.Caso], dict[str, Any]]:
    """Elige en `internos`, entrena con todos ellos y decide `externo`."""
    datos = [Datos.de([p], reglas, extra) for p in internos]
    penalizacion, umbral, f1, _ = elegir(datos)
    modelo = entrenar(*_juntar(datos).matriz(), penalizacion)
    lista = casos_de([externo], decisor(modelo, umbral, reglas, extra))
    return lista, {"penalizacion": penalizacion, "umbral": umbral, "f1_internos": round(f1, 4)}


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    pliegues: int = PLIEGUES,
    k: int = K,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
) -> dict[str, Any]:
    """El JSON de #151: solo entrenamiento, sin texto del corpus."""
    reglas: ReglasDe = {}
    externos = pliegues_topk(indice, codificador, entrenamiento, pliegues, k)
    ids = [d.id for p in externos for d in p.documentos]
    base = casos_de(externos, primer_candidato(UMBRAL_HIBRIDO))

    casos_votos, casos_clasificador, por_pliegue = [], [], []
    for i, externo in enumerate(externos):
        internos = pliegues_internos(indice, codificador, entrenamiento, i, pliegues, k)
        voto, f1_voto = _Buscador(_menciones(internos)).mejor()
        casos_votos += casos_de([externo], votar(voto))

        datos = [Datos.de([p], reglas) for p in internos]
        penalizacion, umbral, f1, curva = elegir(datos)
        modelo = entrenar(*_juntar(datos).matriz(), penalizacion)
        casos_clasificador += casos_de([externo], decisor(modelo, umbral, reglas))
        por_pliegue.append(
            {
                "penalizacion": penalizacion,
                "umbral": umbral,
                "f1_internos": round(f1, 4),
                "f1_internos_por_penalizacion": {str(c): round(v, 4) for c, v in curva.items()},
                "votos": {**voto.a_json(), "f1_internos": round(f1_voto, 4)},
            }
        )

    def metricas(lista: Sequence[charlson_codiesp.Caso]) -> dict[str, Any]:
        return charlson_codiesp.metricas(lista, ids, replicas, por_categoria=False)["micro"]

    # El modelo final, con los 5 pliegues. Su F1 es de validación cruzada sobre ellos, pero
    # maximizado en la rejilla: optimista por la selección.
    datos_externos = [Datos.de([p], reglas) for p in externos]
    penalizacion, umbral, f1_final, curva_final = elegir(datos_externos)
    final = entrenar(*_juntar(datos_externos).matriz(), penalizacion)
    inicio = time.perf_counter()
    casos_de(externos, decisor(final, umbral, reglas))
    segundos = time.perf_counter() - inicio
    menciones = sum(len(d.menciones_diagnostico) for p in externos for d in p.documentos)

    modelo_embeddings = CANDIDATOS[nombre_modelo]
    pesos = final.pesos_por_atributo()
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": {**distemist.entorno(), "modelo": modelo_embeddings.nombre},
        "modelo": {
            "clave": nombre_modelo,
            "nombre": modelo_embeddings.nombre,
            "revision": modelo_embeddings.revision,
            "licencia": modelo_embeddings.licencia,
            "transformers": version_de("transformers"),
        },
        "indice": meta_indice,
        "particion": "train (validación cruzada anidada); desarrollo y prueba no se leen",
        "pliegues": pliegues,
        "k_guardado": k,
        "rejilla": {"penalizaciones": list(PENALIZACIONES), "umbrales": list(UMBRALES)},
        "atributos": list(ATRIBUTOS),
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": len(entrenamiento),
        "por_pliegue": por_pliegue,
        "metodos": {
            "hibrido_exclusiones": metricas(base),
            "votos": metricas(casos_votos),
            "clasificador": metricas(casos_clasificador),
        },
        "comparaciones": {
            "votos_menos_hibrido": comparar(base, casos_votos, ids, replicas),
            "clasificador_menos_hibrido": comparar(base, casos_clasificador, ids, replicas),
            "clasificador_menos_votos": comparar(casos_votos, casos_clasificador, ids, replicas),
        },
        "recuperadas": {
            "votos": recuperadas(externos, base, casos_votos),
            "clasificador": recuperadas(externos, base, casos_clasificador),
        },
        "final": {
            "penalizacion": penalizacion,
            "umbral": umbral,
            "f1_validacion_cruzada_optimista": round(f1_final, 4),
            "f1_por_penalizacion": {str(c): round(v, 4) for c, v in curva_final.items()},
            "parametros": len(ATRIBUTOS) + 1,
            "pesos": dict(sorted(pesos.items(), key=lambda x: -abs(x[1]))),
            "sesgo": round(final.sesgo, 4),
            "pares_entrenamiento": int(_juntar(datos_externos).matriz()[0].shape[0]),
        },
        # Con las reglas ya calculadas (el pipeline ya las corre); sin codificar ni buscar
        "costo": {
            "milisegundos_por_mencion": round(1000 * segundos / max(menciones, 1), 3),
            "milisegundos_por_documento": round(1000 * segundos / len(ids), 3),
            "menciones_por_documento": round(menciones / len(ids), 1),
        },
    }


def tabla(resultados: dict[str, Any]) -> str:
    """F1 en validación cruzada anidada y diferencias pareadas."""
    lineas = [
        "| Segunda etapa (VC anidada, train) | P | R | F1 [IC 95 %] |",
        "|---|---|---|---|",
    ]
    for nombre, m in resultados["metodos"].items():
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} [{m['ic95']['f1'][0]:.3f}; {m['ic95']['f1'][1]:.3f}] |"
        )
    lineas += ["", "| Comparación pareada | Δ F1 [IC 95 %] |", "|---|---|"]
    for nombre, c in resultados["comparaciones"].items():
        d = c["f1"]
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {d['diferencia']:+.3f} "
            f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}] |"
        )
    return "\n".join(lineas)
