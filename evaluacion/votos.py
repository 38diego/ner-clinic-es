"""Segunda etapa por votos: categoría y jerarquía de la CIE-10 (PRD `normalizacion-f1`, FR2, #150).

En vez de que el primer candidato decida solo, los `k` primeros candidatos del top-k votan
(`votar`). Cada candidato pesa:

- su similitud (`temperatura=None`, suma de similitudes), o
- `exp((s - s_max) / temperatura)`, un softmax: con temperatura baja manda el primero y
  con alta pesan parecido todos.

**Dónde se vota** (`nivel`):

- `categoria`: gana la categoría de Charlson con más peso. "Ninguna" (los candidatos sin
  categoría) es una opción más, así que el voto también puede abstenerse.
- `codigo_3`, `bloque` o `capitulo`: primero gana el grupo de la CIE-10 con más peso (el
  código de 3 caracteres, el bloque o el capítulo) y después, **dentro de ese grupo**, la
  categoría con más peso. Es el voto jerárquico que propuso Diego (2026-11-06): varios
  vecinos de un mismo bloque pesan más que uno suelto, y el capítulo decide la región
  antes de la categoría. La jerarquía sale de `jerarquia_cie10`.

Si gana una categoría, el mejor candidato de esa categoría entre los votantes tiene que
pasar el `umbral` de similitud. Ese candidato da el código. Con `k=1` es exactamente la
decisión de hoy (`dos_etapas.primer_candidato`), así que la búsqueda puede quedarse con
ella. Después se aplican las exclusiones de las reglas, como en el híbrido.

**Umbral por categoría** (`umbrales`): opcional. Se ajusta de a una categoría, de mayor a
menor número de menciones gold (solo las que tienen al menos `SOPORTE_MINIMO` en el ajuste),
y se queda solo si sube el F1. Es para los tumores bajo el umbral global.

**Elección sin fuga** (`validacion_anidada`). Para cada pliegue externo de entrenamiento,
la configuración se elige en sus pliegues internos (`dos_etapas.pliegues_internos`) y se
mide en el externo. El F1 de todos los externos juntos es la estimación honesta.
`configuracion_final` elige con todos los pliegues: es la que se medirá en desarrollo (#154),
y su F1 en esos mismos pliegues es optimista.

    uv run python -m evaluacion votos [--modelo sapbert]
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion import charlson_codiesp, distemist
from evaluacion.cache import version_de
from evaluacion.codiesp import DocumentoCodificado
from evaluacion.dos_etapas import (
    CAUSAS,
    UMBRAL_HIBRIDO,
    Decisor,
    K,
    Pliegue,
    Propuesta,
    Resultado,
    causa_perdida,
    comparar,
    pliegues_internos,
    pliegues_topk,
    primer_candidato,
)
from evaluacion.dos_etapas import casos as casos_de
from evaluacion.normalizador import CANDIDATOS, Codificador, Indice, excluida
from evaluacion.sinonimos_codiesp import PLIEGUES
from ner_clinic.charlson import quan
from ner_clinic.normalizacion.decision import pesos_de as _pesos
from ner_clinic.tipos import CategoriaCharlson

RUTA_RESULTADOS = Path("evaluacion/resultados/dos_etapas_votos.json")
NIVELES = ("categoria", "codigo_3", "bloque", "capitulo")
KS = (1, 3, 5, 10, 20)
TEMPERATURAS: tuple[float | None, ...] = (None, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
UMBRALES = tuple(round(0.3 + 0.02 * i, 2) for i in range(26))  # 0,30 a 0,80
SOPORTE_MINIMO = 20


@dataclass(frozen=True)
class Configuracion:
    nivel: str = "categoria"
    k: int = 1
    temperatura: float | None = None
    umbral: float = UMBRAL_HIBRIDO
    umbrales: tuple[tuple[CategoriaCharlson, float], ...] = field(default=())

    def umbral_de(self, categoria: CategoriaCharlson) -> float:
        return dict(self.umbrales).get(categoria, self.umbral)

    def a_json(self) -> dict[str, Any]:
        return {
            "nivel": self.nivel,
            "k": self.k,
            "temperatura": self.temperatura,
            "umbral": self.umbral,
            "umbrales": {c.value: u for c, u in self.umbrales},
        }


def _ganador(claves: Sequence[Any], pesos: Sequence[float]) -> Any:
    """La clave con más peso; a igual peso, la que aparece primero en el top."""
    suma: dict[Any, float] = {}
    for clave, peso in zip(claves, pesos, strict=True):
        suma[clave] = suma.get(clave, 0.0) + peso
    return max(suma, key=suma.__getitem__)


def votantes(configuracion: Configuracion, top: Sequence[Propuesta]) -> list[Propuesta]:
    """Los candidatos de la categoría ganadora (sin aplicar el umbral), en orden del top.

    Vacío si gana "ninguna". Es lo que se muestra al lado de la categoría: "3 de 5
    candidatos son tumor maligno: C34.9, C34.1, C78.0".
    """
    top = list(top[: configuracion.k])
    pesos = _pesos(top, configuracion.temperatura)
    if configuracion.nivel != "categoria":
        grupos = [getattr(p, configuracion.nivel) for p in top]
        grupo = _ganador(grupos, pesos)
        elegidos = [i for i, g in enumerate(grupos) if g == grupo]
        top = [top[i] for i in elegidos]
        pesos = [pesos[i] for i in elegidos]
    categoria = _ganador([p.categoria for p in top], pesos)
    return [p for p in top if categoria is not None and p.categoria is categoria]


def votar(configuracion: Configuracion) -> Decisor:
    """La segunda etapa por votos con `configuracion` (ver el docstring del módulo)."""

    def decidir(texto: str, top: Sequence[Propuesta]) -> Resultado:
        a_favor = votantes(configuracion, top)
        if not a_favor:
            return None
        mejor = max(a_favor, key=lambda p: p.similitud)
        if mejor.similitud < configuracion.umbral_de(mejor.categoria):
            return None
        return mejor.categoria, mejor.codigo_categoria

    return decidir


# --- Búsqueda rápida -----------------------------------------------------------------


@dataclass(frozen=True)
class _Mencion:
    texto: str
    gold: CategoriaCharlson | None
    top: tuple[Propuesta, ...]


def _menciones(pliegues: Iterable[Pliegue]) -> list[_Mencion]:
    return [
        _Mencion(
            charlson_codiesp.texto_en(d.texto, m),
            quan.categoria_de(m.codigo),
            tuple(p.topk[charlson_codiesp.texto_en(d.texto, m)]),
        )
        for p in pliegues
        for d in p.documentos
        for m in d.menciones_diagnostico
    ]


_Voto = tuple[CategoriaCharlson, float] | None


class _Buscador:
    """Evalúa muchas configuraciones sobre las mismas menciones, con el F1 micro.

    El umbral se aplica al final, así que cada (nivel, k, temperatura) se vota una sola
    vez y se barren los umbrales sobre lo votado. Las exclusiones se guardan por (texto,
    categoría).
    """

    def __init__(self, menciones: Sequence[_Mencion]):
        self.menciones = menciones
        self._excluida: dict[tuple[str, CategoriaCharlson], bool] = {}
        self._votos: dict[tuple[str, int, float | None], list[_Voto]] = {}

    def _excluye(self, texto: str, categoria: CategoriaCharlson) -> bool:
        clave = (texto, categoria)
        if clave not in self._excluida:
            self._excluida[clave] = excluida(texto, categoria)
        return self._excluida[clave]

    def votos(self, configuracion: Configuracion) -> list[_Voto]:
        """Por mención, la categoría ganadora y la similitud de su mejor candidato, ya con
        las exclusiones y sin umbral."""
        clave = (configuracion.nivel, configuracion.k, configuracion.temperatura)
        if clave in self._votos:
            return self._votos[clave]
        salida = []
        for m in self.menciones:
            a_favor = votantes(configuracion, m.top)
            if not a_favor or self._excluye(m.texto, a_favor[0].categoria):
                salida.append(None)
            else:
                salida.append((a_favor[0].categoria, max(p.similitud for p in a_favor)))
        self._votos[clave] = salida
        return salida

    def f1(self, votos: Sequence[_Voto], c: Configuracion) -> float:
        vp = fp = fn = 0
        for m, voto in zip(self.menciones, votos, strict=True):
            prediccion = voto[0] if voto is not None and voto[1] >= c.umbral_de(voto[0]) else None
            acierto = m.gold is not None and prediccion is m.gold
            vp += acierto
            fp += prediccion is not None and not acierto
            fn += m.gold is not None and not acierto
        return 2 * vp / (2 * vp + fp + fn) if vp else 0.0

    def mejor(
        self,
        niveles: Sequence[str] = NIVELES,
        ks: Sequence[int] = KS,
        temperaturas: Sequence[float | None] = TEMPERATURAS,
        umbrales: Sequence[float] = UMBRALES,
    ) -> tuple[Configuracion, float]:
        """La configuración de mayor F1. A igual F1, la primera en el orden de la rejilla
        (nivel, k chico, suma de similitudes) con el umbral **más alto**, como en #131: el
        que deja menos códigos dudosos."""
        mejor, f1_mejor = Configuracion(), -1.0
        for nivel in niveles:
            for k in ks:
                for temperatura in temperaturas:
                    base = Configuracion(nivel, k, temperatura)
                    votos = self.votos(base)
                    for umbral in sorted(umbrales, reverse=True):
                        c = replace(base, umbral=umbral)
                        f1 = self.f1(votos, c)
                        if f1 > f1_mejor + 1e-12:
                            mejor, f1_mejor = c, f1
        return mejor, f1_mejor

    def por_categoria(
        self, base: Configuracion, umbrales: Sequence[float] = UMBRALES
    ) -> tuple[Configuracion, float]:
        """Ajusta un umbral por categoría sobre `base`, de a una, si sube el F1."""
        votos = self.votos(base)
        actual, f1_actual = base, self.f1(votos, base)
        soporte = Counter(m.gold for m in self.menciones if m.gold is not None)
        for categoria, n in soporte.most_common():
            if n < SOPORTE_MINIMO:
                break
            for umbral in sorted(umbrales, reverse=True):  # a igual F1, el más alto
                propios = dict(actual.umbrales) | {categoria: umbral}
                ordenados = tuple(sorted(propios.items(), key=lambda x: x[0].value))
                c = replace(actual, umbrales=ordenados)
                f1 = self.f1(votos, c)
                if f1 > f1_actual + 1e-12:
                    actual, f1_actual = c, f1
        return actual, f1_actual


def elegir(
    pliegues: Sequence[Pliegue], niveles: Sequence[str] = NIVELES, por_categoria: bool = False
) -> tuple[Configuracion, float]:
    """La mejor configuración en `pliegues` (sus menciones juntas)."""
    buscador = _Buscador(_menciones(pliegues))
    configuracion, f1 = buscador.mejor(niveles)
    if por_categoria:
        configuracion, f1 = buscador.por_categoria(configuracion)
    return configuracion, f1


# --- Evaluación ----------------------------------------------------------------------

VARIANTES = {
    "votos": {"niveles": NIVELES, "por_categoria": False},
    "votos_umbral_por_categoria": {"niveles": NIVELES, "por_categoria": True},
    **{f"solo_{n}": {"niveles": (n,), "por_categoria": False} for n in NIVELES},
}


def validacion_anidada(
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    externos: Sequence[Pliegue],
    n: int = PLIEGUES,
    k: int = K,
    variantes: dict[str, dict[str, Any]] = VARIANTES,
) -> dict[str, list[tuple[Configuracion, float]]]:
    """Por variante, la configuración elegida en los internos de cada pliegue externo y su
    F1 en esos internos. `externos[i]` es el pliegue `i` de `dos_etapas.pliegues_topk`."""
    elegidas: dict[str, list[tuple[Configuracion, float]]] = {v: [] for v in variantes}
    for i in range(len(externos)):
        internos = pliegues_internos(indice, codificador, entrenamiento, i, n, k)
        buscador = _Buscador(_menciones(internos))
        for nombre, opciones in variantes.items():
            configuracion, f1 = buscador.mejor(opciones["niveles"])
            if opciones["por_categoria"]:
                configuracion, f1 = buscador.por_categoria(configuracion)
            elegidas[nombre].append((configuracion, f1))
    return elegidas


def casos_anidados(
    externos: Sequence[Pliegue], configuraciones: Sequence[Configuracion]
) -> list[charlson_codiesp.Caso]:
    """Cada pliegue externo decidido con la configuración elegida sin verlo."""
    return [
        caso
        for pliegue, configuracion in zip(externos, configuraciones, strict=True)
        for caso in casos_de([pliegue], votar(configuracion))
    ]


def recuperadas(
    externos: Sequence[Pliegue],
    base: Sequence[charlson_codiesp.Caso],
    nuevos: Sequence[charlson_codiesp.Caso],
    umbral: float = UMBRAL_HIBRIDO,
) -> dict[str, Any]:
    """De las menciones que la base pierde, cuántas recupera `nuevos`, por causa; y cuántos
    falsos positivos agrega o quita."""
    topk = [p.topk for p in externos for d in p.documentos for _ in d.menciones_diagnostico]
    reglas = charlson_codiesp.METODOS["reglas_corrector"]
    perdidas = Counter({c: 0 for c in CAUSAS})
    recuperadas_ = Counter({c: 0 for c in CAUSAS})
    nuevas_perdidas = falsos_nuevos = falsos_quitados = 0
    for b, nuevo, tablas in zip(base, nuevos, topk, strict=True):
        causa = causa_perdida(b, tablas[b.texto], umbral, reglas(b.texto))
        if causa is not None:
            perdidas[causa] += 1
            recuperadas_[causa] += nuevo.prediccion is nuevo.gold
        elif b.gold is not None and nuevo.prediccion is not nuevo.gold:
            nuevas_perdidas += 1
        fp_base = b.prediccion is not None and b.prediccion is not b.gold
        fp_nuevo = nuevo.prediccion is not None and nuevo.prediccion is not nuevo.gold
        falsos_nuevos += fp_nuevo and not fp_base
        falsos_quitados += fp_base and not fp_nuevo
    return {
        "perdidas_base": dict(perdidas),
        "recuperadas": dict(recuperadas_),
        "aciertos_base_perdidos": nuevas_perdidas,
        "falsos_positivos_nuevos": falsos_nuevos,
        "falsos_positivos_quitados": falsos_quitados,
    }


def evaluar(
    nombre_modelo: str,
    indice: Indice,
    codificador: Codificador,
    entrenamiento: Sequence[DocumentoCodificado],
    meta_indice: dict[str, Any],
    pliegues: int = PLIEGUES,
    k: int = K,
    replicas: int = distemist.REPLICAS_BOOTSTRAP,
    variantes: dict[str, dict[str, Any]] = VARIANTES,
) -> dict[str, Any]:
    """El JSON de #150: solo entrenamiento, sin texto del corpus."""
    externos = pliegues_topk(indice, codificador, entrenamiento, pliegues, k)
    ids = [d.id for p in externos for d in p.documentos]
    base = casos_de(externos, primer_candidato(UMBRAL_HIBRIDO))
    elegidas = validacion_anidada(
        indice, codificador, entrenamiento, externos, pliegues, k, variantes
    )

    def metricas(lista: Sequence[charlson_codiesp.Caso]) -> dict[str, Any]:
        return charlson_codiesp.metricas(lista, ids, replicas, por_categoria=False)["micro"]

    resultados_variantes = {}
    for nombre, lista in elegidas.items():
        nuevos = casos_anidados(externos, [c for c, _ in lista])
        resultados_variantes[nombre] = {
            "por_pliegue": [
                {"configuracion": c.a_json(), "f1_internos": round(f1, 4)} for c, f1 in lista
            ],
            "anidada": metricas(nuevos),
            "contra_hibrido": comparar(base, nuevos, ids, replicas),
            "recuperadas": recuperadas(externos, base, nuevos),
        }

    finales = {}
    for nombre in ("votos", "votos_umbral_por_categoria"):
        if nombre not in variantes:
            continue
        configuracion, f1 = elegir(externos, **variantes[nombre])
        finales[nombre] = {
            "configuracion": configuracion.a_json(),
            "f1_validacion_cruzada_optimista": round(f1, 4),
        }

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
        "particion": "train (validación cruzada anidada); desarrollo y prueba no se leen",
        "pliegues": pliegues,
        "k_guardado": k,
        "rejilla": {
            "niveles": list(NIVELES),
            "k": list(KS),
            "temperaturas": list(TEMPERATURAS),
            "umbrales": list(UMBRALES),
            "soporte_minimo_por_categoria": SOPORTE_MINIMO,
        },
        "bootstrap": {
            "replicas": replicas,
            "semilla": distemist.SEMILLA_BOOTSTRAP,
            "nivel": 0.95,
            "unidad": "documento",
        },
        "documentos": len(entrenamiento),
        "hibrido_exclusiones": metricas(base),
        "variantes": resultados_variantes,
        "configuracion_final": finales,
    }


def tabla(resultados: dict[str, Any]) -> str:
    """F1 en validación cruzada anidada y diferencia pareada contra el híbrido."""
    h = resultados["hibrido_exclusiones"]
    lineas = [
        "| Segunda etapa (VC anidada, train) | P | R | F1 [IC 95 %] | Δ F1 contra el híbrido "
        "[IC 95 %] | Recuperadas top-5 | FP nuevos / quitados |",
        "|---|---|---|---|---|---|---|",
        f"| híbrido (primer candidato) | {h['precision']:.3f} | {h['recall']:.3f} | "
        f"{h['f1']:.3f} [{h['ic95']['f1'][0]:.3f}; {h['ic95']['f1'][1]:.3f}] | — | — | — |",
    ]
    for nombre, v in resultados["variantes"].items():
        m, d, r = v["anidada"], v["contra_hibrido"]["f1"], v["recuperadas"]
        lineas.append(
            f"| {nombre.replace('_', ' ')} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} [{m['ic95']['f1'][0]:.3f}; {m['ic95']['f1'][1]:.3f}] | "
            f"{d['diferencia']:+.3f} [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}] | "
            f"{r['recuperadas']['top5']}/{r['perdidas_base']['top5']} | "
            f"{r['falsos_positivos_nuevos']} / {r['falsos_positivos_quitados']} |"
        )
    return "\n".join(lineas)
