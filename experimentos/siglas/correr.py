"""Experimento de siglas (issue #8, FR3b del PRD): ¿vale la pena la lista blanca?

Corre el modelo real sobre las notas sintéticas de `notas.jsonl` (una sola carga del
modelo y una sola inferencia por nota) y compara tres variantes del filtro de longitud de
`extraer_candidatos`:

- `tesis`: descarta las entidades de 4 caracteres o menos, sin excepciones.
- `lista_blanca`: igual, salvo las siglas de `SIGLAS_CLINICAS` (comportamiento actual).
- `ninguno`: sin filtro de longitud.

Para cada variante reporta las siglas esperadas que sobreviven, las que el modelo no
detecta nunca y las entidades cortas (4 caracteres o menos) que entran sin ser una
entidad esperada (ruido). Imprime las tablas en Markdown y guarda el detalle en
`resultados.json`.

Uso: `uv run python experimentos/siglas/correr.py`

Todas las notas son sintéticas, escritas para este experimento.
"""

import json
import platform
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import transformers

from ner_clinic.extraccion import (
    FILTROS_LONGITUD,
    LONGITUD_MAXIMA_DESCARTE,
    NOMBRE_MODELO,
    PALABRAS_CORTAS_CLINICAS,
    SIGLAS_CLINICAS,
    ExtractorHF,
    extraer_candidatos,
)
from ner_clinic.limpieza import limpiar
from ner_clinic.segmentacion import segmentar

CARPETA = Path(__file__).parent
NOTAS = CARPETA / "notas.jsonl"
RESULTADOS = CARPETA / "resultados.json"


class ExtractorConCache:
    """Envuelve un extractor y memoriza la salida por fragmento.

    Así el modelo corre una sola vez por fragmento aunque se evalúen tres variantes, y se
    guardan las entidades crudas (antes de fusión y filtros) para saber qué detectó el
    modelo.
    """

    def __init__(self, extractor):
        self.extractor = extractor
        self.cache: dict[str, list[dict]] = {}
        self.segundos_inferencia = 0.0

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        nuevos = [f for f in dict.fromkeys(fragmentos) if f not in self.cache]
        if nuevos:
            inicio = time.perf_counter()
            for fragmento, entidades in zip(nuevos, self.extractor.extraer(nuevos), strict=True):
                self.cache[fragmento] = entidades
            self.segundos_inferencia += time.perf_counter() - inicio
        return [[dict(e) for e in self.cache[f]] for f in fragmentos]


def cargar_notas() -> list[dict]:
    with NOTAS.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def ocurrencias(texto: str, entidad: str) -> list[tuple[int, int]]:
    """Rangos de `entidad` en `texto` como palabra completa (sin distinguir mayúsculas)."""
    patron = re.compile(r"(?<!\w)" + re.escape(entidad) + r"(?!\w)", re.IGNORECASE)
    return [m.span() for m in patron.finditer(texto)]


def se_solapan(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def rangos_crudos(texto: str, extractor: ExtractorConCache) -> list[tuple[int, int, str]]:
    """Entidades crudas del modelo (sin fusión ni filtros) con posiciones de la nota."""
    rangos = []
    for fragmento in segmentar(texto):
        for e in extractor.cache.get(fragmento.texto, []):
            inicio = fragmento.desplazamiento + e["start"]
            fin = fragmento.desplazamiento + e["end"]
            rangos.append((inicio, fin, texto[inicio:fin]))
    return rangos


def evaluar() -> dict:
    notas = cargar_notas()
    extractor = ExtractorConCache(ExtractorHF())

    textos = [limpiar(n["texto"]).texto for n in notas]
    inicio_total = time.perf_counter()
    # Carga del modelo aparte, para que no cuente como tiempo de inferencia.
    extractor.extractor.extraer(["Paciente con HTA."])
    segundos_carga = time.perf_counter() - inicio_total

    salidas = {v: [] for v in FILTROS_LONGITUD}
    for texto in textos:
        for variante in FILTROS_LONGITUD:
            candidatos, _ = extraer_candidatos(texto, extractor, filtro_longitud=variante)
            salidas[variante].append(candidatos)
    segundos_total = time.perf_counter() - inicio_total

    # --- Siglas esperadas: detección cruda del modelo y supervivencia por variante -------
    siglas_ocurrencias = Counter()
    siglas_modelo = Counter()
    siglas_por_variante = {v: Counter() for v in FILTROS_LONGITUD}
    nombres_total = 0
    nombres_por_variante = Counter()
    ruido = {v: defaultdict(list) for v in FILTROS_LONGITUD}
    no_esperadas: dict[str, list[str]] = {v: [] for v in FILTROS_LONGITUD}
    perdidas: dict[str, list[str]] = {v: [] for v in FILTROS_LONGITUD}
    crudas_por_sigla: dict[str, list[str]] = defaultdict(list)

    for i, (nota, texto) in enumerate(zip(notas, textos, strict=True)):
        crudas = rangos_crudos(texto, extractor)
        esperados: list[tuple[int, int]] = []
        for entidad in nota["entidades"]:
            rangos = ocurrencias(texto, entidad["texto"])
            esperados.extend(rangos)
            if entidad["sigla"]:
                sigla = entidad["texto"].upper()
                siglas_ocurrencias[sigla] += 1
                tocadas = [c for c in crudas if any(se_solapan(c[:2], r) for r in rangos)]
                if tocadas:
                    siglas_modelo[sigla] += 1
                    crudas_por_sigla[sigla].extend(c[2] for c in tocadas)
            else:
                nombres_total += 1
            for variante in FILTROS_LONGITUD:
                encontrada = any(
                    se_solapan((m.inicio, m.fin), r)
                    for c in salidas[variante][i]
                    for m in c.menciones
                    for r in rangos
                )
                if not encontrada:
                    perdidas[variante].append(f"{nota['id']}: {entidad['texto']}")
                elif entidad["sigla"]:
                    siglas_por_variante[variante][entidad["texto"].upper()] += 1
                else:
                    nombres_por_variante[variante] += 1

        for variante in FILTROS_LONGITUD:
            for candidato in salidas[variante][i]:
                esperada = any(
                    se_solapan((m.inicio, m.fin), r) for m in candidato.menciones for r in esperados
                )
                if esperada:
                    continue
                no_esperadas[variante].append(f"{nota['id']}: {candidato.texto}")
                if len(" ".join(candidato.texto.split())) <= LONGITUD_MAXIMA_DESCARTE:
                    ruido[variante][candidato.texto].append(nota["id"])

    total_siglas = sum(siglas_ocurrencias.values())
    return {
        "modelo": NOMBRE_MODELO,
        "entorno": {
            "procesador": platform.processor() or platform.machine(),
            "cpu": _nombre_cpu(),
            "hilos_torch": torch.get_num_threads(),
            "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "tiempos_segundos": {
            "carga_modelo": round(segundos_carga, 2),
            "inferencia": round(extractor.segundos_inferencia, 2),
            "total": round(segundos_total, 2),
        },
        "notas": len(notas),
        "siglas": {
            "ocurrencias": dict(sorted(siglas_ocurrencias.items())),
            "detectadas_por_modelo": dict(sorted(siglas_modelo.items())),
            "texto_crudo_del_modelo": {k: sorted(set(v)) for k, v in crudas_por_sigla.items()},
            "nunca_detectadas": sorted(s for s in siglas_ocurrencias if siglas_modelo[s] == 0),
            "en_lista_blanca": sorted(SIGLAS_CLINICAS | PALABRAS_CORTAS_CLINICAS),
        },
        "variantes": {
            v: {
                "siglas_conservadas": sum(siglas_por_variante[v].values()),
                "siglas_esperadas": total_siglas,
                "siglas_conservadas_por_sigla": dict(sorted(siglas_por_variante[v].items())),
                "nombres_conservados": nombres_por_variante[v],
                "nombres_esperados": nombres_total,
                "candidatos": sum(len(c) for c in salidas[v]),
                "candidatos_no_esperados": no_esperadas[v],
                "esperadas_perdidas": perdidas[v],
                "ruido_corto": {k: sorted(set(ids)) for k, ids in sorted(ruido[v].items())},
            }
            for v in FILTROS_LONGITUD
        },
    }


def _nombre_cpu() -> str:
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def imprimir(r: dict) -> None:
    e, t = r["entorno"], r["tiempos_segundos"]
    print(f"Modelo: {r['modelo']} | {r['notas']} notas sintéticas")
    print(f"CPU: {e['cpu']} ({e['hilos_torch']} hilos torch, {e['dispositivo']})")
    print(f"torch {e['torch']}, transformers {e['transformers']}, Python {e['python']}")
    print(
        f"Tiempo: carga {t['carga_modelo']} s, inferencia {t['inferencia']} s, "
        f"total {t['total']} s\n"
    )

    print(
        "| Variante | Siglas conservadas | Nombres completos | Candidatos | No esperados "
        "| Ruido corto (≤4) |"
    )
    print("|---|---|---|---|---|---|")
    for v, d in r["variantes"].items():
        print(
            f"| {v} | {d['siglas_conservadas']}/{d['siglas_esperadas']} "
            f"| {d['nombres_conservados']}/{d['nombres_esperados']} | {d['candidatos']} "
            f"| {len(d['candidatos_no_esperados'])} | {len(d['ruido_corto'])} |"
        )

    s = r["siglas"]
    print("\n| Sigla | Ocurrencias | Modelo la detecta | En lista blanca | Texto crudo |")
    print("|---|---|---|---|---|")
    for sigla, n in s["ocurrencias"].items():
        crudo = ", ".join(f"`{x}`" for x in s["texto_crudo_del_modelo"].get(sigla, []))
        blanca = "sí" if sigla in s["en_lista_blanca"] else "no"
        print(
            f"| {sigla} | {n} | {s['detectadas_por_modelo'].get(sigla, 0)} | {blanca} | {crudo} |"
        )

    print(f"\nSiglas que el modelo no detecta nunca: {', '.join(s['nunca_detectadas']) or '—'}")
    for v, d in r["variantes"].items():
        ejemplos = ", ".join(f"`{k}` ({', '.join(ids)})" for k, ids in d["ruido_corto"].items())
        print(f"Ruido corto en {v}: {ejemplos or '—'}")
    for v, d in r["variantes"].items():
        print(f"\nNo esperados en {v}: {'; '.join(d['candidatos_no_esperados']) or '—'}")
        print(f"Esperadas perdidas en {v}: {'; '.join(d['esperadas_perdidas']) or '—'}")


def main() -> None:
    resultados = evaluar()
    RESULTADOS.write_text(json.dumps(resultados, ensure_ascii=False, indent=2) + "\n")
    imprimir(resultados)
    print(f"\nDetalle en {RESULTADOS}")


if __name__ == "__main__":
    main()
