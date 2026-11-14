"""Muestra reproducible de errores del pipeline para el análisis a mano (FR7, #46).

Toma el detalle de TP/FP/FN que deja el runner (`data/evaluacion/distemist_test_errores.json`,
fuera del repo porque lleva texto del corpus) y saca, con semilla fija, 50 falsos
positivos y 50 falsos negativos estrictos de una variante (por defecto `pipeline`).

Para cada error guarda el documento, el span, un fragmento de contexto de máximo 120
caracteres y lo que sirve para clasificarlo a mano:

- en un falso positivo, las menciones gold que se solapan con él (si hay alguna, el error
  estricto es un acierto laxo: un problema de límite del span);
- en un falso negativo, las predicciones que se solapan con él.

La clasificación (`categoria` y `nota`) la escribe una persona en el JSON de salida;
`conteo` y `tabla_categorias` resumen el archivo ya clasificado y `validar` revisa que las
categorías existan y que los fragmentos no pasen del límite.

Uso:

    uv run python -m evaluacion.errores muestra \
        --detalle data/evaluacion/distemist_test_errores.json \
        --raiz data/distemist --salida ARCHIVO.json
    uv run python -m evaluacion.errores tabla evaluacion/resultados/errores.json

Los fragmentos vienen de DisTEMIST (Miranda-Escalada et al., 2022), CC BY 4.0.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from evaluacion.corpus import RAIZ_DATOS, leer_particion

SEMILLA = 46
TAMANO_MUESTRA = 50
REVISIONES = 10
MAX_CONTEXTO = 120

CATEGORIAS = (
    "limite_span",
    "sintoma_no_anotado",
    "sigla",
    "error_gold",
    "ruido",
    "no_detectada",
    "fusion_indebida",
    "perdida_deduplicacion",
    "otra",
)

DESCRIPCION_CATEGORIAS = {
    "limite_span": "Límite del span: sobra o falta texto",
    "sintoma_no_anotado": "Síntoma o hallazgo que el gold no anota",
    "sigla": "Sigla",
    "error_gold": "Error o inconsistencia del gold",
    "ruido": "Ruido real (no es enfermedad)",
    "no_detectada": "Enfermedad no detectada",
    "fusion_indebida": "Fusión indebida",
    "perdida_deduplicacion": "Mención perdida en la deduplicación",
    "otra": "Otra",
}

ATRIBUCION = (
    "Los fragmentos de contexto vienen del corpus DisTEMIST (Miranda-Escalada A, Gascó L, "
    "Lima-López S, et al. Overview of DisTEMIST at BioASQ: Automatic detection and "
    "normalization of diseases from clinical texts. CEUR Workshop Proceedings 3180, 2022; "
    "https://doi.org/10.5281/zenodo.7614764), con licencia CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). Se citan sin cambios, recortados."
)


def _solapa(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["inicio"] < b["fin"] and b["inicio"] < a["fin"]


def contexto(texto: str, inicio: int, fin: int, maximo: int = MAX_CONTEXTO) -> str:
    """Fragmento de `texto` alrededor de `[inicio, fin)` con `maximo` caracteres o menos.

    Centra el span y reparte el resto a los dos lados. Si el span solo ya pasa del
    máximo, lo recorta. Los saltos de línea se cambian por espacios.
    """
    largo = fin - inicio
    if largo >= maximo:
        desde, hasta = inicio, inicio + maximo
    else:
        sobra = maximo - largo
        desde = max(0, inicio - sobra // 2)
        hasta = min(len(texto), desde + maximo)
        desde = max(0, hasta - maximo)
    return " ".join(texto[desde:hasta].split())


def _errores_de(detalle: dict, variante: str, modo: str) -> tuple[list[dict], list[dict]]:
    """FP y FN de `variante` en `modo`, cada uno con su documento y lo que se le solapa."""
    falsos_positivos, falsos_negativos = [], []
    for doc in detalle["variantes"][variante][modo]:
        gold = [v["gold"] for v in doc["verdaderos"]] + doc["falsos_negativos"]
        predichas = [v["prediccion"] for v in doc["verdaderos"]] + doc["falsos_positivos"]
        for fp in doc["falsos_positivos"]:
            solapadas = [g for g in gold if _solapa(fp, g)]
            falsos_positivos.append(
                {"documento": doc["documento"], **fp, "gold_solapado": solapadas}
            )
        for fn in doc["falsos_negativos"]:
            solapadas = [p for p in predichas if _solapa(fn, p)]
            falsos_negativos.append(
                {"documento": doc["documento"], **fn, "prediccion_solapada": solapadas}
            )
    return falsos_positivos, falsos_negativos


def muestrear(
    detalle: dict,
    textos: dict[str, str],
    variante: str = "pipeline",
    modo: str = "estricto",
    n: int = TAMANO_MUESTRA,
    semilla: int = SEMILLA,
    revisiones: int = REVISIONES,
) -> dict[str, Any]:
    """Muestra de `n` FP y `n` FN de `variante`, con contexto y `revisiones` ids a revisar.

    Usa un solo `random.Random(semilla)`: primero los FP, luego los FN y al final los ids
    para revisión, así el resultado depende solo del detalle y de la semilla.
    """
    rng = random.Random(semilla)
    falsos_positivos, falsos_negativos = _errores_de(detalle, variante, modo)
    errores = []
    for tipo, poblacion in (("FP", falsos_positivos), ("FN", falsos_negativos)):
        elegidos = rng.sample(poblacion, min(n, len(poblacion)))
        for i, error in enumerate(elegidos, start=1):
            texto = textos[error["documento"]]
            item = {
                "id": f"{tipo}-{i:02d}",
                "tipo": tipo,
                **error,
                "contexto": contexto(texto, error["inicio"], error["fin"]),
                "categoria": None,
                "nota": "",
            }
            if tipo == "FP":
                item["acierto_laxo"] = bool(error["gold_solapado"])
            errores.append(item)
    ids = [e["id"] for e in errores]
    return {
        "atribucion": ATRIBUCION,
        "particion": detalle["particion"],
        "variante": variante,
        "modo": modo,
        "semilla": semilla,
        "poblacion": {"FP": len(falsos_positivos), "FN": len(falsos_negativos)},
        "categorias": DESCRIPCION_CATEGORIAS,
        "revisar": sorted(rng.sample(ids, min(revisiones, len(ids)))),
        "errores": errores,
    }


def validar(muestra: dict[str, Any]) -> list[str]:
    """Problemas del archivo clasificado: categorías vacías o desconocidas, contexto largo."""
    problemas = []
    for error in muestra["errores"]:
        if error["categoria"] not in CATEGORIAS:
            problemas.append(f"{error['id']}: categoría {error['categoria']!r} no válida")
        if len(error["contexto"]) > MAX_CONTEXTO:
            problemas.append(f"{error['id']}: contexto de {len(error['contexto'])} caracteres")
    return problemas


def conteo(muestra: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Errores por categoría y tipo (FP, FN), en el orden de `CATEGORIAS`."""
    por_tipo = Counter((e["categoria"], e["tipo"]) for e in muestra["errores"])
    return {
        categoria: {"FP": por_tipo[(categoria, "FP")], "FN": por_tipo[(categoria, "FN")]}
        for categoria in CATEGORIAS
        if por_tipo[(categoria, "FP")] or por_tipo[(categoria, "FN")]
    }


def tabla_categorias(muestra: dict[str, Any]) -> str:
    """Tabla Markdown del conteo por categoría, con totales."""
    lineas = ["| Categoría | FP | FN | Total |", "|---|---|---|---|"]
    total_fp = total_fn = 0
    for categoria, n in conteo(muestra).items():
        total_fp += n["FP"]
        total_fn += n["FN"]
        nombre = DESCRIPCION_CATEGORIAS[categoria]
        lineas.append(f"| {nombre} | {n['FP']} | {n['FN']} | {n['FP'] + n['FN']} |")
    lineas.append(f"| **Total** | {total_fp} | {total_fn} | {total_fp + total_fn} |")
    laxos = sum(1 for e in muestra["errores"] if e.get("acierto_laxo"))
    fps = sum(1 for e in muestra["errores"] if e["tipo"] == "FP")
    lineas.append("")
    lineas.append(f"FP que son acierto laxo (se solapan con un gold): {laxos} de {fps}.")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluacion.errores", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    p_muestra = sub.add_parser("muestra", help="Saca la muestra de errores sin clasificar.")
    p_muestra.add_argument("--detalle", type=Path, required=True)
    p_muestra.add_argument("--raiz", type=Path, default=RAIZ_DATOS)
    p_muestra.add_argument("--variante", default="pipeline")
    p_muestra.add_argument("--salida", type=Path, required=True)

    p_tabla = sub.add_parser("tabla", help="Valida un archivo clasificado e imprime la tabla.")
    p_tabla.add_argument("archivo", type=Path)

    args = parser.parse_args(argv)
    if args.comando == "muestra":
        detalle = json.loads(args.detalle.read_text(encoding="utf-8"))
        textos = {d.id: d.texto for d in leer_particion(args.raiz, detalle["particion"])}
        muestra = muestrear(detalle, textos, variante=args.variante)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(
            json.dumps(muestra, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{len(muestra['errores'])} errores de {muestra['poblacion']} en {args.salida}; "
            f"revisar: {', '.join(muestra['revisar'])}"
        )
    else:
        muestra = json.loads(args.archivo.read_text(encoding="utf-8"))
        problemas = validar(muestra)
        if problemas:
            raise SystemExit("\n".join(problemas))
        print(tabla_categorias(muestra))


if __name__ == "__main__":
    main()
