"""Cuenta la cobertura del set sintético reservado (`notas.jsonl`, issue #44).

Uso, desde la raíz del repo:

    uv run python evaluacion/sintetico/cobertura.py

Imprime la tabla de cobertura en Markdown. El README la incluye tal cual y
`tests/test_set_sintetico.py` comprueba que coincide, así que los conteos no se editan a
mano. Solo lee las anotaciones: no corre el pipeline ni el modelo.
"""

import json
import re
from collections import Counter
from pathlib import Path

RUTA_NOTAS = Path(__file__).with_name("notas.jsonl")

ESTADOS = ("confirmada", "negada", "familiar")

CATEGORIAS = (
    "infarto_miocardio",
    "insuficiencia_cardiaca",
    "vascular_periferica",
    "cerebrovascular",
    "demencia",
    "pulmonar_cronica",
    "reumatologica",
    "ulcera_peptica",
    "hepatica_leve",
    "diabetes",
    "diabetes_complicada",
    "hemiplejia",
    "renal_cronica",
    "tumor_maligno",
    "hepatica_severa",
    "tumor_metastasico",
    "vih_sida",
)

JERARQUIAS = {
    "diabetes": ("diabetes", "diabetes_complicada"),
    "hepática": ("hepatica_leve", "hepatica_severa"),
    "tumor": ("tumor_maligno", "tumor_metastasico"),
}

# Una sigla es una entidad escrita toda en mayúsculas, sin espacios (p. ej. HTA, DM2, EPOC).
PATRON_SIGLA = re.compile(r"[A-ZÁÉÍÓÚÑ]{2,}[0-9]?")

FORMAS_NEGACION = {
    "niega": r"\bniega\b",
    "sin": r"\bsin\b",
    "no presenta": r"\bno presenta\b",
    "se descarta": r"\bse descarta\b",
    "negativo para": r"\bnegativo para\b",
    "ni / o (enumeración)": r"\b(ni|o)\b",
}

PARENTESCOS = {
    "madre": r"\bmadre\b",
    "padre": r"\bpadre\b",
    "hermano/a": r"\bherman[oa]s?\b",
    "abuelo/a": r"\babuel[oa]s?\b",
    "tío/a": r"\bt[ií][oa]s?\b",
    "hijo/a": r"\bhij[oa]s?\b",
    "primo/a": r"\bprim[oa]s?\b",
    "nieto/a": r"\bniet[oa]s?\b",
}

MAX_UNA_LINEA = 120
MIN_LARGA = 1500


def cargar(ruta: Path = RUTA_NOTAS) -> list[dict]:
    """Lee el JSONL, una nota por línea."""
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def es_sigla(texto: str) -> bool:
    return PATRON_SIGLA.fullmatch(texto) is not None


def contexto_previo(nota: str, entidad: str) -> str:
    """Texto de la misma oración (o del mismo rótulo) que va antes de la entidad.

    Busca la primera aparición como palabra completa, para que "AR" no coincida dentro de
    "paciente".
    """
    m = re.search(rf"(?<!\w){re.escape(entidad)}(?!\w)", nota)
    pos = m.start() if m else nota.find(entidad)
    bajo = nota.lower()
    inicio = max(bajo.rfind(".", 0, pos), bajo.rfind("\n", 0, pos), bajo.rfind(":", 0, pos))
    return bajo[inicio + 1 : pos]


def _ultimo(patrones: dict[str, str], fragmento: str) -> str | None:
    """La forma cuyo patrón aparece más cerca del final del fragmento."""
    mejor, pos_mejor = None, -1
    for nombre, patron in patrones.items():
        for m in re.finditer(patron, fragmento):
            if m.start() > pos_mejor:
                mejor, pos_mejor = nombre, m.start()
    return mejor


def contar(notas: list[dict]) -> dict:
    """Conteos de cobertura del set."""
    entidades = [(n, e) for n in notas for e in n["entidades"]]
    estados = Counter(e["estado"] for _, e in entidades)
    confirmadas = Counter(
        e["categoria"] for _, e in entidades if e["estado"] == "confirmada" and "categoria" in e
    )
    jerarquias = Counter()
    for n in notas:
        cats = {
            e["categoria"]
            for e in n["entidades"]
            if e["estado"] == "confirmada" and "categoria" in e
        }
        for nombre, (leve, severa) in JERARQUIAS.items():
            if leve in cats and severa in cats:
                jerarquias[nombre] += 1
    negaciones = Counter()
    parentescos = Counter()
    for n, e in entidades:
        previo = contexto_previo(n["texto"], e["texto"])
        if e["estado"] == "negada":
            negaciones[_ultimo(FORMAS_NEGACION, previo) or "otra"] += 1
        elif e["estado"] == "familiar":
            parentescos[_ultimo(PARENTESCOS, previo) or "otro"] += 1
    return {
        "notas": len(notas),
        "entidades": len(entidades),
        "estados": {s: estados[s] for s in ESTADOS},
        "categorias": {c: confirmadas[c] for c in CATEGORIAS},
        "jerarquias": {j: jerarquias[j] for j in JERARQUIAS},
        "siglas": sum(es_sigla(e["texto"]) for _, e in entidades),
        "notas_con_registrados": sum("registrados" in n for n in notas),
        "entidades_con_registrado": sum("registrado" in e for _, e in entidades),
        "una_linea": sum(
            "\n" not in n["texto"] and len(n["texto"]) <= MAX_UNA_LINEA for n in notas
        ),
        "largas": sum(len(n["texto"]) > MIN_LARGA for n in notas),
        "negaciones": dict(negaciones.most_common()),
        "parentescos": dict(parentescos.most_common()),
    }


def tabla(c: dict) -> str:
    """Tabla de cobertura en Markdown."""
    filas = [
        "| Qué | Cantidad |",
        "|---|---|",
        f"| Notas | {c['notas']} |",
        f"| Entidades anotadas | {c['entidades']} |",
    ]
    filas += [f"| Entidades en estado `{s}` | {n} |" for s, n in c["estados"].items()]
    filas += [
        f"| Entidades que son siglas | {c['siglas']} |",
        f"| Notas con diagnósticos registrados | {c['notas_con_registrados']} |",
        f"| Entidades con `registrado` esperado | {c['entidades_con_registrado']} |",
        f"| Notas de una línea (≤ {MAX_UNA_LINEA} caracteres) | {c['una_linea']} |",
        f"| Notas de más de {MIN_LARGA} caracteres | {c['largas']} |",
    ]
    filas += [f"| Notas con jerarquía leve/severa: {j} | {n} |" for j, n in c["jerarquias"].items()]
    filas += ["", "| Categoría de Charlson | Entidades confirmadas |", "|---|---|"]
    filas += [f"| `{cat}` | {n} |" for cat, n in c["categorias"].items()]
    filas += ["", "| Forma de negación (la más cercana) | Entidades negadas |", "|---|---|"]
    filas += [f"| {f} | {n} |" for f, n in c["negaciones"].items()]
    filas += ["", "| Parentesco (el más cercano) | Entidades familiares |", "|---|---|"]
    filas += [f"| {p} | {n} |" for p, n in c["parentescos"].items()]
    return "\n".join(filas)


if __name__ == "__main__":
    print(tabla(contar(cargar())))
