"""Acuerdo entre los dos anotadores del set sintético reservado (issue #44).

`anotaciones/a.jsonl` es la anotación original (la de quien escribió las notas) y
`anotaciones/b.jsonl` la del segundo anotador, que anotó a ciegas: solo vio los textos y
los criterios del README. `notas.jsonl` es el gold final, después de adjudicar las
discrepancias con las reglas del README.

Uso, desde la raíz del repo:

    uv run python evaluacion/sintetico/acuerdo.py

Imprime la tabla de acuerdo en Markdown y la lista de discrepancias. El README incluye la
tabla tal cual y `tests/test_set_sintetico.py` comprueba que coincide.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CARPETA = Path(__file__).with_name("anotaciones")


def cargar(ruta: Path) -> dict[str, dict]:
    with ruta.open(encoding="utf-8") as f:
        return {(n := json.loads(linea))["id"]: n for linea in f if linea.strip()}


def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().split())


def coinciden(a: str, b: str) -> bool:
    """Dos entidades son la misma si una contiene a la otra, sin distinguir mayúsculas."""
    a, b = _normalizar(a), _normalizar(b)
    return a in b or b in a


@dataclass
class Acuerdo:
    pares: list[tuple[str, dict, dict]] = field(default_factory=list)
    solo_a: list[tuple[str, dict]] = field(default_factory=list)
    solo_b: list[tuple[str, dict]] = field(default_factory=list)
    indices: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def f1_entidades(self) -> float:
        n = len(self.pares)
        return 2 * n / (2 * n + len(self.solo_a) + len(self.solo_b))

    @property
    def acuerdo_estado(self) -> float:
        return sum(a["estado"] == b["estado"] for _, a, b in self.pares) / len(self.pares)

    @property
    def kappa_estado(self) -> float:
        """Kappa de Cohen del estado sobre las entidades que anotaron los dos."""
        n = len(self.pares)
        cuenta_a = Counter(a["estado"] for _, a, _ in self.pares)
        cuenta_b = Counter(b["estado"] for _, _, b in self.pares)
        esperado = sum(cuenta_a[k] * cuenta_b[k] for k in cuenta_a) / n / n
        return (self.acuerdo_estado - esperado) / (1 - esperado)

    @property
    def confirmadas(self) -> list[tuple[str, dict, dict]]:
        return [p for p in self.pares if p[1]["estado"] == p[2]["estado"] == "confirmada"]

    @property
    def acuerdo_categoria(self) -> float:
        pares = self.confirmadas
        return sum(a.get("categoria") == b.get("categoria") for _, a, b in pares) / len(pares)


def comparar(a: dict[str, dict], b: dict[str, dict]) -> Acuerdo:
    """Empareja las entidades de cada nota (prefiere el mismo estado) y compara índices."""
    acuerdo = Acuerdo()
    for id_, nota_a in a.items():
        nota_b = b[id_]
        usadas: set[int] = set()
        for entidad in nota_a["entidades"]:
            candidatas = [
                j
                for j, otra in enumerate(nota_b["entidades"])
                if j not in usadas and coinciden(entidad["texto"], otra["texto"])
            ]
            candidatas.sort(key=lambda j: nota_b["entidades"][j]["estado"] != entidad["estado"])
            if candidatas:
                usadas.add(candidatas[0])
                acuerdo.pares.append((id_, entidad, nota_b["entidades"][candidatas[0]]))
            else:
                acuerdo.solo_a.append((id_, entidad))
        acuerdo.solo_b += [(id_, e) for j, e in enumerate(nota_b["entidades"]) if j not in usadas]
        acuerdo.indices.append((id_, nota_a["indice"], nota_b["indice"]))
    return acuerdo


def tabla(acuerdo: Acuerdo) -> str:
    notas = len(acuerdo.indices)
    exactos = sum(x == y for _, x, y in acuerdo.indices)
    filas = [
        ("Entidades que anotaron los dos", str(len(acuerdo.pares))),
        ("Solo el anotador A / solo el B", f"{len(acuerdo.solo_a)} / {len(acuerdo.solo_b)}"),
        ("F1 de acuerdo en entidades", f"{acuerdo.f1_entidades:.3f}"),
        ("Acuerdo en el estado", f"{acuerdo.acuerdo_estado:.3f}"),
        ("Kappa de Cohen del estado", f"{acuerdo.kappa_estado:.3f}"),
        ("Acuerdo en la categoría (confirmadas)", f"{acuerdo.acuerdo_categoria:.3f}"),
        ("Índice de Charlson igual", f"{exactos} de {notas} notas"),
    ]
    lineas = ["| Medida | Valor |", "|---|---|"]
    lineas += [f"| {medida} | {valor} |" for medida, valor in filas]
    return "\n".join(lineas)


def discrepancias(acuerdo: Acuerdo) -> list[str]:
    lineas = [
        f"{i}: estado {a['texto']!r} {a['estado']} / {b['estado']}"
        for i, a, b in acuerdo.pares
        if a["estado"] != b["estado"]
    ]
    lineas += [
        f"{i}: categoría {a['texto']!r} {a.get('categoria')} / {b.get('categoria')}"
        for i, a, b in acuerdo.confirmadas
        if a.get("categoria") != b.get("categoria")
    ]
    lineas += [f"{i}: solo A {e['texto']!r} ({e['estado']})" for i, e in acuerdo.solo_a]
    lineas += [f"{i}: solo B {e['texto']!r} ({e['estado']})" for i, e in acuerdo.solo_b]
    lineas += [f"{i}: índice {x} / {y}" for i, x, y in acuerdo.indices if x != y]
    return lineas


if __name__ == "__main__":
    resultado = comparar(cargar(CARPETA / "a.jsonl"), cargar(CARPETA / "b.jsonl"))
    print(tabla(resultado))
    print()
    print("\n".join(discrepancias(resultado)))
