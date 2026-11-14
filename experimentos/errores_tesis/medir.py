"""Mide el efecto de corregir los errores heredados de la tesis (issue #24).

Corre `analizar()` con el modelo real sobre notas sintéticas anotadas a mano (las de
`experimentos/siglas/notas.jsonl` y las de los casos de error en
`experimentos/errores_tesis/notas.jsonl`) y compara contra lo esperado:

- recall: entidades esperadas que aparecen en la salida (coincidencia sin distinguir
  mayúsculas, por contención en cualquier sentido);
- estado correcto: de las encontradas, cuántas tienen el estado anotado;
- registrado correcto: si la nota trae `registrados`, cuántas entidades con `registrado`
  anotado salen con el valor esperado;
- índice correcto: si la nota trae `indice`, cuántas notas dan exactamente ese índice;
- entidades de salida e índice de Charlson total.

Uso (desde la raíz del repo que se quiere medir):

    uv run python experimentos/errores_tesis/medir.py <notas.jsonl> <salida.json>

Para medir "antes", se corre el mismo script con `uv run --project <worktree>` sobre un
worktree del commit anterior a las correcciones.
"""

import json
import sys
import time
from pathlib import Path

from ner_clinic import analizar


def _coincide(esperado: str, obtenido: str) -> bool:
    e, o = esperado.lower(), obtenido.lower()
    return e in o or o in e


def medir(ruta_notas: Path) -> dict:
    notas = [json.loads(linea) for linea in ruta_notas.read_text().splitlines() if linea]
    analizar("Calentamiento: paciente con HTA.")  # carga el modelo fuera de la medición

    esperadas = encontradas = estado_ok = salida_total = indice_total = 0
    reg_total = reg_ok = ind_total = ind_ok = 0
    perdidas: list[str] = []
    estados_mal: list[str] = []
    otros_mal: list[str] = []
    inicio = time.perf_counter()
    for nota in notas:
        resultado = analizar(nota["texto"], diagnosticos_registrados=nota.get("registrados"))
        if "indice" in nota:
            ind_total += 1
            if resultado.indice_charlson == nota["indice"]:
                ind_ok += 1
            else:
                otros_mal.append(
                    f"{nota['id']}: índice esperado {nota['indice']}, "
                    f"salió {resultado.indice_charlson}"
                )
        salida_total += len(resultado.entidades)
        indice_total += resultado.indice_charlson
        for esperada in nota["entidades"]:
            esperadas += 1
            match = next(
                (e for e in resultado.entidades if _coincide(esperada["texto"], e.texto)), None
            )
            if match is None:
                perdidas.append(f"{nota['id']}: {esperada['texto']}")
                continue
            encontradas += 1
            if "registrado" in esperada:
                reg_total += 1
                if match.registrado == esperada["registrado"]:
                    reg_ok += 1
                else:
                    otros_mal.append(
                        f"{nota['id']}: {esperada['texto']} registrado esperado "
                        f"{esperada['registrado']}, salió {match.registrado}"
                    )
            if match.estado.value == esperada["estado"]:
                estado_ok += 1
            else:
                estados_mal.append(
                    f"{nota['id']}: {esperada['texto']} "
                    f"(esperado {esperada['estado']}, salió {match.estado.value})"
                )
    return {
        "notas": len(notas),
        "entidades_esperadas": esperadas,
        "encontradas": encontradas,
        "recall": round(encontradas / esperadas, 4),
        "estado_correcto": estado_ok,
        "precision_estado": round(estado_ok / encontradas, 4) if encontradas else 0.0,
        "registrado_correcto": f"{reg_ok}/{reg_total}",
        "indice_correcto": f"{ind_ok}/{ind_total}",
        "entidades_salida": salida_total,
        "indice_charlson_total": indice_total,
        "segundos": round(time.perf_counter() - inicio, 2),
        "perdidas": perdidas,
        "estados_mal": estados_mal,
        "otros_mal": otros_mal,
    }


if __name__ == "__main__":
    ruta_notas, ruta_salida = Path(sys.argv[1]), Path(sys.argv[2])
    resultado = medir(ruta_notas)
    ruta_salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in resultado.items() if not isinstance(v, list)}))
