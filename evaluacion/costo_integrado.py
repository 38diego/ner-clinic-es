"""Costo de la API con el normalizador integrado, en Docker con 2 vCPU y 4 GiB (#167).

#154 midió el costo por la ruta de la evaluación. Aquí se mide lo que correría en
producción: la API (`ner_clinic_api`) por HTTP, con `latencia_docker.medir` (arranque en
frío, las 250 notas de prueba de DisTEMIST, notas de 2.000 caracteres, una de 5.000 y
memoria pico).

Imágenes, construidas desde el árbol limpio:

- **`ner-clinic-es:base`**: la etapa `base` del `Dockerfile` (API y NER).
- **`ner-clinic-es-sapbert`**: la anterior más SapBERT, su tokenizador y el índice final
  (`artefactos.RUTA_INDICE`) en `RUTA_INDICE_EN_IMAGEN`. El clasificador ya va en el
  paquete.

Corridas, en este orden:

1. `reglas`: `ner-clinic-es:base` con las reglas.
2. `normalizador`: `ner-clinic-es-sapbert` con `NER_NORMALIZADOR=sapbert`.
3. `control`: `ner-clinic-es-sapbert` con las reglas. Separa lo que cuesta usar el
   normalizador de lo que cambia por la imagen o por el orden de las corridas.

    uv run python -m evaluacion.costo_integrado

El JSON solo trae métricas: ningún texto de las notas.
"""

import argparse
import tempfile
from pathlib import Path
from typing import Any

RUTA_RESULTADOS = Path("evaluacion/resultados/costo_integrado.json")
IMAGEN_BASE = "ner-clinic-es:base"
IMAGEN_SAPBERT = "ner-clinic-es-sapbert"
RUTA_INDICE_EN_IMAGEN = "/opt/normalizacion/indice_final.pt"
CORRIDAS = (
    ("reglas", IMAGEN_BASE, {"NER_NORMALIZADOR": "reglas"}),
    ("normalizador", IMAGEN_SAPBERT, {"NER_NORMALIZADOR": "sapbert"}),
    ("control", IMAGEN_SAPBERT, {"NER_NORMALIZADOR": "reglas"}),
)


def dockerfile_sapbert(base: str = IMAGEN_BASE) -> str:
    """La imagen de la API más SapBERT (con su tokenizador) y el índice final."""
    from ner_clinic.normalizacion.indice import SAPBERT

    tokenizador, revision = SAPBERT.tokenizador or (SAPBERT.nombre, SAPBERT.revision)
    return f"""FROM {base}
USER root
RUN HF_HUB_OFFLINE=0 python -c "\\
from huggingface_hub import snapshot_download; \\
snapshot_download('{SAPBERT.nombre}', revision='{SAPBERT.revision}', \\
allow_patterns=['config.json', 'model.safetensors']); \\
snapshot_download('{tokenizador}', revision='{revision}', \\
allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json', \\
'sentencepiece.bpe.model'])"
COPY indice_final.pt {RUTA_INDICE_EN_IMAGEN}
ENV NER_INDICE_NORMALIZADOR={RUTA_INDICE_EN_IMAGEN}
USER ner
"""


def resumen_comparado(corridas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Lo que suma el normalizador sobre las reglas (y sobre el control, en la misma
    imagen): arranque, p95 de las notas de 2.000 caracteres y memoria pico."""

    def valores(m: dict[str, Any]) -> dict[str, float]:
        return {
            "arranque_s": m["arranque_en_frio_s"],
            "p95_2000_s": m["notas_2000_caracteres"]["servidor"]["p95_s"],
            "mediana_2000_s": m["notas_2000_caracteres"]["servidor"]["mediana_s"],
            "p95_distemist_s": m["distemist_test"]["servidor"]["p95_s"],
            "memoria_pico_mb": m["memoria_pico_mb"]["cgroup_memory_peak"],
            "imagen_mb": m["imagen"]["tamano_mb"],
        }

    por_corrida = {nombre: valores(m) for nombre, m in corridas.items()}
    norm = por_corrida["normalizador"]
    return {
        "por_corrida": por_corrida,
        "normalizador_menos_reglas": {
            k: round(norm[k] - por_corrida["reglas"][k], 3) for k in norm
        },
        "normalizador_menos_control": {
            k: round(norm[k] - por_corrida["control"][k], 3) for k in norm
        },
    }


def _docker(*argumentos: str) -> str:  # pragma: no cover - usa Docker
    from evaluacion.latencia_docker import _docker as docker

    return docker(*argumentos)


def construir(repo: Path, indice: Path) -> None:  # pragma: no cover - usa Docker
    _docker("build", "--target", "base", "-t", IMAGEN_BASE, str(repo))
    with tempfile.TemporaryDirectory(dir=indice.parent) as contexto:
        (Path(contexto) / "Dockerfile").write_text(dockerfile_sapbert(), encoding="utf-8")
        (Path(contexto) / "indice_final.pt").hardlink_to(indice)
        _docker("build", "-t", IMAGEN_SAPBERT, contexto)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - usa Docker
    from datetime import UTC, datetime

    from evaluacion import latencia_docker
    from evaluacion.__main__ import _cambios_sin_commitear
    from evaluacion.corpus import RAIZ_DATOS
    from evaluacion.distemist import entorno, guardar_json
    from ner_clinic.normalizacion import artefactos

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raiz", type=Path, default=RAIZ_DATOS)
    parser.add_argument("--salida", type=Path, default=RUTA_RESULTADOS)
    parser.add_argument("--puerto", type=int, default=latencia_docker.PUERTO_HOST)
    args = parser.parse_args(argv)
    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe y no se sobrescribe.")
    cambios = _cambios_sin_commitear()
    if cambios:
        raise SystemExit("El costo se mide con el árbol limpio:\n" + "\n".join(cambios))

    construir(Path.cwd(), artefactos.RUTA_INDICE.resolve())
    corridas = {}
    for nombre, imagen, variables in CORRIDAS:
        print(f"--- {nombre} ({imagen})")
        corridas[nombre] = latencia_docker.medir(imagen, args.raiz, args.puerto, variables)
    resultados = {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": entorno(),
        "orden": [n for n, _, _ in CORRIDAS],
        "comparacion": resumen_comparado(corridas),
        "corridas": corridas,
    }
    guardar_json(resultados, args.salida)
    print(resultados["comparacion"])


if __name__ == "__main__":  # pragma: no cover
    main()
