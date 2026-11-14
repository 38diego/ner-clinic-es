"""Costo de despliegue del normalizador por embeddings (PRD `normalizacion`, NFR, #134).

Compara dos imágenes con los recursos de Cloud Run (`--cpus=2 --memory=4g`):

- **Base**: la imagen de la API (`ner-clinic-es`), con el modelo NER.
- **Candidata**: la base más SapBERT, su tokenizador y el índice del catálogo con los
  sinónimos de CodiEsp entrenamiento (`ner-clinic-es-normalizacion`).

En cada contenedor se mide, sin pasar por HTTP:

1. **Carga**: segundos hasta tener los modelos (y el índice) listos y calentados.
2. **Notas de 2.000 caracteres**: las notas de prueba de DisTEMIST que pasan de 2.000
   caracteres, recortadas a 2.000 (como `latencia_docker`). Por nota, el pipeline
   (`analizar`) y, en la candidata, además la normalización de todas sus entidades en un
   lote. Mediana, p95 y máximo.
3. **Memoria pico** del contenedor (`memory.peak` del cgroup).

Más el tamaño de cada imagen (`docker image inspect`, como `latencia_docker`).

**Con `--final` (#154)** se mide además una tercera imagen (`ner-clinic-es-normalizacion-final`):
la candidata con el índice que suma los sinónimos de DisTEMIST y CANTEMIST (#153) y, por
nota, la segunda etapa: el top-20 de cada entidad, las reglas y el clasificador de #151
guardado en el JSON de la medición final (`medicion_final.RUTA_RESULTADOS`). Se reporta el
tiempo de la segunda etapa por nota y el total con ella y sin ella, en las mismas notas.

Uso (con `ner-clinic-es` ya construida desde un árbol limpio):

    uv run python -m evaluacion.costo_normalizacion [--salida ARCHIVO.json]
    uv run python -m evaluacion.costo_normalizacion --final   # costo_normalizacion_final.json

El JSON solo trae métricas: ningún texto de las notas.
"""

import argparse
import json
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion.latencia_docker import CARACTERES_MEDIA, CPUS, MEMORIA, recortar, resumen

SALIDA = Path("evaluacion/resultados/costo_normalizacion.json")
SALIDA_FINAL = Path("evaluacion/resultados/costo_normalizacion_final.json")
IMAGEN_BASE = "ner-clinic-es"
IMAGEN_CANDIDATA = "ner-clinic-es-normalizacion"
IMAGEN_FINAL = "ner-clinic-es-normalizacion-final"
RESULTADOS_EN_CONTENEDOR = "/medicion/evaluacion/resultados/normalizacion_final_desarrollo.json"
RUTA_INDICE_EN_IMAGEN = "/opt/normalizacion/indice.pt"
RAIZ_INDICE = Path("data/costo_normalizacion")
MODELO = "sapbert"


def notas_de_2000(textos: list[str]) -> list[str]:
    """Las notas de más de `CARACTERES_MEDIA` caracteres, recortadas a ese largo."""
    return [recortar(t, CARACTERES_MEDIA) for t in textos if len(t) > CARACTERES_MEDIA]


def dockerfile_candidata(base: str = IMAGEN_BASE) -> str:
    """La imagen candidata (o la final): la base más SapBERT, su tokenizador y el índice."""
    from evaluacion.normalizador import CANDIDATOS

    modelo = CANDIDATOS[MODELO]
    tokenizador, revision_tokenizador = modelo.tokenizador or (modelo.nombre, modelo.revision)
    return f"""FROM {base}
USER root
RUN HF_HUB_OFFLINE=0 python -c "\\
from huggingface_hub import snapshot_download; \\
snapshot_download('{modelo.nombre}', revision='{modelo.revision}', \\
allow_patterns=['config.json', 'model.safetensors']); \\
snapshot_download('{tokenizador}', revision='{revision_tokenizador}', \\
allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json', \\
'sentencepiece.bpe.model'])"
COPY indice.pt {RUTA_INDICE_EN_IMAGEN}
USER ner
"""


# --- Dentro del contenedor -----------------------------------------------------------


def decidir_entidades(normalizador, textos: list[str], decidir) -> list:
    """La segunda etapa sobre las entidades de una nota ya buscadas en `normalizador`: el
    `decidir` de #151 sobre el top-k de cada una y, después, las exclusiones."""
    from evaluacion.dos_etapas import propuestas
    from evaluacion.normalizador import con_exclusiones

    mapear = con_exclusiones(lambda texto: decidir(texto, propuestas(normalizador, texto)))
    return [mapear(texto) for texto in textos]


def medir_dentro(
    raiz: Path, con_normalizador: bool, segunda_etapa: Path | None = None, hibrido: bool = False
) -> dict[str, Any]:  # pragma: no cover
    """Corre en el contenedor: carga, latencia por nota y memoria pico.

    Con `segunda_etapa` (el JSON de la medición final), cada entidad pasa además por el
    clasificador de #151 sobre su top-20, con las reglas y las exclusiones. Con `hibrido`,
    por el mismo camino pero decidiendo con el primer candidato (el híbrido de #133), para
    comparar las dos imágenes con el mismo código.
    """
    import torch

    from evaluacion.corpus import leer_particion
    from evaluacion.normalizador import CANDIDATOS, CodificadorHF, Indice, Normalizador
    from ner_clinic.pipeline import analizar

    inicio = time.perf_counter()
    analizar("Paciente con HTA.")  # carga y calienta el NER
    codificador = indice = decidir = None
    if con_normalizador:
        codificador = CodificadorHF(CANDIDATOS[MODELO])
        guardado = torch.load(RUTA_INDICE_EN_IMAGEN, weights_only=True)
        indice = Indice(
            guardado["codigos"], guardado["codigos"], guardado["vectores"],
            guardado.get("origenes"),
        )  # fmt: skip
        indice.buscar(codificador(["hipertensión arterial"]))
    if segunda_etapa is not None:
        from evaluacion import clasificador, medicion_final

        datos = json.loads(Path(segunda_etapa).read_text(encoding="utf-8"))
        modelo, umbral = medicion_final.modelo_de_json(
            datos["clasificador_final"][medicion_final.ELEGIDO]
        )
        decidir = clasificador.decisor(modelo, umbral)
    elif hibrido:
        from evaluacion.dos_etapas import UMBRAL_HIBRIDO, primer_candidato

        decidir = primer_candidato(UMBRAL_HIBRIDO)
    if decidir is not None:
        from evaluacion.dos_etapas import K

        normalizador = Normalizador(indice, codificador, K)
        decidir_entidades(normalizador, ["hipertensión arterial"], decidir)
    carga = time.perf_counter() - inicio

    notas = notas_de_2000([d.texto for d in leer_particion(raiz, "test")])
    pipeline, total, sin_segunda, entidades = [], [], [], 0
    for nota in notas:
        inicio = time.perf_counter()
        resultado = analizar(nota)
        pipeline.append(time.perf_counter() - inicio)
        textos = [e.texto for e in resultado.entidades]
        entidades += len(textos)
        if codificador is not None and textos:
            if decidir is None:
                indice.buscar(codificador(textos))
            else:
                normalizador = Normalizador(indice, codificador, K)
                normalizador.precalcular(textos)
        sin_segunda.append(time.perf_counter() - inicio)
        if decidir is not None and textos:
            decidir_entidades(normalizador, textos, decidir)
        total.append(time.perf_counter() - inicio)
    caracteres = [len(n) for n in notas]
    pico = int(Path("/sys/fs/cgroup/memory.peak").read_text())
    salida = {
        "carga_s": round(carga, 1),
        "entidades": entidades,
        "pipeline": resumen(pipeline, caracteres),
        "total": resumen(total, caracteres),
        "memoria_pico_mib": round(pico / 2**20),
    }
    if decidir is not None:
        salida["sin_segunda_etapa"] = resumen(sin_segunda, caracteres)
        segunda = [t - s for t, s in zip(total, sin_segunda, strict=True)]
        salida["segunda_etapa"] = resumen(segunda, caracteres)
    return salida


# --- Fuera del contenedor ------------------------------------------------------------


def _docker(*argumentos: str) -> str:  # pragma: no cover - usa Docker
    return subprocess.run(
        ["docker", *argumentos], check=True, capture_output=True, text=True
    ).stdout.strip()


def _guardar_indice(destino: Path, con_fuentes: bool = False) -> dict[str, Any]:  # pragma: no cover
    """El índice del catálogo con los sinónimos de CodiEsp entrenamiento (y, con
    `con_fuentes`, los de DisTEMIST y CANTEMIST de #153), en un archivo."""
    import torch

    from evaluacion import codiesp, sinonimos_codiesp
    from evaluacion.normalizador import CANDIDATOS, CodificadorHF, indice_en_disco

    modelo = CANDIDATOS[MODELO]
    codificador = CodificadorHF(modelo)
    indice, meta = indice_en_disco(codiesp.leer_catalogo(), modelo, lambda: codificador)
    entrenamiento = codiesp.leer_particion(codiesp.RAIZ_CODIESP, "train")
    sinonimos = sinonimos_codiesp.sinonimos_de(entrenamiento)
    if con_fuentes:
        sinonimos += _sinonimos_de_fuentes(entrenamiento, sinonimos)
    indice = indice.con_sinonimos(sinonimos, codificador)
    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"codigos": indice.codigos, "vectores": indice.vectores, "origenes": indice.origenes},
        destino,
    )
    return {
        "filas": len(indice.codigos),
        "catalogo": meta["catalogo"],
        "bytes": destino.stat().st_size,
    }


def _sinonimos_de_fuentes(entrenamiento, propios) -> list:  # pragma: no cover - lee corpus
    """Los sinónimos de #153 que entran en la medición final, sin desarrollo ni prueba."""
    from evaluacion import codiesp, medicion_final, sinonimos_publicos
    from evaluacion.__main__ import _leer_fuentes

    args = argparse.Namespace(
        raiz=codiesp.RAIZ_CODIESP,
        cantemist=sinonimos_publicos.RAIZ_CANTEMIST,
        distemist=sinonimos_publicos.RAIZ_DISTEMIST,
    )
    desarrollo = codiesp.leer_particion(codiesp.RAIZ_CODIESP, "dev")
    fuentes, _, ids = _leer_fuentes(args, entrenamiento, desarrollo)
    return sinonimos_publicos.sinonimos_externos(
        medicion_final.externas_de(fuentes),
        [*ids["dev"], *ids["test"]],
        (s.texto for s in propios),
    )


def _medir_imagen(
    imagen: str,
    raiz: Path,
    con_normalizador: bool,
    segunda_etapa: bool = False,
    hibrido: bool = False,
) -> dict[str, Any]:  # pragma: no cover
    repo = Path(__file__).resolve().parent
    argumentos = [
        "run", "--rm", f"--cpus={CPUS}", f"--memory={MEMORIA}",
        "-v", f"{repo}:/medicion/evaluacion:ro",
        "-v", f"{Path(raiz).resolve()}:/datos:ro",
        "-e", "PYTHONPATH=/medicion",
        imagen, "python", "-m", "evaluacion.costo_normalizacion", "--dentro", "--raiz", "/datos",
    ]  # fmt: skip
    if con_normalizador:
        argumentos.append("--con-normalizador")
    if segunda_etapa:
        argumentos += ["--segunda-etapa", RESULTADOS_EN_CONTENEDOR]
    if hibrido:
        argumentos.append("--hibrido")
    medicion = json.loads(_docker(*argumentos).splitlines()[-1])
    medicion["imagen"] = {
        "nombre": imagen,
        "id": _docker("image", "inspect", "--format", "{{.Id}}", imagen),
        "tamano_mb": round(int(_docker("image", "inspect", "--format", "{{.Size}}", imagen)) / 1e6),
    }
    return medicion


def _construir(imagen: str, carpeta: Path, con_fuentes: bool) -> dict[str, Any]:  # pragma: no cover
    indice = _guardar_indice(carpeta / "indice.pt", con_fuentes)
    with tempfile.TemporaryDirectory(dir=carpeta) as contexto:
        (Path(contexto) / "Dockerfile").write_text(dockerfile_candidata(), encoding="utf-8")
        (Path(contexto) / "indice.pt").hardlink_to(carpeta / "indice.pt")
        _docker("build", "-t", imagen, contexto)
    return indice


def medir(raiz: Path, final: bool = False) -> dict[str, Any]:  # pragma: no cover
    """Base y candidata; con `final`, además la imagen final con la segunda etapa."""
    indice = _construir(IMAGEN_CANDIDATA, RAIZ_INDICE, con_fuentes=False)
    from evaluacion.distemist import entorno

    resultados = {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": entorno(),
        "docker_run": {"cpus": CPUS, "memoria": MEMORIA},
        "notas": "DisTEMIST test de más de 2.000 caracteres, recortadas a 2.000; sin texto aquí",
        "indice": indice,
    }
    if final:
        resultados["indice_final"] = _construir(IMAGEN_FINAL, RAIZ_INDICE / "final", True)
    resultados["base"] = _medir_imagen(IMAGEN_BASE, raiz, con_normalizador=False)
    # Con `final`, la candidata va por el mismo camino que la final (top-20 y decisión)
    resultados["candidata"] = _medir_imagen(IMAGEN_CANDIDATA, raiz, True, hibrido=final)
    if final:
        resultados["final"] = _medir_imagen(IMAGEN_FINAL, raiz, True, segunda_etapa=True)
    return resultados


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    from evaluacion.corpus import RAIZ_DATOS
    from evaluacion.distemist import guardar_json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raiz", type=Path, default=RAIZ_DATOS)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument(
        "--final", action="store_true", help="Mide también la imagen final con la segunda etapa"
    )
    parser.add_argument("--dentro", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--con-normalizador", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--segunda-etapa", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hibrido", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dentro:
        print(
            json.dumps(
                medir_dentro(args.raiz, args.con_normalizador, args.segunda_etapa, args.hibrido)
            )
        )
        return
    salida = args.salida or (SALIDA_FINAL if args.final else SALIDA)
    resultados = medir(args.raiz, args.final)
    guardar_json(resultados, salida)
    for nombre in ("base", "candidata", "final"):
        if nombre not in resultados:
            continue
        r = resultados[nombre]
        print(
            f"{nombre}: imagen {r['imagen']['tamano_mb']} MB, carga {r['carga_s']} s, "
            f"p95 {r['total']['p95_s']} s, memoria pico {r['memoria_pico_mib']} MiB"
        )
    if "final" in resultados:
        print(f"segunda etapa: p95 {resultados['final']['segunda_etapa']['p95_s']} s por nota")
    print(f"Resultados en {salida}")


if __name__ == "__main__":  # pragma: no cover
    main()
