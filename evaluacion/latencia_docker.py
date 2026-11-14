"""Latencia de la imagen Docker con los recursos de Cloud Run (FR3 del PRD de despliegue).

Arranca el contenedor con `--cpus=2 --memory=4g` y mide, vía la API (HTTP):

1. **Arranque en frío:** desde `docker run` hasta que `/api/salud` dice que el modelo está
   cargado (proceso, carga del modelo y calentamiento).
2. **Las 250 notas de prueba de DisTEMIST**, una por una: mediana, p95, máximo y
   milisegundos por cada 1.000 caracteres.
3. **Notas de 2.000 caracteres:** las notas de prueba que pasan de 2.000 caracteres,
   recortadas a 2.000 (el criterio de éxito del PRD es p95 ≤ 5 s).
4. **Una nota de 5.000 caracteres:** la nota de prueba más larga recortada a 5.000, varias
   veces.
5. **Memoria pico** del contenedor: `memory.peak` del cgroup y, como respaldo, el máximo
   que ve `docker stats` durante la medición.

Uso (la imagen ya construida con `docker build -t ner-clinic-es .`, la completa):

    uv run python -m evaluacion.latencia_docker [--imagen ner-clinic-es]
        [--raiz data/distemist] [--salida evaluacion/resultados/latencia_docker.json]

El JSON solo trae métricas: ningún texto de las notas.
"""

import argparse
import json
import math
import platform
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluacion.corpus import RAIZ_DATOS, leer_particion

SALIDA = Path("evaluacion/resultados/latencia_docker.json")
CPUS = "2"
MEMORIA = "4g"
PUERTO_HOST = 18080
# Topes altos: la medición hace unas 400 peticiones desde la misma IP, y hay notas de
# prueba de más de 5.000 caracteres (la más larga tiene 6.329).
ENTORNO_MEDICION = {
    "TOPE_IP_DIA": "100000",
    "TOPE_GLOBAL_DIA": "100000",
    "MAX_CARACTERES": "10000",
}
CARACTERES_MEDIA = 2000
CARACTERES_LARGA = 5000
REPETICIONES_LARGA = 3
ESPERA_MAXIMA_S = 600


def percentil(valores: Sequence[float], p: float) -> float:
    """Percentil `p` (0-100) por el método del rango más cercano: siempre es un valor medido."""
    if not valores:
        raise ValueError("No hay valores")
    ordenados = sorted(valores)
    rango = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[rango - 1]


def resumen(segundos: Sequence[float], caracteres: Sequence[int]) -> dict[str, Any]:
    """Mediana, p95 y máximo en segundos, y milisegundos por cada 1.000 caracteres."""
    if len(segundos) != len(caracteres):
        raise ValueError("Cada tiempo necesita su número de caracteres")
    return {
        "notas": len(segundos),
        "caracteres_mediana": statistics.median(caracteres),
        "caracteres_max": max(caracteres),
        "mediana_s": round(statistics.median(segundos), 3),
        "p95_s": round(percentil(segundos, 95), 3),
        "max_s": round(max(segundos), 3),
        "ms_por_1000_caracteres": round(sum(segundos) * 1000 / sum(caracteres) * 1000, 1),
    }


def recortar(texto: str, caracteres: int) -> str:
    """Los primeros `caracteres` del texto, sin partir la última palabra si se puede."""
    if len(texto) <= caracteres:
        return texto
    corte = texto[:caracteres]
    espacio = corte.rfind(" ")
    return corte[:espacio] if espacio > caracteres * 0.9 else corte


def a_bytes(cantidad: str) -> int:
    """`'1.5GiB'` o `'812MiB'` (formato de `docker stats`) a bytes."""
    unidades = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "kB": 10**3, "MB": 10**6}
    unidades["GB"] = 10**9
    for unidad in sorted(unidades, key=len, reverse=True):
        if cantidad.endswith(unidad):
            return round(float(cantidad[: -len(unidad)]) * unidades[unidad])
    raise ValueError(f"Cantidad desconocida: {cantidad!r}")


def _post(url: str, cuerpo: dict) -> tuple[float, dict]:  # pragma: no cover - usa la red
    datos = json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(
        url, data=datos, headers={"Content-Type": "application/json"}, method="POST"
    )
    inicio = time.perf_counter()
    with urllib.request.urlopen(peticion, timeout=300) as respuesta:
        contenido = json.loads(respuesta.read())
    return time.perf_counter() - inicio, contenido


def _salud(url: str) -> dict | None:  # pragma: no cover - usa la red
    try:
        with urllib.request.urlopen(url, timeout=2) as respuesta:
            return json.loads(respuesta.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None


def _docker(*argumentos: str) -> str:  # pragma: no cover - usa Docker
    return subprocess.run(
        ["docker", *argumentos], check=True, capture_output=True, text=True
    ).stdout.strip()


class _MuestreoMemoria:  # pragma: no cover - usa Docker
    """Muestrea `docker stats` en un hilo y guarda el máximo de memoria usada."""

    def __init__(self, contenedor: str):
        self.contenedor = contenedor
        self.maximo = 0
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)

    def __enter__(self) -> "_MuestreoMemoria":
        self._hilo.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._parar.set()
        self._hilo.join()

    def _bucle(self) -> None:
        while not self._parar.is_set():
            try:
                uso = _docker("stats", "--no-stream", "--format", "{{.MemUsage}}", self.contenedor)
                self.maximo = max(self.maximo, a_bytes(uso.split("/")[0].strip()))
            except (subprocess.CalledProcessError, ValueError):
                pass
            self._parar.wait(1)


def _memoria_pico_cgroup(contenedor: str) -> int | None:  # pragma: no cover - usa Docker
    try:
        return int(_docker("exec", contenedor, "cat", "/sys/fs/cgroup/memory.peak"))
    except (subprocess.CalledProcessError, ValueError):
        return None


def _medir(
    url: str, textos: Sequence[str], progreso: Callable[[int], None] | None = None
) -> tuple[list[float], list[float]]:  # pragma: no cover - usa la red
    """Tiempos de cliente (HTTP completo) y de servidor (`meta.segundos`) por nota."""
    cliente, servidor = [], []
    for i, texto in enumerate(textos, 1):
        segundos, contenido = _post(url, {"texto": texto})
        cliente.append(segundos)
        servidor.append(contenido["meta"]["segundos"])
        if progreso:
            progreso(i)
    return cliente, servidor


def medir(
    imagen: str, raiz: Path, puerto: int, entorno_extra: dict[str, str] | None = None
) -> dict[str, Any]:  # pragma: no cover
    """Mide `imagen`; `entorno_extra` agrega variables de entorno al contenedor (p. ej.
    `NER_NORMALIZADOR`, #167)."""
    documentos = leer_particion(raiz, "test")
    textos = [d.texto for d in documentos]
    medias = [recortar(t, CARACTERES_MEDIA) for t in textos if len(t) >= CARACTERES_MEDIA]
    larga = recortar(max(textos, key=len), CARACTERES_LARGA)

    base = f"http://127.0.0.1:{puerto}"
    nombre = f"latencia-ner-{int(time.time())}"
    variables = {**ENTORNO_MEDICION, **(entorno_extra or {})}
    entorno = [a for k, v in variables.items() for a in ("-e", f"{k}={v}")]

    inicio = time.perf_counter()
    _docker(
        "run", "-d", "--rm", "--name", nombre, "--cpus", CPUS, "--memory", MEMORIA,
        "-p", f"127.0.0.1:{puerto}:8080", *entorno, imagen,
    )  # fmt: skip
    try:
        with _MuestreoMemoria(nombre) as memoria:
            while (salud := _salud(f"{base}/api/salud")) is None or not salud["modelo_cargado"]:
                if time.perf_counter() - inicio > ESPERA_MAXIMA_S:
                    raise TimeoutError("El contenedor no cargó el modelo a tiempo")
                time.sleep(0.25)
            arranque = time.perf_counter() - inicio
            print(f"Arranque en frío: {arranque:.1f} s")

            url = f"{base}/api/analizar"

            def avance(i: int) -> None:
                if i % 50 == 0:
                    print(f"  {i}/{len(textos)} notas")

            cliente, servidor = _medir(url, textos, avance)
            cliente_2k, servidor_2k = _medir(url, medias)
            cliente_5k, servidor_5k = _medir(url, [larga] * REPETICIONES_LARGA)
            pico_cgroup = _memoria_pico_cgroup(nombre)
            pico_stats = memoria.maximo
    finally:
        subprocess.run(["docker", "stop", nombre], capture_output=True)

    caracteres = [len(t) for t in textos]
    return {
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mapeo": salud["mapeo"],
        "imagen": {
            "nombre": imagen,
            "id": _docker("image", "inspect", "--format", "{{.Id}}", imagen),
            "tamano_mb": round(
                int(_docker("image", "inspect", "--format", "{{.Size}}", imagen)) / 1e6
            ),
        },
        "docker_run": {"cpus": CPUS, "memoria": MEMORIA, "entorno": variables},
        "host": {"cpu": _nombre_cpu(), "maquina": platform.machine()},
        "notas": "DisTEMIST test (250 casos clínicos públicos); ningún texto en este archivo",
        "tiempo": (
            "`cliente`: petición HTTP completa desde el host; `servidor`: `meta.segundos` "
            "de la API (solo `analizar()`)"
        ),
        "arranque_en_frio_s": round(arranque, 1),
        "distemist_test": {
            "cliente": resumen(cliente, caracteres),
            "servidor": resumen(servidor, caracteres),
        },
        "notas_2000_caracteres": {
            "cliente": resumen(cliente_2k, [len(t) for t in medias]),
            "servidor": resumen(servidor_2k, [len(t) for t in medias]),
        },
        "nota_5000_caracteres": {
            "caracteres": len(larga),
            "repeticiones": REPETICIONES_LARGA,
            "cliente_s": [round(s, 3) for s in cliente_5k],
            "servidor_s": [round(s, 3) for s in servidor_5k],
        },
        "memoria_pico_mb": {
            "cgroup_memory_peak": None if pico_cgroup is None else round(pico_cgroup / 2**20),
            "docker_stats_max": round(pico_stats / 2**20),
        },
    }


def _nombre_cpu() -> str:  # pragma: no cover - depende de la máquina
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - usa Docker
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--imagen", default="ner-clinic-es")
    parser.add_argument("--raiz", type=Path, default=RAIZ_DATOS)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--puerto", type=int, default=PUERTO_HOST)
    args = parser.parse_args(argv)

    resultados = medir(args.imagen, args.raiz, args.puerto)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
