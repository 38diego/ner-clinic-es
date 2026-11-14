"""Negación e incertidumbre del pipeline sobre NUBes, texto clínico real (#59, #87).

NUBes (Lima et al., LREC 2020) tiene unas 29.700 frases de informes clínicos reales en
español, anonimizadas y barajadas, con la negación y la incertidumbre anotadas a mano:
claves (marcadores), alcances y eventos. Se distribuye en GitHub con licencia CC BY-SA
3.0 ES. Aquí se usa la carpeta `NUBes/` (10 muestras, `SAMPLE-001` a `SAMPLE-010`); la
`SAMPLE-001` la anotaron dos personas y un árbitro y las demás una sola.

Formato (BRAT standoff, ver las guías de anotación del repositorio). Cada `.txt` trae una
frase por línea y su `.ann`, offsets en caracteres sobre el `.txt` completo:

    T1  NegSynMarker 180 182    No
    T2  DISO 183 190            cefalea
    T3  Phrase 183 214          cefalea ni focalidad ni vomitos
    R1  Scope Arg1:T1 Arg2:T3

- Claves: `NegSynMarker` ("no", "sin"), `NegLexMarker` ("niega", "descartar", "ausencia
  de"), `NegMorMarker` ("afebril", "asintomático"), `UncertLexMarker` ("posible",
  "compatible con", "sugestivo de") y `UncertSynMarker` (la "o" que duda).
- Alcances: el argumento 2 de una relación `Scope` o `DiscScope` (las partes de un
  alcance discontinuo) que sale de una clave. Es un `Phrase` o directamente el evento
  (`DISO`, `PROC`, `SUBS`, `BODY`, `OTHER`) cuando el alcance y el evento coinciden.
  La clase del alcance (negación o incertidumbre) la da el tipo de la clave, no el del
  alcance.
- `NegMorMarker` no tiene alcance: según las guías, el alcance es la palabra misma. Aquí
  la palabra cuenta como una región de negación.
- `DiscMarker` une las dos partes de una clave discontinua ("no ... claro"); el alcance
  sale de la segunda parte, así que no hace falta seguirlo.

Ningún alcance cruza un salto de línea (se verifica al leer), así que cada región se
asigna a su frase y queda en posiciones de la frase.

Método de la evaluación:

1. Cada frase se analiza sola con el pipeline (`pipeline._analizar`, que es
   `analizar()`). No se juntan frases: la deduplicación fusionaría entidades de frases
   distintas, que en NUBes están barajadas y no son de la misma nota (y hasta #73 la
   clasificación por texto filtraba la negación de una frase a la misma palabra en
   otra). Para no pagar una llamada al modelo por frase,
   antes se llena la caché del modelo (`cache.ExtractorConCache`) con todas las frases,
   por lotes y ordenadas por largo; el análisis frase por frase después solo lee la
   caché. Cada frase es un fragmento propio, igual que si se analizara sola.
2. Cada mención de cada entidad del resultado se cruza con las regiones gold de su
   frase. La mención cae en una región si al menos la mitad de sus caracteres están en
   ella. Si cae en varias, manda la más corta (la más específica: "posible X sin Y" deja
   "Y" negada dentro de un alcance incierto). Estado esperado: `negada` (región de
   negación), `incierta` (región de incertidumbre) o `afirmada` (ninguna).
3. El estado obtenido es el de la entidad de la mención: `confirmada`, `negada`,
   `posible` o `familiar` (NUBes no anota familiares; se reportan como "salió
   familiar"). Desde #73 el pipeline clasifica cada mención por su posición, así que es
   el de la mención. El estado se lee como texto (`Estado.value`). `Estado.POSIBLE`
   existe desde #88, pero sin claves que lo disparen (son de #83): hasta entonces la
   columna sale en ceros.
4. Matriz de confusión con `metricas.matriz_confusion` (filas negada, incierta y
   afirmada; columnas confirmada, negada, posible y familiar) y dos métricas:
   - P/R/F1 de `negada` sobre las menciones esperadas `negada` o `afirmada`: las
     `incierta` no cuentan, igual que antes de #87.
   - P/R/F1 de `posible` sobre todas las menciones: VP es una `incierta` que sale
     `posible`; FP, una que sale `posible` sin ser `incierta`; FN, una `incierta` que
     no sale `posible`.

Además se cuantifica el cambio de la clasificación por texto (`docs/errores-tesis.md`) a
la clasificación por posición (#73): para cada mención se recalcula el estado que le
daba la búsqueda por texto de la tesis, con los mismos intervalos de `contexto`, y se
cuentan las menciones cuyo estado cambia. Hasta #73 era al revés: el pipeline
clasificaba por texto y aquí se recalculaba por posición (por contención; el pipeline
usa el inicio de la mención).

El corpus nunca se commitea: se descarga en `data/nubes/`, que está en `.gitignore`. El
JSON versionado solo lleva métricas y fragmentos cortos (120 caracteres o menos) de la
muestra de discrepancias, con atribución.
"""

import hashlib
import random
import re
import shutil
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from evaluacion.cache import ExtractorConCache
from evaluacion.corpus import ErrorCorpus, _descomprimir
from evaluacion.distemist import entorno, memoria_pico_mib
from evaluacion.errores import MAX_CONTEXTO, contexto
from evaluacion.metricas import Conteos, bootstrap_ic, matriz_confusion, micro_prf
from ner_clinic import contexto as contexto_pipeline
from ner_clinic.extraccion import Extractor, siglas_para_el_modelo
from ner_clinic.limpieza import limpiar
from ner_clinic.pipeline import _analizar
from ner_clinic.segmentacion import segmentar
from ner_clinic.tipos import Estado

COMMIT_NUBES = "91c81c1f5eee2b5b9d377cc78c2442e8e8b198a1"
URL_NUBES = (
    "https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus/archive/"
    f"{COMMIT_NUBES}.zip"
)
SHA256_NUBES = "cd3cea101057660560cb423afa7152c6d2996953524e141c96af97fcd4a7983e"
RAIZ_NUBES = Path("data/nubes")
NOMBRE_ZIP = f"nubes-{COMMIT_NUBES[:7]}.zip"
CARPETA_CORPUS = f"NUBes-negation-uncertainty-biomedical-corpus-{COMMIT_NUBES}"
MUESTRAS = tuple(f"SAMPLE-{i:03d}" for i in range(1, 11))

# Partición del PRD `negacion` (FR1): durante el desarrollo solo se mira la de desarrollo;
# la reservada se corre una vez al final. La `SAMPLE-001` (doble anotación) es reservada.
MUESTRAS_DESARROLLO = MUESTRAS[1::2]
MUESTRAS_RESERVADAS = MUESTRAS[0::2]
Particion = Literal["desarrollo", "reservada", "todo"]
PARTICIONES: dict[str, tuple[str, ...]] = {
    "desarrollo": MUESTRAS_DESARROLLO,
    "reservada": MUESTRAS_RESERVADAS,
    "todo": MUESTRAS,
}

CARPETA_RESULTADOS = Path("evaluacion/resultados")
CARPETA_DETALLE = Path("data/evaluacion")
LOTE_INFERENCIA = 256
SEMILLA = 59
TAMANO_MUESTRA = 30
REPLICAS_BOOTSTRAP = 1000

Clase = Literal["negacion", "incertidumbre"]
Esperado = Literal["negada", "incierta", "afirmada"]
ESPERADOS: tuple[str, ...] = ("negada", "incierta", "afirmada")
# Estados del pipeline como texto.
POSIBLE = Estado.POSIBLE.value
OBTENIDOS: tuple[str, ...] = ("confirmada", "negada", POSIBLE, "familiar")

CLAVES: dict[str, Clase] = {
    "NegSynMarker": "negacion",
    "NegLexMarker": "negacion",
    "NegMorMarker": "negacion",
    "UncertSynMarker": "incertidumbre",
    "UncertLexMarker": "incertidumbre",
}
RELACIONES_ALCANCE = ("Scope", "DiscScope")

# Discrepancias que van a la muestra, en el orden en que se sortean.
DISCREPANCIAS = (("negada", "confirmada"), ("afirmada", "negada"))

CATEGORIAS = {
    "clave_no_reconocida": "El pipeline tiene la clave, pero no en esta forma («descartó»)",
    "clave_ausente": "Clave que el pipeline no tiene",
    "alcance_mal_cortado": "Alcance mal cortado: la negación llega muy lejos o se queda corta",
    "por_texto": "Clasificación por texto: la misma palabra está negada en otra parte",
    "disparador_falso": "Disparador falso: la clave está dentro de otra palabra o de un nombre",
    "error_gold": "Error u omisión del gold, o convención de NUBes",
    "otra": "Otra",
}

ATRIBUCION = (
    "Los fragmentos vienen del corpus NUBes (Lima López S, Perez N, Cuadros M, Rigau G. "
    "NUBes: A Corpus of Negation and Uncertainty in Spanish Clinical Texts. LREC 2020, "
    "pp. 5772-5781; https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-"
    f"corpus, commit {COMMIT_NUBES[:7]}), con licencia CC BY-SA 3.0 ES "
    "(https://creativecommons.org/licenses/by-sa/3.0/es/). Se citan recortados, sin "
    "otros cambios."
)


# --- Descarga ------------------------------------------------------------------------


def sha256_de(ruta: Path) -> str:
    """SHA-256 en hexadecimal de un archivo, leído por bloques."""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(destino: Path = RAIZ_NUBES, url: str = URL_NUBES, sha256: str = SHA256_NUBES) -> Path:
    """Baja NUBes en el commit `COMMIT_NUBES`, verifica su SHA-256 y lo descomprime.

    Como `corpus.descargar`: no vuelve a bajar el zip si ya está y su hash coincide, ni a
    descomprimir si la carpeta ya existe, y si el hash no coincide borra lo bajado y falla.

    Devuelve la carpeta del corpus (`destino/NUBes-negation-...-<commit>`).
    """
    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / NOMBRE_ZIP

    if not (zip_path.exists() and sha256_de(zip_path) == sha256):
        parcial = zip_path.with_suffix(".zip.parcial")
        with urllib.request.urlopen(url) as respuesta, parcial.open("wb") as f:
            shutil.copyfileobj(respuesta, f)
        obtenido = sha256_de(parcial)
        if obtenido != sha256:
            parcial.unlink()
            raise ErrorCorpus(
                f"El SHA-256 de {url} no coincide: se esperaba {sha256} y se obtuvo "
                f"{obtenido}. No se descomprimió nada."
            )
        parcial.replace(zip_path)

    corpus = destino / CARPETA_CORPUS
    if not corpus.is_dir():
        _descomprimir(zip_path, destino)
    return corpus


def muestras_de(particion: Particion | None, muestras: Sequence[str] | None = None) -> list[str]:
    """Las muestras a evaluar: las pedidas con `--muestras` o las de la `particion`.

    Sin ninguna de las dos, todas. Las dos a la vez no se aceptan.
    """
    if particion is not None and muestras is not None:
        raise ValueError("Use --particion o --muestras, no las dos.")
    if muestras is not None:
        return list(muestras)
    return list(PARTICIONES[particion or "todo"])


# --- Lectura -------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """Tramo de una frase afectado por una clave: `frase.texto[inicio:fin]`.

    `clave` es el texto de la clave que la abre (la palabra misma en un `NegMorMarker`).
    """

    inicio: int
    fin: int
    clase: Clase
    clave: str


@dataclass(frozen=True)
class Frase:
    """Una línea de un archivo de NUBes con sus regiones gold en posiciones de la frase."""

    id: str
    texto: str
    regiones: list[Region] = field(default_factory=list)


@dataclass(frozen=True)
class _Texto:
    tipo: str
    inicio: int
    fin: int
    texto: str


def _leer_ann(ruta: Path, texto: str) -> tuple[dict[str, _Texto], list[tuple[str, str, str]]]:
    """Anotaciones de texto (`T`) y relaciones (`R`) de un `.ann`, con offsets verificados."""
    textos: dict[str, _Texto] = {}
    relaciones: list[tuple[str, str, str]] = []
    with ruta.open(encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            linea = linea.rstrip("\n")
            if not linea.strip():
                continue
            partes = linea.split("\t")
            if linea.startswith("T"):
                tipo, _, offsets = partes[1].partition(" ")
                if ";" in offsets:
                    raise ErrorCorpus(f"{ruta.name}:{numero}: anotación discontinua no soportada")
                inicio, fin = (int(x) for x in offsets.split())
                # BRAT escribe los saltos de línea del span como espacios.
                if texto[inicio:fin].replace("\n", " ") != partes[2]:
                    raise ErrorCorpus(
                        f"{ruta.name}:{numero}: [{inicio}:{fin}] {partes[2]!r} no coincide "
                        f"con {texto[inicio:fin]!r}"
                    )
                textos[partes[0]] = _Texto(tipo, inicio, fin, partes[2])
            elif linea.startswith("R"):
                relacion, arg1, arg2 = partes[1].split()
                relaciones.append((relacion, arg1.split(":", 1)[1], arg2.split(":", 1)[1]))
    return textos, relaciones


def _regiones_de(
    textos: dict[str, _Texto], relaciones: list[tuple[str, str, str]]
) -> list[tuple[int, int, Clase, str]]:
    """Regiones del archivo en offsets del `.txt`: alcances y negaciones morfológicas."""
    regiones = []
    for relacion, clave, alcance in relaciones:
        if relacion not in RELACIONES_ALCANCE:
            continue
        marcador = textos[clave]
        if marcador.tipo not in CLAVES:
            raise ErrorCorpus(f"Alcance que sale de algo que no es una clave: {marcador.tipo}")
        destino = textos[alcance]
        regiones.append((destino.inicio, destino.fin, CLAVES[marcador.tipo], marcador.texto))
    for anotacion in textos.values():
        if anotacion.tipo == "NegMorMarker":
            regiones.append((anotacion.inicio, anotacion.fin, "negacion", anotacion.texto))
    return regiones


def leer_archivo(ruta_txt: Path) -> list[Frase]:
    """Frases de un `.txt` de NUBes (una por línea) con las regiones de su `.ann`."""
    with ruta_txt.open(encoding="utf-8", newline="") as f:
        texto = f.read()
    textos, relaciones = _leer_ann(ruta_txt.with_suffix(".ann"), texto)

    lineas: list[tuple[int, int]] = []
    inicio = 0
    for linea in texto.split("\n"):
        lineas.append((inicio, inicio + len(linea)))
        inicio += len(linea) + 1

    por_linea: dict[int, list[Region]] = {}
    for a, b, clase, clave in _regiones_de(textos, relaciones):
        numero = next((i for i, (ini, fin) in enumerate(lineas) if ini <= a < fin), None)
        if numero is None or b > lineas[numero][1]:
            raise ErrorCorpus(f"{ruta_txt.name}: la región [{a}:{b}] cruza un salto de línea")
        desplazamiento = lineas[numero][0]
        por_linea.setdefault(numero, []).append(
            Region(a - desplazamiento, b - desplazamiento, clase, clave)
        )

    return [
        Frase(
            id=f"{ruta_txt.stem}:{numero + 1}",
            texto=texto[ini:fin],
            regiones=sorted(por_linea.get(numero, []), key=lambda r: (r.inicio, r.fin)),
        )
        for numero, (ini, fin) in enumerate(lineas)
        if texto[ini:fin].strip()
    ]


def _carpeta_corpus(raiz: Path) -> Path:
    """Acepta la carpeta del corpus o la que la contiene."""
    anidada = Path(raiz) / CARPETA_CORPUS
    return anidada if anidada.is_dir() else Path(raiz)


def leer(raiz: Path, muestras: Iterable[str] | None = None) -> list[Frase]:
    """Frases de las `muestras` de NUBes (`SAMPLE-001`...), en orden de archivo y línea.

    Sin `muestras`, lee todas las carpetas `SAMPLE-*` que haya.
    """
    carpeta = _carpeta_corpus(raiz) / "NUBes"
    if muestras is None:
        muestras = sorted(p.name for p in carpeta.glob("SAMPLE-*") if p.is_dir())
    frases: list[Frase] = []
    for muestra in muestras:
        archivos = sorted((carpeta / muestra).glob("*.txt"))
        if not archivos:
            raise ErrorCorpus(f"No hay archivos .txt en {carpeta / muestra}")
        for archivo in archivos:
            frases.extend(leer_archivo(archivo))
    return frases


# --- Estado esperado -----------------------------------------------------------------


def _solape(a_inicio: int, a_fin: int, b_inicio: int, b_fin: int) -> int:
    return max(0, min(a_fin, b_fin) - max(a_inicio, b_inicio))


def region_de(inicio: int, fin: int, regiones: Iterable[Region]) -> Region | None:
    """La región gold que cubre la mención `[inicio, fin)`, o `None`.

    La cubre si al menos la mitad de los caracteres de la mención están en ella. Si la
    cubren varias, gana la más corta; a igual largo, la de negación.
    """
    largo = max(fin - inicio, 1)
    cubren = [r for r in regiones if 2 * _solape(inicio, fin, r.inicio, r.fin) >= largo]
    if not cubren:
        return None
    return min(cubren, key=lambda r: (r.fin - r.inicio, r.clase != "negacion", r.inicio))


def esperado_de(region: Region | None) -> Esperado:
    if region is None:
        return "afirmada"
    return "negada" if region.clase == "negacion" else "incierta"


# --- Estado por texto (la clasificación de la tesis, hasta #73) ----------------------


def estado_por_texto(texto: str, entidad: str) -> str:
    """Estado que le daba a `entidad` la clasificación por texto de la tesis.

    Busca el texto de la entidad, sin distinguir mayúsculas, dentro de cada intervalo de
    `contexto.intervalos` del texto limpio, con la misma prioridad (familiar, posible,
    negada, confirmada): si la palabra está negada en un bloque, todas sus menciones salen
    negadas. Es el `contexto.clasificar` que el pipeline usó hasta #73; se conserva aquí
    solo para medir cuántas menciones cambian.
    """
    limpio = limpiar(texto).texto
    if not entidad.strip():
        return Estado.CONFIRMADA.value
    patron = re.compile(re.escape(entidad.strip()), re.IGNORECASE)
    estados = {
        i.estado
        for i in contexto_pipeline.intervalos(limpio)
        if patron.search(limpio[i.inicio : i.fin])
    }
    for estado in contexto_pipeline.PRIORIDAD:
        if estado in estados:
            return estado.value
    return Estado.CONFIRMADA.value


# --- Corrida -------------------------------------------------------------------------


@dataclass(frozen=True)
class Evaluada:
    """Una mención del pipeline cruzada con el gold de su frase."""

    frase: str
    texto: str
    inicio: int
    fin: int
    entidad: str
    esperado: Esperado
    obtenido: str
    por_texto: str
    clave_gold: str | None


def fragmentos_de(frases: Iterable[Frase]) -> list[str]:
    """Los fragmentos que el pipeline le pasará al modelo, sin repetir, del más corto al
    más largo (para que los lotes tengan poco relleno). Llevan las siglas en mayúscula,
    como los ve el modelo (`siglas_para_el_modelo`, #114): si no, la caché guardaría un
    texto que nadie pide."""
    vistos = dict.fromkeys(
        siglas_para_el_modelo(f.texto)
        for frase in frases
        for f in segmentar(limpiar(frase.texto).texto)
    )
    return sorted(vistos, key=len)


def precalentar(
    frases: Sequence[Frase], extractor: Extractor, lote: int = LOTE_INFERENCIA
) -> float:
    """Pasa por `extractor` (con caché) los fragmentos de todas las frases, por lotes.

    Devuelve los segundos que tardó. Con la caché llena, el análisis frase por frase ya
    no llama al modelo.
    """
    inicio = time.perf_counter()
    fragmentos = fragmentos_de(frases)
    for desde in range(0, len(fragmentos), lote):
        extractor.extraer(fragmentos[desde : desde + lote])
    return time.perf_counter() - inicio


def evaluar_frase(frase: Frase, extractor: Extractor) -> list[Evaluada]:
    """Analiza la frase sola y cruza cada mención con las regiones gold."""
    resultado = _analizar(frase.texto, extractor=extractor)
    evaluadas = []
    for entidad in resultado.entidades:
        for i, mencion in enumerate(entidad.menciones):
            region = region_de(mencion.inicio, mencion.fin, frase.regiones)
            evaluadas.append(
                Evaluada(
                    frase=frase.id,
                    texto=mencion.texto,
                    inicio=mencion.inicio,
                    fin=mencion.fin,
                    entidad=entidad.texto,
                    esperado=esperado_de(region),
                    # El de la mención: NUBes anota cada una por su contexto (#208).
                    obtenido=entidad.estado_de_mencion(i).value,
                    por_texto=estado_por_texto(frase.texto, entidad.texto),
                    clave_gold=region.clave if region else None,
                )
            )
    return evaluadas


def correr(frases: Sequence[Frase], extractor: Extractor) -> list[list[Evaluada]]:
    """Menciones evaluadas de cada frase, en el orden de `frases`."""
    return [evaluar_frase(frase, extractor) for frase in frases]


# --- Métricas ------------------------------------------------------------------------


def conteos_negada(evaluadas: Iterable[Evaluada]) -> Conteos:
    """VP, FP y FN de `negada` sobre las menciones esperadas `negada` o `afirmada`."""
    vp = fp = fn = 0
    for e in evaluadas:
        if e.esperado == "incierta":
            continue
        negada = e.obtenido == Estado.NEGADA.value
        if e.esperado == "negada":
            vp += negada
            fn += not negada
        else:
            fp += negada
    return Conteos(vp, fp, fn)


def conteos_posible(evaluadas: Iterable[Evaluada]) -> Conteos:
    """VP, FP y FN de `posible` (esperado: `incierta`) sobre todas las menciones."""
    vp = fp = fn = 0
    for e in evaluadas:
        posible = e.obtenido == POSIBLE
        if e.esperado == "incierta":
            vp += posible
            fn += not posible
        else:
            fp += posible
    return Conteos(vp, fp, fn)


def matriz(evaluadas: Sequence[Evaluada]) -> dict[str, dict[str, int]]:
    """Matriz esperado x obtenido: filas negada/incierta/afirmada; columnas los estados."""
    clases = list(dict.fromkeys(ESPERADOS + OBTENIDOS))
    completa = matriz_confusion(
        [e.esperado for e in evaluadas], [e.obtenido for e in evaluadas], clases
    )
    return {fila: {col: completa[fila][col] for col in OBTENIDOS} for fila in ESPERADOS}


def _redondear(valor: float) -> float:
    return round(valor, 4)


def _metricas(
    por_frase: Sequence[Sequence[Evaluada]],
    contar: Callable[[Iterable[Evaluada]], Conteos],
    replicas: int,
) -> dict[str, Any]:
    """P/R/F1 de los `conteos` de cada frase, con intervalo del 95 % por bootstrap."""
    conteos = [contar(evaluadas) for evaluadas in por_frase]
    prf = micro_prf(conteos)
    ic = bootstrap_ic(conteos, n=replicas, semilla=SEMILLA)
    return {
        "precision": _redondear(prf.precision),
        "recall": _redondear(prf.recall),
        "f1": _redondear(prf.f1),
        "ic95": {
            nombre: [_redondear(i.inferior), _redondear(i.superior)]
            for nombre, i in ic._asdict().items()
        },
        "vp": sum(c.vp for c in conteos),
        "fp": sum(c.fp for c in conteos),
        "fn": sum(c.fn for c in conteos),
    }


def metricas_negada(
    por_frase: Sequence[Sequence[Evaluada]], replicas: int = REPLICAS_BOOTSTRAP
) -> dict[str, Any]:
    """P/R/F1 de `negada` con intervalo del 95 % por bootstrap sobre frases."""
    return _metricas(por_frase, conteos_negada, replicas)


def metricas_posible(
    por_frase: Sequence[Sequence[Evaluada]], replicas: int = REPLICAS_BOOTSTRAP
) -> dict[str, Any]:
    """P/R/F1 de `posible` con intervalo del 95 % por bootstrap sobre frases."""
    return _metricas(por_frase, conteos_posible, replicas)


def muestra_de(id_frase: str) -> str:
    """`SAMPLE-001` para la frase `sample-001.traum.chico:12`."""
    return id_frase.split(".", 1)[0].upper()


def por_muestra(evaluadas: Sequence[Evaluada]) -> dict[str, dict[str, Any]]:
    """P/R/F1 de `negada` por muestra de NUBes (la `SAMPLE-001` tiene doble anotación)."""
    grupos: dict[str, list[Evaluada]] = {}
    for e in evaluadas:
        grupos.setdefault(muestra_de(e.frase), []).append(e)
    return {m: _prf_dict(conteos_negada(grupos[m])) for m in sorted(grupos)}


def por_texto(evaluadas: Sequence[Evaluada]) -> dict[str, Any]:
    """Cuánto cambia clasificar por posición (el pipeline, desde #73) y no por texto.

    Cuenta las menciones cuyo estado obtenido difiere del que les daba la clasificación
    por texto de la tesis (`transiciones`: `por texto->obtenido`), cuántas de ellas
    aciertan con el gold en cada caso y las métricas de `negada` que tendría la
    clasificación por texto.
    """

    def acierta(esperado: str, estado: str) -> bool:
        return (esperado == "negada") == (estado == Estado.NEGADA.value)

    cambian = [e for e in evaluadas if e.obtenido != e.por_texto]
    comparables = [e for e in cambian if e.esperado != "incierta"]
    return {
        "menciones_que_cambian": len(cambian),
        "transiciones": dict(Counter(f"{e.por_texto}->{e.obtenido}" for e in cambian)),
        "aciertan_por_texto": sum(acierta(e.esperado, e.por_texto) for e in comparables),
        "aciertan_por_posicion": sum(acierta(e.esperado, e.obtenido) for e in comparables),
        "metricas_por_texto": _prf_dict(
            conteos_negada(Evaluada(**{**e.__dict__, "obtenido": e.por_texto}) for e in evaluadas)
        ),
    }


def _prf_dict(conteos: Conteos) -> dict[str, Any]:
    prf = micro_prf([conteos])
    return {
        "precision": _redondear(prf.precision),
        "recall": _redondear(prf.recall),
        "f1": _redondear(prf.f1),
        "vp": conteos.vp,
        "fp": conteos.fp,
        "fn": conteos.fn,
    }


# --- Muestra de discrepancias --------------------------------------------------------


def _clave_muestra(item: dict[str, Any]) -> tuple[str, int, int]:
    return (item["frase"], item["inicio"], item["fin"])


def muestrear(
    evaluadas: Sequence[Evaluada],
    textos: dict[str, str],
    n: int = TAMANO_MUESTRA,
    semilla: int = SEMILLA,
) -> list[dict[str, Any]]:
    """Muestra fija de discrepancias: la mitad de cada tipo de `DISCREPANCIAS`.

    Cada una lleva un contexto de `MAX_CONTEXTO` caracteres o menos y los campos
    `categoria` y `nota` vacíos, para clasificar a mano.
    """
    rng = random.Random(semilla)
    por_tipo = n // len(DISCREPANCIAS)
    muestra = []
    for esperado, obtenido in DISCREPANCIAS:
        poblacion = [e for e in evaluadas if (e.esperado, e.obtenido) == (esperado, obtenido)]
        for e in rng.sample(poblacion, min(por_tipo, len(poblacion))):
            muestra.append(
                {
                    "frase": e.frase,
                    "inicio": e.inicio,
                    "fin": e.fin,
                    "mencion": e.texto,
                    "esperado": e.esperado,
                    "obtenido": e.obtenido,
                    "por_texto": e.por_texto,
                    "clave_gold": e.clave_gold,
                    "contexto": contexto(textos[e.frase], e.inicio, e.fin),
                    "categoria": None,
                    "nota": "",
                }
            )
    return muestra


def conservar_clasificacion(
    muestra: list[dict[str, Any]], anterior: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Copia `categoria` y `nota` de un resultado anterior a las mismas discrepancias."""
    if not anterior:
        return muestra
    previas = {_clave_muestra(d): d for d in anterior.get("discrepancias", [])}
    for item in muestra:
        previa = previas.get(_clave_muestra(item))
        if previa is not None and previa.get("contexto") == item["contexto"]:
            item["categoria"] = previa.get("categoria")
            item["nota"] = previa.get("nota", "")
    return muestra


def validar(resultados: dict[str, Any]) -> list[str]:
    """Problemas de la muestra: categorías vacías o desconocidas, contexto largo."""
    problemas = []
    for i, item in enumerate(resultados["discrepancias"], start=1):
        if item["categoria"] not in CATEGORIAS:
            problemas.append(f"#{i} {item['frase']}: categoría {item['categoria']!r} no válida")
        if len(item["contexto"]) > MAX_CONTEXTO:
            problemas.append(
                f"#{i} {item['frase']}: contexto de {len(item['contexto'])} caracteres"
            )
    return problemas


# --- Evaluación completa -------------------------------------------------------------


def evaluar(
    frases: Sequence[Frase],
    extractor: Extractor,
    muestras: Sequence[str],
    cache: ExtractorConCache | None = None,
    lote: int = LOTE_INFERENCIA,
    replicas: int = REPLICAS_BOOTSTRAP,
    anterior: dict[str, Any] | None = None,
    limite: int | None = None,
    particion: Particion | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Corre la evaluación y devuelve `(resultados, detalle)`.

    `resultados` es el JSON versionado: métricas y la muestra de discrepancias con
    fragmentos cortos. `detalle` trae todas las menciones evaluadas con su texto, para
    guardar fuera del repo. Si `extractor` es un `ExtractorConCache`, antes de analizar se
    llena la caché por lotes (`precalentar`). `anterior` es un resultado previo del que se
    conserva la clasificación a mano de la muestra. `particion` (`desarrollo`,
    `reservada`, `todo` o `None` si se pidieron `muestras` sueltas) solo queda registrada.
    """
    segundos_inferencia = precalentar(frases, extractor, lote) if cache is not None else 0.0
    inicio = time.perf_counter()
    por_frase = correr(frases, extractor)
    segundos_analisis = time.perf_counter() - inicio

    evaluadas = [e for lista in por_frase for e in lista]
    textos = {f.id: f.texto for f in frases}
    muestra = conservar_clasificacion(muestrear(evaluadas, textos), anterior)
    clases_region = Counter(r.clase for f in frases for r in f.regiones)

    resultados = {
        "corpus": {
            "nombre": "NUBes",
            "commit": COMMIT_NUBES,
            "sha256_zip": SHA256_NUBES,
            "licencia": "CC BY-SA 3.0 ES",
            "particion": particion,
            "muestras": list(muestras),
            "limite": limite,
            "frases": len(frases),
            "frases_con_negacion": sum(
                any(r.clase == "negacion" for r in f.regiones) for f in frases
            ),
            "frases_con_incertidumbre": sum(
                any(r.clase == "incertidumbre" for r in f.regiones) for f in frases
            ),
            "regiones": dict(clases_region),
        },
        "fecha": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entorno": entorno(),
        "metodo": {
            "unidad": "mención de entidad del pipeline",
            "analisis": "frase por frase, con la caché del modelo llenada por lotes",
            "cobertura": "al menos la mitad de la mención dentro de la región; manda la más corta",
            "negmormarker": "la palabra misma cuenta como región de negación",
            "negada": "P/R/F1 sobre las menciones esperadas negada o afirmada",
            "posible": "P/R/F1 sobre todas las menciones; lo esperado es incierta",
        },
        "menciones_evaluadas": len(evaluadas),
        "frases_con_menciones": sum(1 for lista in por_frase if lista),
        "matriz": matriz(evaluadas),
        "negada": metricas_negada(por_frase, replicas),
        "negada_por_muestra": por_muestra(evaluadas),
        "posible": metricas_posible(por_frase, replicas),
        "salio_posible": sum(e.obtenido == POSIBLE for e in evaluadas),
        "salio_familiar": sum(e.obtenido == Estado.FAMILIAR.value for e in evaluadas),
        "por_texto": por_texto(evaluadas),
        "tiempos": {
            "segundos_inferencia": round(segundos_inferencia, 2),
            "segundos_analisis": round(segundos_analisis, 2),
            "lote": lote,
        },
        "cache": (
            None
            if cache is None
            else {"aciertos": cache.aciertos, "fallos": cache.fallos, "llamadas": cache.llamadas}
        ),
        "memoria_pico_mib": memoria_pico_mib(),
        "categorias": CATEGORIAS,
        "atribucion": ATRIBUCION,
        "discrepancias": muestra,
    }
    detalle = {
        "corpus": resultados["corpus"],
        "menciones": [e.__dict__ for e in evaluadas],
    }
    return resultados, detalle


def ruta_resultados(salida: Path | None) -> Path:
    if salida is None:
        return CARPETA_RESULTADOS / "nubes.json"
    return salida if salida.suffix == ".json" else salida / "nubes.json"


def ruta_detalle(detalle: Path | None) -> Path:
    if detalle is None:
        return CARPETA_DETALLE / "nubes_menciones.json"
    return detalle if detalle.suffix == ".json" else detalle / "nubes_menciones.json"


def tabla_matriz(resultados: dict[str, Any]) -> str:
    """Matriz de confusión en Markdown."""
    lineas = [
        "| Esperado \\ obtenido | " + " | ".join(OBTENIDOS) + " | Total |",
        "|---|" + "---|" * (len(OBTENIDOS) + 1),
    ]
    for fila, columnas in resultados["matriz"].items():
        valores = [columnas[c] for c in OBTENIDOS]
        lineas.append(f"| {fila} | " + " | ".join(map(str, valores)) + f" | {sum(valores)} |")
    return "\n".join(lineas)


def resumen(resultados: dict[str, Any]) -> str:
    n = resultados["negada"]
    s = resultados["posible"]
    t = resultados["tiempos"]
    p = resultados["por_texto"]
    return (
        f"Negada: P {n['precision']:.3f} R {n['recall']:.3f} F1 {n['f1']:.3f} "
        f"(VP {n['vp']}, FP {n['fp']}, FN {n['fn']}); "
        f"Posible: P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} "
        f"(VP {s['vp']}, FP {s['fp']}, FN {s['fn']}); "
        f"{resultados['menciones_evaluadas']} menciones en {resultados['corpus']['frases']} "
        f"frases; salió familiar: {resultados['salio_familiar']}; por texto cambian "
        f"{p['menciones_que_cambian']}. Inferencia {t['segundos_inferencia']} s, análisis "
        f"{t['segundos_analisis']} s."
    )


def tabla_discrepancias(resultados: dict[str, Any]) -> str:
    """Conteo de la muestra por categoría y tipo de discrepancia, en Markdown."""
    tipos = [f"{e}->{o}" for e, o in DISCREPANCIAS]
    conteo = Counter(
        (d["categoria"], f"{d['esperado']}->{d['obtenido']}") for d in resultados["discrepancias"]
    )
    lineas = [
        "| Categoría | " + " | ".join(tipos) + " | Total |",
        "|---|" + "---|" * (len(tipos) + 1),
    ]
    for categoria, descripcion in CATEGORIAS.items():
        valores = [conteo[(categoria, t)] for t in tipos]
        if any(valores):
            lineas.append(
                f"| {descripcion} | " + " | ".join(map(str, valores)) + f" | {sum(valores)} |"
            )
    return "\n".join(lineas)
