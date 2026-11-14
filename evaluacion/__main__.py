"""Línea de comandos de la evaluación.

Uso:

    uv run python -m evaluacion descargar [--corpus distemist|codiesp|cantemist]
        [--destino data/distemist | data/codiesp] [--destino-catalogo data/codiesp_codes]
    uv run python -m evaluacion distemist [--raiz data/distemist] [--particion test|train]
        [--limite N] [--variantes crudo pipeline ...] [--salida ARCHIVO.json|CARPETA]
        [--detalle ARCHIVO.json|CARPETA] [--cache data/cache] [--sin-tiempos]
    uv run python -m evaluacion sintetico [--notas evaluacion/sintetico/notas.jsonl]
        [--salida evaluacion/resultados/sintetico.json] [--cache data/cache]
    uv run python -m evaluacion nubes [--raiz data/nubes]
        [--particion desarrollo|reservada|todo | --muestras SAMPLE-001 ...] [--limite N]
        [--salida ARCHIVO.json|CARPETA] [--detalle ARCHIVO.json|CARPETA]
        [--cache data/cache] [--lote 256] [--batch 32]
    uv run python -m evaluacion robustez [--raiz data/distemist]
        [--notas evaluacion/sintetico/notas.jsonl] [--solo distemist|sintetico]
        [--semillas 0 1 2 3 4 5] [--limite N] [--salida ARCHIVO.json] [--cache data/cache]
        [--correcciones [ARCHIVO.json]]
    uv run python -m evaluacion codiesp [--raiz data/codiesp] [--particiones train dev]
        [--salida evaluacion/resultados/codiesp_reglas_desarrollo.json]
        [--detalle data/evaluacion/codiesp_reglas_errores.json]
    uv run python -m evaluacion normalizacion --modelo sapbert|e5_base|e5_small|bsc_bio_ehr
        [--raiz data/codiesp] [--catalogo data/codiesp_codes] [--indices data/normalizacion]
        [--salida ARCHIVO.json]
    uv run python -m evaluacion normalizacion --tabla   # compara los JSON ya medidos
    uv run python -m evaluacion sinonimos [--modelo sapbert] [--raiz data/codiesp]
        [--catalogo data/codiesp_codes] [--indices data/normalizacion] [--pliegues 5]
        [--salida ARCHIVO.json]
    uv run python -m evaluacion prueba [--modelo sapbert]   # CodiEsp prueba, una sola vez (#134)
    uv run python -m evaluacion dos-etapas [--modelo sapbert] [--k 20] [--pliegues 5]
        [--raiz data/codiesp] [--catalogo data/codiesp_codes] [--indices data/normalizacion]
        [--salida evaluacion/resultados/dos_etapas_arnes.json]
    uv run python -m evaluacion votos [--modelo sapbert] [--pliegues 5]   # #150, solo train
    uv run python -m evaluacion clasificador [--modelo sapbert] [--pliegues 5]   # #151
    uv run python -m evaluacion reordenador --reordenador mmarco|bge_base|bge_m3   # #152
    uv run python -m evaluacion fuentes [--modelo sapbert]   # #153, CANTEMIST y DisTEMIST
    uv run python -m evaluacion final [--modelo sapbert]   # #154, desarrollo una sola vez
    uv run python -m evaluacion artefactos   # #164, índice y clasificador de la librería
    uv run python -m evaluacion extremo   # #167, reglas contra normalizador por documento
    uv run python -m evaluacion sintetico --normalizador --salida ...   # #167
    uv run python -m evaluacion sintetico --hibrido --salida ...        # #180
"""

import argparse
import json
import subprocess
from pathlib import Path

from evaluacion import (
    artefactos,
    charlson_codiesp,
    clasificador,
    codiesp,
    contexto_charlson,
    dos_etapas,
    extremo_codiesp,
    fuentes_sinonimos,
    medicion_final,
    normalizacion_codiesp,
    normalizador,
    nubes,
    prueba_codiesp,
    reordenador,
    robustez,
    sinonimos_codiesp,
    sinonimos_publicos,
    votos,
)
from evaluacion.cache import RAIZ_CACHE, ExtractorConCache
from evaluacion.corpus import RAIZ_DATOS, descargar, leer_con_reporte, leer_particion
from evaluacion.distemist import (
    REPLICAS_BOOTSTRAP,
    VARIANTES,
    evaluar,
    guardar_json,
    resumen_tiempos,
    ruta_detalle,
    ruta_resultados,
    tabla,
)
from ner_clinic.extraccion import ExtractorHF


def _distemist(args: argparse.Namespace) -> None:
    documentos = leer_particion(args.raiz, args.particion)
    if args.limite is not None:
        documentos = documentos[: args.limite]

    modelo = ExtractorHF()
    con_cache = ExtractorConCache(modelo, carpeta=args.cache)
    resultados, detalle = evaluar(
        documentos,
        con_cache,
        args.particion,
        variantes=args.variantes,
        extractor_tiempos=None if args.sin_tiempos else modelo,
        replicas=args.bootstrap,
        limite=args.limite,
        cache=con_cache,
    )

    salida = ruta_resultados(args.salida, args.particion)
    archivo_detalle = ruta_detalle(args.detalle, args.particion)
    guardar_json(resultados, salida)
    guardar_json(detalle, archivo_detalle)

    print(f"DisTEMIST {args.particion}: {resultados['documentos']} documentos")
    print(tabla(resultados))
    print(resumen_tiempos(resultados))
    cache = resultados["cache"]
    print(
        f"Variantes en {resultados['segundos_variantes']} s; caché: {cache['aciertos']} "
        f"fragmentos leídos, {cache['fallos']} inferidos."
    )
    print(f"Resultados en {salida}; TP/FP/FN con texto en {archivo_detalle}")


def _sintetico(args: argparse.Namespace) -> None:
    if args.salida.exists():
        raise SystemExit(
            f"{args.salida} ya existe: el set es reservado y se mide una sola vez. "
            "Si se vuelve a medir, use otra --salida (p. ej. sintetico_v2.json)."
        )
    notas = contexto_charlson.leer_notas(args.notas)
    con_cache = ExtractorConCache(ExtractorHF(), carpeta=args.cache)
    normalizador = None
    if args.normalizador or args.hibrido:
        from ner_clinic.normalizacion import artefactos as libreria

        normalizador = libreria.cargar()
    from evaluacion.distemist import VARIANTES

    if args.variante not in VARIANTES:
        raise SystemExit(f"Variante desconocida: {args.variante}. Opciones: {list(VARIANTES)}")
    variante = VARIANTES[args.variante]
    if variante.crudo or not variante.deduplicar:
        raise SystemExit(f"La variante {args.variante} no cambia la configuración de extracción.")
    resultados = contexto_charlson.evaluar(
        notas,
        con_cache,
        ruta_notas=args.notas,
        cache=con_cache,
        normalizador=normalizador,
        configuracion=variante.configuracion,
        modo_mapeo="hibrido" if args.hibrido else "normalizador",
    )
    guardar_json(resultados, args.salida)

    print(f"Set sintético: {len(notas)} notas")
    print(contexto_charlson.tabla(resultados))
    print(f"Resultados en {args.salida}")


def _nubes(args: argparse.Namespace) -> None:
    muestras = nubes.muestras_de(args.particion, args.muestras)
    particion = args.particion or ("todo" if args.muestras is None else None)
    corpus = nubes.descargar(args.raiz)
    frases = nubes.leer(corpus, muestras)
    if args.limite is not None:
        frases = frases[: args.limite]

    salida = nubes.ruta_resultados(args.salida)
    anterior = json.loads(salida.read_text(encoding="utf-8")) if salida.exists() else None
    con_cache = ExtractorConCache(ExtractorHF(batch_size=args.batch), carpeta=args.cache)
    resultados, detalle = nubes.evaluar(
        frases,
        con_cache,
        muestras,
        cache=con_cache,
        lote=args.lote,
        anterior=anterior,
        limite=args.limite,
        particion=particion,
    )
    archivo_detalle = nubes.ruta_detalle(args.detalle)
    guardar_json(resultados, salida)
    guardar_json(detalle, archivo_detalle)

    nombre = f"partición {particion}: " if particion else ""
    print(f"NUBes {nombre}{', '.join(muestras)}: {len(frases)} frases")
    print(nubes.tabla_matriz(resultados))
    print(nubes.resumen(resultados))
    sin_clasificar = sum(d["categoria"] is None for d in resultados["discrepancias"])
    print(
        f"Resultados en {salida}; menciones con texto en {archivo_detalle}; "
        f"{sin_clasificar} discrepancias de la muestra sin clasificar."
    )


def _robustez(args: argparse.Namespace) -> None:
    if args.correcciones is not None:
        # El listado trae texto del corpus: solo se escribe en data/, que no se versiona.
        if not args.correcciones.resolve().is_relative_to(Path("data").resolve()):
            raise SystemExit(
                f"{args.correcciones}: el listado tiene texto del corpus y va en data/."
            )
        documentos = leer_particion(args.raiz, "test")
        if args.limite is not None:
            documentos = documentos[: args.limite]
        con_cache = ExtractorConCache(ExtractorHF(), carpeta=args.cache)
        filas = robustez.correcciones_en(documentos, con_cache)
        guardar_json({"documentos": len(documentos), "entidades": filas}, args.correcciones)
        print(f"{len(filas)} entidades con categoría por corrección en {args.correcciones}")
        return
    if args.salida.exists():
        raise SystemExit(
            f"{args.salida} ya existe y no se sobrescribe: use otra --salida "
            "(p. ej. evaluacion/resultados/robustez_despues.json)."
        )
    documentos = None
    if args.solo != "sintetico":
        documentos = leer_particion(args.raiz, "test")
        if args.limite is not None:
            documentos = documentos[: args.limite]
    notas = None if args.solo == "distemist" else contexto_charlson.leer_notas(args.notas)

    con_cache = ExtractorConCache(ExtractorHF(), carpeta=args.cache)
    resultados = robustez.evaluar(
        con_cache,
        documentos=documentos,
        notas=notas,
        ruta_notas=None if notas is None else args.notas,
        semillas=args.semillas,
        replicas=args.bootstrap,
        limite=args.limite,
        cache=con_cache,
    )
    guardar_json(resultados, args.salida)

    print(robustez.tabla(resultados))
    print(f"Resultados en {args.salida}")


def _descargar_codiesp(args: argparse.Namespace) -> None:
    corpus = codiesp.descargar_corpus(args.destino or codiesp.RAIZ_CODIESP)
    print(f"Corpus en {corpus}")
    for particion in codiesp.PARTICIONES:
        lectura = codiesp.leer_con_reporte(corpus, particion)
        diagnosticos = sum(len(d.menciones_diagnostico) for d in lectura.documentos)
        print(
            f"  {particion}: {len(lectura.documentos)} documentos, {diagnosticos} menciones "
            f"de diagnóstico, {len(lectura.errados)} menciones que no cuadran"
        )
    catalogo = codiesp.descargar_catalogo(args.destino_catalogo)
    print(f"Catálogo en {catalogo}: {len(codiesp.leer_catalogo(catalogo))} códigos")


def _codiesp(args: argparse.Namespace) -> None:
    # El detalle trae texto del corpus: solo se escribe en data/, que no se versiona.
    if not args.detalle.resolve().is_relative_to(Path("data").resolve()):
        raise SystemExit(f"{args.detalle}: el detalle tiene texto del corpus y va en data/.")
    por_particion = {p: codiesp.leer_particion(args.raiz, p) for p in args.particiones}
    resultados, detalle = charlson_codiesp.evaluar(por_particion, replicas=args.bootstrap)
    guardar_json(resultados, args.salida)
    guardar_json(detalle, args.detalle)

    print(charlson_codiesp.tabla(resultados))
    print()
    print(charlson_codiesp.tabla_categorias(resultados))
    print(f"Resultados en {args.salida}; errores con texto en {args.detalle}")


def _bytes_modelo(modelo: normalizador.ModeloEmbeddings) -> int | None:
    """Tamaño de los pesos del modelo en la caché de Hugging Face."""
    from huggingface_hub import try_to_load_from_cache

    ruta = try_to_load_from_cache(modelo.nombre, "model.safetensors", revision=modelo.revision)
    return Path(ruta).stat().st_size if isinstance(ruta, str) else None


def _normalizacion(args: argparse.Namespace) -> None:
    if args.tabla:
        medidos = [
            json.loads(normalizacion_codiesp.ruta_resultados(m).read_text(encoding="utf-8"))
            for m in normalizador.CANDIDATOS
            if normalizacion_codiesp.ruta_resultados(m).is_file()
        ]
        reglas = None
        if charlson_codiesp.RUTA_RESULTADOS.is_file():
            reglas = json.loads(charlson_codiesp.RUTA_RESULTADOS.read_text(encoding="utf-8"))
        print(normalizacion_codiesp.tabla(medidos, reglas))
        return
    if args.modelo is None:
        raise SystemExit("Falta --modelo (o --tabla).")
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    por_particion = {p: codiesp.leer_particion(args.raiz, p) for p in args.particiones}

    bytes_modelo = _bytes_modelo(modelo)
    codificador = normalizador.CodificadorHF(modelo)
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    resultados = normalizacion_codiesp.evaluar(
        args.modelo,
        normalizador.Normalizador(indice, codificador),
        por_particion,
        meta,
        bytes_modelo=bytes_modelo,
        replicas=args.bootstrap,
    )
    salida = args.salida or normalizacion_codiesp.ruta_resultados(args.modelo)
    guardar_json(resultados, salida)
    print(normalizacion_codiesp.tabla([resultados]))
    print(f"Resultados en {salida}")


def _sinonimos(args: argparse.Namespace) -> None:
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    por_particion = {p: codiesp.leer_particion(args.raiz, p) for p in ("train", "dev")}
    # De prueba solo los nombres de los archivos, para verificar que no entra al índice
    ids_prueba = codiesp.ids_de(args.raiz, "test")

    codificador = normalizador.CodificadorHF(modelo)
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    resultados = sinonimos_codiesp.evaluar(
        args.modelo,
        indice,
        codificador,
        por_particion,
        meta,
        ids_prueba,
        replicas=args.bootstrap,
        pliegues=args.pliegues,
    )
    salida = args.salida or sinonimos_codiesp.ruta_resultados(args.modelo)
    guardar_json(resultados, salida)
    print(sinonimos_codiesp.tabla(resultados))
    print(f"Resultados en {salida}")


def _dos_etapas(args: argparse.Namespace) -> None:
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    desarrollo = codiesp.leer_particion(args.raiz, "dev")
    ids_prueba = codiesp.ids_de(args.raiz, "test")

    codificador = dos_etapas.CodificadorEnDisco(
        lambda: normalizador.CodificadorHF(modelo),
        dos_etapas.ruta_vectores(modelo, args.indices),
    )
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    try:
        resultados = dos_etapas.evaluar(
            args.modelo,
            indice,
            codificador,
            entrenamiento,
            desarrollo,
            meta,
            ids_prueba,
            k=args.k,
            pliegues=args.pliegues,
            replicas=args.bootstrap,
        )
    finally:
        codificador.guardar()
    guardar_json(resultados, args.salida)
    print(dos_etapas.tabla(resultados))
    print(f"Resultados en {args.salida}")


def _reordenador(args: argparse.Namespace) -> None:
    modelo = normalizador.CANDIDATOS[args.modelo]
    reordenante = reordenador.CANDIDATOS[args.reordenador]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    codificador = dos_etapas.CodificadorEnDisco(
        lambda: normalizador.CodificadorHF(modelo),
        dos_etapas.ruta_vectores(modelo, args.indices),
    )
    memoria_antes = reordenador.memoria_residente_mib()
    modelo_real = reordenador.ReordenadorHF(reordenante)
    modelo_real([("hipertensión", "Hipertensión esencial")])  # los pesos ya en memoria
    memoria_modelo = reordenador.memoria_residente_mib() - memoria_antes
    puntuar = reordenador.PuntuacionesEnDisco(
        lambda: modelo_real, reordenador.ruta_puntuaciones(reordenante, args.indices)
    )
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    try:
        resultados = reordenador.evaluar(
            args.reordenador,
            args.modelo,
            indice,
            codificador,
            puntuar,
            entrenamiento,
            meta,
            pliegues=args.pliegues,
            replicas=args.bootstrap,
            modelo_real=modelo_real,
        )
    finally:
        codificador.guardar()
        puntuar.guardar()
    resultados["costo"]["memoria_modelo_mib"] = round(memoria_modelo)
    salida = args.salida or reordenador.ruta_resultados(args.reordenador)
    guardar_json(resultados, salida)
    print(reordenador.tabla(resultados))
    print(f"Resultados en {salida}")


def _leer_fuentes(args: argparse.Namespace, entrenamiento, desarrollo):
    """CANTEMIST y DisTEMIST sin los documentos de CodiEsp desarrollo y prueba (#153).
    Los textos de desarrollo solo se usan para buscar casos compartidos con CANTEMIST."""
    ids = {p: codiesp.ids_de(args.raiz, p) for p in codiesp.PARTICIONES}
    cantemist, descartes_c = sinonimos_publicos.leer_cantemist(args.cantemist)
    distemist_, descartes_d = sinonimos_publicos.leer_distemist(args.distemist)
    compartidos = sinonimos_publicos.documentos_compartidos(
        sinonimos_publicos.textos_cantemist(args.cantemist),
        {d.id: d.texto for d in [*entrenamiento, *desarrollo]},
    )
    fuentes, reporte = fuentes_sinonimos.preparar(
        cantemist,
        distemist_,
        {"cantemist": descartes_c, "distemist": descartes_d},
        ids,
        compartidos,
    )
    return fuentes, reporte, ids


def _fuentes(args: argparse.Namespace) -> None:
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    desarrollo = codiesp.leer_particion(args.raiz, "dev")  # solo para buscar casos compartidos
    fuentes, reporte, _ = _leer_fuentes(args, entrenamiento, desarrollo)
    codificador = dos_etapas.CodificadorEnDisco(
        lambda: normalizador.CodificadorHF(modelo),
        dos_etapas.ruta_vectores(modelo, args.indices),
    )
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    try:
        resultados = fuentes_sinonimos.evaluar(
            args.modelo,
            indice,
            codificador,
            entrenamiento,
            meta,
            fuentes,
            reporte,
            pliegues=args.pliegues,
            replicas=args.bootstrap,
        )
    finally:
        codificador.guardar()
    guardar_json(resultados, args.salida)
    print(fuentes_sinonimos.tabla(resultados))
    print(f"Resultados en {args.salida}")


def _final(args: argparse.Namespace) -> None:
    salida = args.salida or medicion_final.RUTA_RESULTADOS
    if salida.exists():
        raise SystemExit(
            f"{salida} ya existe: desarrollo se mide una sola vez y no se sobrescribe."
        )
    cambios = _cambios_sin_commitear()
    if cambios:
        raise SystemExit(
            "La medición final se corre con el árbol limpio; hay cambios sin commitear:\n"
            + "\n".join(cambios)
        )
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    desarrollo = codiesp.leer_particion(args.raiz, "dev")
    fuentes, reporte, ids = _leer_fuentes(args, entrenamiento, desarrollo)
    perdidas = None
    if args.arnes.is_file():
        arnes = json.loads(args.arnes.read_text(encoding="utf-8"))
        perdidas = arnes["desarrollo"]["errores"]["detalle"]
    codificador = dos_etapas.CodificadorEnDisco(
        lambda: normalizador.CodificadorHF(modelo),
        dos_etapas.ruta_vectores(modelo, args.indices),
    )
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    try:
        resultados = medicion_final.evaluar(
            args.modelo,
            indice,
            codificador,
            entrenamiento,
            desarrollo,
            meta,
            fuentes,
            reporte,
            ids["test"],
            perdidas,
            pliegues=args.pliegues,
            replicas=args.bootstrap,
        )
    finally:
        codificador.guardar()
    guardar_json(resultados, salida)
    print(medicion_final.tabla(resultados))
    print(f"Resultados en {salida}")


def _artefactos(args: argparse.Namespace) -> None:
    """Índice final y clasificador de la librería, con la comprobación contra #154."""
    from ner_clinic.normalizacion import artefactos as libreria

    cambios = _cambios_sin_commitear()
    if cambios:
        raise SystemExit(
            "Los artefactos se construyen con el árbol limpio (el commit queda en el JSON); "
            "hay cambios sin commitear:\n" + "\n".join(cambios)
        )
    modelo = normalizador.SAPBERT
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    desarrollo = codiesp.leer_particion(args.raiz, "dev")
    prohibidos = {p: codiesp.ids_de(args.raiz, p) for p in ("dev", "test")}
    codificador = dos_etapas.CodificadorEnDisco(
        lambda: normalizador.CodificadorHF(modelo),
        dos_etapas.ruta_vectores(modelo, args.indices),
    )
    indice, _ = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    try:
        final, meta_indice, clasificador_ = artefactos.construir(
            indice, codificador, entrenamiento, prohibidos, args.pliegues
        )
        libreria.guardar_indice(final, args.indice_final, meta_indice)
        guardar_json(clasificador_, args.clasificador)
        segunda = libreria.cargar(args.indice_final, args.clasificador, lambda: codificador)
        micro_154, eleccion_154 = artefactos.de_154(args.medicion_154)
        comprobacion = artefactos.comprobar(segunda, desarrollo, micro_154)
    finally:
        codificador.guardar()
    # Lo mismo con SapBERT en vivo, sin los vectores en caché de #154: es lo que corre la
    # librería. Se reporta; la que tiene que coincidir es la de arriba.
    en_vivo = libreria.cargar(
        args.indice_final, args.clasificador, lambda: normalizador.CodificadorHF(modelo)
    )
    comprobacion["en_vivo"] = artefactos.comprobar(en_vivo, desarrollo, micro_154)
    eleccion = clasificador_["meta"]["eleccion"]
    comprobacion["eleccion_154"] = (
        None if eleccion_154 is None else {k: eleccion_154[k] for k in ("penalizacion", "umbral")}
    )
    comprobacion["eleccion_coincide"] = eleccion_154 is not None and all(
        eleccion[k] == eleccion_154[k] for k in ("penalizacion", "umbral")
    )
    clasificador_["meta"]["comprobacion_desarrollo"] = comprobacion
    guardar_json(clasificador_, args.clasificador)
    print(
        f"Índice: {meta_indice['filas']:,} filas en {args.indice_final}\n"
        f"Clasificador: penalización {eleccion['penalizacion']}, umbral {eleccion['umbral']} "
        f"en {args.clasificador}\n"
        f"Comprobación en desarrollo: F1 {comprobacion['f1_desarrollo']} "
        f"(#154: {comprobacion['f1_154']}); con SapBERT en vivo: "
        f"{comprobacion['en_vivo']['f1_desarrollo']}"
    )
    if not (comprobacion["coincide"] and comprobacion["eleccion_coincide"]):
        raise SystemExit("Los artefactos no reproducen el método medido en #154.")


def _extremo(args: argparse.Namespace) -> None:
    """Reglas, normalizador e híbrido de punta a punta en CodiEsp desarrollo (#167, #180)."""
    from ner_clinic.normalizacion import artefactos as libreria

    salida = args.salida or extremo_codiesp.RUTA_RESULTADOS
    if salida.exists():
        raise SystemExit(f"{salida} ya existe: se mide una sola vez y no se sobrescribe.")
    cambios = _cambios_sin_commitear()
    if cambios:
        raise SystemExit(
            "La medición se corre con el árbol limpio; hay cambios sin commitear:\n"
            + "\n".join(cambios)
        )
    desarrollo = codiesp.leer_particion(args.raiz, "dev")
    normalizador = libreria.cargar(args.indice_final)
    meta = libreria.leer_clasificador()["meta"]
    con_cache = ExtractorConCache(ExtractorHF(), carpeta=args.cache)
    resultados = extremo_codiesp.evaluar(
        desarrollo,
        con_cache,
        normalizador,
        replicas=args.bootstrap,
        meta_normalizador={
            "huella_indice": meta["huella_indice"],
            "commit_clasificador": meta["entorno"]["commit"],
            "umbral": meta["eleccion"]["umbral"],
        },
    )
    guardar_json(resultados, salida)
    print(extremo_codiesp.tabla(resultados))
    print(f"Resultados en {salida}")


def _clasificador(args: argparse.Namespace) -> None:
    _segunda_etapa(args, clasificador.evaluar, clasificador.tabla)


def _votos(args: argparse.Namespace) -> None:
    _segunda_etapa(args, votos.evaluar, votos.tabla)


def _segunda_etapa(args: argparse.Namespace, evaluar, tabla) -> None:
    """Una segunda etapa elegida por validación cruzada anidada en entrenamiento."""
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    codificador = dos_etapas.CodificadorEnDisco(
        lambda: normalizador.CodificadorHF(modelo),
        dos_etapas.ruta_vectores(modelo, args.indices),
    )
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    try:
        resultados = evaluar(
            args.modelo,
            indice,
            codificador,
            entrenamiento,
            meta,
            pliegues=args.pliegues,
            replicas=args.bootstrap,
        )
    finally:
        codificador.guardar()
    guardar_json(resultados, args.salida)
    print(tabla(resultados))
    print(f"Resultados en {args.salida}")


def _cambios_sin_commitear() -> list[str]:
    salida = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return salida.splitlines()


def _prueba(args: argparse.Namespace) -> None:
    salida = args.salida or prueba_codiesp.RUTA_RESULTADOS
    if salida.exists():
        raise SystemExit(
            f"{salida} ya existe: CodiEsp prueba se mide una sola vez y no se sobrescribe."
        )
    cambios = _cambios_sin_commitear()
    if cambios:
        raise SystemExit(
            "La medición final se corre con el árbol limpio; hay cambios sin commitear:\n"
            + "\n".join(cambios)
        )
    modelo = normalizador.CANDIDATOS[args.modelo]
    catalogo = codiesp.leer_catalogo(args.catalogo)
    entrenamiento = codiesp.leer_particion(args.raiz, "train")
    prueba = codiesp.leer_particion(args.raiz, "test")

    codificador = normalizador.CodificadorHF(modelo)
    indice, meta = normalizador.indice_en_disco(
        catalogo, modelo, lambda: codificador, raiz=args.indices
    )
    resultados = prueba_codiesp.evaluar(
        args.modelo, indice, codificador, entrenamiento, prueba, meta, replicas=args.bootstrap
    )
    guardar_json(resultados, salida)
    print(prueba_codiesp.tabla(resultados))
    print(f"Resultados en {salida}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluacion", description=__doc__)
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    p_descargar = subcomandos.add_parser(
        "descargar",
        help="Baja de Zenodo DisTEMIST, CodiEsp y su catálogo CIE-10, o CANTEMIST, verifica el MD5 "
        "y lo descomprime.",
    )
    p_descargar.add_argument(
        "--corpus", choices=["distemist", "codiesp", "cantemist"], default="distemist"
    )
    p_descargar.add_argument(
        "--destino", type=Path, help="Por defecto, data/distemist o data/codiesp."
    )
    p_descargar.add_argument(
        "--destino-catalogo",
        type=Path,
        default=codiesp.RAIZ_CATALOGO,
        help="Solo con --corpus codiesp; con distemist se ignora.",
    )

    p_distemist = subcomandos.add_parser(
        "distemist",
        help="Corre el modelo crudo, el pipeline y la ablación sobre una partición.",
    )
    p_distemist.add_argument("--raiz", type=Path, default=RAIZ_DATOS, help="Carpeta del corpus.")
    p_distemist.add_argument("--particion", choices=("test", "train"), default="test")
    p_distemist.add_argument(
        "--limite", type=int, default=None, help="Solo los primeros N documentos."
    )
    p_distemist.add_argument(
        "--variantes",
        nargs="+",
        choices=list(VARIANTES),
        default=None,
        help="Por defecto, todas en test; crudo y pipeline en train.",
    )
    p_distemist.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="JSON de resultados (o carpeta). Por defecto evaluacion/resultados/.",
    )
    p_distemist.add_argument(
        "--detalle",
        type=Path,
        default=None,
        help="JSON con TP/FP/FN y texto (o carpeta). Por defecto data/evaluacion/.",
    )
    p_distemist.add_argument("--cache", type=Path, default=RAIZ_CACHE)
    p_distemist.add_argument(
        "--sin-tiempos", action="store_true", help="No mide el pipeline sin caché."
    )
    p_distemist.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_sintetico = subcomandos.add_parser(
        "sintetico",
        help="Mide estado, Charlson, índice y registrado sobre el set sintético reservado.",
    )
    p_sintetico.add_argument("--notas", type=Path, default=contexto_charlson.RUTA_NOTAS)
    p_sintetico.add_argument(
        "--salida",
        type=Path,
        default=contexto_charlson.RUTA_RESULTADOS,
        help="JSON de resultados; no se sobrescribe si ya existe.",
    )
    p_sintetico.add_argument("--cache", type=Path, default=RAIZ_CACHE)
    mapeo_sintetico = p_sintetico.add_mutually_exclusive_group()
    mapeo_sintetico.add_argument(
        "--normalizador",
        action="store_true",
        help="La categoría de Charlson la da solo el normalizador (#167), no las reglas.",
    )
    mapeo_sintetico.add_argument(
        "--hibrido",
        action="store_true",
        help="Primero las reglas y, para lo que no mapean, el normalizador (#180).",
    )
    p_sintetico.add_argument(
        "--variante",
        default="pipeline",
        help="Variante de la ablación de DisTEMIST cuya configuración de extracción se usa "
        "(p. ej. fusion_barreras o sin_fusion, #55).",
    )

    p_nubes = subcomandos.add_parser(
        "nubes",
        help="Mide la negación y la incertidumbre del pipeline en NUBes (lo descarga si falta).",
    )
    p_nubes.add_argument("--raiz", type=Path, default=nubes.RAIZ_NUBES, help="Carpeta del corpus.")
    seleccion = p_nubes.add_mutually_exclusive_group()
    seleccion.add_argument(
        "--particion",
        choices=list(nubes.PARTICIONES),
        default=None,
        help="Desarrollo (muestras pares), reservada (impares) o todo. Por defecto, todo.",
    )
    seleccion.add_argument(
        "--muestras",
        nargs="+",
        choices=nubes.MUESTRAS,
        default=None,
        help="Muestras sueltas, en vez de una partición.",
    )
    p_nubes.add_argument("--limite", type=int, default=None, help="Solo las primeras N frases.")
    p_nubes.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="JSON de resultados (o carpeta). Por defecto evaluacion/resultados/nubes.json.",
    )
    p_nubes.add_argument(
        "--detalle",
        type=Path,
        default=None,
        help="JSON con las menciones y su texto (o carpeta). Por defecto data/evaluacion/.",
    )
    p_nubes.add_argument("--cache", type=Path, default=RAIZ_CACHE)
    p_nubes.add_argument(
        "--lote", type=int, default=nubes.LOTE_INFERENCIA, help="Fragmentos por llamada."
    )
    p_nubes.add_argument("--batch", type=int, default=32, help="batch_size del modelo.")

    p_robustez = subcomandos.add_parser(
        "robustez",
        help="Prueba de estrés: DisTEMIST y set sintético en minúscula, sin tildes y con errores.",
    )
    p_robustez.add_argument("--raiz", type=Path, default=RAIZ_DATOS, help="Carpeta de DisTEMIST.")
    p_robustez.add_argument("--notas", type=Path, default=contexto_charlson.RUTA_NOTAS)
    p_robustez.add_argument(
        "--solo", choices=("distemist", "sintetico"), default=None, help="Por defecto, los dos."
    )
    p_robustez.add_argument(
        "--semillas",
        type=int,
        nargs="+",
        default=[robustez.SEMILLA_DESARROLLO, *robustez.SEMILLAS_REPORTE],
        help="Semillas de los errores en el set sintético (0 = desarrollo, 1-5 = reporte).",
    )
    p_robustez.add_argument(
        "--limite", type=int, default=None, help="Solo los primeros N documentos de DisTEMIST."
    )
    p_robustez.add_argument(
        "--salida",
        type=Path,
        default=robustez.RUTA_RESULTADOS,
        help="JSON de resultados; no se sobrescribe si ya existe.",
    )
    p_robustez.add_argument(
        "--correcciones",
        type=Path,
        nargs="?",
        const=robustez.RUTA_CORRECCIONES,
        default=None,
        help="Solo lista las entidades de DisTEMIST prueba con categoría por corrección "
        "(con texto del corpus: solo bajo data/; por defecto data/evaluacion/). Ignora "
        "--salida, --solo, --semillas y --bootstrap.",
    )
    p_robustez.add_argument("--cache", type=Path, default=RAIZ_CACHE)
    p_robustez.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_codiesp = subcomandos.add_parser(
        "codiesp",
        help="Línea base de las reglas de Charlson en CodiEsp (solo entrenamiento y desarrollo).",
    )
    p_codiesp.add_argument(
        "--raiz", type=Path, default=codiesp.RAIZ_CODIESP, help="Carpeta de CodiEsp."
    )
    p_codiesp.add_argument(
        "--particiones",
        nargs="+",
        choices=charlson_codiesp.PARTICIONES_DESARROLLO,
        default=list(charlson_codiesp.PARTICIONES_DESARROLLO),
        help="La de prueba no se acepta: queda para la medición final (#134).",
    )
    p_codiesp.add_argument("--salida", type=Path, default=charlson_codiesp.RUTA_RESULTADOS)
    p_codiesp.add_argument(
        "--detalle",
        type=Path,
        default=charlson_codiesp.RUTA_DETALLE,
        help="JSON con los errores y su texto; solo bajo data/.",
    )
    p_codiesp.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_normalizacion = subcomandos.add_parser(
        "normalizacion",
        help="Normalización por embeddings en CodiEsp entrenamiento y desarrollo (#131).",
    )
    p_normalizacion.add_argument("--modelo", choices=list(normalizador.CANDIDATOS))
    p_normalizacion.add_argument(
        "--tabla", action="store_true", help="Solo compara los resultados ya medidos."
    )
    p_normalizacion.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_normalizacion.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_normalizacion.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_normalizacion.add_argument(
        "--particiones",
        nargs="+",
        choices=charlson_codiesp.PARTICIONES_DESARROLLO,
        default=list(charlson_codiesp.PARTICIONES_DESARROLLO),
    )
    p_normalizacion.add_argument("--salida", type=Path, default=None)
    p_normalizacion.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_sinonimos = subcomandos.add_parser(
        "sinonimos",
        help="Sinónimos de CodiEsp entrenamiento en el índice, medidos en desarrollo (#132).",
    )
    p_sinonimos.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_sinonimos.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_sinonimos.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_sinonimos.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_sinonimos.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_sinonimos.add_argument("--salida", type=Path, default=None)
    p_sinonimos.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_prueba = subcomandos.add_parser(
        "prueba",
        help="Medición final en CodiEsp prueba, una sola vez y con el árbol limpio (#134).",
    )
    p_prueba.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_prueba.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_prueba.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_prueba.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_prueba.add_argument("--salida", type=Path, default=None)
    p_prueba.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_dos_etapas = subcomandos.add_parser(
        "dos-etapas",
        help="Arnés de la decisión en dos etapas: top-k por pliegue, comparación pareada, "
        "causas de las perdidas y techo (#149).",
    )
    p_dos_etapas.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_dos_etapas.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_dos_etapas.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_dos_etapas.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_dos_etapas.add_argument("--k", type=int, default=dos_etapas.K)
    p_dos_etapas.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_dos_etapas.add_argument("--salida", type=Path, default=dos_etapas.RUTA_RESULTADOS)
    p_dos_etapas.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_votos = subcomandos.add_parser(
        "votos",
        help="Segunda etapa por votos (categoría y jerarquía CIE-10), elegida por validación "
        "cruzada anidada en entrenamiento (#150).",
    )
    p_votos.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_votos.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_votos.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_votos.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_votos.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_votos.add_argument("--salida", type=Path, default=votos.RUTA_RESULTADOS)
    p_votos.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_clasificador = subcomandos.add_parser(
        "clasificador",
        help="Segunda etapa entrenada: regresión logística sobre el top-k, elegida por "
        "validación cruzada anidada en entrenamiento (#151).",
    )
    p_clasificador.add_argument(
        "--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert"
    )
    p_clasificador.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_clasificador.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_clasificador.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_clasificador.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_clasificador.add_argument("--salida", type=Path, default=clasificador.RUTA_RESULTADOS)
    p_clasificador.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_reordenador = subcomandos.add_parser(
        "reordenador",
        help="Segunda etapa con un reordenador local, solo y como atributos del clasificador "
        "(#152).",
    )
    p_reordenador.add_argument("--reordenador", choices=list(reordenador.CANDIDATOS), required=True)
    p_reordenador.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_reordenador.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_reordenador.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_reordenador.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_reordenador.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_reordenador.add_argument("--salida", type=Path, default=None)
    p_reordenador.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_fuentes = subcomandos.add_parser(
        "fuentes",
        help="Sinónimos de CANTEMIST y DisTEMIST en la validación cruzada anidada (#153).",
    )
    p_fuentes.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_fuentes.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_fuentes.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_fuentes.add_argument("--cantemist", type=Path, default=sinonimos_publicos.RAIZ_CANTEMIST)
    p_fuentes.add_argument("--distemist", type=Path, default=sinonimos_publicos.RAIZ_DISTEMIST)
    p_fuentes.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_fuentes.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_fuentes.add_argument("--salida", type=Path, default=fuentes_sinonimos.RUTA_RESULTADOS)
    p_fuentes.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_final = subcomandos.add_parser(
        "final",
        help="Medición final en CodiEsp desarrollo: el clasificador con DisTEMIST y "
        "CANTEMIST contra el híbrido, una sola vez y con el árbol limpio (#154).",
    )
    p_final.add_argument("--modelo", choices=list(normalizador.CANDIDATOS), default="sapbert")
    p_final.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_final.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_final.add_argument("--cantemist", type=Path, default=sinonimos_publicos.RAIZ_CANTEMIST)
    p_final.add_argument("--distemist", type=Path, default=sinonimos_publicos.RAIZ_DISTEMIST)
    p_final.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_final.add_argument("--arnes", type=Path, default=dos_etapas.RUTA_RESULTADOS)
    p_final.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_final.add_argument("--salida", type=Path, default=None)
    p_final.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    p_artefactos = subcomandos.add_parser(
        "artefactos",
        help="Índice final y clasificador sin fuentes para la librería, con la comprobación "
        "contra #154 (#164).",
    )
    p_artefactos.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_artefactos.add_argument("--catalogo", type=Path, default=codiesp.RAIZ_CATALOGO)
    p_artefactos.add_argument("--indices", type=Path, default=normalizador.RAIZ_INDICES)
    p_artefactos.add_argument("--pliegues", type=int, default=sinonimos_codiesp.PLIEGUES)
    p_artefactos.add_argument("--indice-final", type=Path, default=None)
    p_artefactos.add_argument("--clasificador", type=Path, default=None)
    p_artefactos.add_argument("--medicion-154", type=Path, default=artefactos.RUTA_MEDICION_154)

    p_extremo = subcomandos.add_parser(
        "extremo",
        help="Reglas, normalizador e híbrido de punta a punta en CodiEsp desarrollo, por "
        "documento (#167, #180).",
    )
    p_extremo.add_argument("--raiz", type=Path, default=codiesp.RAIZ_CODIESP)
    p_extremo.add_argument("--indice-final", type=Path, default=None)
    p_extremo.add_argument("--cache", type=Path, default=RAIZ_CACHE)
    p_extremo.add_argument("--salida", type=Path, default=None)
    p_extremo.add_argument("--bootstrap", type=int, default=REPLICAS_BOOTSTRAP)

    args = parser.parse_args(argv)
    if args.comando == "extremo":
        from ner_clinic.normalizacion import artefactos as libreria

        args.indice_final = args.indice_final or libreria.RUTA_INDICE
        _extremo(args)
    elif args.comando == "artefactos":
        from ner_clinic.normalizacion import artefactos as libreria

        args.indice_final = args.indice_final or libreria.RUTA_INDICE
        args.clasificador = args.clasificador or libreria.RUTA_CLASIFICADOR
        _artefactos(args)
    elif args.comando == "descargar" and args.corpus == "codiesp":
        _descargar_codiesp(args)
    elif args.comando == "descargar" and args.corpus == "cantemist":
        destino = args.destino or sinonimos_publicos.RAIZ_CANTEMIST
        print(f"CANTEMIST en {sinonimos_publicos.descargar_cantemist(destino)}")
    elif args.comando == "descargar":
        corpus = descargar(args.destino or RAIZ_DATOS)
        print(f"Corpus en {corpus}")
        for particion in ("test", "train"):
            lectura = leer_con_reporte(corpus, particion)
            print(
                f"  {particion}: {len(lectura.documentos)} documentos, "
                f"{lectura.total_menciones} menciones, "
                f"{len(lectura.errados)} offsets que no cuadran"
            )
    elif args.comando == "final":
        _final(args)
    elif args.comando == "distemist":
        _distemist(args)
    elif args.comando == "sintetico":
        _sintetico(args)
    elif args.comando == "nubes":
        _nubes(args)
    elif args.comando == "robustez":
        _robustez(args)
    elif args.comando == "codiesp":
        _codiesp(args)
    elif args.comando == "normalizacion":
        _normalizacion(args)
    elif args.comando == "sinonimos":
        _sinonimos(args)
    elif args.comando == "prueba":
        _prueba(args)
    elif args.comando == "dos-etapas":
        _dos_etapas(args)
    elif args.comando == "votos":
        _votos(args)
    elif args.comando == "clasificador":
        _clasificador(args)
    elif args.comando == "reordenador":
        _reordenador(args)
    elif args.comando == "fuentes":
        _fuentes(args)


if __name__ == "__main__":
    main()
