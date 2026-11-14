---
name: normalizacion
description: Experimento de normalización por similitud semántica (mención → CIE-10 por embeddings → Charlson con la tabla de Quan) frente a las reglas hechas a mano, medido en CodiEsp; incluye el filtro de longitud (#56)
status: backlog
created: 2026-11-05T03:06:05Z
---

# PRD: normalizacion

## Executive Summary

Hoy la categoría de Charlson sale de **17 reglas regex escritas a mano**, heredadas de la
tesis, más un corrector de ortografía con 64 palabras clave (#115) y una lista de 30 siglas
(#114). El epic de robustez (#111) mostró el límite de ese enfoque: **una lista hecha a
mano solo cubre lo que alguien previó**.

- En DisTEMIST con la nota en minúscula, solo 13 de las 90 siglas del gold están en la
  lista. El pipeline recupera 3 de las 21 que pierde.
- Las reglas no reconocen "accidente cerebrovascular" ni "ictus" escritos completos
  (#123), ni un tumor "sin metástasis" (#120).
- Cada hueco nuevo es otra regex, otra lista y otro test.

Este subproyecto prueba otra metodología, elegida por Diego el 2026-11-05 (opción B):

1. **Mención → código CIE-10** por similitud de coseno entre embeddings de la mención y de
   las descripciones de un catálogo CIE-10 en español.
2. **CIE-10 → categoría de Charlson** con la tabla publicada de **Quan et al. (2005)**.

Ninguno de los dos pasos se escribe a mano: el catálogo es público y la tabla, publicada.
Es un **experimento**: se mide contra las reglas actuales en datos públicos con CIE-10
anotado (**CodiEsp**) y solo se integra al pipeline si gana. El filtro de longitud (#56)
entra aquí, porque decide qué menciones llegan a la normalización.

## Problem Statement

- **Manualidad.** Mantener las reglas exige prever cada forma de escribir una enfermedad.
  Los huecos aparecen solos (#120, #123, #63) y cada corrección es trabajo manual.
- **Sin medición de CIE-10 ni de Charlson contra un gold.** El reporte lo dice como límite:
  no había un corpus público en español con CIE-10. Las categorías solo se miden en el set
  sintético, que escribimos nosotros y ya no es ciego.
- **Siglas y menciones cortas.** El filtro de longitud descarta 39 de las 85 siglas de
  DisTEMIST que el modelo encuentra (#56). La lista blanca es otra lista a mano.
- **Antecedente.** En la tesis se probaron embeddings para el mapeo CIE-10 y no dieron buenos
  resultados (`Emmbeding.ipynb`). El experimento tiene que mostrar con datos qué cambia: un
  modelo entrenado para enlazar sinónimos médicos y un corpus con CIE-10 para medirlo.

## User Stories

**US1. Auditor o médico en la demo**
Como lector, quiero que una enfermedad escrita de cualquier forma razonable ("accidente
cerebrovascular", "ictus", "ACV", "acv isquémico") tenga su categoría de Charlson, y ver de
dónde salió.
- Criterios de aceptación:
  - Cada entidad con categoría muestra el código CIE-10, su descripción del catálogo y la
    similitud ("I63.9 *Infarto cerebral, no especificado* · similitud 0,91").
  - Si la similitud no llega al umbral, la entidad sale sin categoría, como hoy.

**US2. Quien mantiene el proyecto**
Como mantenedor, quiero que un hueco nuevo se resuelva agregando sinónimos al catálogo o
ajustando un umbral medido, no escribiendo otra regex.

**US3. Quien lee el portafolio**
Como cliente, quiero ver la categoría de Charlson medida contra un gold público de CIE-10,
comparada con las reglas, también con notas mal escritas.
- Criterios de aceptación:
  - `docs/evaluacion.md` trae una sección con P/R/F1 de categoría de Charlson y de CIE-10
    en CodiEsp **prueba**, por método, con IC del 95 %, y la prueba de estrés de #112.

## Functional Requirements

**FR1. Datos y licencias (antes de escribir código del experimento)**
- **CodiEsp** v1.4 (Zenodo, CC BY 4.0): 500 casos de entrenamiento, 250 de desarrollo y 250
  de prueba, con códigos CIE-10-ES por documento. CodiEsp-X trae además el fragmento de texto
  y su posición. Se descarga con su MD5, como DisTEMIST, a `data/` (no se versiona).
- **Catálogo CIE-10 en español** con descripciones: el de CIE-10-ES que acompaña a CodiEsp
  o el del Ministerio de Sanidad. Se busca también la tabla de referencia de MinSalud
  Colombia (SISPRO), que es un pendiente del proyecto. Se documentan la fuente, la versión
  y la licencia de cada uno.
- **Tabla de Quan (2005)**: códigos CIE-10 de cada una de las 17 categorías de Charlson, con
  la cita del artículo. Se cruza contra una implementación publicada (p. ej. el paquete
  `coder` de rOpenSci) y se versiona como datos con tests.
- **Particiones.** CodiEsp entrenamiento y desarrollo para ajustar (umbrales, sinónimos);
  CodiEsp **prueba** solo para la medición final. DisTEMIST entrenamiento para decidir el
  filtro (FR6); DisTEMIST prueba solo para la medición final.

**FR2. Gold de categoría de Charlson**
- Cada código de CodiEsp-X se pasa por la tabla de Quan. Cada mención del gold queda con su
  categoría de Charlson o "ninguna".
- Antes de usarlo se verifica qué codifica CodiEsp. Si solo codifica diagnósticos
  afirmados del paciente y no los negados ni los familiares, eso define qué se compara.
  Queda documentado.

**FR3. Línea base: las reglas actuales en CodiEsp**
- `mapear` con la corrección de #115, sobre el texto de cada mención del gold de CodiEsp-X.
  Se mide P/R/F1 de categoría de Charlson y cuántas menciones reciben categoría.
- Las mismas cifras con las variantes de la prueba de estrés (`evaluacion/robustez.py`).

**FR4. Normalización por embeddings**
- Índice de embeddings de las descripciones del catálogo CIE-10, más los sinónimos que se
  agreguen (FR4b). Búsqueda por similitud de coseno (top-k).
- **Modelos candidatos.** Se comparan al menos dos, y uno debe tener licencia clara para
  redistribuirlo en la imagen Docker pública:
  - SapBERT multilingüe (`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`, base
    XLM-R, unos 0,3B parámetros). Está entrenado para enlazar sinónimos médicos, pero su
    página no declara licencia y se entrenó con UMLS, que tiene licencia propia: hay que
    resolverlo antes de distribuirlo.
  - Un modelo con licencia clara: el codificador biomédico en español del BSC
    (`bsc-bio-ehr-es`, el que ya usa el NER) con *pooling*, o un modelo de embeddings
    multilingüe de propósito general.
- **Umbral de similitud** para aceptar un código, ajustado en CodiEsp desarrollo. Por
  debajo del umbral, la entidad queda sin código.
- **FR4b. Sinónimos.** Las menciones de CodiEsp entrenamiento con su código gold se agregan
  al índice como sinónimos. Es la forma "sin regex" de cubrir siglas y formas coloquiales.
  Se mide con ellos y sin ellos.

**FR5. Híbrido (opción C, si hace falta)**
- Embeddings para el código y, después, las exclusiones de las reglas que el catálogo no
  capta (p. ej. "gestacional", "in situ", "sin metástasis"), solo si FR4 falla en esos casos
  medidos. Cada exclusión que quede se justifica con un caso de desarrollo.

**FR6. Filtro de longitud (#56)**
- Se decide con DisTEMIST **entrenamiento** y con el set sintético: lista blanca, sin
  filtro, o "corta pero normalizable" (una mención corta pasa si la normalización la lleva a
  un código con similitud alta). Es esta última la que reemplazaría la lista blanca de siglas.

**FR7. Medición final y decisión**
- **CodiEsp prueba:** P/R/F1 de categoría de Charlson (micro y por categoría) y exactitud de
  CIE-10 a 3 caracteres (top-1 y top-5), por método (reglas, embeddings, híbrido), con IC del
  95 % y con las variantes de la prueba de estrés.
- **Índice de Charlson por documento** (CodiEsp-D, de punta a punta: NER, contexto y
  normalización), si FR2 confirma que el gold por documento es comparable.
- **DisTEMIST prueba y set sintético:** el NER y la prueba de estrés no empeoran.
- **Costo:** tamaño de la imagen, memoria y latencia p95 en 2 vCPU (`evaluacion/latencia_docker.py`).
- **Decisión documentada:** si un método gana en CodiEsp prueba sin pasarse del costo, se
  integra al pipeline en un epic aparte. Si no, se quedan las reglas y el reporte dice por
  qué.

## Non-Functional Requirements

- **Privacidad.** Todo corre local. Nada de APIs externas con texto clínico (un LLM queda
  fuera por eso).
- **Explicabilidad.** Cada categoría trae el código, la descripción del catálogo y la
  similitud. Es el reemplazo de "qué regla aplicó".
- **Costo de despliegue.** La imagen no supera unos 2 GB comprimida (hoy 855 MB). La memoria
  en uso cabe en 4 GiB con el NER. La latencia p95 de una nota de 2.000 caracteres no supera
  1 s en 2 vCPU (hoy 0,48 s).
- **Reproducible.** Descargas con MD5, revisión fija de cada modelo, índice construido por
  script, resultados en `evaluacion/resultados/` con commit y versiones.
- **Licencias** de catálogo, tabla, corpus y modelos verificadas y citadas antes de publicar.
- Un PR por sub-issue. Tests sin red con modelos y catálogos de juguete.

## Success Criteria

- En CodiEsp prueba, el mejor método supera a las reglas en **F1 de categoría de Charlson**
  con IC que no se solapan, o las iguala con un recall claramente mayor sin perder más de
  0,02 de precisión.
- Con la variante de errores al 10 % de la prueba de estrés, la caída de F1 de categoría es
  menor que la de las reglas con el corrector de #115.
- "accidente cerebrovascular", "ictus" y "linfoma de Hodgkin sin metástasis" tienen su
  categoría correcta sin una regla escrita para ellos.
- El costo de despliegue cumple los NFR.
- Las metas son metas, no condiciones para mergear el reporte. Si no se cumplen, el reporte
  dice cuánto se llegó y por qué, y las reglas se quedan.

## Constraints & Assumptions

- CodiEsp son casos clínicos publicados en España, como DisTEMIST, no notas de una IPS
  colombiana. Es el mejor gold público con CIE-10 en español, pero no es el texto final.
- CIE-10-ES (España) y la CIE-10 que usa Colombia (la de la OMS) comparten la mayoría de los
  códigos de 3 y 4 caracteres que usa la tabla de Quan, pero no todos: `docs/datos.md` (#129)
  lista los prefijos que faltan en la CIE-10-ES y los códigos propios de la CIE-10-CM.
- La tabla de Quan se diseñó para datos administrativos de egresos. Aplicada a menciones en
  texto es una aproximación, y queda dicho así.
- Las reglas actuales también son una forma de la tabla de Quan escrita como regex, así que
  la comparación es justa.

## Out of Scope

- Integrar el método ganador al pipeline, a la API y a la demo: va en un epic aparte si gana.
- Afinar (*fine-tuning*) un modelo de embeddings propio. Se considera solo si los modelos
  existentes quedan cerca y hay un indicio claro de mejora.
- Un LLM como normalizador (privacidad y costo).
- La negación con la clave mal escrita (#126) y el modelo de negación (#77).
- CIE-11.

## Dependencies

- Epic #111 (robustez), cerrado: `evaluacion/robustez.py` y las variantes del texto.
- Epic #39 (evaluación): runners, caché del modelo y descarga con MD5.
- Issue #56, que pasa a ser sub-issue de este epic. #120 y #123 sirven como casos de prueba.
- Diego: confirmar la elección del modelo cuando se resuelva la licencia (FR4).
