---
name: pipeline-core
description: Librería Python que extrae comorbilidades de una nota clínica en español, las clasifica por contexto y las mapea a CIE-10 y al índice de Charlson
status: backlog
created: 2026-10-29T00:05:53Z
---

# PRD: pipeline-core

## Executive Summary

Primer subproyecto de `ner-clinic-es`: el núcleo del pipeline como librería Python
(`ner_clinic`), reescrita a partir de la lógica de la tesis (`03_Fase_NLP/PipelineNER.ipynb`
y `04_Perfil_Morbimortalidad/MapeoCIE10 OK.ipynb`), con funciones puras y tests.

Recibe una nota clínica en texto libre (y opcionalmente el diagnóstico principal) y
devuelve cada comorbilidad detectada con su posición en el texto, su estado
(confirmada / negada / antecedente familiar), su categoría de Charlson y un código
CIE-10 sugerido, además del índice de Charlson de la nota.

Es la base sobre la que se construyen después la evaluación, la API y la demo.

## Problem Statement

La lógica que funcionó en la tesis vive en notebooks acoplados a pandas, a rutas locales
y a datos reales de pacientes. Así no se puede reutilizar en una API, no se puede probar
y no se puede mostrar en un portafolio. Además pierde la posición de cada entidad en el
texto (las convierte a minúsculas sin offsets), lo que impide resaltarlas en una demo.

## User Stories

**US1. Integrador (la API y la demo de los siguientes subproyectos)**
Como desarrollador que integra el núcleo, quiero llamar a una sola función con una nota
y recibir un resultado estructurado y serializable, para no conocer los detalles internos.
- Criterios de aceptación:
  - `analizar(texto, diagnostico_principal=None)` devuelve un objeto con la lista de
    entidades y el índice de Charlson.
  - El resultado se convierte a JSON sin pasos adicionales.
  - Cada entidad trae `texto`, `inicio`, `fin`, `estado`, `categoria_charlson`
    (o vacío), `cie10` (o vacío) y `fuente_descarte` cuando se descartó.
  - `texto_original[inicio:fin]` coincide con lo que se detectó.

**US2. Auditor de cuentas médicas (cliente final, a través de la demo)**
Como auditor, quiero ver qué comorbilidades tiene el paciente, cuáles niega y cuáles son
de la familia, para no tener que leer toda la nota.
- Criterios de aceptación:
  - "Niega EPOC" sale como negada; "Madre con cáncer de mama" sale como antecedente
    familiar; "HTA en tratamiento" sale como confirmada.
  - Una comorbilidad repetida o mal escrita en la misma nota aparece una sola vez.

**US3. Codificador / analista de calidad**
Como codificador, quiero un CIE-10 sugerido y el índice de Charlson, para priorizar
historias y apoyar la codificación.
- Criterios de aceptación:
  - Las confirmadas que caen en una de las 17 categorías de Charlson traen categoría
    y código CIE-10 (p. ej. "diabetes mellitus tipo 2" → `E11.9`).
  - Las que no son de Charlson salen con estado pero sin código, marcadas como
    "fuera de Charlson".
  - El índice de Charlson suma solo las confirmadas, con los pesos estándar.

**US4. Diego (mantenedor)**
Como mantenedor, quiero poder correr los tests sin descargar el modelo ni tener GPU,
para iterar rápido y en CI.
- Criterios de aceptación:
  - El modelo NER está detrás de una interfaz; los tests usan un extractor falso.
  - `pytest` corre completo sin red y en menos de 30 segundos.
  - Hay un test de integración marcado aparte que sí usa el modelo real.

## Functional Requirements

**FR1. Limpieza del texto**
Quitar encabezados tipo folio, líneas de guiones y saltos de línea repetidos, como
`limpiar_encabezados_historia` de la tesis, **conservando un mapa de posiciones** para
traducir los offsets del texto limpio al texto original.

**FR2. Segmentación (chunking)**
Partir el texto en fragmentos de hasta 2.000 caracteres por párrafo y luego por oración,
como `segmentar_historia`, conservando el offset de cada fragmento.

**FR3. Extracción NER**
- Modelo `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` con `aggregation_strategy="simple"`,
  en lote, en GPU si hay y si no en CPU.
- Fusionar entidades partidas a 18 caracteres o menos de distancia, salvo que haya
  barrera (salto de línea, coma, punto, guion aislado o numeral de lista).
- Completar palabras cortadas a izquierda y derecha.
- Descartar entidades de 4 caracteres o menos, **salvo las de una lista blanca de siglas
  clínicas** (HTA, EPOC, IAM, ACV, ICC, ERC, IRC, DM, DM2, VIH, FA, TEP, etc.); descartar
  también las que terminan en ` per`, ` end` o ` inf`, y las de la lista negra
  administrativa (`diagnostico`, `antecedentes`, `paciente`, etc.). La lista blanca es la
  única desviación intencional frente a la tesis: el filtro original elimina las siglas,
  que en una nota corta suelen ser la única mención de la comorbilidad.

**FR3b. Experimento de siglas**
Antes de fijar la lista blanca, medir con el modelo real sobre un conjunto de notas
sintéticas con siglas y nombres completos:
- si el modelo detecta las siglas como entidad (y cuáles no);
- cuántas entidades cortas de ruido entran con tres variantes del filtro: el de la tesis,
  con lista blanca y sin filtro de longitud.
El resultado queda documentado en `docs/experimentos/siglas.md` con la decisión tomada.
- Quitar viñetas numéricas al inicio.

**FR4. Deduplicación interna**
Una entidad contenida en otra más larga, o con similitud Jaro-Winkler de 0,85 o más
frente a otra ya guardada, se descarta y queda registrado con cuál se fusionó.

**FR5. Clasificación por contexto**
Con las reglas regex de la tesis:
- **Familiar** si aparece en un bloque de antecedentes familiares o en una frase con
  parentesco (madre, padre, hermano, tío, abuelo, hijo, primo, familiar).
- Si no es familiar, **negada** si aparece en un bloque de negación (`no`, `niega`,
  `sin`, `descartado`, `negativo`, `sospecha de`, etc., con las barreras de la tesis).
- Si no, **confirmada**.
La prioridad es familiar, luego negada, luego confirmada, como en la tesis.

**FR6. Filtro contra el diagnóstico principal** (solo si se envía)
Una confirmada se descarta si comparte una palabra significativa (más de 3 letras, sin
stopwords ni términos genéricos CIE-10) con el diagnóstico principal, por coincidencia
exacta o con similitud Jaro-Winkler o Levenshtein de 0,88 o más. Se registra el motivo.

**FR7. Mapeo a Charlson y CIE-10**
Las 17 categorías de la tesis, cada una con su regex de inclusión, su regex de exclusión
y su función de subclasificación al código CIE-10, aplicadas en el mismo orden (la
primera categoría que coincide gana):
IAM, insuficiencia cardiaca, enfermedad vascular periférica, cerebrovascular, demencia,
pulmonar crónica, reumatológica, úlcera péptica, hepática leve, diabetes sin
complicación, diabetes complicada, hemiplejía/paraplejía, renal crónica, tumor maligno,
hepática severa, tumor metastásico, VIH/SIDA.
El texto se normaliza (minúsculas, sin tildes, sin caracteres de control) antes de
aplicar las reglas, como `limpiar_texto_clinico`.

**FR8. Índice de Charlson**
Suma de pesos estándar (1, 2, 3 o 6 según la categoría) sobre las categorías presentes
entre las confirmadas, contando cada categoría una sola vez. Si el paciente tiene la
forma leve y la severa de una misma enfermedad (diabetes o hepática), cuenta solo la
severa. Sin ajuste por edad en este subproyecto.

**FR9. Trazabilidad**
El resultado incluye las entidades descartadas (por basura, duplicado interno o
diagnóstico principal) con el motivo, como hacía la tesis, para que un auditor vea por
qué se excluyó algo.

## Non-Functional Requirements

- **Python 3.11+**, empaquetado con `pyproject.toml`, código y docstrings en español.
- **Sin pandas en el núcleo.** Funciones puras sobre `str` y dataclasses.
- **Modelo inyectable**: el NER se pasa como dependencia; se carga una sola vez y se
  reutiliza entre llamadas.
- **Rendimiento en CPU**: una nota de hasta 2.000 caracteres se procesa en menos de
  5 s en CPU de portátil, sin contar la primera carga del modelo.
- **Tests**: cada fase tiene tests unitarios con notas sintéticas y casos tomados de
  la lógica de la tesis (fusión, negación, familiar, deduplicación, filtro por
  diagnóstico, cada una de las 17 categorías).
- **Privacidad**: ningún dato real de pacientes en código, tests, fixtures ni ejemplos.
  Solo notas sintéticas escritas para el proyecto.

## Success Criteria

- `pytest` en verde, sin red, en menos de 30 s, con cobertura de 85 % o más en
  `ner_clinic`.
- El ejemplo del README se actualiza a lo que de verdad produce el núcleo con el modelo
  real, y queda como test de integración.
- Las 17 categorías tienen al menos un caso positivo y un caso de exclusión probados.
- `docs/experimentos/siglas.md` reporta, para las tres variantes del filtro, cuántas
  siglas detecta el modelo y cuántas entidades de ruido entran, y justifica la variante
  elegida.
- Para cada entidad, `texto_original[inicio:fin]` coincide con su `texto`.
- Procesar una nota de 2.000 caracteres en CPU toma menos de 5 s.

## Constraints & Assumptions

- Se porta la lógica de la tesis tal cual, incluidos sus umbrales (18, 0,85, 0,88,
  más de 4 caracteres), con la única excepción de la lista blanca de siglas (FR3), que
  se confirma o se descarta con el experimento de FR3b. Mejorarla (por ejemplo, NegEx o un modelo de negación) queda para
  después de tener métricas en el subproyecto de evaluación.
- Se supone que el modelo y el corpus permiten uso en un proyecto de portafolio; la
  verificación de licencias sigue pendiente y no bloquea este
  subproyecto porque el modelo no se redistribuye.
- La deduplicación de folios copiados (fase 1 de la tesis, `SequenceMatcher` 0,85) no
  aplica porque la entrada es una sola nota.

## Out of Scope

- API REST, Docker, demo web y despliegue (subproyectos 3 y 4).
- Evaluación con métricas sobre DisTEMIST (subproyecto 2).
- Entrada de historia completa con varios folios y su deduplicación.
- Mapeo a CIE-10 de enfermedades fuera de las 17 categorías de Charlson.
- Ajuste del índice de Charlson por edad.
- Reentrenar o ajustar el modelo NER.
- Cambiar las reglas de contexto o de mapeo frente a la tesis.

## Dependencies

- `transformers` y `torch` (modelo NER), `rapidfuzz` (Jaro-Winkler y Levenshtein).
- Modelo `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` descargado de Hugging Face (solo para el
  test de integración y el uso real).
- Código de la tesis (privado) como referencia de la lógica;
  no se copia nada más que la lógica, reescrita.
