---
name: evaluacion
description: Métricas del modelo, del pipeline y de cada decisión heredada de la tesis sobre DisTEMIST, más contexto y Charlson sobre un set sintético reservado
status: backlog
created: 2026-10-30T01:21:41Z
---

# PRD: evaluacion

## Executive Summary

Segundo subproyecto de `ner-clinic-es`. El núcleo (`analizar()`) está terminado y medido
solo con notas sintéticas escritas por nosotros para encontrar errores. Este subproyecto
lo mide de forma independiente y reproducible:

1. **NER sobre DisTEMIST** (corpus público, 250 casos clínicos de prueba con gold standard
   de enfermedades): el modelo crudo, el pipeline completo y una ablación de cada decisión
   de la tesis (fusión a 18 caracteres, expansión de palabras, filtro de longitud, lista
   blanca de siglas, lista negra).
2. **Contexto y Charlson** sobre un set sintético nuevo de unas 100 notas, reservado: no se
   usa para ajustar reglas.

El resultado es `docs/evaluacion.md`, la tabla de métricas del README y un comando que
cualquiera puede correr para obtener las mismas cifras.

## Problem Statement

- Las únicas cifras del proyecto vienen de 55 notas sintéticas escritas para encontrar
  errores o para el experimento de siglas. Sirven para mostrar que las correcciones
  funcionan, no para decir qué tan bien funciona el sistema.
- La validación de la tesis la hizo un solo anotador no experto sobre 50 historias. Su
  recall (98,5 %) es probablemente optimista, y no se puede publicar ni reproducir.
- Las decisiones heredadas de la tesis (umbral de 18 caracteres, filtro de 4 caracteres,
  lista negra) nunca se midieron por separado. No sabemos cuáles ayudan y cuáles estorban.
- Un cliente (IPS, auditoría médica) o un reclutador va a preguntar "¿qué tan bien
  funciona?". Hoy no hay una respuesta con datos públicos.

## User Stories

**US1. Cliente o reclutador que lee el portafolio**
Como posible cliente, quiero ver precisión, recall y F1 sobre un corpus público conocido,
con sus límites explicados, para confiar en el sistema sin tener que probarlo yo.
- Criterios de aceptación:
  - El README tiene una tabla corta con P/R/F1 del modelo crudo y del pipeline sobre
    DisTEMIST, y enlaza a `docs/evaluacion.md`.
  - El reporte dice en lenguaje claro qué mide cada cifra y qué no mide (p. ej. el
    contexto se mide con notas sintéticas, no con texto real).

**US2. Diego (mantenedor)**
Como mantenedor, quiero saber qué aporta cada paso del pipeline, para decidir con datos
qué conservar, qué cambiar y qué quitar de la lógica de la tesis.
- Criterios de aceptación:
  - Hay una tabla de ablación: cada fila apaga o cambia una decisión y muestra P/R/F1.
  - Cada fila trae una conclusión (se conserva, se cambia o queda abierta). Los cambios
    al pipeline van en issues aparte, no en este subproyecto.

**US3. Quien quiera reproducir las cifras**
Como revisor técnico, quiero correr un solo comando y obtener las mismas métricas, para
verificar que no están maquilladas.
- Criterios de aceptación:
  - Un comando descarga el corpus (con verificación de checksum), corre la evaluación y
    escribe los resultados en JSON.
  - Correrlo dos veces da las mismas cifras.
  - El reporte registra la versión del modelo, de `transformers`, de `torch`, el commit
    del repo y la CPU.

**US4. Auditor (a través de la demo)**
Como auditor, quiero saber qué tan seguido el sistema se equivoca al decir negada o
familiar, para saber cuánto revisar a mano.
- Criterios de aceptación:
  - El reporte trae la matriz de confusión del estado (confirmada / negada / familiar) y
    la exactitud del índice de Charlson, sobre el set sintético reservado.

## Functional Requirements

**FR1. Descarga del corpus**
- Script que baja `distemist_zenodo.zip` de Zenodo (registro 7614764, CC BY 4.0),
  verifica su MD5 y lo descomprime en `data/distemist/`. `data/` ya está en `.gitignore`;
  el corpus no se commitea.
- Si el archivo ya está y el checksum coincide, no se vuelve a bajar.

**FR2. Lectura del gold standard**
- Leer `test_annotated/text_files/*.txt` y `subtrack1_entities` (TSV: `filename`, `mark`,
  `label`, `off0`, `off1`, `span`) en estructuras propias.
- Verificar al leer que `texto[off0:off1] == span` en todas las anotaciones. Si alguna no
  coincide (codificación, saltos de línea), se reporta y no se sigue con offsets errados.
- Lo mismo para `train_set`, que se usa solo en FR6.

**FR3. Métricas de NER**
- Micro precisión, recall y F1 a nivel de mención, en dos variantes:
  - **Estricta**: mismo `inicio` y `fin` que el gold.
  - **Laxa**: la predicción se solapa con una mención gold (cada gold empareja con una
    sola predicción).
- Las predicciones del pipeline son **todas las menciones** de todas las entidades de
  `Resultado.entidades`, con cualquier estado. DisTEMIST anota también las enfermedades
  negadas, así que el estado no filtra (supuesto a verificar en FR2 con ejemplos del gold).
- Intervalo de confianza del 95 % por bootstrap sobre documentos, para que dos filas de
  la ablación no se lean como distintas si no lo son.
- Las funciones de métricas son puras y tienen tests unitarios con casos pequeños
  escritos a mano.

**FR4. Modelo crudo y pipeline completo**
- **Modelo crudo**: salida del modelo con `aggregation_strategy="simple"` y ventana
  deslizante, sin fusión, expansión ni filtros. Se compara con lo que reporta la ficha del
  modelo (P 0,754, R 0,759, F1 0,757).
- **Pipeline completo**: `analizar(texto)` sin diagnósticos registrados.
- La inferencia del modelo se hace una sola vez por documento y se reutiliza en todas las
  variantes (como `ExtractorConCache` del experimento de siglas).

**FR5. Ablación de las decisiones de la tesis**
Cada fila parte del pipeline completo y cambia una sola cosa:
- sin fusión de entidades cercanas, y con la fusión a otras distancias (p. ej. 0, 10, 18,
  30 caracteres);
- sin expansión de palabras cortadas;
- filtro de longitud: `tesis`, `lista_blanca` (actual) y `ninguno`;
- sin lista negra administrativa;
- sin el descarte de sufijos truncados (` per`, ` end`, ` inf`);
- sin deduplicación interna (debería afectar solo la vista de entidades, no las menciones;
  se verifica).
Si un paso no se puede apagar desde fuera, se agrega un parámetro con el comportamiento
actual por defecto. La API pública de `analizar()` no cambia.

**FR6. Chequeo de contaminación**
La ficha del modelo no dice si se entrenó con la partición de prueba. Se mide el modelo
crudo también sobre `train_set` y se compara con la prueba:
- F1 en prueba parecido al de la ficha y claramente menor que en entrenamiento: se
  asume que la prueba no se vio.
- F1 en prueba casi igual al de entrenamiento: se reporta que las cifras pueden ser
  optimistas. Las conclusiones de la ablación siguen valiendo porque son comparaciones
  relativas.
Se busca además en el paper o el repositorio del modelo qué datos usó, y se cita.

**FR7. Análisis de errores**
- Muestra fija (semilla fija) de 100 errores del pipeline completo: 50 falsos positivos y
  50 falsos negativos, en modo estricto.
- Cada error se clasifica en una categoría: límite del span (sobra o falta texto), síntoma
  o hallazgo que el gold no anota, sigla, error del gold, ruido real, enfermedad que no se
  detectó, fusión indebida, etc.
- El reporte trae la tabla de categorías con ejemplos cortos. Los fragmentos citados vienen
  de DisTEMIST (público, CC BY 4.0) y se atribuyen.

**FR8. Set sintético reservado para contexto y Charlson**
- Unas 100 notas sintéticas nuevas en `evaluacion/sintetico/notas.jsonl`, con el mismo
  formato que `experimentos/errores_tesis/notas.jsonl`: entidades con su estado, índice de
  Charlson esperado y, en algunas, diagnósticos registrados con `registrado` esperado.
- Variedad pedida: confirmadas, negadas, familiares, siglas, mayúsculas y minúsculas,
  abreviaturas, formas de escribir la negación, varias categorías de Charlson y
  jerarquías leve/severa, notas cortas y largas.
- Las escribe Claude. Las valida una **doble anotación ciega**: un segundo anotador
  independiente (un subagente que no vio las anotaciones) anota las 100 notas con los
  mismos criterios escritos. Se reporta el acuerdo (kappa de Cohen para el estado) y cada
  discrepancia se adjudica con una regla escrita. Se corrigen errores de anotación, no se
  ajustan las notas al sistema. (Cambio del 2026-10-31: reemplaza la revisión de 20 notas
  por Diego, que no tiene formación clínica.)
- **Reservado**: se mide una sola vez y esa cifra es la que se reporta. Si después se
  cambian reglas por lo que se vio, el reporte lo dice y la cifra nueva queda aparte.

**FR8b. Negación sobre texto clínico real (NUBes)**
- NUBes (Lima et al., LREC 2020; CC BY-SA 3.0 ES) tiene unas 29.700 frases de historias
  clínicas reales en español, anonimizadas, con claves y alcances de negación e
  incertidumbre. Es la medición principal de la negación. El set sintético queda para los
  familiares, Charlson y `registrado`, que no tienen corpus público.
- Cada entidad que detecta el pipeline se cruza con los alcances gold. El estado esperado
  es `negada` si cae dentro de un alcance de negación y `confirmada` si no cae en ninguno.
  Si cae en un alcance de incertidumbre, se reporta aparte.
- Matriz de confusión, P/R/F1 de `negada` y una muestra de discrepancias clasificadas.
  Issue #59.

**FR9. Métricas de contexto y Charlson**
Sobre el set de FR8:
- recall de entidades esperadas (como `medir.py`);
- matriz de confusión del estado y exactitud por clase;
- precisión y recall por categoría de Charlson (de las confirmadas);
- exactitud del índice: exacto y con diferencia de 1 punto o menos;
- exactitud de `registrado`.

**FR10. Rendimiento**
Durante la corrida sobre DisTEMIST se mide el tiempo por documento en CPU (mediana y
p95), la longitud en caracteres y el uso de memoria pico. Con eso se estima el tiempo por
análisis en Cloud Run (2 vCPU) para ajustar el timeout y el tamaño máximo de nota del
siguiente subproyecto.

**FR11. Reporte**
- `docs/evaluacion.md`: método, versiones, tablas (modelo crudo, pipeline, ablación,
  contaminación, errores, negación en NUBes, contexto, Charlson, rendimiento), conclusiones y límites.
- Resultados completos en JSON versionados en `evaluacion/resultados/`. Solo métricas y
  los fragmentos citados de DisTEMIST, nunca el corpus completo.
- README: tabla corta de resultados y la casilla "Evaluación sobre corpus público con
  métricas" marcada.

## Non-Functional Requirements

- **Reproducible**: un comando (`uv run python -m evaluacion` o similar) hace todo. Sin
  aleatoriedad fuera de semillas fijas. La inferencia es determinista en CPU.
- **CI sin red ni modelo**: los tests de lectura del corpus y de métricas usan archivos de
  ejemplo pequeños escritos a mano. La corrida real queda marcada `modelo` y se hace en
  local.
- **Tiempo**: la corrida completa (prueba + entrenamiento + ablaciones + set sintético) en
  CPU de portátil cabe en 1 h, gracias a que la inferencia se hace una vez por documento.
- **Código del núcleo intacto**: si hace falta exponer un parámetro para la ablación, el
  valor por defecto mantiene el comportamiento actual y los 4.238 tests siguen en verde.
- **Privacidad**: solo DisTEMIST (público) y notas sintéticas. Nada del repo de la tesis.
- **Atribución**: el reporte y el README citan DisTEMIST (Miranda-Escalada et al., 2022)
  con su licencia CC BY 4.0.

## Success Criteria

- El modelo crudo sobre la prueba de DisTEMIST da un F1 estricto a ±0,03 del 0,757 de su
  ficha, o el reporte explica la diferencia (p. ej. modo de evaluación o ventana).
- Hay P/R/F1 estrictos y laxos, con intervalos del 95 %, del modelo crudo, del pipeline
  completo y de cada fila de la ablación (al menos 8 filas).
- El chequeo de contaminación (FR6) tiene una conclusión explícita.
- 100 errores clasificados, con su tabla de categorías.
- Set sintético de al menos 100 notas, con al menos 20 revisadas por Diego, y sus
  métricas de estado, Charlson y `registrado` reportadas.
- Tiempo por documento (mediana y p95) en CPU reportado, con la estimación para Cloud Run.
- Correr la evaluación dos veces da los mismos números.
- CI en verde, sin red, con cobertura de 85 % o más.

## Constraints & Assumptions

- DisTEMIST anota enfermedades, no síntomas, y anota también las negadas. Lo primero
  coincide con el alcance del modelo. Lo segundo se verifica en FR2.
- DisTEMIST son casos clínicos publicados en revistas, no notas de una IPS colombiana:
  hay menos siglas, menos errores de ortografía y otro estilo. El reporte lo dice. Las
  reglas de limpieza pensadas para folios casi no se ejercitan.
- Las notas sintéticas de FR8 las escribimos nosotros, que conocemos las reglas. Las
  cifras de familiares y Charlson son una cota optimista, y el reporte lo dice. Ninguna
  anotación del proyecto la validó un clínico.
- NUBes viene de hospitales de España, no de Colombia, y no anota familiares.
- Si el chequeo de contaminación sale mal, no hay otro corpus público equivalente a mano
  (CANTEMIST solo anota tumores; CARMEN-I requiere acceso acreditado). En ese caso se
  reportan las cifras con la advertencia.

## Out of Scope

- Cambiar reglas o umbrales del pipeline por lo que muestre la ablación: se proponen en
  issues nuevos.
- Evaluar el linking a SNOMED CT (subtrack 2 de DisTEMIST) o el CIE-10 contra un gold:
  DisTEMIST no tiene CIE-10, y mapear de SNOMED a CIE-10 no es trivial y tiene su propia
  licencia.
- Otros corpus (CANTEMIST, CARMEN-I, LivingNER). NUBes sí entra (FR8b).
- Reentrenar o ajustar el modelo.
- Anotación manual de estado sobre textos de DisTEMIST.
- API, Docker y demo (subproyectos siguientes).

## Dependencies

- Núcleo `ner_clinic` (epic #4, cerrado).
- Corpus DisTEMIST en Zenodo, registro 7614764 (`distemist_zenodo.zip`, MD5
  `bc84461ad07b59a94c79ca9c648952c2`), CC BY 4.0.
- Modelo `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` (CC BY 4.0, según su ficha).
- Diego: revisar al menos 20 notas del set sintético antes de la medición.
