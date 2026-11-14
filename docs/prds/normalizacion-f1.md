---
name: normalizacion-f1
description: Subir el F1 de categoría de Charlson del normalizador por embeddings con una decisión en dos etapas (el top-k propone, una segunda etapa decide) y más sinónimos públicos; elegido con validación cruzada en CodiEsp entrenamiento
status: backlog
created: 2026-11-07T03:04:11Z
---

# PRD: normalizacion-f1

## Executive Summary

El epic #128 dejó un normalizador por embeddings (SapBERT con los sinónimos de CodiEsp
entrenamiento, más las exclusiones de las reglas) que en CodiEsp prueba gana a las reglas:
F1 de categoría de Charlson 0,874 contra 0,792. Diego decidió no integrarlo todavía
(2026-11-06). Antes hay dos frentes, cada uno por su lado: subir el F1 y resolver la
licencia de SapBERT. Este PRD cubre el primero.

**La meta es el F1, no solo el recall** (Diego, 2026-11-06). Cada comorbilidad que se
reporta tiene que serlo y hay que encontrarlas casi todas.

Hoy el primer candidato del índice decide solo: si su similitud pasa el umbral, su
categoría es la respuesta. La propuesta es **separar las dos cosas**:

1. **Primera etapa, recuperar.** El índice trae un top-k de candidatos: código CIE-10,
   descripción, similitud y origen (catálogo o sinónimo).
2. **Segunda etapa, decidir.** Otro componente mira ese top-k completo y elige la
   categoría de Charlson o "ninguna". Puede votar por categoría o por la jerarquía de la
   CIE-10, o ser un clasificador pequeño entrenado. Así el primer candidato no decide
   solo.

A eso se suman más sinónimos de corpus públicos, para los tumores sobre todo.

## Problem Statement

En desarrollo, el híbrido con exclusiones pierde **52 de las 333 menciones** gold con
categoría y agrega 21 falsos positivos. Así se reparten las 52 perdidas en un análisis
hecho después de #134 (no versionado; FR1 lo versiona):

| Causa | Menciones | Qué la arreglaría |
|---|---|---|
| La categoría correcta está en el top-5, pero no de primera (vecinos como Q25, D49, S06, R20) | 26 | Una segunda etapa que mire todo el top-k |
| El primer candidato es correcto, pero queda bajo el umbral (todas son tumores) | 12¹ | Un umbral o una decisión por categoría, o más sinónimos de tumores |
| Solo las reglas las recuperan | 4 | Las reglas como un voto más |
| Ruido de anotación | ~9 | Nada: quedan como límite |
| Otra categoría | 1 | — |

¹ El reparto de `docs/normalizacion.md` (#133) cuenta 10 bajo el umbral con otra
taxonomía: allí las causas son excluyentes y se asignan en otro orden.

- Tumor maligno suma 27 de las 52.
- 47 de las 52 son menciones que no están entre los sinónimos. El índice generaliza
  peor fuera de lo que memorizó.
- La categoría correcta está entre los cinco primeros candidatos en el 96 % de las
  menciones de desarrollo. **El techo de una buena segunda etapa es alto.**

**Advertencia sobre los datos.** Este análisis sale de desarrollo, que ya se usó en #132 y
#133 para elegir el método. CodiEsp prueba ya se usó una vez (#134) y no se vuelve a usar
para ajustar. Por eso:
- Todas las elecciones de este epic se hacen con **validación cruzada en entrenamiento**.
- Desarrollo se mide una vez al final, y el reporte dice que es optimista.

**Sobre los capítulos de la CIE-10 (propuesta de Diego).** El capítulo solo es demasiado
grueso para Charlson. El IX (circulatorio) mezcla infarto, insuficiencia cardiaca, ACV,
enfermedad vascular periférica e hipertensión, y la hipertensión sola no cuenta en
Charlson (en Quan, I11.0, I13.0 e I13.2 sí cuentan, como insuficiencia cardiaca).
Pero la jerarquía (capítulo → bloque → código de 3 caracteres) sí sirve para que los
candidatos voten: varios vecinos del mismo bloque pesan más que uno suelto.

## User Stories

**US1. Médico o auditor en la demo**
Como lector, quiero que la lista de comorbilidades tenga casi todas las de la nota y casi
ninguna de más, y ver por qué salió cada una.
- Criterios de aceptación:
  - Cada categoría muestra el código y la descripción del candidato que la sostiene.
  - Cuando decide la segunda etapa, muestra también los candidatos que votaron
    ("3 de 5 candidatos son tumor maligno: C34.9, C34.1, C78.0").

**US2. Quien mantiene el proyecto**
Como mantenedor, quiero que cada mejora se elija con un protocolo fijo (validación cruzada
en entrenamiento) y que desarrollo no se gaste en ajustes.
- Criterios de aceptación: cada variante deja su F1 por pliegue en
  `evaluacion/resultados/`, y desarrollo se mide en un solo issue al final.

**US3. Quien lee el portafolio**
Como cliente, quiero ver cuánto mejora cada pieza (segunda etapa, más sinónimos) y lo que
cuesta en latencia y memoria.

## Functional Requirements

**FR1. Arnés de dos etapas**
- La primera etapa guarda, por mención, el top-k del índice con k configurable (10 o 20).
  Cada candidato trae código, similitud, origen, categoría de Quan, código de 3
  caracteres, bloque y capítulo. Se calcula una vez por índice y se reutiliza: ninguna
  segunda etapa vuelve a codificar. Los vectores de catálogo y sinónimos se guardan por
  texto, así que rehacer un índice cuesta poco.
- Validación cruzada de 5 pliegues por documento en CodiEsp entrenamiento, como en #132.
  Los sinónimos de cada pliegue salen solo de los otros cuatro.
- **Comparación pareada.** La diferencia de F1 contra el híbrido actual, con un IC por
  bootstrap pareado por documento.
- **Línea base.** El híbrido con exclusiones (umbral 0,69) se reproduce en el arnés.
- **Validación cruzada anidada.** Para lo que se entrena o elige dentro de un pliegue
  externo, el top-k de los pliegues internos sale de un índice **sin** los sinónimos del
  pliegue externo: 5 índices externos y 20 internos.
- **Análisis de errores en entrenamiento.** El reparto de la tabla de arriba se repite en
  los pliegues de entrenamiento, para confirmar que no es propio de desarrollo.
- **Las 52 menciones perdidas de desarrollo se versionan** por id de documento, posición
  y causa, sin texto, para medir en FR6 cuántas se recuperan.

**FR2. Segunda etapa sin entrenar: votos**
- **Voto por categoría.** La suma de similitudes, o un softmax con temperatura, de los
  candidatos de cada categoría de Charlson dentro del top-k, con "ninguna" como una
  categoría más.
- **Voto jerárquico.** El mismo voto por código de 3 caracteres y por bloque, y de ahí a
  la categoría. El capítulo entra como desempate, no como decisión.
- Los parámetros (k, temperatura, umbral global o por categoría) se eligen por
  validación cruzada.

**FR3. Segunda etapa entrenada: clasificador sobre el top-k**
- Un modelo pequeño y explicable, por ejemplo regresión logística o árboles con
  *gradient boosting*. Por cada par (mención, categoría candidata) decide si se acepta.
- Atributos:
  - similitudes y márgenes;
  - votos por categoría y por jerarquía;
  - origen del candidato;
  - largo de la mención y si es sigla;
  - **la salida de las reglas como un atributo más**, no como respaldo fijo;
  - si aplica una exclusión de las reglas (#133, #144);
  - la puntuación del reordenador de FR4, si se mide.
- Se entrena solo con los pliegues de entrenamiento, con la validación cruzada anidada de
  FR1 para sus hiperparámetros: el pliegue que se evalúa no entra ni en el índice ni en el
  entrenamiento. Sus pesos o importancias se reportan.

**FR4. Segunda etapa con un reordenador (*cross-encoder*)**
- Un reordenador local, con licencia clara y multilingüe, sobre el top-k.
- Se mide su F1 y su costo. Se descarta si no mejora sobre FR2 o FR3, o si suma más de
  0,2 s al p95 de una nota de 2.000 caracteres.

**FR5. Más sinónimos públicos**
- **CANTEMIST** (tumores en español, códigos CIE-O). La conducta del código de morfología
  y la topografía dan la categoría: /6 metastásico; /3 maligno, salvo los de piel que Quan
  excluye (C44, como el basocelular); /0, /1 y /2 (*in situ*) sin categoría; /9 se revisa.
  Ataca de frente a tumor maligno.
  - Se verifican la licencia y la correspondencia CIE-O → Charlson.
  - El código CIE-10 de cada sinónimo, si hace falta, sale de una tabla publicada.
- **DisTEMIST-linking** (enfermedades con códigos SNOMED CT). Primero se verifica si la
  licencia de SNOMED permite usar y redistribuir esos códigos. Si no, se descarta y se
  documenta por qué.
- **Fuga por documentos compartidos.** Los 1.000 documentos de CodiEsp están también en
  DisTEMIST, con el prefijo `es-` en el id (de desarrollo, 190 en DisTEMIST entrenamiento
  y 60 en prueba; de prueba, 181 y 69). CANTEMIST puede compartir casos. Las menciones de documentos de CodiEsp desarrollo o prueba no
  entran nunca como sinónimos, y en la validación cruzada tampoco las del pliegue que se
  evalúa. Se verifica por id de documento, extendiendo `verificar_origen`.
- Cada fuente se mide con y sin ella en la validación cruzada. El índice resultante tiene
  que caber en la imagen.

**FR6. Medición final, costo y decisión**
- La mejor combinación de FR2 a FR5, elegida en entrenamiento, se mide una vez en CodiEsp
  desarrollo junto al híbrido actual. Se reportan:
  - P/R/F1 de categoría micro y por categoría, con IC;
  - el IC pareado de la diferencia;
  - las variantes de la prueba de estrés;
  - el F1 en las menciones no vistas.
- Costo con `evaluacion/costo_normalizacion.py`.
- Reporte en `docs/normalizacion.md` y resumen en `docs/evaluacion.md`, con la decisión.

## Non-Functional Requirements

- **Privacidad.** Solo corpus públicos o frases sintéticas. Nada de los egresos de la
  tesis en el índice ni en el entrenamiento: el índice viajaría en la imagen pública, y
  esas notas no tienen código por mención.
- **Todo local.** Nada de APIs externas con texto clínico.
- **Explicable.** La segunda etapa dice qué candidatos la sostienen.
- **Costo.** Con 2 vCPU y memoria dentro de 4 GiB. El híbrido actual ya mide un p95 de
  1,05 s (unos 0,87 s con la base de `latencia_docker`), en el límite de 1 s. Por eso la
  meta es sobre lo que se agrega: la segunda etapa sin reordenador suma 0,05 s o menos al
  p95, y el reordenador 0,2 s o menos. Bajar el p95 total queda para el epic de
  integración.
- **Reproducible.** Semillas fijas, revisión fija de cada modelo, resultados con commit y
  versiones, y ningún texto del corpus en los JSON versionados.
- Un PR por sub-issue. Tests sin red, con índices y modelos de juguete.

## Success Criteria

- **Validación cruzada en entrenamiento.** La variante elegida supera al híbrido actual
  en F1 de categoría, y el IC pareado de la diferencia no incluye el cero. Como la
  variante se elige con esa misma validación cruzada, la cifra es optimista; la de
  desarrollo es la que cuenta.
- **Desarrollo, una sola medición.** El F1 pasa de 0,885 y la precisión no cae más de
  0,01. Meta orientativa: 0,90 o más.
- Se recupera al menos la mitad de las 26 menciones con la categoría correcta en el top-5
  (versionadas en FR1).
- Con errores al 10 %, el F1 no cae más que el del híbrido actual.
- El costo agregado cumple el NFR.
- Las metas son metas, no condiciones para mergear el reporte. Si no se cumplen, el
  reporte dice cuánto se llegó y por qué.

## Constraints & Assumptions

- No queda una partición limpia de CodiEsp: prueba ya se usó y desarrollo se miró. La
  cifra final de desarrollo es optimista, y el reporte lo dice.
- La segunda etapa entrenada aprende el criterio de los anotadores de CodiEsp, que puede
  no ser el de una IPS colombiana.
- La licencia de SapBERT se resuelve aparte. Si al final hay que cambiar de modelo, la
  segunda etapa y los sinónimos se vuelven a medir con el nuevo.

## Out of Scope

- Integrar el normalizador al pipeline, la API y la demo. Va en otro epic, después de
  esta decisión y la de la licencia.
- Afinar (*fine-tuning*) el modelo de embeddings.
- Un LLM como juez (privacidad y costo).
- Usar los egresos de la tesis como sinónimos o como datos de entrenamiento.
- Volver a medir en CodiEsp prueba.
- CIE-11.

## Dependencies

- Epic #128, cerrado: `evaluacion/normalizador.py`, `evaluacion/sinonimos_codiesp.py`,
  los pliegues de #132 y `evaluacion/costo_normalizacion.py`.
- Descarga de CANTEMIST (y de DisTEMIST-linking si su licencia lo permite), con MD5.
- Diego: confirmar la decisión final y, aparte, la licencia del modelo.
