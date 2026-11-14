---
name: negacion
description: Corregir la clasificación de negación del pipeline (enumeraciones, "no presenta", claves que faltan, alcance) y clasificar por posición, medido en una partición reservada de NUBes
status: backlog
created: 2026-11-02T02:13:14Z
---

# PRD: negacion

## Executive Summary

La evaluación (epic #39, [`docs/evaluacion.md`](../evaluacion.md)) mostró que la negación
es el punto débil del pipeline:

- En historias clínicas reales (NUBes) deja sin negar 1 de cada 4 enfermedades negadas
  (recall 0,754).
- Es la causa más frecuente de errores en el índice de Charlson del set sintético (6 de 14).

Las causas están identificadas en cuatro issues:

- #62: enumeraciones ("niega HTA, DM, EPOC" solo niega HTA).
- #61: "no presenta X" no niega X.
- #65: faltan claves ("descartó", "nunca", "tampoco", "ningún").
- #66: el alcance cruza ";", ":" y paréntesis, y "negativo" niega hacia atrás la frase
  entera.

Este subproyecto las corrige y además cambia la clasificación **de buscar el texto a mirar
la posición** de cada mención. El cambio se mide con una partición de NUBes que no se usa
para desarrollar, de modo que el "antes y después" del reporte sea honesto.

## Problem Statement

- Una negada que sale confirmada **suma al índice de Charlson y oculta un posible
  subregistro**. Es el error más caro para un auditor. En la demo (lista de comorbilidades
  con click al contexto), además, el usuario vería como confirmada una enfermedad que la nota
  niega.
- `contexto.clasificar` busca el **texto** de la entidad dentro de bloques negados. Si la
  misma palabra aparece negada en una parte y confirmada en otra, gana la negación (o la
  de familiar) para todas las menciones. Es un límite heredado de la tesis
  ([`docs/errores-tesis.md`](../errores-tesis.md)). En notas largas, donde las palabras se
  repiten, pesa más que en NUBes.
- Si se ajustan las reglas mirando NUBes completo y después se reporta NUBes completo, la
  cifra queda inflada.

## User Stories

**US1. Auditor o médico que usa la demo**
Como lector de la lista de comorbilidades, quiero que las enfermedades que la nota niega
salgan como negadas, para no tomar como antecedente algo que el paciente no tiene.
- Criterios de aceptación:
  - "Niega HTA, DM, EPOC" deja las tres negadas.
  - "No presenta EPOC" niega EPOC.
  - "Se descartó TEP" niega TEP.
  - "Niega fiebre pero presenta diabetes" deja la diabetes confirmada.

**US2. Auditor**
Como auditor, quiero que una enfermedad negada en un párrafo y confirmada en otro salga
con los dos estados, cada uno con sus menciones, para ver las dos cosas al hacer click.
- Criterios de aceptación:
  - En "Niega IAM previo. … Ingresa por IAM con elevación del ST", el IAM sale confirmado
    (con la mención del ingreso) y negado (con la de los antecedentes), como dos entidades.
  - El índice de Charlson suma la confirmada.

**US3. Quien lee el portafolio**
Como cliente o reclutador, quiero ver el antes y el después medidos en datos que no se
usaron para ajustar, para confiar en que la mejora es real.
- Criterios de aceptación:
  - `docs/evaluacion.md` trae una sección "Negación: antes y después" con P/R/F1 de
    `negada` en la partición reservada de NUBes (con IC del 95 %), y con el set sintético
    medido de nuevo y marcado como "segunda medición, después de ver los errores".

## Functional Requirements

**FR1. Partición de NUBes y línea base**
- Desarrollo: muestras pares (SAMPLE-002, 004, 006, 008 y 010).
- Reservada: muestras impares (SAMPLE-001, 003, 005, 007 y 009). SAMPLE-001 es la que
  tiene doble anotación.
- Antes de cambiar nada, se mide la línea base del código actual en las dos particiones
  (`nubes_linea_base_desarrollo.json`, `nubes_linea_base_reservada.json`).
- Durante el desarrollo **solo** se mira la partición de desarrollo. La reservada se corre
  una vez al final (FR7).

**FR2. Clasificación por posición**
- `contexto` calcula **intervalos** del texto (alcances de negación y bloques de
  familiares) y clasifica cada **mención** según su posición. Ya no busca el texto de la
  entidad.
- Las menciones de un mismo candidato que caen en estados distintos se separan en una
  entidad por estado. Cada entidad conserva sus menciones, y las entidades siguen en orden
  de aparición.
- Prioridad de un intervalo sobre otro, igual que hoy: familiar, luego negada, luego
  confirmada.
- Este paso **no cambia las reglas**: con las mismas regex, la única diferencia permitida
  es la de texto contra posición. Se verifica con los tests existentes de `contexto` y con
  la línea base de desarrollo, que solo puede cambiar en las menciones del caso
  "por texto" (#59 midió 23 en el corpus completo).

**FR3. Enumeraciones (#62)**
El alcance de una negación sigue a través de comas, "y", "o" y "ni" mientras siga una
enumeración, y se corta en las barreras reales ("pero", "aunque", "refiere", "presenta",
punto, salto de línea…). "Niega fiebre y presenta diabetes" sigue sin negar la diabetes.

**FR4. "No presenta" (#61)**
"no presenta", "no presentó" y "no ha presentado" son disparadores. "Presenta" sigue
siendo barrera cuando no va justo después de un disparador.

**FR5. Claves que faltan (#65)**
- Las formas del verbo "descartar", hacia adelante y hacia atrás: "se descartó", "descarta",
  "descartándose", "TEP descartado"…
- "tampoco", "nunca", "ningún/ninguna", "en ningún momento".
- "cede", "desaparición de" y "se resuelve" **no** se agregan: para un auditor, una
  enfermedad resuelta no es lo mismo que una negada. Queda documentado.

**FR6. Alcance (#66)**
- El alcance se corta en ";", ":", paréntesis y " - ".
- `\b` antes de "negativ" y "descartad": "seronegativa" no niega.
- El patrón hacia atrás ("X negativo", "X descartado") se limita a unas pocas palabras, no
  a toda la cláusula.
- Las barreras de cláusula ("que", "por", "debido a", "ya que") se agregan solo si mejoran
  el F1 en la partición de desarrollo.

**FR7. Medición final y reporte**
- NUBes reservada: una sola corrida con el código final (`nubes_despues_reservada.json`),
  comparada con su línea base.
- Set sintético: `sintetico_v2.json`, marcado como segunda medición. La de #45 no se toca.
- DisTEMIST: se verifica que el NER no cambia. Las menciones son las mismas, porque el
  contexto no toca los spans.
- `docs/evaluacion.md` y el README se actualizan con el antes y el después. Los issues #61,
  #62, #65 y #66 se cierran con su cifra.

## Non-Functional Requirements

- **Un PR por sub-issue**, en serie: todos tocan `contexto.py`.
- Cada regla nueva lleva tests con frases sintéticas cortas (no fragmentos de NUBes) y un
  caso que **no** debe cambiar. Los tests existentes siguen en verde, salvo los que fijaban
  el comportamiento que se corrige, y esos se explican en el PR.
- Cada PR reporta el F1 de `negada` en la partición de desarrollo, antes y después. Ningún
  PR puede bajarlo sin una razón escrita.
- La API pública (`analizar`, `Resultado.to_dict`) no cambia de forma. Una entidad puede
  aparecer dos veces con estados distintos, y eso se documenta en `tipos.py` y en el README.
- El rendimiento por nota no empeora más de un 10 % (las regex corren sobre la nota, no
  sobre el modelo).

## Success Criteria

- Recall de `negada` en la partición reservada de NUBes de **0,85 o más** (la línea base
  del corpus completo es 0,754), sin que la precisión baje de 0,80.
- En el set sintético (segunda medición), la exactitud de `negada` sube de 0,742 a 0,90 o
  más, y el índice exacto, de 86 a 90 o más.
- Ninguna regresión en confirmadas ni familiares del set sintético de más de 1 punto.
- Los ejemplos de US1 y US2 son tests que pasan.
- CI en verde, con cobertura de 85 % o más.

## Constraints & Assumptions

- Los umbrales de éxito son metas, no condiciones para mergear. Si no se alcanzan, el
  reporte dice cuánto se llegó y por qué.
- Las particiones de NUBes son de hospitales de España. Las formas colombianas ("niega",
  "se descarta", "no presenta") se cubren con tests y con el set sintético.
- El set sintético ya no es ciego para estas reglas: vimos sus errores en #45. Por eso su
  segunda medición se reporta como tal, y la cifra con peso es la de NUBes reservada.

## Out of Scope

- Un estado nuevo "incierta" ("sospecha de", "probable"). Hoy "sospecha de" cuenta como
  negada. Cambiarlo toca la API y el índice, así que va en un issue aparte si hace falta.
- "Resuelta" ("cede", "desaparece") como estado.
- Modelos de negación entrenados (NegEx-ML, BERT de negación). Primero las reglas.
- Cambios en la detección de entidades (fusión #55, filtro #56) y en las reglas de
  Charlson (#63).
- API, Docker y demo.

## Dependencies

- Epic #39 (evaluación), cerrado: runner de NUBes (`evaluacion/nubes.py`, con
  `--muestras`), set sintético y caché del modelo.
- Issues #61, #62, #65 y #66, que pasan a ser sub-issues de este epic.
