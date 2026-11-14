---
name: incertidumbre
description: Estado nuevo "posible" para las sospechas (ni confirmadas ni negadas), índice de Charlson si se confirman y corrección de la lista que niega de más (#83, #84)
status: backlog
created: 2026-11-02T23:41:25Z
---

# PRD: incertidumbre

## Executive Summary

Hoy el pipeline tiene tres estados: confirmada, negada y familiar. Una **sospecha** ("sospecha
de TEP", "probable neumonía", "no se descarta IAM") tiene que caer en uno de ellos:

- "sospecha de" sale **negada**, por herencia de la tesis.
- "no se descarta" también sale negada, desde #65 (issue #83).
- "probable" o "compatible con" salen **confirmadas**.

Ninguna de las dos cosas es correcta: una sospecha no es un diagnóstico ni un descarte.

Este subproyecto agrega un cuarto estado, **`posible`**:

- se detecta con reglas explicables, igual que la negación;
- **no suma** al índice de Charlson, pero el resultado trae aparte el índice que daría
  **si se confirmaran las posibles**;
- se mide con NUBes, que anota la incertidumbre (unas 1.300 claves en la partición de
  desarrollo).

También se corrige #84: la lista de #62 niega una enfermedad de la cláusula siguiente.

Decisiones de Diego (2026-11-02): estado nuevo "posible"; no suma al índice pero se muestra
aparte el índice si se confirman.

## Problem Statement

- Para un auditor, una sospecha es información distinta de un antecedente y de un descarte:
  - Si se marca **negada**, se esconde algo que hay que estudiar (una sospecha de cáncer se
    ve igual que "niega cáncer").
  - Si se marca **confirmada**, infla el índice y el subregistro con algo que quizá no
    existe.
- En NUBes, de 1.288 menciones en un alcance de incertidumbre, 199 salen negadas y 1.086
  confirmadas (`evaluacion/resultados/nubes.json`).
- #84: "sin secuelas motoras, demencia vascular leve" niega la demencia. Es la única
  regresión del set sintético v2.

## User Stories

**US1. Auditor o médico en la demo**
Como lector, quiero ver aparte las enfermedades sospechadas ("a confirmar"), con su click al
contexto, para saber qué falta estudiar sin confundirlo con lo que el paciente tiene o no
tiene.
- Criterios de aceptación:
  - "Sospecha de TEP", "probable neumonía", "No se descarta IAM", "SCA vs. pericarditis" y
    "lesión compatible con metástasis" dan `posible`.
  - "Se descartó TEP" y "niega TEP" siguen dando `negada`. "TEP" sin calificar da
    `confirmada`.

**US2. Codificador o analista**
Como analista, quiero el índice de Charlson con lo confirmado y, aparte, el índice si se
confirman las sospechas, para dimensionar el riesgo sin mezclar las dos cosas.
- Criterios de aceptación:
  - `Resultado` tiene `indice_charlson` (solo confirmadas, como hoy) y
    `indice_charlson_si_se_confirman` (confirmadas + posibles, con la misma jerarquía).
  - "Paciente con DM2. Sospecha de cáncer metastásico" da 1 y 7.

**US3. Quien integra la API**
Como integrador, quiero que el cambio sea compatible: un valor más en `estado` y un campo
más en el resultado, sin romper los existentes.

## Functional Requirements

**FR1. Medición de `posible` en NUBes**
- El runner de NUBes compara `incierta` (esperado) con `posible` (obtenido) y reporta
  P/R/F1 de `posible` con IC, además de las de `negada`, que no cambian de definición.
- La matriz de confusión pasa a tener 4 columnas: confirmada, negada, posible y familiar.
- Antes de cambiar el pipeline, se mide la línea base en la partición de desarrollo
  (`posible` da 0, porque el estado no existe todavía).

**FR2. Lista que niega de más (#84)**
- La continuación de listas de #62 no debe saltar a la cláusula siguiente cuando la coma
  separa dos ideas ("sin secuelas motoras, demencia vascular leve").
- Se decide con la partición de desarrollo y con tests. "Niega HTA, DM, EPOC" y "sin
  cefalea, diplopía, fotofobia" siguen negando las tres.

**FR3. Estado `posible` en el pipeline**
- `Estado.POSIBLE = "posible"`.
- Las posibles reciben categoría de Charlson y CIE-10 (para mostrarlas), pero **no suman**
  a `indice_charlson`.
- Campo nuevo `Resultado.indice_charlson_si_se_confirman`, que también va en `to_dict()`.
- `cruzar_con_registrados` trata a las posibles como hoy a las negadas
  (`registrado=None`).
- README, docstrings de `tipos.py` y `analizar()` actualizados.

**FR4. Reglas de incertidumbre (#83)**
- Intervalos de incertidumbre en `contexto`, con la misma mecánica que la negación: un
  disparador, un alcance hacia adelante con cortes, el patrón corto hacia atrás y la
  clasificación por inicio dentro.
- Claves candidatas, ordenadas por frecuencia en NUBes:
  - "probable(mente)", "posible(mente)", "compatible(s) con", "sospecha de", "parece",
    "sugestivo/a(s) de", "valorar", "?", "pudiera", "al parecer", "dudoso/a", "se plantea",
    "impresiona de", "vs/versus";
  - las formas negadas de descartar: "no se descarta", "no se puede descartar", "sin
    descartar", "a descartar", "para descartar".
- Cada clave entra solo si mejora el F1 de `posible` sin bajar el F1 de `negada` en
  desarrollo. Las que se descartan quedan con su cifra en el docstring.
- Prioridad: familiar > posible > negada > confirmada. Así, "no se descarta X" (donde "no"
  dispara negación) queda posible.
- "sospecha de" y "bajo riesgo de" salen de los disparadores de negación.

**FR5. Medición final y reporte**
- **NUBes reservada:** una corrida final. Es la **segunda** vez que se corre la reservada
  (la primera fue #74). En medio solo se usó la partición de desarrollo, y el reporte lo
  dice.
- **Set sintético v3:** sirve para comprobar que negadas, familiares e índice no empeoran
  frente a la v2. El set no anota `posible`, así que una entidad del gold que salga posible
  cuenta como error de estado y se lista.
- **DisTEMIST:** el NER no cambia.
- `docs/evaluacion.md` (sección nueva) y README.

## Non-Functional Requirements

- Un PR por sub-issue, en serie (todos tocan `contexto.py` o `tipos.py`).
- Tests con frases sintéticas cortas, nunca fragmentos de NUBes.
- La partición reservada se corre solo en la medición final.
- `to_dict()` sigue siendo compatible: los campos existentes no cambian.
- El rendimiento no empeora: hay un test de tiempo con texto adversario, como en #66.

## Success Criteria

- F1 de `posible` en la partición reservada de NUBes de **0,70 o más**. La incertidumbre
  es más difusa que la negación; la meta es razonable para reglas, no ambiciosa.
- F1 de `negada` en la reservada sin bajar de 0,85. Era 0,864 después de #74, y quitar
  "sospecha de" de la negación debería ayudarle.
- Set sintético v3 sin regresiones frente a la v2 en negadas, familiares ni índice exacto,
  y con sin086 (#84) corregida.
- Los ejemplos de US1 y US2 son tests.

## Constraints & Assumptions

- NUBes anota la incertidumbre con criterio de lingüistas sobre texto de España. "Compatible
  con" es incertidumbre para NUBes, pero en radiología colombiana a veces se lee como casi
  confirmación. La regla se decide con datos y la duda queda documentada.
- El set sintético no se reanota con `posible`. Sigue midiendo lo que ya medía.

## Out of Scope

- Estado "resuelta" ("cede", "se resolvió").
- Grados de certeza (probable frente a posible).
- El modelo de negación e incertidumbre (#77).
- API, Docker y demo (subproyecto siguiente; la demo mostrará las posibles aparte).

## Dependencies

- Epic #71 (negación), cerrado: `contexto` por posición y runner de NUBes con particiones.
- Issues #83 y #84, que pasan a ser sub-issues de este epic.
