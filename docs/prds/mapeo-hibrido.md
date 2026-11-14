---
name: mapeo-hibrido
description: Combinar reglas y normalizador en un solo mapeo a Charlson (reglas primero, normalizador de respaldo), con los huecos de las reglas cerrados y el mismo camino en la librería, la API y Docker
status: backlog
created: 2026-11-08T22:00:00Z
---

# PRD: mapeo-hibrido

## Executive Summary

Hoy la categoría de Charlson sale de **uno de dos** mapeos: las reglas o el normalizador
(SapBERT + regresión logística, epic #162). Cada uno gana donde el otro pierde:

| | Reglas | Normalizador |
|---|---|---|
| CodiEsp desarrollo, de punta a punta (F1) | 0,724 | **0,764** |
| Set sintético, índice exacto (sin fusión) | **96 %** | 82 % |

El normalizador conoce formas que las reglas no ("ictus", "falla renal crónica"); las
reglas no se equivocan con las siglas locales que el normalizador confunde ("ICC") ni con
los tramos largos. Además, la API usa por defecto el normalizador y la imagen Docker usa
las reglas (no lleva SapBERT), así que la demo y la API no dan lo mismo.

Este epic (decidido con Diego el 2026-11-08) los combina: **primero las reglas y, si no
dan categoría, el normalizador**. Antes cierra los huecos conocidos de las reglas (#123,
#120, #63), porque en el híbrido las reglas mandan.

## Problem Statement

- **Dos caminos que no se hablan.** Quien usa la API recibe el normalizador; quien usa
  la imagen o la demo, las reglas. Las cifras del README no dicen cuál es "el sistema".
- **Cada método pierde donde el otro acierta** (tabla de arriba).
- **Huecos de las reglas con issue abierto:** "accidente cerebrovascular" e "ictus" (#123),
  "sin metástasis" (#120), fibrosis pulmonar, insuficiencia arterial y cirrosis Child-Pugh C
  (#63). En el híbrido, una regla que da la categoría equivocada ya no la corrige nadie.

## User Stories

**US1. Médico o auditor.** Como usuario de la API o la demo, quiero que "ictus" y "EPOC"
salgan con su categoría aunque una esté escrita de forma poco común, sin que el sistema
confunda siglas que las reglas ya conocen.
- Cada entidad dice si la categoría salió de las reglas o del normalizador (`mapeo`).

**US2. Quien integra la librería.** Como desarrollador, quiero un parámetro para elegir el
modo (`hibrido` o `normalizador`) y que sin normalizador todo quede igual que hoy.

**US3. Quien despliega.** Como mantenedor, quiero una sola configuración: la API arranca en
híbrido y, si la imagen no trae los artefactos del normalizador, queda en reglas y lo dice
en `/salud`, sin otra variable de entorno.

## Functional Requirements

- **FR1. Reglas (#123, #120, #63).**
  - "accidente/enfermedad/evento cerebrovascular" e "ictus" → cerebrovascular.
  - "arteriopatía periférica (obliterante)" e "insuficiencia arterial crónica de miembros" →
    vascular periférica.
  - "fibrosis pulmonar" y "enfermedad pulmonar intersticial" → pulmonar crónica (J84).
  - Un tumor "sin metástasis" → tumor maligno; los metastásicos no cambian.
  - Cirrosis con Child-Pugh C, descompensada o con ascitis → hepatopatía severa; Child-Pugh
    A sigue leve. "Hepatitis alcohólica" → leve (K70.1, como en Quan).
  - Cada caso con su test; el set sintético sin regresiones.
- **FR2. Modo híbrido en `analizar()`.** Con normalizador, el modo por defecto es
  `hibrido`: cada entidad (y cada diagnóstico registrado) se mapea con las reglas y solo
  las que quedan sin categoría pasan por el normalizador, en un solo lote. El modo
  `normalizador` conserva el comportamiento de #162. Sin normalizador, solo reglas.
- **FR3. Medición.** Reglas, normalizador e híbrido en el set sintético y en CodiEsp
  desarrollo de punta a punta, con las reglas ya corregidas.
- **FR4. API y Docker.** `NER_NORMALIZADOR` acepta `reglas`, `sapbert` (solo el normalizador)
  e `hibrido`. Por defecto `hibrido` (si la medición lo respalda, FR3). En `hibrido`, si
  faltan los artefactos, la API arranca con reglas, lo registra y `/salud` lo informa;
  `sapbert` sigue fallando si faltan. El Dockerfile deja de fijar `reglas`.

## Non-Functional Requirements

- Sin texto de corpus en los JSON versionados. CodiEsp prueba no se vuelve a medir.
- El híbrido no cuesta más que el normalizador: SapBERT corre solo para lo que las reglas
  no mapean.

## Success Criteria

Regla de decisión, declarada antes de medir (FR3). El híbrido queda por defecto en la API
si cumple las dos:

1. En CodiEsp desarrollo de punta a punta, su F1 supera al de las reglas.
2. En el set sintético, su índice exacto no cae más de 1 punto porcentual frente a las
   reglas.

Si no las cumple, la API queda en `reglas` por defecto y el modo híbrido sigue disponible.

## Constraints & Assumptions

- El set sintético ya se ha usado muchas veces: sus cifras son optimistas para las reglas,
  que se corrigieron mirándolo. CodiEsp desarrollo también se miró (#132, #133, #154).
- La licencia de los pesos de SapBERT sigue pendiente: la imagen pública no lo lleva (epic
  de despliegue).

## Out of Scope

- Llevar SapBERT a la imagen Docker y resolver su licencia (epic #97).
- Reentrenar el clasificador del normalizador.
- La negación (#136, #137, #126) y las siglas locales del normalizador.

## Dependencies

- Epic #162 (normalizador integrado), #55 (sin fusión), #176 (cruce por categoría).
