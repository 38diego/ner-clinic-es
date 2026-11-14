---
name: integracion-normalizador
description: Llevar al pipeline y a la API el normalizador por embeddings con el clasificador de #151 (SapBERT + sinónimos de CodiEsp, sin fuentes externas), como opción activable, con la medición de extremo a extremo y la licencia de SapBERT documentada como pendiente
status: backlog
created: 2026-11-08T16:05:14Z
---

# PRD: integracion-normalizador

## Executive Summary

El epic #148 dejó elegido el normalizador: SapBERT recupera los 20 códigos CIE-10 más
parecidos a la mención (en el catálogo CIE-10-ES y los sinónimos de CodiEsp
entrenamiento) y una regresión logística sobre ese top-20 decide la categoría de Charlson.
En desarrollo, el F1 de categoría pasa de 0,885 con el híbrido a 0,914. Diego decidió
adoptarlo **sin** los sinónimos de DisTEMIST y CANTEMIST (2026-11-08).

Hoy todo eso vive en `evaluacion/`: el pipeline (`ner_clinic`) sigue mapeando con las
reglas regex. Este PRD lo lleva a la librería y a la API **como una opción activable**.
Las reglas no se van: son el mapeo por defecto, el respaldo si faltan los artefactos y
una de las señales del clasificador.

**Licencia de SapBERT** (Diego, 2026-11-08): se integra ya y se resuelve después. Su
código es MIT, pero los pesos en Hugging Face no declaran licencia y se entrenaron con
UMLS, que tiene licencia propia. Queda documentado como pendiente, y la imagen pública
de la demo no lleva SapBERT hasta resolverlo (eso es del epic de la demo).

## Problem Statement

- **El mejor método no llega a quien usa el proyecto.** La API y la demo dan la categoría
  de las reglas (F1 0,792 en CodiEsp prueba contra 0,874 del normalizador con umbral,
  #134), y lo medido en #148 no se ve fuera de los JSON de evaluación.
- **El código del normalizador es de experimento.** Está repartido en `evaluacion/`
  (`normalizador`, `dos_etapas`, `votos`, `clasificador`), mezclado con validación
  cruzada, bootstrap y lectura de corpus. La librería no puede depender de eso.
- **No se ha medido de extremo a extremo.** #148 midió sobre las menciones gold de
  CodiEsp. En el pipeline las menciones las corta el NER (más largas, más cortas,
  fusionadas), y la entidad tiene un nombre y varias menciones. El efecto en el índice de
  Charlson de una nota no se conoce.
- **El clasificador final no está guardado.** El JSON de #154 tiene los pesos del modelo
  con fuentes; del modelo sin fuentes solo quedaron la penalización (0,001) y el umbral
  (0,30).

## User Stories

**US1. Médico o auditor que usa la API**
Como usuario, quiero que una comorbilidad escrita de forma poco común ("ca de colon con
siembras hepáticas", "IRC estadio 4") salga con su categoría de Charlson, y saber de dónde
salió ("parecido a 'insuficiencia renal crónica, estadio 4', del catálogo CIE-10").
- La entidad dice si la categoría salió de las reglas o del normalizador.
- Si salió del normalizador, trae el candidato que más pesó, su origen y su similitud.

**US2. Quien integra la librería**
Como desarrollador, quiero activar el normalizador con un parámetro y que, si no lo
activo, el pipeline se comporte exactamente igual que hoy.
- `analizar()` sin el normalizador da el mismo `Resultado` que antes del epic.
- Si los artefactos no están, el error dice qué falta y cómo construirlo.

**US3. Quien mantiene el proyecto**
Como mantenedor, quiero un solo código del normalizador: el de la librería, y que la
evaluación lo reutilice.
- Un test comprueba que la librería decide igual que lo medido en #154 sobre las
  menciones de desarrollo.

**US4. Quien lee el portafolio**
Como lector, quiero ver cuánto cambia el índice de Charlson de una nota con el
normalizador y cuánto cuesta, no solo el F1 sobre menciones gold.

## Functional Requirements

**FR1. El normalizador en la librería**
- Un módulo `ner_clinic.normalizacion` con solo lo que hace falta en ejecución:
  - cargar el índice (vectores, códigos, textos, origen) y validar su huella;
  - el codificador de SapBERT con la revisión fija;
  - el top-20 por similitud, en lote para todas las entidades de una nota;
  - los 39 atributos del par (mención, categoría) y la regresión logística cargada de un
    JSON (pesos, media, desviación, umbral);
  - las exclusiones de las reglas.
- `evaluacion/` importa estas piezas de la librería en vez de tener su copia. Los
  resultados ya versionados no se recalculan.
- `torch` y `transformers` ya son dependencias. No se agrega scikit-learn.

**FR2. Artefactos reproducibles**
- Un comando construye el índice (catálogo CIE-10-ES 2018 + sinónimos de CodiEsp
  entrenamiento, sin fuentes externas) en `data/normalizacion/` (no se versiona: pesa
  unos 300 MB).
- Otro entrena el clasificador final sin fuentes con los 5 pliegues de entrenamiento,
  con la misma elección de penalización y umbral de #154, y lo guarda en un JSON
  **versionado dentro del paquete**: 39 pesos, sesgo, media, desviación, umbral, commit,
  revisión del modelo y huella del índice. Sin texto de ningún corpus.
- Si se vuelve a entrenar con los mismos datos, la penalización y el umbral coinciden con
  los de #154 (0,001 y 0,30).

**FR3. El normalizador en el pipeline**
- `analizar(..., normalizador=None)`: con `None`, todo como hoy. Con un `Normalizador`
  cargado, el paso 6 usa la segunda etapa para las confirmadas y las posibles.
- Qué texto se normaliza: el nombre de la entidad y, si no da categoría, sus menciones,
  en el mismo orden que `_mapear` hoy (#113). Las reglas + el corrector siguen
  calculándose, porque son atributos del clasificador.
- `Entidad` gana dos campos, en `to_dict()` también:
  - `mapeo`: `"reglas"` o `"normalizador"`;
  - `candidato`: con el normalizador, el texto, el código, el origen (catálogo o
    sinónimo de CodiEsp) y la similitud del mejor candidato de la categoría elegida.
    El texto sale del índice (catálogo o CodiEsp, públicos), nunca de la nota.
- `cie10` es el código del candidato elegido; `correcciones` sigue saliendo de las reglas.

**FR4. La API**
- Variable de entorno `NER_NORMALIZADOR` (`reglas` por defecto, `sapbert`). Con
  `sapbert`, la app carga el índice y el clasificador al arrancar y la nota de
  calentamiento pasa también por el normalizador.
- `GET /api/salud` dice qué mapeo está activo. El OpenAPI documenta los campos nuevos.
- El front no cambia en este epic.

**FR5. Medición de extremo a extremo y costo**
- Con el pipeline completo (NER incluido), reglas contra normalizador:
  - **Set sintético** (100 notas, el de #45): categoría por entidad e índice de Charlson
    exacto y a ±1. Va en un archivo aparte (`sintetico_v5.json`); el set ya se midió
    varias veces y el reporte lo dice.
  - **CodiEsp desarrollo, por documento**: precisión, recall y F1 de las categorías de
    Charlson de la nota contra las que salen de sus códigos gold, pareado con bootstrap
    por documento. Optimista, porque desarrollo ya se miró.
- CodiEsp prueba no se toca.
- Costo en Docker con 2 vCPU: latencia por nota (mediana y p95), memoria pico, tamaño de
  la imagen y arranque en frío, con y sin normalizador.
- Con eso, Diego decide si la API lo activa por defecto.

**FR6. Documentación y licencias**
- README y `docs/`: cómo construir los artefactos, cómo activarlo, qué cambia en la
  salida y los resultados de FR5.
- La licencia de SapBERT queda escrita como pendiente en `docs/datos.md` y en el README:
  código MIT, pesos sin licencia declarada, entrenado con UMLS. Antes de publicar una
  imagen con SapBERT o de usarlo con un cliente hay que resolverla.
- La atribución CC BY 4.0 del catálogo CIE-10-ES y de CodiEsp se cita donde se usan.

## Non-Functional Requirements

- **Sin texto de la nota en ningún lado nuevo**: el normalizador no registra ni guarda
  menciones. Los logs de la API siguen sin texto.
- **Costo**: con el normalizador, el p95 por nota en Docker con 2 vCPU sube 0,5 s o
  menos sobre el pipeline con reglas, y la memoria pico cabe holgada en 4 GiB (en #154,
  1,0 GiB).
- **Importar `ner_clinic` no carga `torch` ni el índice**, como hoy.
- **Tests sin red ni modelos**: con un codificador falso (trigramas, como en
  `tests/test_normalizador.py`) y un índice chico.

## Success Criteria

- `analizar()` sin normalizador: todos los tests actuales pasan sin cambiarlos.
- Equivalencia: sobre las menciones gold de CodiEsp desarrollo, la librería da las mismas
  decisiones que el método medido en #154 sin fuentes (F1 0,921).
- FR5 medido y versionado, con commit y versiones, desde un árbol limpio.
- La API con `NER_NORMALIZADOR=sapbert` responde con `mapeo` y `candidato`, y `salud` lo
  reporta.
- El p95 y la memoria cumplen lo de los requisitos no funcionales.

## Constraints & Assumptions

- Todo corre local: ninguna API externa recibe texto clínico.
- SapBERT se descarga de Hugging Face con revisión fija; no se versiona en el repo.
- El índice tarda unos 35 min en CPU en construirse (#131). Se construye una vez y se
  reutiliza mientras su huella coincida.
- CodiEsp prueba está gastada: no se usa para nada en este epic.
- El clasificador se entrenó con menciones gold; las del NER pueden ser distintas. FR5
  mide cuánto importa, sin reajustar nada con desarrollo.

## Out of Scope

- La demo (front, imagen pública, Cloud Run): va en el epic de despliegue o de la demo,
  y depende de la licencia.
- Resolver la licencia de SapBERT o cambiar de modelo.
- Volver a ajustar el clasificador, el umbral o el índice con resultados de extremo a
  extremo.
- Los sinónimos de DisTEMIST y CANTEMIST (descartados en #154).
- El job de Docker en CI sigue en pausa.

## Dependencies

- Epic #148 cerrado (clasificador y decisión).
- Datos locales: catálogo CIE-10-ES y CodiEsp en `data/`, como en #148.
- Diego: decidir con FR5 si la API activa el normalizador por defecto.
