---
name: robustez
description: Que el pipeline aguante notas escritas a la carrera (siglas en minúscula, sin tildes, errores de ortografía), con una prueba de estrés medida en DisTEMIST y en el set sintético
status: backlog
created: 2026-11-05T01:06:01Z
---

# PRD: robustez

## Executive Summary

Al probar la demo, Diego escribió "epoc" en minúscula y el pipeline no lo reconoció. Una
exploración rápida (2026-11-04, set sintético, exploratoria) mostró tres problemas
distintos:

1. **Siglas en minúscula.** El modelo distingue mayúsculas en las siglas: "pte con EPOC"
   la detecta y "pte con epoc" no; "HTA, DM, EPOC" da las tres y "hta, dm, epoc" ninguna.
   Con el set sintético pasado a minúscula, el índice de Charlson exacto baja de 92 a 80
   de 100. Poner en mayúscula las siglas conocidas antes del modelo lo sube a 89, sin
   cambiar las notas que ya funcionaban (92).
2. **Errores de ortografía.** El modelo los tolera bastante ("eppoc", "diabetis mellitus"
   salen detectados), pero las reglas de Charlson son regex y no les asignan categoría.
   Con errores inyectados en el 5–10 % de las palabras, el índice exacto baja de 92 a
   85–86.
3. **Bug de la fusión.** "EPOC, epoc y eppoc" se fusionan en una entidad (bien), pero esa
   entidad toma el nombre "eppoc", que no mapea: el índice da 0 aunque la nota dice EPOC
   dos veces bien escrito.

Este subproyecto corrige los tres y agrega al reporte una **prueba de estrés**: las mismas
mediciones con el texto en minúscula, sin tildes y con errores, antes y después.

## Problem Statement

- Las notas reales se escriben rápido: en minúscula, sin tildes, con errores de digitación.
  El set sintético y DisTEMIST están bien escritos, así que las cifras del reporte son las
  de un caso favorable.
- En la demo, cualquiera que la pruebe va a escribir en minúscula. Si "epoc" no sale, la
  primera impresión es que el sistema no sirve.
- Una enfermedad detectada pero sin categoría **no suma al índice**: es un falso negativo
  silencioso, porque se ve en la lista y el índice no la cuenta.

## User Stories

**US1. Médico o auditor que prueba la demo**
Como lector, quiero que "epoc", "hta" o "dm2" escritos en minúscula salgan igual que en
mayúscula.
- Criterios de aceptación:
  - "pte con epoc", "hta, dm, epoc" y "paciente de 70 años con epoc oxigenorequiriente"
    dan las mismas entidades y el mismo índice que con las siglas en mayúscula.
  - La mención que se resalta en la nota es el texto tal como lo escribió el médico (en
    minúscula), con las mismas posiciones.

**US2. Auditor**
Como auditor, quiero que una enfermedad mal escrita sume al índice si es claramente esa
enfermedad, y que no se inventen enfermedades a partir de palabras parecidas.
- Criterios de aceptación:
  - "diabetis mellitus tipo 2" da diabetes; "eppoc" da enfermedad pulmonar crónica;
    "cirrocis" da enfermedad hepática.
  - "EPOC, epoc y eppoc" da una entidad con categoría y el índice suma 1.
  - Palabras cortas o comunes parecidas a una clave no reciben categoría (se fijan con
    tests).

**US3. Quien lee el portafolio**
Como cliente, quiero saber cuánto empeora el sistema con notas mal escritas, medido con
datos que no se usaron para ajustarlo.
- Criterios de aceptación:
  - `docs/evaluacion.md` trae una sección "Robustez: notas mal escritas" con F1 laxo y
    estricto de DisTEMIST (prueba) en cada variante del texto, antes y después, con IC.

## Functional Requirements

**FR1. Variantes del texto y línea base**
- Un módulo de evaluación genera variantes **del mismo largo** que el original, para que
  los spans anotados sigan sirviendo:
  - `minusculas`; `minusculas_sin_tildes`;
  - `errores_<p>`: en una fracción `p` de las palabras de 4 letras o más, una sustitución
    de una letra o el intercambio de dos letras vecinas (sin insertar ni borrar, para no
    mover los offsets), con semilla fija.
- Se verifica por documento que la variante tiene el mismo largo; si no (algún carácter
  Unicode que cambia de largo al pasar a minúscula), el documento se excluye y se cuenta.
- Antes de cambiar el pipeline se mide la línea base: DisTEMIST prueba (F1 laxo y estricto
  por variante) y set sintético (índice exacto y recall de categorías por variante).

**FR2. Siglas en mayúscula para el modelo**
- Antes del NER, las siglas conocidas se pasan a mayúscula en **una copia** del texto que
  solo ve el modelo. El largo no cambia, así que las posiciones sirven igual y las
  menciones se recortan del texto real.
- La lista sale de `siglas.SIGLAS`, sin las que en minúscula son palabras comunes
  (`_SOLO_MAYUSCULAS`: "les", "ar"), más las que use el NER que falten (p. ej. "IRC").
  Cada sigla que se agregue o se quite se decide con la línea base y queda en el
  docstring.
- Se aplica en `analizar()` y en `menciones_del_modelo()`, para que la fila del modelo
  crudo de la evaluación vea lo mismo.

**FR3. Categoría con todas las menciones (bug de la fusión)**
- Si el texto de la entidad no mapea a Charlson, se prueba con el texto de cada una de sus
  menciones y gana la primera que mapea, en orden de aparición.
- El nombre que se muestra no cambia (sigue la regla de #110).

**FR4. Fallback difuso en las reglas de Charlson**
- Si ninguna regla aplica, `mapear` corrige las palabras del texto contra un vocabulario
  de palabras clave de las reglas ("diabetes", "epoc", "cirrosis", "insuficiencia",
  "demencia"…) con RapidFuzz (ya es dependencia) y vuelve a correr las reglas sobre el
  texto corregido.
- Solo palabras de 5 letras o más y a distancia de edición 1 (Damerau-Levenshtein, para
  contar el intercambio de letras vecinas). Una palabra que ya está en el vocabulario no
  se toca. Si una palabra queda a distancia 1 de dos claves distintas, no se corrige.
- La regla que aplicó por corrección queda marcada en la entidad
  (`Entidad.categoria_por_correccion: bool`, también en `to_dict()`), para que la demo
  pueda decir "¿quiso decir…?" y el auditor sepa que no es una coincidencia exacta.
- Los umbrales se fijan con la semilla de desarrollo (FR6), nunca con las de reporte.

**FR5. Demo**
- La entidad con `categoria_por_correccion` muestra una marca discreta en la lista (p. ej.
  "corregido: diabetis → diabetes").

**FR6. Medición final y reporte**
- **DisTEMIST prueba** con todas las variantes, antes y después: es la cifra con peso, porque
  no se usó para ajustar nada de este subproyecto. Mide el NER (FR2) con el F1 global y con
  el recall en las menciones que son siglas; DisTEMIST no anota Charlson.
- **Set sintético con errores inyectados**: los umbrales del FR4 se ajustan con la semilla
  0; el reporte usa las semillas 1 a 5 (media y rango). El reporte dice que el set ya se
  vio en mediciones anteriores y que los errores inyectados no son la distribución real de
  errores de un médico: es una prueba de estrés, no una garantía.
- **Falsos positivos del fallback**: en DisTEMIST prueba con el texto original, se listan
  las entidades que solo reciben categoría por corrección, y se revisan a mano para el
  reporte (cuántas son y cuántas están bien).
- El texto original no empeora: F1 de DisTEMIST y set sintético v3 iguales o mejores.
- `docs/evaluacion.md` (sección nueva), README y `Resultado`/`Entidad` documentados.

## Non-Functional Requirements

- Un PR por sub-issue, en serie (FR2, FR3 y FR4 tocan el pipeline o las reglas).
- Tests con frases sintéticas cortas, incluidos los casos que **no** deben cambiar.
- El tiempo por nota no sube más de un 5 % (las siglas y la corrección corren sobre el
  texto, no sobre el modelo; la corrección solo corre cuando ninguna regla aplicó).
- `to_dict()` compatible: solo se agrega un campo.
- La caché del modelo de la evaluación se indexa por el texto que ve el modelo, así que las
  variantes no chocan con lo ya cacheado.

## Success Criteria

- DisTEMIST prueba, variante `minusculas`, **recall laxo en las siglas** del gold: el FR2
  recupera al menos la mitad de lo que se pierde frente al original, y se reporta también
  la diferencia sobre las mismas 90 siglas. (Ajuste del 2026-11-05, con la línea base de
  #112 y antes de tocar el pipeline: el F1 global de DisTEMIST no baja en minúscula,
  0,860 → 0,862, porque solo el 3,5 % de sus menciones son siglas, así que una meta sobre
  el F1 global no distinguía el efecto. En esas 90 siglas, el modelo crudo pasa de 85 a
  51 encontradas y el pipeline de 46 a 25.)
- Las siglas que agrega el FR2 **no** se sacan de los falsos negativos de DisTEMIST prueba:
  esa partición no se usa para ajustar.
- Fuera de este subproyecto: en el texto original, el pipeline pierde 39 de las 85 siglas
  que el modelo encuentra, por el filtro de longitud (issue #56). El FR2 no lo arregla:
  con el texto en minúscula, el techo del pipeline sigue siendo el del original.
- DisTEMIST prueba, texto original: F1 laxo y estricto no bajan más de 0,005.
- Set sintético, variante `minusculas`: índice exacto de 88 o más (línea base exploratoria:
  80).
- Set sintético con errores al 10 % (semillas 1 a 5): el índice exacto medio sube al menos
  3 puntos frente a la línea base.
- Set sintético v3 con el texto original sin regresiones (índice exacto 92).
- Falsos positivos del fallback en DisTEMIST: menos del 10 % de las entidades corregidas.
- Los ejemplos de US1 y US2 son tests.

## Constraints & Assumptions

- No se puede mirar notas reales para calibrar los errores. Las variantes son una
  aproximación: minúsculas y sin tildes son comunes en notas digitadas en Colombia; los
  errores inyectados son uniformes y los de un médico no lo son (abreviaturas propias,
  letras vecinas en el teclado, palabras pegadas).
- Las métricas del set sintético ya no son ciegas; el peso lo lleva DisTEMIST.
- Las metas son metas, no condiciones para mergear: si no se alcanzan, el reporte dice
  cuánto se llegó y por qué.

## Out of Scope

- Corregir la ortografía del texto que ve el modelo (solo se cambian siglas a mayúscula).
- Errores que insertan o borran letras en la prueba de estrés de DisTEMIST (mueven los
  offsets). El fallback del FR4 sí los cubre, y se prueban con tests y con el set
  sintético.
- Palabras pegadas o separadas ("insuficienciacardiaca"), abreviaturas propias de cada
  médico y siglas nuevas fuera de las de `siglas.py`.
- Reentrenar o ajustar el modelo con texto en minúscula (iría con #77 o un experimento
  aparte).
- Las reglas de negación e incertidumbre con errores ("nieaga"): se mide si empeoran, pero
  no se corrigen aquí.

## Dependencies

- Epic de evaluación (#39): runner de DisTEMIST, caché del modelo y set sintético.
- `siglas.py` (#24) y las reglas de Charlson.
- Va antes del despliegue (#102): la demo pública debe aguantar texto en minúscula.
