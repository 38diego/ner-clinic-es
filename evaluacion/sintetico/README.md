# Set sintético reservado (contexto y Charlson)

100 notas clínicas **100 % sintéticas**, escritas para este proyecto y anotadas a mano, para
medir el estado de cada entidad (confirmada, negada, familiar), el índice de Charlson y el
cruce con diagnósticos registrados (PRD `docs/prds/evaluacion.md`, FR8 y FR9; issue #44).

Ninguna nota sale de datos de pacientes: no hay nombres, cédulas ni instituciones. El estilo
imita notas de IPS colombianas (abreviaturas, mayúsculas, alguna errata a propósito, como
"pacinete" o "hemodialisis").

## Set reservado

- Las notas se anotaron desde lo que dice el texto, como lo haría un codificador. **No se
  corrió `analizar()` ni el modelo sobre ellas** antes de anotarlas ni de validarlas.
- No se usan para ajustar reglas. Se miden una sola vez y esa cifra es la que se reporta. Si
  después se cambian reglas por lo que se vio, el reporte lo dice y la cifra nueva va aparte.
- Límites:
  - Las escribimos nosotros, que conocemos las reglas, así que las cifras son una cota
    optimista.
  - Ninguna anotación la validó un clínico.

## Validación: doble anotación ciega

Un segundo anotador independiente (B) anotó las 100 notas sin ver la anotación original
(A): solo tuvo los textos y los criterios de este README. Las dos anotaciones están en
`anotaciones/a.jsonl` y `anotaciones/b.jsonl`. La tabla sale de
`uv run python evaluacion/sintetico/acuerdo.py`. Dos entidades son la misma si una
contiene a la otra.

| Medida | Valor |
|---|---|
| Entidades que anotaron los dos | 333 |
| Solo el anotador A / solo el B | 4 / 6 |
| F1 de acuerdo en entidades | 0.985 |
| Acuerdo en el estado | 1.000 |
| Kappa de Cohen del estado | 1.000 |
| Acuerdo en la categoría (confirmadas) | 0.995 |
| Índice de Charlson igual | 98 de 100 notas |

**Cuidado al leer el acuerdo:** los dos anotadores son instancias del mismo modelo (Claude),
con los mismos criterios escritos. Así que el acuerdo mide sobre todo que las anotaciones
son consistentes y que no hay descuidos, no que dos expertos independientes coincidan.
Sirvió para encontrar los 10 casos de abajo, no para validar el criterio clínico.

### Adjudicación

Cada discrepancia se resolvió con una regla, que quedó agregada a los criterios:

| Nota | A | B | Regla | Gold final |
|---|---|---|---|---|
| sin038, sin098 | "hemiparesia derecha" como hemiplejía | no la anota | La hemiparesia del examen de un ACV agudo es un signo del evento; solo la secuelar es hemiplejía | B (índices 1 y 2) |
| sin013, sin096 | — | "cardiopatías" negada | Los términos genéricos de enfermedad se anotan, sin categoría | B |
| sin064 | — | "infecciones oportunistas" negada | Igual que la anterior | B |
| sin072 | — | "riñón de mieloma" | Una complicación con nombre propio se anota aunque su categoría ya esté cubierta | B, sin categoría |
| sin089 | "DM2 con nefropatía y retinopatía diabéticas" | "DM2" + "nefropatía" + "retinopatía diabéticas" | Granularidad: la mención completa es una sola entidad; el índice es el mismo | A |
| sin097 | "cancer gastrico metastasico" en la impresión diagnóstica | solo la primera mención | Las menciones con forma distinta en la nota se anotan todas | A |
| sin099 | "VIH" y "SIDA" | solo "VIH" | Igual que la anterior; la categoría suma una vez | A |

## Formato (`notas.jsonl`)

Una nota por línea, con el formato de `experimentos/errores_tesis/notas.jsonl` y un campo
opcional más por entidad:

| Campo | Qué es |
|---|---|
| `id` | `sin001` … `sin100` |
| `texto` | la nota |
| `registrados` | opcional: diagnósticos registrados (descripciones CIE-10) |
| `entidades[].texto` | la enfermedad tal como aparece literal en la nota |
| `entidades[].estado` | `confirmada`, `negada` o `familiar` |
| `entidades[].categoria` | opcional: categoría de Charlson esperada (valores de `CategoriaCharlson`); solo si la entidad pertenece a una |
| `entidades[].registrado` | opcional: si la entidad corresponde a un diagnóstico de `registrados`; en esas notas lo llevan todas las confirmadas |
| `indice` | índice de Charlson esperado de la nota |

`categoria` se agregó para medir precisión y recall por categoría (FR9). También va en
negadas y familiares, aunque no sumen.

## Criterios de anotación

- Se anotan **enfermedades** (lo que busca el modelo entrenado con DisTEMIST), no síntomas
  ni signos (disnea, fiebre, ascitis, afasia) ni hábitos (tabaquismo).
- **Índice**: pesos de Charlson et al. (1987) sin ajuste por edad. Solo suman las
  confirmadas, cada categoría una vez, y con jerarquía leve → severa (diabetes, hepática,
  tumor): si están las dos, solo suma la severa.
- Decisiones clínicas de `docs/errores-tesis.md`: diabetes sin tipo con cetoacidosis es
  diabetes sin complicación crónica (E14.1); ACV con infarto o isquemia es cerebrovascular;
  VIH sin SIDA pesa 6.
- **Hepática**: hepatitis crónica, hepatitis autoinmune o alcohólica y cirrosis compensada
  sin hipertensión portal son leves; cirrosis con hipertensión portal, várices esofágicas,
  encefalopatía o Child-Pugh C son severas.
- **Diabetes complicada**: con daño de órgano blanco (nefropatía, retinopatía, neuropatía,
  pie diabético). La diabetes gestacional no es categoría de Charlson.
- **Reumatológica**: lupus, artritis reumatoide, polimiositis, polimialgia reumática y
  esclerosis sistémica. Espondilitis anquilosante, artritis psoriásica y Sjögren no cuentan.
- **Tumor**: leucemia, linfoma y mieloma son tumor maligno; "metastásico" o metástasis a
  distancia es tumor metastásico.
- **Renal**: ERC en cualquier estadio, como el pipeline (convención del proyecto).
- Úlceras por presión o arteriales no son úlcera péptica; HTA, FA, dislipidemia e
  hipotiroidismo no tienen categoría.
- **Hemiplejía**: solo la hemiparesia o hemiplejía secuelar o establecida. La del examen de
  un ACV agudo es un signo del evento y no se anota.
- Los términos genéricos de enfermedad ("cardiopatías", "infecciones oportunistas") se anotan
  sin categoría. Las complicaciones con nombre propio ("riñón de mieloma") también, aunque
  su categoría ya esté cubierta.
- Si una enfermedad aparece con formas distintas en la nota ("VIH" y "SIDA"), se anotan
  todas; la categoría suma una vez.
- `registrado`: `true` si alguno de los `registrados` describe la misma enfermedad, aunque
  sea con otro nivel de detalle.

## Cobertura

Generada con `uv run python evaluacion/sintetico/cobertura.py`, que solo lee las
anotaciones. `tests/test_set_sintetico.py` comprueba el formato, los mínimos del issue #44
y que esta tabla sea la salida actual del script.

| Qué | Cantidad |
|---|---|
| Notas | 100 |
| Entidades anotadas | 339 |
| Entidades en estado `confirmada` | 217 |
| Entidades en estado `negada` | 71 |
| Entidades en estado `familiar` | 51 |
| Entidades que son siglas | 115 |
| Notas con diagnósticos registrados | 18 |
| Entidades con `registrado` esperado | 69 |
| Notas de una línea (≤ 120 caracteres) | 58 |
| Notas de más de 1500 caracteres | 6 |
| Notas con jerarquía leve/severa: diabetes | 6 |
| Notas con jerarquía leve/severa: hepática | 4 |
| Notas con jerarquía leve/severa: tumor | 5 |

| Categoría de Charlson | Entidades confirmadas |
|---|---|
| `infarto_miocardio` | 7 |
| `insuficiencia_cardiaca` | 8 |
| `vascular_periferica` | 5 |
| `cerebrovascular` | 6 |
| `demencia` | 7 |
| `pulmonar_cronica` | 12 |
| `reumatologica` | 8 |
| `ulcera_peptica` | 7 |
| `hepatica_leve` | 10 |
| `diabetes` | 14 |
| `diabetes_complicada` | 8 |
| `hemiplejia` | 5 |
| `renal_cronica` | 10 |
| `tumor_maligno` | 11 |
| `hepatica_severa` | 10 |
| `tumor_metastasico` | 8 |
| `vih_sida` | 7 |

| Forma de negación (la más cercana) | Entidades negadas |
|---|---|
| niega | 35 |
| se descarta | 17 |
| no presenta | 7 |
| sin | 6 |
| negativo para | 4 |
| ni / o (enumeración) | 2 |

| Parentesco (el más cercano) | Entidades familiares |
|---|---|
| madre | 16 |
| padre | 11 |
| hermano/a | 10 |
| abuelo/a | 4 |
| tío/a | 4 |
| hijo/a | 4 |
| nieto/a | 1 |
| primo/a | 1 |

La forma de negación y el parentesco se cuentan por la palabra clave más cercana antes de la
entidad, en la misma oración. "Sigla" es una entidad escrita toda en mayúsculas y sin
espacios (HTA, DM2, EPOC).
