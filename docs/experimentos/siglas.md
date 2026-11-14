# Experimento de siglas (FR3b, issue #8)

## Pregunta

La tesis descartaba toda entidad de 4 caracteres o menos. Eso elimina siglas como HTA, DM2
o EPOC, que en una nota corta suelen ser la única mención de la comorbilidad. FR3 propone
una lista blanca de siglas (`SIGLAS_CLINICAS`). Antes de darla por buena hay que medir con
el modelo real:

1. si el modelo detecta las siglas como entidad, y cuáles no;
2. cuánto ruido corto entra con cada variante del filtro de longitud.

## Método

- **Notas**: 40 notas **sintéticas** escritas desde cero para el experimento
  (`experimentos/siglas/notas.jsonl`), con el estilo de notas de ingreso, evolución y
  valoración prequirúrgica en Colombia. Mezclan siglas y nombres completos, negaciones
  ("niega HTA"), antecedentes familiares ("padre con HTA") y siglas que no son enfermedad:
  signos vitales (TA, FC, FR, PA), laboratorios (Hb, Cr, BUN, PCR, TSH), exámenes (EKG,
  TAC, Rx), medicamentos (ASA, IECA), vías (VO, IV, SC) y administrativas (EPS, IPS, SOAT,
  UCI). Las notas s36 a s40 se agregaron después de una primera corrida para probar sin
  texto pegado las siglas candidatas a la lista blanca, palabras cortas (asma, gota, tos)
  y el pronombre "les" junto a la sigla LES.
- **Anotación**: cada nota tiene a mano sus entidades esperadas (79 siglas y 73 nombres
  completos), con estado confirmada, negada o familiar.
- **Modelo**: `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` con `ExtractorHF`, cargado una sola
  vez. Cada nota pasa por el modelo una vez y las tres variantes reusan esa salida.
- **Variantes** del parámetro `filtro_longitud` de `extraer_candidatos`:
  - `tesis`: descarta 4 caracteres o menos, sin excepciones;
  - `lista_blanca`: igual, salvo `SIGLAS_CLINICAS` (por defecto);
  - `ninguno`: sin filtro de longitud (los demás filtros de basura siguen).
- **Métricas**: una entidad esperada cuenta como conservada si alguna mención de un
  candidato se solapa con ella (así "ACV hemorrágico" conserva ACV). La detección del
  modelo se mide sobre su salida cruda, antes de fusión y filtros. Es **ruido corto** todo
  candidato de 4 caracteres o menos que no se solapa con ninguna entidad esperada.
- **Reproducir**: `uv run python experimentos/siglas/correr.py`. Imprime las tablas y guarda
  el detalle en `experimentos/siglas/resultados.json`.

## Hardware y tiempo

- Intel Core Ultra 7 255U, solo CPU (12 hilos de torch), 30 GB de RAM.
- torch 2.15.0+cpu, transformers 5.21.0, Python 3.12.
- Carga del modelo: unos 15 s la primera vez (desde la caché de Hugging Face en disco) y
  unos 3 s con el disco en caché del sistema operativo.
- Inferencia de las 40 notas: unos 1,8 s (unos 45 ms por nota). Comando completo, con
  la importación de torch: unos 35 s.

## Resultados

### Detección del modelo

El modelo **detecta como entidad todas las siglas probadas**: 78 de 79 ocurrencias, de 29
siglas distintas. La única que se escapa es "VIH" en "Niega VIH." (s21); en otras dos notas
sí la detecta. No hay ninguna sigla que el modelo no detecte nunca.

| Sigla | Ocurrencias | Detectadas | En la lista blanca original | Texto crudo del modelo |
|---|---|---|---|---|
| ACV | 4 | 4 | sí | `ACV`, `ACV hemorrágico` |
| AIT | 1 | 1 | sí | `AIT` |
| DM | 2 | 2 | sí | `DM`, `DM tipo 2` |
| DM1 | 1 | 1 | sí | `DM1` |
| DM2 | 7 | 7 | sí | `DM2`, `DM2 no insulinorre` |
| EAP | 1 | 1 | sí | `EAP` |
| ECV | 2 | 2 | no | `ECV`, `ECV hemorrágico` |
| EPOC | 6 | 6 | sí | `EPOC`, `EPOC exacerbado`, `EPOC oxigenodependiente`, … |
| ERC | 4 | 4 | sí | `ERC`, `ERC estadio 4 secundaria a nefropatía diabética` |
| ERGE | 1 | 1 | no | `ERGE` |
| EVP | 1 | 1 | no | `EVP` |
| FA | 4 | 4 | sí | `FA`, `FA paroxística`, `FA permanente` |
| HIV | 1 | 1 | no | `HIV` |
| HPB | 1 | 1 | no | `HPB` |
| HTA | 13 | 13 | sí | `HTA` |
| IAM | 5 | 5 | sí | `IAM` |
| ICC | 3 | 3 | sí | `ICC` |
| IMA | 1 | 1 | no | `IMA` |
| IRC | 2 | 2 | sí | `IRC` |
| IVU | 2 | 2 | no | `IVU`, `IVU a repetición` |
| LES | 2 | 2 | no | `LES` |
| SAHOS | 1 | 1 | no (5 letras) | `SAHOS` |
| SCA | 1 | 1 | no | `SCA` |
| SIDA | 1 | 1 | sí | `SIDA` |
| TBC | 2 | 2 | no | `TBC`, `TBC pulmonar` |
| TEP | 3 | 3 | sí | `TEP` |
| TIA | 1 | 1 | no | `TIA` |
| TVP | 3 | 3 | no | `TVP`, `TVP en miembro inferior` |
| VIH | 3 | 2 | sí | `VIH`, `coinfección con VIH` |

### Por variante

"Lista blanca original" es la de 16 siglas de #7; "lista blanca ampliada" es la que queda
en este PR (ver la decisión).

| Variante | Siglas conservadas | Nombres completos conservados | Candidatos | Candidatos no esperados | Ruido corto (≤ 4) |
|---|---|---|---|---|---|
| `tesis` | 15/79 (19 %) | 47/73 | 64 | 5 | 0 |
| `lista_blanca` original | 65/79 (82 %) | 47/73 | 114 | 5 | 0 |
| `lista_blanca` ampliada | **78/79 (99 %)** | **50/73** | 130 | 5 | 0 |
| `ninguno` | 78/79 (99 %) | 50/73 | 130 | 5 | 0 |

- **`tesis`** solo conserva las 15 siglas que el modelo fusionó con texto vecino
  ("FA paroxística", "ACV hemorrágico", "EPOC exacerbado"). Pierde 64 de 79: en 33 de las
  40 notas desaparece al menos una comorbilidad que solo estaba escrita como sigla.
- **Lista blanca original** pierde 13 ocurrencias que el modelo sí detecta (la 14.ª es el
  "Niega VIH." que el modelo no marca): SCA, TIA, HIV, ECV e IMA (equivalentes de
  categorías de Charlson que las reglas de mapeo ya reconocen) y TVP, LES, IVU, TBC, ERGE
  y HPB. EVP sobrevivió solo porque la fusión la unió con "úlcera en pie derecho"; sola se
  habría perdido. Pierde además "asma" y "gota", palabras completas de 4 letras.
- **Ruido corto: cero en las tres variantes.** Aun sin filtro de longitud, el modelo no
  marcó ninguna sigla de signos vitales, laboratorio, examen, medicamento, vía ni
  administrativa, ni el pronombre "les" ("Se les explica a los familiares"), ni "tos".
- Los 5 candidatos no esperados son largos y no dependen del filtro: "Alergias" (dos veces),
  "caídas", "lesiones" y "enfermedades crónicas".
- Los nombres completos que se pierden en todas las variantes (disnea, fiebre, dolor
  torácico, ortopnea, palpitaciones…) son síntomas o signos: DisTEMIST anota enfermedades y
  el modelo no los marca. No dependen del filtro de longitud.

## Decisión

> **Nota (#56).** Desde #56 el filtro por defecto es `minimo` y la lista blanca solo
> decide las palabras en minúscula de 2 letras o menos. Detalle en
> [`normalizacion.md`](../normalizacion.md#filtro-de-longitud-56).

**Se mantiene la variante `lista_blanca` (por defecto) y se amplía `SIGLAS_CLINICAS`** con
lo que el experimento mostró que el modelo detecta y el filtro perdía:

- equivalentes de Charlson: ECV, SCA, IMA, TIA, HIV, EVP;
- otras comorbilidades frecuentes en notas colombianas: TVP, LES, IVU, TBC, ERGE, HPB;
- términos completos de 4 letras: ASMA, GOTA.

No se quitó ninguna sigla: ninguna de la lista metió ruido.

Por qué no `ninguno`, si en este conjunto da lo mismo:

- Las notas sintéticas son limpias. El filtro de la tesis se pensó para las notas reales,
  con fragmentos truncados, errores de digitación y restos de plantillas, que no se pueden
  traer a este repo. La lista blanca deja pasar solo términos conocidos y conserva la
  protección de la tesis contra lo demás.
- Con la lista ampliada, `lista_blanca` y `ninguno` dan el mismo resultado aquí: no se
  pierde nada medible por ser conservador.

Por qué no `tesis`: pierde el 81 % de las siglas, y con ellas comorbilidades y puntos de
Charlson.

## Límites y pendientes

- 40 notas escritas por una sola persona: sirven para ver si el modelo detecta las siglas,
  no para estimar precisión ni exhaustividad generales.
- Resuelto en #24: "ASMA" y "GOTA" pasaron a `PALABRAS_CORTAS_CLINICAS`, y las siglas que
  se escriben igual que una palabra común ("LES" frente a "les", "TIA" frente a "tia") solo
  pasan en mayúsculas (`SIGLAS_SENSIBLES_A_MAYUSCULAS`).
- Volver a medir `ninguno` contra `lista_blanca` en el subproyecto de evaluación, con las
  anotaciones de DisTEMIST, que sí permiten contar el ruido corto sobre texto clínico real
  público.
