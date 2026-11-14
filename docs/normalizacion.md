# Experimento de normalización: bitácora

Cifras de desarrollo del experimento de normalización (PRD
[`normalizacion`](prds/normalizacion.md), epic #128). Sirven para ajustar y comparar
métodos. **No son la cifra final**: esa se mide una sola vez en CodiEsp prueba (#134) y va
a [`evaluacion.md`](evaluacion.md). Datos y licencias en [`datos.md`](datos.md).

## Línea base: las reglas actuales (#130)

```bash
uv run python -m evaluacion codiesp   # → evaluacion/resultados/codiesp_reglas_desarrollo.json
```

**Qué se mide.** Las reglas de Charlson sobre el texto de cada mención de diagnóstico del
gold de CodiEsp-X, en entrenamiento y desarrollo (750 casos, 10.637 menciones de
diagnóstico). La categoría gold es la que la tabla de Quan le da al código CIE-10-ES
anotado: 1.100 menciones tienen una. Es la normalización aislada: sin NER ni contexto.

- Verdadero positivo: las reglas dan la misma categoría que el gold.
- Falso positivo: dan una categoría que el gold no tiene (otra categoría o ninguna).
- Falso negativo: no dan la categoría del gold.
- IC del 95 % por bootstrap sobre documentos (1.000 réplicas).

### Resultado

| Método | Variante | Precisión | Recall | F1 |
|---|---|---|---|---|
| Reglas | original | 0,860 [0,832; 0,887] | 0,719 [0,679; 0,757] | 0,783 [0,754; 0,810] |
| Reglas + corrector (#115) | original | 0,860 [0,833; 0,888] | 0,723 [0,683; 0,761] | **0,786** [0,756; 0,813] |
| Reglas | errores en el 10 % de las palabras | 0,858 | 0,636 | 0,731 [0,698; 0,761] |
| Reglas + corrector | errores en el 10 % de las palabras | 0,863 | 0,700 | 0,773 [0,744; 0,802] |

- **En minúscula y sin tildes da exactamente lo mismo** que el original: las reglas ya
  pasan el texto a minúscula y sin tildes.
- **El corrector de #115 sostiene el recall con errores**: con el 10 % de las palabras mal
  escritas, el recall cae 0,08 sin corrector y 0,02 con él.
- Por partición (reglas + corrector): entrenamiento F1 0,797; desarrollo 0,759.
- **Las reglas dan categoría a 924 menciones**; 121 de ellas no tienen categoría en el gold.
- **Código CIE-10.** En el 49,9 % de las menciones gold con categoría, el código de las
  reglas coincide con el gold en sus 3 primeros caracteres. De las 551 que no coinciden,
  297 no reciben código (son los falsos negativos) y 254 reciben otro. Sobre las que sí
  reciben código, el acierto es 68 % (549 de 803). Los códigos distintos son sobre todo
  los que las reglas dan por defecto o de la CIE-10 de la OMS: C80 para cualquier tumor
  fuera de su lista (65), B24 (52) y E14.9 (49), que no existen en la CIE-10-ES.

### Por categoría (reglas + corrector, original)

| Categoría | Menciones gold | P | R | F1 |
|---|---|---|---|---|
| Tumor maligno | 350 | 0,930 | 0,720 | 0,812 |
| Tumor metastásico | 154 | 0,986 | 0,922 | 0,953 |
| Renal crónica | 93 | 0,833 | 0,591 | 0,692 |
| Diabetes | 81 | 0,941 | 0,988 | 0,964 |
| Hepática leve | 63 | 0,473 | 0,413 | 0,441 |
| Cerebrovascular | 56 | 0,571 | 0,357 | 0,440 |
| Pulmonar crónica | 54 | 1,000 | 0,963 | 0,981 |
| VIH/sida | 52 | 0,912 | 1,000 | 0,954 |
| Hemiplejia | 50 | 0,867 | 0,520 | 0,650 |
| Vascular periférica | 35 | 0,722 | 0,371 | 0,491 |
| Hepática severa | 35 | 0,917 | 0,629 | 0,746 |
| Infarto de miocardio | 20 | 0,900 | 0,900 | 0,900 |
| Úlcera péptica | 18 | 1,000 | 0,278 | 0,435 |
| Reumatológica | 16 | 0,733 | 0,688 | 0,710 |
| Insuficiencia cardiaca | 14 | 0,375 | 0,857 | 0,522 |
| Demencia | 5 | 0,455 | 1,000 | 0,625 |
| Diabetes complicada | 4 | 1,000 | 1,000 | 1,000 |

Las categorías con pocas menciones (demencia, diabetes complicada, insuficiencia cardiaca)
tienen IC muy anchos, que están en el JSON. En las que aparecen en muy pocos documentos el
IC no es interpretable: una réplica del bootstrap que no sortea ninguno da F1 = 0 sin casos
(diabetes complicada, en 3 documentos, queda en [0; 1] con 4 aciertos de 4).

### De dónde vienen los errores

Los falsos positivos y negativos con su texto quedan en
`data/evaluacion/codiesp_reglas_errores.json` (no se versiona). Son de dos clases:

**Huecos de las reglas**: formas de escribir que nadie previó. Es lo que los embeddings
tienen que cubrir.

- Cerebrovascular: "hemorragia subaracnoidea", "ictus", "accidente isquémico transitorio",
  "hematoma subdural", "HSD" (#123).
- Úlcera péptica: "ulcus duodenal", "ulcus gástrico".
- Tumor maligno: "seminoma", "glioma", "HCC", "tumor de Wilms", "GIST".
- Hemiplejia: "paresia" (Quan cuenta G83 como hemiplejia).
- Hepática leve: "VHC", "hepatopatía alcohólica".

**Diferencias de definición entre las reglas y Quan**: la regla decide a propósito algo
distinto de la tabla. No son errores de escritura y no hay que "arreglarlos" a ciegas.

- "Insuficiencia renal" sin "crónica", "fracaso renal", "fallo renal" (30 menciones):
  22 de 30 están codificadas N19 (insuficiencia renal no especificada), que Quan cuenta
  como renal crónica, y las otras 8, N18. La regla exige que diga crónica, para no contar una
  insuficiencia renal aguda.
- "Hepatitis B" y "hepatitis C" (28): CodiEsp las codifica B19 (hepatitis viral sin
  especificar), que Quan no cuenta (solo B18, la crónica). La regla las toma como hepática
  leve.
- "Cardiopatía isquémica" (17, I25.9 e I25.10) y "cardiopatía hipertensiva" (3, I11.9):
  Quan no las cuenta como insuficiencia cardiaca; la regla sí.
- 7 tumores metastásicos salen como tumor maligno.

**Falsos positivos de una palabra suelta**: "paraganglioma de cuerpo carotídeo" (D44.6, 3
veces en un caso) sale cerebrovascular; "MAV" (malformación arteriovenosa, 5), también.

### Estado de las menciones gold

CodiEsp anota también las menciones negadas y las de familiares (ver
[`datos.md`](datos.md#qué-anota-codiesp-130)). Con nuestro módulo de contexto, de las 1.100
menciones gold con categoría, 937 salen confirmadas (recall de las reglas 0,726), 76
negadas (0,671), 73 posibles (0,712) y 14 de familiares (0,857). El método de
normalización no cambia con el estado; esta cuenta importa para #134.

## Embeddings sin sinónimos (#131)

```bash
uv run python -m evaluacion normalizacion --modelo sapbert   # uno por modelo, en su proceso
uv run python -m evaluacion normalizacion --tabla            # compara con las reglas
# → evaluacion/resultados/normalizacion_<modelo>_desarrollo.json
```

**Qué se mide.** Lo mismo que en la línea base, con el mismo código
(`charlson_codiesp.metricas`): el texto de cada mención gold se convierte en un vector y se
busca el código del catálogo CIE-10-ES (98.288 códigos) con la descripción más parecida
(coseno). La categoría sale del código con la tabla de Quan. Si la similitud no llega al
umbral, la mención queda sin código. El catálogo va tal cual, sin sinónimos: eso es #132.

- **Umbral.** El de mayor F1 de categoría en **desarrollo** (variante original), en una
  grilla de 0 a 1 cada 0,01. La cifra de entrenamiento es la más limpia: no se usó para
  elegirlo.
- **Un código padre sin categoría se lee como su hijo `.9`.** El catálogo trae los códigos
  de 3 caracteres ("E11 Diabetes mellitus tipo 2"), cuya descripción corta suele ser la
  más parecida a una mención. Quan no le da categoría a E11 (reparte sus hijos entre
  diabetes con y sin complicación). Sin esta regla, el recall de diabetes con e5-small era
  0,04 (en la primera corrida, que no se versionó). El gold de CodiEsp no usa esos códigos
  padre. La regla le cambia la categoría a 13: E10, E11, E13, G83, I09, I27, I42, I73, K55,
  K70, K72, K76 y Z95 (Z95.9, implante cardiovascular sin especificar, es vascular
  periférica en Quan; Z95 abarca también marcapasos y *bypass*).

### Modelos

| Clave | Modelo (revisión fija en `evaluacion/normalizador.py`) | *Pooling* | Licencia | Pesos |
|---|---|---|---|---|
| `sapbert` | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | CLS, en minúscula | **No declarada**; entrenado con UMLS, que tiene licencia propia | 1,1 GB |
| `e5_base` | `intfloat/multilingual-e5-base` | media, "query: " | MIT | 1,1 GB |
| `e5_small` | `intfloat/multilingual-e5-small` | media, "query: " | MIT | 0,47 GB |
| `bsc_bio_ehr` | `PlanTL-GOB-ES/bsc-bio-ehr-es`, sin afinar para similitud | media | Apache-2.0 | 0,5 GB |

SapBERT usa el tokenizador de `FacebookAI/xlm-roberta-base` (MIT): su repo solo trae el
`sentencepiece`, idéntico (mismo SHA-256), que `transformers` 5 no convierte sin el paquete
`sentencepiece`.

### Resultado (entrenamiento + desarrollo, 1.100 menciones gold con categoría)

| Método | Umbral | P | R | F1 [IC 95 %] | F1 entrenamiento | F1 errores 10 % | Código a 3, top-1 / top-5 (todas las menciones) |
|---|---|---|---|---|---|---|---|
| Reglas + corrector (#130) | — | 0,860 | 0,723 | **0,786** [0,756; 0,813] | 0,797 | 0,773 | — |
| SapBERT | 0,54 | 0,729 | 0,684 | **0,706** [0,682; 0,731] | 0,710 | 0,672 | 0,611 / 0,771 |
| e5-base | 0,87 | 0,502 | 0,459 | 0,480 [0,447; 0,514] | 0,476 | 0,465 | 0,373 / 0,557 |
| e5-small | 0,92 | 0,569 | 0,312 | 0,403 [0,362; 0,441] | 0,397 | 0,390 | 0,384 / 0,576 |
| bsc-bio-ehr sin afinar | 0,85 | 0,474 | 0,276 | 0,348 [0,311; 0,384] | 0,343 | 0,316 | 0,182 / 0,260 |

- **SapBERT es el único que se acerca a las reglas**, y sin sinónimos no las alcanza: le
  faltan 0,08 de F1, con IC que no se solapan.
- **Un modelo de similitud general (e5) no sirve aquí**, y uno sin afinar (bsc-bio-ehr)
  menos. Es lo que explica el resultado de la tesis (`Emmbeding.ipynb`): usaba e5-base
  para filtrar 15 candidatos y bsc-bio-ehr con *mean pooling* para elegir, con un umbral de
  0,70. En ese modelo la similitud del primer candidato casi nunca baja de 0,85 (en
  desarrollo, el 4,9 % de las menciones; ninguna baja de 0,70), así que ese umbral no
  filtraba nada. Además
  limpiaba el catálogo quitando "no especificado", "otros" o "debido a", con lo que códigos
  distintos quedaban con la misma descripción, y no tenía un gold para medir.
- **El umbral filtra poco en SapBERT**: el F1 casi no cambia entre 0 y 0,55 y después cae.
  Sus errores tienen similitud alta, así que no se pueden descartar por confianza. En
  e5-small el umbral elegido (0,92) gana 0,002 de F1 en desarrollo a cambio de mucho
  recall (0,31 frente a 0,40 sin umbral): el criterio de elección no favorece el recall,
  que es lo que pesa en la demo. No cambia la conclusión.
- **Las variantes**: la minúscula no le cambia nada a SapBERT (trabaja en minúscula); con
  errores al 10 %, cae 0,034 de F1, más que las reglas con corrector (0,013).
- **El código**: SapBERT acierta el código a 3 caracteres en el 61 % de todas las menciones
  de diagnóstico (77 % entre los cinco primeros). Las reglas no dan código fuera de
  Charlson. Las métricas de código son solo con la variante original; las variantes se
  miden en la categoría.

### Por categoría: reglas frente a SapBERT

| Categoría | Gold | F1 reglas | F1 SapBERT |
|---|---|---|---|
| Tumor maligno | 350 | **0,81** | 0,66 |
| Tumor metastásico | 154 | **0,95** | 0,81 |
| Renal crónica | 93 | 0,69 | **0,75** |
| Diabetes | 81 | **0,96** | 0,85 |
| Hepática leve | 63 | 0,44 | **0,50** |
| Cerebrovascular | 56 | 0,44 | **0,59** |
| Pulmonar crónica | 54 | **0,98** | 0,71 |
| VIH/sida | 52 | 0,95 | 0,97 |
| Hemiplejia | 50 | 0,65 | 0,67 |
| Vascular periférica | 35 | **0,49** | 0,42 |
| Hepática severa | 35 | 0,75 | **0,91** |
| Infarto de miocardio | 20 | **0,90** | 0,62 |
| Úlcera péptica | 18 | 0,43 | **0,87** |
| Reumatológica | 16 | 0,71 | 0,75 |
| Insuficiencia cardiaca | 14 | 0,52 | 0,49 |
| Demencia | 5 | 0,62 | 0,91 |
| Diabetes complicada | 4 | 1,00 | 0,67 |

En negrita, el mejor cuando la diferencia es de 0,05 o más en una categoría con 10 o más
menciones gold.

Los dos métodos fallan en sitios distintos. Las reglas ganan donde la enfermedad se escribe
casi siempre igual (tumores, diabetes, EPOC, infarto); SapBERT, donde hay muchas formas de
escribirla que nadie previó ("ulcus", "hemorragia subaracnoidea", hepatopatías). Es el
argumento para el híbrido (#133).

### Casos de #120 y #123

Con el catálogo real y sin umbral (tests `modelo` de `tests/test_normalizador.py`):

- "accidente cerebrovascular" → cerebrovascular en SapBERT y los dos e5: I63 (infarto
  cerebral) en SapBERT y e5-base, I69 (secuelas) en e5-small.
- "ictus" no lo resuelve **ningún** modelo: en el catálogo solo aparece en "Historia
  familiar de ictus" y en la escala NIHSS. SapBERT lo deja más cerca de "Terremoto" (X34).
  Es el caso típico que resuelven los sinónimos de #132.
- "linfoma de Hodgkin sin metástasis" sale como tumor maligno en los cuatro, pero en
  SapBERT con el código equivocado (C85.9, linfoma **no** Hodgkin): la categoría es la
  correcta por suerte.

### Costo (CPU de portátil, un proceso por modelo; no es la de Cloud Run)

| Modelo | Índice del catálogo | Construirlo | ms por mención (mediana / p95) | ms por mención en lote | Memoria máxima |
|---|---|---|---|---|---|
| SapBERT | 304 MB | 35 min | 30,1 / 66,6 | 15,7 | 2,6 GiB |
| e5-base | 304 MB | 39 min | 28,7 / 62,9 | 18,3 | 2,8 GiB |
| e5-small | 153 MB | 12 min | 21,0 / 44,1 | 6,1 | 2,1 GiB |
| bsc-bio-ehr | 304 MB | 26 min | 31,7 / 72,3 | 10,1 | 2,1 GiB |

- Los tiempos por mención varían un 10 % o más entre corridas (la latencia se mide sobre
  200 menciones de desarrollo); son órdenes de magnitud, no cifras finas.
- El índice se construye **una vez** y se guarda; en la imagen iría ya construido.
- La memoria es la del proceso completo (modelo, catálogo, índice y la evaluación). Con el
  NER al lado, SapBERT no cabría holgado en los 4 GiB de Cloud Run sin reducir el índice
  (p. ej. a float16, la mitad).
- Una nota con 20 menciones son unos 0,3 s más con SapBERT en lote.

### Lo que sigue

- **#132, sinónimos**: agregar al índice las menciones de entrenamiento con su código gold.
  Desde ahí entrenamiento ya no sirve para medir y la cifra de desarrollo pasa a ser la
  limpia.
- **#133, híbrido**: las reglas donde ganan y SapBERT donde no aplican.
- **Licencia de SapBERT**: hay que resolverla antes de llevarlo a la imagen pública.
- Los candidatos repiten descripciones (X34, X34.X, X34.XXX son "Terremoto"): el top-5
  aprovecharía mejor sin duplicados.

## Sinónimos de CodiEsp entrenamiento (#132)

```bash
uv run python -m evaluacion sinonimos --modelo sapbert
# → evaluacion/resultados/sinonimos_sapbert_desarrollo.json
```

**Qué cambia.** Al índice del catálogo se le agregan, como sinónimos, las menciones de
diagnóstico de CodiEsp **entrenamiento** con su código gold. Es la forma "sin regex" de
cubrir siglas y formas coloquiales: "ictus" no está en el catálogo, pero sí en las notas de
entrenamiento, con I63.9. Cada fila del índice guarda su origen, así que un candidato se
explica solo: `I63.9: parecido a 'ictus', mención de CodiEsp entrenamiento`.

- **Sinónimos.** 7.207 menciones de 500 documentos dan **3.338 sinónimos**. Para
  deduplicar, se colapsan los espacios y no se distingue mayúscula de minúscula. 82 textos
  tienen más de un código gold; queda el más frecuente, y el empate se resuelve por el
  código menor. 108 menciones quedan con un código distinto al suyo.
- **Sin fuga.** Desde aquí entrenamiento ya no sirve para medir: todas las cifras son de
  **desarrollo**. `verificar_origen` falla si un sinónimo viene de un documento de
  desarrollo (250 ids) o de prueba (250 ids). De prueba solo se leen los nombres de los
  archivos.
- **Umbral elegido en entrenamiento**, no en desarrollo, para que la cifra de desarrollo
  no quede ajustada:
  - Sin sinónimos, entrenamiento no está en el índice y se usa completo; sale 0,54, igual
    que en #131.
  - Con sinónimos se usa validación cruzada de 5 pliegues por documento: cada pliegue se
    mide con los sinónimos de los otros cuatro. Sale 0,69. Cada pliegue ve unos 4/5 de
    los sinónimos y el índice final todos, así que el umbral puede quedar algo corrido.
- Las reglas se miden con el mismo código, sobre las mismas menciones de desarrollo.

### Resultado (desarrollo, 250 documentos, 333 menciones gold con categoría)

| Método | Umbral | P | R | F1 [IC 95 %] | F1 errores 10 % | F1 no vistas | Código a 3, top-1 / top-5 |
|---|---|---|---|---|---|---|---|
| Reglas + corrector | — | 0,843 | 0,691 | 0,759 [0,704; 0,808] | 0,734 | 0,749 | — |
| SapBERT sin sinónimos | 0,54 | 0,723 | 0,673 | 0,697 [0,651; 0,744] | 0,662 | 0,663 | 0,612 / 0,767 |
| SapBERT con sinónimos | 0,69 | **0,921** | **0,844** | **0,881** [0,852; 0,910] | **0,852** | **0,802** | **0,870 / 0,932** |

- Con sinónimos, SapBERT supera a las reglas por 0,12 de F1, y los intervalos no se
  cruzan. Gana en precisión (de 43 a 24 falsos positivos) y en recall (de 103 a 52
  menciones perdidas).
- En minúscula da lo mismo que el original: SapBERT trabaja en minúscula. Sin tildes cambia
  poco: 0,889 con sinónimos y 0,685 sin ellos. Con 10 % de errores ortográficos pierde
  0,03, como las reglas.
- **Cuánto es memoria.** 2.027 de las 3.430 menciones de desarrollo (59 %) tienen un texto
  idéntico a un sinónimo, y el primer candidato es un sinónimo en 2.638. Es el mismo corpus
  y los mismos anotadores, así que la cifra completa es optimista para otras notas.
- **F1 no vistas** mide solo las 1.403 menciones cuyo texto no está entre los sinónimos
  (189 con categoría). Ahí la ventaja se achica pero sigue: 0,802 contra 0,749 de las
  reglas. Esas menciones se benefician de sinónimos parecidos aunque no idénticos (otra
  escritura u otras palabras alrededor).
- En notas colombianas el vocabulario no es el de CodiEsp, que son casos clínicos de
  España. Los sinónimos que hacen falta allá se agregan igual, como datos (US2).

### Por categoría (F1, desarrollo)

| Categoría | Gold | Reglas | SapBERT sin | SapBERT con |
|---|---|---|---|---|
| Tumor maligno | 112 | 0,780 | 0,618 | 0,810 |
| Tumor metastásico | 40 | 0,961 | 0,845 | 0,961 |
| Renal crónica | 31 | 0,745 | 0,813 | **0,984** |
| Hepática leve | 26 | 0,381 | 0,400 | **0,773** |
| Diabetes | 22 | 1,000 | 0,872 | 1,000 |
| Hemiplejia | 19 | 0,533 | 0,600 | **0,857** |
| Pulmonar crónica | 17 | 1,000 | 0,849 | 1,000 |
| Cerebrovascular | 14 | 0,400 | 0,643 | **0,846** |
| VIH/sida | 13 | 0,897 | 0,963 | 0,963 |
| Vascular periférica | 9 | 0,533 | 0,571 | 0,750 |
| Úlcera péptica | 8 | 0,769 | 0,875 | 0,933 |
| Hepática severa | 7 | 0,444 | 1,000 | 0,875 |
| Infarto de miocardio | 6 | 0,923 | 0,667 | 0,909 |
| Reumatológica | 4 | 0,727 | 0,571 | 1,000 |
| Insuficiencia cardiaca | 3 | 0,286 | 0,333 | 0,667 |
| Demencia | 2 | 1,000 | 1,000 | 1,000 |

Diabetes complicada no tiene menciones gold en desarrollo. En negrita, las diferencias de
0,05 o más con las reglas en categorías con 10 o más menciones gold. Con sinónimos, SapBERT
iguala o supera a las reglas en todas las categorías salvo infarto, donde la diferencia es
de una mención. Con pocas menciones gold, los intervalos son anchos.

### Siglas (71 menciones gold de desarrollo, 29 siglas distintas)

Se cuenta como sigla una mención gold de 2 a 6 letras o dígitos que empieza en mayúscula y
no tiene minúsculas ("VIH", "DM", "EPOC"). Se separan las 25 que están en la lista de 30
siglas que el pipeline sube a mayúscula para el modelo (#114) de las 46 que no están.

| Método | Categoría correcta, con categoría gold (24) | En la lista (19) | Código a 3 top-1 (71) |
|---|---|---|---|
| Reglas + corrector | 19 | 19 | — |
| SapBERT sin sinónimos | 12 | 12 | 22 |
| SapBERT con sinónimos | 19 | 19 | 50 |

- Con sinónimos, SapBERT resuelve las mismas siglas de Charlson que la lista hecha a mano,
  sin una lista. 52 de las 71 menciones sigla están tal cual entre los sinónimos.
- Fuera de la lista, el código a 3 caracteres acierta en 26 de 46 (10 sin sinónimos). La
  mayoría no son de Charlson (CMV, SIADH, WPW); una lista manual no las tendría.
- Las 5 menciones de Charlson fuera de la lista (4 siglas: LH, SK, VHB, VHC) no las
  resuelve ningún método.

### Costo

- Codificar los 3.338 sinónimos toma 58 s y agrega 3.338 filas (3 %) a las 98.288 del
  índice.
- Latencia con sinónimos: mediana de 43 ms por mención (p95 72 ms), 27 ms en lote. La
  memoria máxima del proceso de evaluación fue de 2,9 GiB.

### Lo que sigue

- **#133, híbrido**: en CodiEsp, con sinónimos, el modelo ya iguala o supera a las reglas
  en casi todas las categorías. El híbrido sirve si las reglas aportan donde el modelo no
  tiene sinónimos (notas colombianas), y para no depender solo de un modelo sin licencia
  clara.
- **#134**: la medición final en CodiEsp prueba.
- **Licencia de SapBERT**: sigue pendiente antes de la imagen pública.

## Híbrido: exclusiones de las reglas después del modelo (#133)

Se mide en la misma corrida de #132 (`python -m evaluacion sinonimos`) y queda en el mismo
JSON. Los híbridos usan el índice con sinónimos y su umbral, 0,69.

### Errores de SapBERT con sinónimos en desarrollo, por causa

Hay 75 menciones con una categoría distinta a la del gold: 24 falsos positivos, 52
falsos negativos y una confusión entre categorías, que cuenta en los dos lados. Revisadas
una por una:

| Causa | Menciones | Ejemplos (texto público de CodiEsp) |
|---|---|---|
| El vecino más cercano es otra cosa | 25 | "pérdida de fuerza" → debilidad muscular; "aneurisma aorta" → aneurisma congénito; "hematoma cerebeloso" → hematoma traumático; "necrosis gástrica" → necrosis hepática |
| Benigno y maligno se escriben parecido | 14 | "miomas" → mieloma; "meningioma" → plasmocitoma; "hipernefroma" → hidronefrosis; "leiomiosarcoma" → leiomioma; "LH" → "LB" |
| Neoplasia de comportamiento no especificado | 13 | "neoplasia de encía" es D49.0 en el gold y C03 en el modelo; "neoplasia vesical" es al revés; "adenocarcinoma de ciego" queda junto a "tumor de ciego" (D49.0) |
| Codificación de CodiEsp o definición de Quan | 13 | "hepatitis B" es B19.10 en entrenamiento y B18.1 en desarrollo; "VIH" como Z21; "isquemia miocárdica" (I25.9) contra miocardiopatía isquémica (I25.5); linfoma NK (C86, que Quan 2005 no tiene) |
| Bajo el umbral, con el candidato correcto | 10 | "melanoma cutáneo maligno nodular en hombro izquierdo" da el código exacto con 0,686 |

### Exclusiones (`normalizador.con_exclusiones`): se agregan

Después del modelo se aplica la exclusión de la regla de la categoría elegida. Si
coincide, la mención queda sin categoría. Solo puede quitar categorías.

| | Se activa | Arregla | Quita una confusión | Rompe |
|---|---|---|---|---|
| Entrenamiento (validación cruzada, 5 pliegues) | 6 | 5 | 1 | 0 |
| Desarrollo | 3 | 3 | 0 | 0 |

"Arregla" es un falso positivo menos. "Quita una confusión" es cuando el gold tiene otra
categoría: se va el falso positivo, pero el falso negativo sigue.

- **Desarrollo.** Los tres casos son del grupo benigno/maligno: "meningioma frontal
  izquierdo", "miomas" y "adenomioma de vesícula", que el modelo daba como tumor maligno.
- **Entrenamiento.**
  - "hemangiomas" y "adenoma" de próstata, dados como tumor maligno.
  - "bronquiolitis", dada como EPOC.
  - "neoplasia gástrica" y "úlcera en paladar", dadas como úlcera péptica.
  - Una metástasis dada como tumor primario. Ahí la exclusión quita la categoría
    equivocada, pero la mención no queda con la correcta.
- **Exclusiones de precedencia (#144).** Algunas exclusiones no dicen "no es esta
  enfermedad", sino "le toca a otra regla": la de tumor maligno excluye "metástasis" para
  ceder la mención a tumor metastásico.
  - Si la otra regla toma el texto ("metástasis hepáticas"), el modelo se equivocó y la
    exclusión se aplica.
  - Si la otra regla tampoco lo toma, porque el término está negado o en duda, no hay a
    quién cederlo y la exclusión no se aplica. Es el caso de "linfoma de Hodgkin sin
    metástasis" (#123), que antes del arreglo quedaba sin categoría.

  Ninguna de las cifras de la tabla cambia con el arreglo.
- **Con evidencia** se agregan las exclusiones de tres reglas, las únicas que se activan:
  tumor maligno, pulmonar crónica y úlcera péptica. No rompen nada en ninguna partición.
- **Las otras 14** no se activan en CodiEsp, así que no hay cifra a favor ni en contra. Se
  mantienen porque no cambian nada aquí y son las mismas que ya usan las reglas del
  pipeline. No queda ninguna exclusión descartada.
- El efecto es chico: 3 de 75 errores. Las exclusiones no ayudan con los falsos negativos.
  Tampoco con los malignos que el modelo da como benignos, como "hipernefroma" o
  "leiomiosarcoma".

### Respaldo de las reglas (`con_respaldo`): se mide, no se adopta aquí

Si el modelo, ya con las exclusiones, no da categoría, se usan las reglas + corrector.

| Método | P | R | F1 [IC 95 %] | F1 errores 10 % | F1 no vistas | FP / FN |
|---|---|---|---|---|---|---|
| SapBERT con sinónimos | 0,921 | 0,844 | 0,881 [0,852; 0,910] | 0,852 | 0,802 | 24 / 52 |
| **+ exclusiones** | **0,930** | 0,844 | **0,885** [0,855; 0,913] | 0,856 | 0,809 | 21 / 52 |
| + exclusiones + respaldo de reglas | 0,838 | **0,916** | 0,875 [0,841; 0,907] | **0,866** | **0,833** | 59 / 28 |

- **Lo que gana.** El respaldo recupera 24 falsos negativos. 14 son tumores malignos que
  el modelo dejó en un código sin categoría, como "adenocarcinoma de ciego" o "mandibular
  carcinoma"; las reglas los reconocen por "carcinoma" o "sarcoma".
- **Lo que agrega.** Suma 37 categorías que el gold no tiene.
  - **23 son diferencias de definición** de la línea base (#130):
    - cardiopatía isquémica o hipertensiva como insuficiencia cardiaca: 9 (en Quan no
      cuentan sin falla: las reglas lo corrigen en #201);
    - hepatitis B o C no especificada (B19): 8;
    - carcinoma de piel (C44): 4;
    - VIH asintomático (Z21): 2.

    Para quien lee la nota no son errores clínicos, pero en CodiEsp y en Quan cuentan
    como falsos positivos.
  - **Las otras 14 son reglas más amplias que Quan:**
    - cerebrovascular: 5 (MAV, isquemia retiniana, disección carotídea, hematoma
      traumático);
    - hemiplejia: 3 ("mielitis transversa");
    - reumatológica: 3 (esclerodactilia, arteritis de la temporal);
    - renal crónica, infarto y vascular periférica: 1 cada una.
- **Por qué no se adopta en este issue.** Su F1 es 0,01 menor y su recall 0,07 mayor. Para
  la demo pesa más el recall, porque lo que no se detecta el lector no lo ve. Para el
  índice de Charlson, en cambio, un falso positivo suma puntos. Es una decisión de
  producto, que se toma al llevar el normalizador al pipeline.

### Conclusión

El híbrido con exclusiones queda como el mejor método en desarrollo: F1 0,885. Aporta
poco sobre el modelo con sinónimos (+0,004), pero no rompe nada en entrenamiento ni en
desarrollo, y cuesta una regex por mención. Es el que se mide una vez en CodiEsp prueba
(#134), junto a las reglas y al modelo con sinónimos solo.

Los errores que quedan no los arregla otra regla:
- Para los de vecino equivocado y bajo el umbral hacen falta más sinónimos, o un
  reordenamiento del top-5. La categoría correcta está entre los cinco primeros en el 96 %
  de las menciones.
- Los de codificación y definición son del gold, no del método.

## Filtro de longitud (#56)

```bash
uv run python -m evaluacion distemist --particion train \
    --variantes pipeline filtro_lista_blanca filtro_ninguno \
    --salida evaluacion/resultados/filtro_longitud_distemist_train.json \
    --detalle data/evaluacion/filtro_longitud_distemist_train_detalle.json
```

**El problema.** El pipeline descartaba toda mención de 4 caracteres o menos que no
estuviera en una lista blanca de siglas hecha a mano. En DisTEMIST prueba, el modelo crudo
encuentra 85 de las 90 siglas del gold y el pipeline se quedaba con 46 (#117). Es la misma
pregunta de este epic: una lista manual frente a algo que no dependa de ella.

**Cómo se decidió.** Con DisTEMIST **entrenamiento** y con el set sintético. DisTEMIST
prueba queda para #134. El modelo NER se afinó con DisTEMIST entrenamiento, así que ahí sus
cifras son altas, pero la comparación entre filtros es justa: el modelo es el mismo y solo
cambia lo que se descarta.

| Filtro (DisTEMIST entrenamiento, 750 documentos) | P estricta | R estricta | F1 estricta [IC 95 %] | F1 laxa | VP / FP |
|---|---|---|---|---|---|
| Lista blanca (el de antes) | 0,839 | 0,759 | 0,797 [0,787; 0,807] | 0,907 | 6.124 / 1.176 |
| Sin filtro | 0,840 | 0,779 | 0,809 [0,798; 0,819] | 0,918 | 6.284 / 1.196 |
| **Mínimo (el de ahora)** | 0,840 | 0,779 | **0,809** [0,798; 0,819] | **0,918** | 6.284 / 1.194 |

Sin filtro y mínimo son equivalentes: la diferencia es de 2 falsos positivos.

- **Lo que hacía la lista blanca.** Descartaba 160 aciertos por cada 20 falsos positivos
  que evitaba. Los aciertos eran siglas que la lista no tenía (EC, HCC, GIST, PKU, ELA, LCG,
  WPW, MAV…) y palabras cortas como "lúes" y "coma". Los falsos positivos eran sobre todo
  siglas o palabras cortas que el gold no anota, o anota con otro borde ("DR", "MAVC",
  "masa"). Una misma sigla puede ser acierto en un documento y falso positivo en otro, como
  "MAV". Solo 4 eran basura: "de", "en", "hipo" y "cara".
- **El filtro mínimo.** Una mención de 4 caracteres o menos pasa, salvo en tres casos:
  - no tiene letras ("1", ",");
  - es una palabra en minúscula de 2 letras o menos que no está en la lista ("de", "en").
    Con mayúscula inicial pasa: "Mx" (metástasis) es un acierto, y "En" o "De" al inicio
    de frase no aparecieron como entidad;
  - es una sigla ambigua que no está en mayúscula ("les", "tía", con o sin tilde). Sigue
    pasando solo en mayúsculas (#24).

  Conserva los 160 aciertos y quita 2 de los 20 falsos positivos. La lista blanca sigue
  existiendo, pero solo para las siglas en minúscula de 2 letras ("dm", "fa"); ya no hace
  falta para las siglas en mayúscula.
- **Set sintético** (100 notas, con el texto original y en minúscula): el filtro mínimo da
  exactamente lo mismo que la lista blanca. El recall de entidades es 331/339 y 330/339, y
  el índice exacto 92 y 89 de 100. Las notas sintéticas no traen siglas fuera de la lista.
- **Descartada: "corta pero normalizable".** Esta opción deja pasar una mención corta si el
  normalizador la lleva a un código con similitud alta. Exigiría SapBERT dentro del
  pipeline solo para filtrar, y eso depende de la decisión de #134 (licencia y memoria).
  El filtro mínimo ya recupera los aciertos sin necesitar el modelo.

## Medición final (#134)

La medición única en CodiEsp prueba, el costo en Docker, las licencias, el chequeo de las
metas y la recomendación están en
[`evaluacion.md`, sección 8](evaluacion.md#8-normalización-a-cie-10-codiesp). En resumen:
- El híbrido con exclusiones da F1 0,874, contra 0,792 de las reglas.
- En las menciones no vistas empatan: 0,757 contra 0,755.
- La licencia de SapBERT no está declarada y la p95 es de 1,05 s.
- Recomendación: las reglas se quedan.

## Decisión en dos etapas: el arnés (#149, epic #148)

PRD: [`prds/normalizacion-f1.md`](prds/normalizacion-f1.md). La meta es subir el F1 de
categoría del híbrido. Para eso se separan dos pasos:
- **Recuperar:** el índice trae un top-k de candidatos.
- **Decidir:** otro componente mira todo el top-k y elige.

Este issue arma la base para comparar segundas etapas (`evaluacion/dos_etapas.py`):
- **Top-k una vez por índice.** Cada pliegue guarda el top-20 de sus menciones, con
  código, similitud, origen, categoría de Quan, código de 3 caracteres y capítulo. El
  bloque CIE-10 queda para #150, que es quien lo usa: el catálogo de CodiEsp no trae los
  bloques.
- **Vectores en disco.** Los de catálogo, sinónimos y menciones se guardan por texto en
  `data/normalizacion/`. La segunda corrida no carga el modelo.
- **Validación cruzada anidada** (`pliegues_internos`). Para ajustar algo sin ver el
  pliegue externo `i`, cada pliegue `j` se busca en un índice sin los sinónimos de `i` ni
  de `j`.
- **Comparación pareada** (`comparar`): el IC de la diferencia por bootstrap pareado por
  documento.

```bash
uv run python -m evaluacion dos-etapas --modelo sapbert --k 20
```

### Reproduce lo de antes

| Método | Validación cruzada en entrenamiento: P / R / F1 [IC 95 %] | Desarrollo: F1 |
|---|---|---|
| Reglas + corrector | 0,868 / 0,737 / 0,797 [0,763; 0,826] | — |
| SapBERT con sinónimos, umbral 0,69 | 0,882 / 0,825 / 0,853 [0,828; 0,875] | — |
| Híbrido con exclusiones | 0,889 / 0,825 / 0,856 [0,831; 0,878] | 0,885 (21 FP, 52 FN) |

- El 0,853 es el punto 0,69 de la curva de #132 (0,8525).
- El 0,885 y sus errores son los de #133.
- **La comparación pareada sí detecta lo que los IC sueltos no.** Las exclusiones suben el
  F1 en 0,0035, con un IC pareado de [0,0011; 0,0064]: es poco, pero no es ruido. Contra
  las reglas, el híbrido sube 0,059 [0,031; 0,089].

### Por qué se pierden las menciones

| Causa | Entrenamiento (VC) | Desarrollo |
|---|---|---|
| Primer candidato correcto, pero bajo el umbral | 23 | 12 |
| Categoría correcta entre el 2.º y el 5.º candidato | 62 | 26 |
| Categoría correcta entre el 6.º y el 20.º | 26 | 11 |
| Fuera del top-20, pero las reglas la dan | 8 | 0 |
| Ni el top-20 ni las reglas | 15 | 3 |
| Una exclusión quitó la correcta | 0 | 0 |
| **Total** | **134** de 767 con categoría | **52** de 333 |

El análisis hecho a mano en desarrollo después de #134 contaba un "ruido de anotación"
(~9) que aquí no existe: esas menciones caen en alguna de las causas de arriba.

- **El patrón se repite en entrenamiento.** Ahí el 83 % de las perdidas tiene la
  categoría correcta en el top-20, contando el primer candidato bajo el umbral (111 de
  134); en desarrollo, el 94 % (49 de 52). No era algo propio de desarrollo.
- En desarrollo, "solo las reglas" da 0 y no 4 como en el análisis a mano: ese análisis
  miraba las reglas antes que el top-20, y aquí cuentan solo si la categoría no está en
  el top-20.
- **Tumor maligno domina en las dos particiones:** 61 de 134 en entrenamiento y 27 de 52
  en desarrollo. En entrenamiento le siguen tumor metastásico (20) y cerebrovascular (19).
- **Los falsos positivos son casi todos de una misma clase:** el gold no tiene categoría
  y el modelo pone una. Son 74 de 79 en entrenamiento y 20 de 21 en desarrollo. Una
  segunda etapa que sepa abstenerse también sube la precisión.

### Techo de una segunda etapa

Es el F1 de un oráculo que acierta si la categoría correcta está en el top-k, y si no se
abstiene:

| k | Entrenamiento (VC) | Desarrollo |
|---|---|---|
| 1 | 0,922 | 0,936 |
| 5 | 0,967 | 0,979 |
| 10 | 0,973 | 0,992 |
| 20 | 0,985 | 0,996 |

Con k = 1 el oráculo solo elige bien el umbral, y ya pasa del híbrido (0,856). Entre lo
que hay hoy y el techo con k = 5 quedan 0,11 de F1: es el margen de #150 a #152.

### Las menciones perdidas, versionadas

Las 52 de desarrollo quedan en `evaluacion/resultados/dos_etapas_arnes.json`
(`desarrollo.errores.detalle`). Cada una trae documento, posición, código gold, categoría
y causa, sin texto. #154 mide contra esa lista cuántas se recuperan.

## Segunda etapa por votos (#150)

En vez de que el primer candidato decida solo, los `k` primeros votan
(`evaluacion/votos.py`):
- **Peso de cada candidato:** su similitud, o un softmax con temperatura.
- **"Ninguna" vota también:** los candidatos sin categoría suman a esa opción, así que el
  voto puede abstenerse.
- **Voto jerárquico:** primero gana el grupo de la CIE-10 con más peso (código de 3
  caracteres, bloque o capítulo, de `docs/datos.md`) y después, dentro de ese grupo, la
  categoría.
- **Umbral:** el mejor candidato de la categoría ganadora tiene que pasarlo. Con `k = 1`
  es la decisión de hoy.

**Cómo se eligió, sin fuga.** Hay una rejilla de nivel, k (1 a 20), temperatura (suma,
0,01 a 0,5) y umbral (0,30 a 0,80). En cada pliegue externo de entrenamiento se elige con
sus pliegues internos, cuyos índices no tienen los sinónimos del externo, y se mide en el
externo. Desarrollo no se usó.

```bash
uv run python -m evaluacion votos --modelo sapbert
```

### Resultado (validación cruzada anidada en entrenamiento, 500 documentos, 767 menciones con categoría)

| Segunda etapa | P | R | F1 [IC 95 %] | Diferencia de F1 contra el híbrido [IC pareado] | Recupera del top-5 | Falsos positivos nuevos / quitados |
|---|---|---|---|---|---|---|
| Híbrido (primer candidato, 0,69) | 0,889 | 0,825 | 0,856 [0,831; 0,878] | — | — | — |
| **Votos (nivel elegido en cada pliegue)** | **0,920** | **0,836** | **0,876** [0,850; 0,898] | **+0,020** [+0,003; +0,036] | 17 de 62 | 16 / 39 |
| Solo voto por categoría¹ | 0,927 | 0,833 | 0,878 [0,851; 0,900] | +0,022 [+0,006; +0,038] | 14 de 62 | 10 / 39 |
| Lo anterior más un umbral por categoría | 0,920 | 0,828 | 0,872 [0,845; 0,894] | +0,016 [−0,001; +0,032] | 17 de 62 | 15 / 39 |
| Voto por capítulo, luego categoría | 0,886 | 0,864 | 0,875 [0,852; 0,898] | +0,019 [+0,001; +0,037] | 23 de 62 | 32 / 26 |
| Voto por bloque, luego categoría | 0,869 | 0,864 | 0,867 [0,843; 0,889] | +0,011 [−0,006; +0,027] | 20 de 62 | 43 / 22 |
| Voto por código de 3, luego categoría | 0,857 | 0,858 | 0,857 [0,834; 0,880] | +0,001 [−0,014; +0,018] | 15 de 62 | 50 / 19 |

¹ Las filas "solo …" fijan el nivel de antemano. Son para entender cada nivel, no para
elegir: escoger la mejor de ellas mirando los externos tendría sesgo de selección. La
estimación del procedimiento que se usará es la fila **Votos**, que elige el nivel dentro
de cada pliegue.

- **Los votos suben el F1 en 0,020, y el IC pareado no incluye el cero.** La mayor
  parte de la ganancia es precisión: quitan 39 falsos positivos y agregan 16.
  - Los falsos positivos del híbrido son casi todos menciones sin categoría en el gold
    (#149). El voto de "ninguna" quita los que la mayoría del top-k no respalda.
  - Con el nivel fijo en categoría, se recuperan 17 de las 23 menciones bajo el umbral y
    14 de las 62 del top-5, y se pierden 25 que el híbrido acertaba.
- **El voto jerárquico recupera más, pero agrega más falsos positivos.** El capítulo
  recupera 23 de las 62 del top-5, el que más. Pero agrupar por capítulo o bloque junta
  candidatos de categorías distintas, el grupo ganador suele arrastrar una categoría, y
  la precisión no sube como con el voto por categoría (0,886 por capítulo y 0,869 por
  bloque, contra 0,927). El voto por capítulo también mejora al híbrido (+0,019, con el
  IC justo por encima de cero) y, con el bloque, da la mayor ganancia de recall (+0,039).
  En F1 queda detrás del voto por categoría.
- **El capítulo como nivel, no solo como desempate.** El issue lo planteaba como
  desempate. Pero con similitudes continuas los empates exactos casi no ocurren, y así no
  habría cambiado nada. Por eso se probó como un nivel completo: el capítulo decide la
  región y la categoría se vota dentro. En un pliegue de cinco, la búsqueda lo eligió.
- **El umbral por categoría no ayuda.** Elegido en los internos, empeora en el externo
  (0,872 contra 0,876): con pocas menciones por categoría se sobreajusta. Se descarta.
- **La elección es estable.** En 4 de 5 pliegues externos se eligió el voto por
  categoría, con k entre 10 y 20, temperatura entre 0,05 y 0,1 y umbral entre 0,52 y
  0,56. Con temperatura 0,05 y un candidato a 0,05 por debajo del primero, este pesa
  e⁻¹ ≈ 0,37 del primero: votan los vecinos cercanos.
- **Con votos, el umbral de similitud baja de 0,69 a unos 0,54.** Parte de lo que hacía
  el umbral (no dar categoría a lo dudoso) lo hace ahora el voto de "ninguna". A igual F1
  se prefiere el umbral más alto.

**Configuración para #154** (elegida con los 5 pliegues juntos; su F1 en esos mismos
pliegues, 0,880, es optimista): voto por categoría, k = 20, temperatura 0,05 y umbral
0,54, más las exclusiones de las reglas.

El techo con k = 5 era 0,967 (#149). Del margen entre 0,856 y 0,967, el voto cierra unas
dos décimas partes. Lo que falta son menciones en las que la mayoría del top-k apunta a
otra categoría: ahí hace falta una decisión que mire más que la similitud. Eso se prueba
en #151 (clasificador) y #152 (reordenador).

## Segunda etapa entrenada: clasificador sobre el top-k (#151)

Los votos de #150 deciden solo con la similitud. Aquí una **regresión logística** aprende a
decidir con más señales (`evaluacion/clasificador.py`):

- **Pares.** Por cada mención se arman pares (mención, categoría candidata): las
  categorías del top-20, más la de las reglas aunque no esté en el top.
- **Atributos de cada par** (22 más uno por categoría, ninguno con texto):
  - similitudes y márgenes;
  - la fracción del voto de #150, con varias temperaturas;
  - el peso del capítulo y del bloque;
  - el origen del candidato (catálogo o sinónimo);
  - el largo de la mención y si es sigla;
  - si las reglas dan esta categoría u otra, y si la exclusión aplica.
- **Decisión.** Gana el par más probable si pasa el umbral; si no, queda sin categoría.
  Después van las exclusiones, como siempre.
- **Por qué torch y no scikit-learn.** La regresión logística es de torch, que ya es
  dependencia. scikit-learn no está en el proyecto, y un *gradient boosting* pediría
  agregarlo.

**Sin fuga.** Es la validación cruzada anidada de #150. Para cada pliegue externo, los
pares de los internos salen de índices sin los sinónimos del externo. La penalización
L2 y el umbral se eligen dejando fuera un interno a la vez. Después se entrena con los
cuatro internos y se decide el externo. En la misma vuelta se eligen los votos con los
mismos internos, para compararlos pareados.

```bash
uv run python -m evaluacion clasificador --modelo sapbert
```

### Resultado (validación cruzada anidada en entrenamiento, 767 menciones con categoría)

| Segunda etapa | P | R | F1 [IC 95 %] | VP / FP / FN |
|---|---|---|---|---|
| Híbrido (primer candidato, 0,69) | 0,889 | 0,825 | 0,856 [0,831; 0,878] | 633 / 79 / 134 |
| Votos (#150) | 0,920 | 0,836 | 0,876 [0,850; 0,898] | 641 / 56 / 126 |
| **Clasificador** | **0,916** | **0,881** | **0,898** [0,875; 0,919] | 676 / 62 / 91 |

| Comparación pareada | Δ P | Δ R | Δ F1 [IC 95 %] |
|---|---|---|---|
| Clasificador menos híbrido | +0,027 | +0,056 | **+0,042** [+0,025; +0,063] |
| Clasificador menos votos | −0,004 | +0,046 | **+0,023** [+0,005; +0,042] |

- **El clasificador gana a los dos, y los IC pareados no incluyen el cero.**
  - Frente al híbrido sube la precisión y el recall.
  - Frente a los votos la precisión queda igual (−0,004, con un IC de [−0,025; +0,017]),
    pero el recall sube +0,046 [+0,022; +0,072].
- **Advertencia sobre el conjunto de atributos.** La primera versión tenía además la
  similitud del primer candidato. Era redundante: es la similitud menos el margen. Con
  ella el clasificador daba 0,893, y +0,017 [−0,001; +0,035] contra los votos. Se quitó
  en la revisión para que los pesos se pudieran leer, no para subir el F1. Aun así, el
  cambio se hizo después de ver un resultado anidado, y eso deja algo de optimismo. Por
  eso la cifra que cuenta es la de desarrollo en #154.
- **Qué recupera.** De las 134 que el híbrido pierde, recupera:
  - 37 de las 62 en el top-5 (los votos, 17);
  - 20 de las 23 bajo el umbral;
  - 6 de las 8 que solo daban las reglas;
  - 3 de las 26 entre el 6.º y el 20.º candidato.

  Pierde 23 que el híbrido acertaba, mete 22 falsos positivos y quita 39.
- **Qué aprendió.** Son los pesos del modelo final sobre atributos estandarizados. Varios
  atributos están correlacionados (los votos con distintas temperaturas, la similitud con
  los márgenes) y las columnas de categoría suman 1, como el sesgo. Por eso la L2 reparte
  el peso entre ellos: cada peso se lee como dirección, no como tamaño exacto.
  - **Señales a favor:** los votos (temperatura 0,1: +0,85; 0,05: +0,57), que las reglas
    den esta categoría (+0,78) y los márgenes contra otra categoría (+0,74) y contra el
    primero (+0,71).
  - **Señal en contra:** que las reglas den otra categoría (−0,53). Las reglas sirven
    como segunda opinión.
- **La rejilla alcanza.** El F1 casi no cambia con la penalización: con los 5 pliegues
  va de 0,892 (1e-1) a 0,900 (1e-3). En cada pliegue se eligió entre 1e-6 y 1e-3, con un
  umbral de probabilidad entre 0,30 y 0,45.
- **Costo.** El modelo tiene 40 parámetros. Decide en unos 0,03 ms por mención y 0,44 ms
  por documento de entrenamiento (14 menciones en promedio), con las reglas ya
  calculadas (el pipeline ya las corre) y sin contar la codificación ni la búsqueda.

**Modelo para #154:** penalización 1e-3 y umbral 0,30, entrenado con los 5 pliegues. El
F1 de 0,900 que da en esos pliegues sale de validación cruzada, pero es el máximo de la
rejilla, así que es optimista por la selección. Los pesos completos están en
`evaluacion/resultados/dos_etapas_clasificador.json`.

**Cuánto falta.** El techo de un oráculo con el top-20 era 0,985 (#149). Este
clasificador puede ir un poco más allá, porque también mira las reglas. Del margen entre
0,856 y 0,985, cierra un tercio. Lo que sigue:
- el reordenador (#152), como otra señal;
- más sinónimos públicos (#153) para los tumores.

## Segunda etapa con reordenador: se descarta (#152)

Un reordenador (*cross-encoder*) lee la mención junto a la descripción de cada candidato,
en vez de comparar dos vectores calculados por separado (`evaluacion/reordenador.py`). Se
probó de dos formas, con los 5 o los 10 primeros candidatos:
- **Solo:** sus puntuaciones reemplazan a la similitud, y se elige con la búsqueda de #150.
- **Como atributos del clasificador de #151:** la mejor puntuación de la categoría, el
  margen contra otra y su fracción de la suma.

La elección es la misma validación cruzada anidada de #150 y #151. Desarrollo no se usó.

```bash
uv run python -m evaluacion reordenador --reordenador mmarco   # o bge_base, bge_m3
```

### Candidatos

| Clave | Modelo | Parámetros | Pesos | Licencia | Reparo |
|---|---|---|---|---|---|
| `mmarco` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | 0,47 GB | Apache 2.0 | Se entrenó con mMARCO, derivado de MS MARCO, cuyos términos son solo para investigación no comercial: la misma clase de duda que SapBERT |
| `bge_base` | `BAAI/bge-reranker-base` | 278M | 1,1 GB | MIT | Solo declara inglés y chino; su tarjeta no declara los datos |
| `bge_m3` | `BAAI/bge-reranker-v2-m3` | 568M | 2,3 GB | Apache 2.0 | Su tarjeta no declara los datos. Sus pesos solos son más que toda la imagen candidata de #134 (2,0 GB) |

Quedaron fuera antes de medir:
- `jina-reranker-v2`: CC BY-NC, no permite uso comercial.
- `gte-multilingual-reranker-base`: exige ejecutar código remoto.
- Qwen3-Reranker y mxbai-rerank: son decodificadores de 0,5B parámetros o más.

### Resultado (validación cruzada anidada en entrenamiento, 767 menciones con categoría)

La referencia es el clasificador de #151: F1 0,898 [0,875; 0,919].

| Reordenador | k | Solo: F1 | Clasificador con reordenador: F1 | Diferencia contra el clasificador [IC pareado] | Pliegues que mejora | Suma a una nota (mediana, dos corridas) |
|---|---|---|---|---|---|---|
| `mmarco` | 5 | 0,841 | 0,897 | −0,001 [−0,011; +0,008] | 4 de 5 | 0,23 a 0,55 s |
| `mmarco` | 10 | 0,787 | 0,901 | +0,002 [−0,005; +0,011] | 4 de 5 | 0,39 a 0,95 s |
| `bge_base` | 5 | 0,821 | 0,897 | −0,001 [−0,013; +0,010] | 2 de 5 | 0,86 a 2,19 s |
| `bge_base` | 10 | 0,786 | 0,894 | −0,004 [−0,016; +0,007] | 3 de 5 | 1,55 a 3,63 s |
| `bge_m3` | 5 | 0,841 | 0,903 | +0,005 [−0,002; +0,012] | 4 de 5 | 3,47 a 7,47 s |
| `bge_m3` | 10 | 0,827 | 0,898 | +0,000 [−0,008; +0,009] | 2 de 5 | 5,67 a 12,30 s |

**Sobre el costo.** Es para 8 entidades por nota (el promedio de las notas de 2.000
caracteres de #134), con pares reales del top-k y 2 hilos en el portátil. No es una
medición en Docker. Dos corridas del mismo código dieron valores que difieren unas 2,4
veces; no se investigó por qué. El JSON versionado trae la corrida más lenta. La memoria
que suma cada modelo cargado es de 540 MiB (`mmarco`), 769 MiB (`bge_base`) y 1.603 MiB
(`bge_m3`).

- **Solo no mejora a nadie.** Los seis quedan por debajo del híbrido (0,856) en el punto,
  y todos por debajo del clasificador con IC pareados que excluyen el cero. No se comparó
  de forma pareada contra el híbrido. Una explicación posible, no probada, es que son
  modelos de búsqueda general y no de terminología médica.
- **`mmarco` y `bge_m3` con k = 5 dan los mismos conteos** (620 VP, 87 FP y 147 FN), pero
  es coincidencia: sus errores son distintos.
- **Como atributos no aporta.** Ninguno de los seis IC pareados excluye el cero. El mejor
  punto es `bge_m3` con k = 5 (+0,005), y su k = 10 da +0,000: no hay una tendencia que
  sostenga esa cifra. Además se eligió entre seis opciones, así que aun ese punto es
  optimista.
- **Ninguno cumple el costo.** El límite del PRD es sumar 0,2 s o menos al p95. La
  mediana es una cota inferior del p95, y el mejor valor medido, `mmarco` con k = 5, ya
  da 0,23 s. Vale para este portátil; en esa variante el margen es pequeño. La decisión
  no depende de esto: sin ganancia de F1, el reordenador sobra a cualquier costo.

**Decisión: se descartan los tres.** No mejoran al clasificador y se pasan del costo. El
modelo para #154 sigue siendo el de #151, sin atributos del reordenador. No se
buscó abaratarlos (por ejemplo, saltando las menciones con coincidencia exacta o
cuantizando), porque no hay ganancia de F1 que pagar. Un reordenador entrenado con
terminología médica en español podría dar otro resultado, pero no se encontró uno con
licencia clara entre los candidatos revisados.

## Más sinónimos públicos: CANTEMIST y DisTEMIST (#153)

Los sinónimos de CodiEsp entrenamiento (#132) solo cubren lo que se anotó en 500 notas, y
27 de las 52 pérdidas en desarrollo eran de tumor maligno (#149). Aquí se suman al índice
menciones de dos corpus públicos más. La fuente, la licencia, cómo se pasa de CIE-O a
Charlson y el cruce de ids con CodiEsp están en [`datos.md`](datos.md#sinónimos-de-cantemist-y-distemist-153).
El código está en `evaluacion/sinonimos_publicos.py` (lectura y fuga) y
`evaluacion/fuentes_sinonimos.py` (medición).

**Cómo se mide.** Con los mismos 5 pliegues de #149, cada configuración suma sus fuentes
a los sinónimos de CodiEsp de cada índice de pliegue. Las menciones de los documentos que
se evalúan no entran (DisTEMIST son notas de CodiEsp), y las de CodiEsp desarrollo y
prueba no entran nunca. Se mide el híbrido de hoy (umbral 0,69 fijo) y el clasificador
de #151, reentrenado en cada configuración con validación cruzada anidada. Cada
configuración se compara contra `base` (solo CodiEsp) con el mismo método, de forma
pareada por documento. Las cinco configuraciones se declararon antes de medir; `ambas`,
DisTEMIST más la variante léxica de CANTEMIST, era la combinación prevista.

    uv run python -m evaluacion descargar --corpus cantemist
    uv run python -m evaluacion fuentes

### Resultado (validación cruzada anidada en entrenamiento, 767 menciones con categoría)

`evaluacion/resultados/dos_etapas_fuentes.json`, medido sobre `64d708e`. `base`
reproduce las cifras de #151 (0,856 y 0,898).

| Fuentes | Filas del índice | Método | P | R | F1 [IC 95 %] | Δ F1 vs base [IC] | F1 tumor maligno | Δ tumor maligno [IC] |
|---|---|---|---|---|---|---|---|---|
| base | 101.626 | híbrido | 0,889 | 0,825 | 0,856 [0,831; 0,878] | — | 0,778 | — |
| | | clasificador | 0,916 | 0,881 | 0,898 [0,875; 0,919] | — | 0,863 | — |
| distemist | 101.796 | híbrido | 0,889 | 0,827 | 0,857 | +0,001 [+0,000; +0,003] | 0,781 | +0,003 [+0,000; +0,009] |
| | | clasificador | 0,915 | 0,884 | 0,899 | +0,001 [−0,008; +0,011] | 0,847 | −0,016 [−0,041; +0,007] |
| cantemist | 105.445 | híbrido | 0,809 | 0,859 | 0,833 | **−0,023** [−0,042; −0,005] | 0,841 | **+0,063** [+0,026; +0,107] |
| | | clasificador | 0,946 | 0,867 | 0,905 | +0,006 [−0,009; +0,021] | 0,917 | **+0,055** [+0,021; +0,094] |
| cantemist_lexico | 104.126 | híbrido | 0,891 | 0,851 | 0,871 | **+0,015** [+0,002; +0,027] | 0,826 | **+0,048** [+0,019; +0,084] |
| | | clasificador | 0,938 | 0,872 | 0,904 | +0,006 [−0,006; +0,018] | 0,891 | **+0,028** [+0,003; +0,056] |
| ambas | 104.289 | híbrido | 0,891 | 0,853 | 0,871 | **+0,015** [+0,003; +0,029] | 0,828 | **+0,050** [+0,022; +0,086] |
| | | clasificador | 0,938 | 0,880 | 0,908 [0,885; 0,928] | +0,009 [−0,003; +0,022] | 0,896 | **+0,033** [+0,009; +0,062] |

En negrita, los IC pareados que excluyen el cero. Son 5 configuraciones, 2 métodos y
varias métricas sin corrección por comparaciones múltiples: un IC que apenas excluye el
cero (como +0,003 en tumor maligno con `cantemist_lexico`) es una señal débil. Las filas
del índice son las del índice con los 5 pliegues: 98.288 del catálogo, 3.338 sinónimos de
CodiEsp y los externos.

- **DisTEMIST no aporta.** Suma 170 sinónimos, porque sus notas utilizables son las
  mismas de CodiEsp entrenamiento, que ya están como sinónimos. El F1 cambia en una
  milésima.
- **CANTEMIST sube tumor maligno**, con los dos métodos y con IC que excluyen el cero. Con
  el clasificador, el F1 de tumor maligno pasa de 0,863 a 0,891 (léxica), 0,896 (`ambas`)
  o 0,917 (completa).
- **CANTEMIST completo rompe tumor metastásico en el híbrido**, como se temía al declarar
  la variante léxica. Sus falsos positivos de metastásico pasan de 3 a 82 y el F1 general
  cae 0,023. Ya sin etiquetas TNM, la causa son las menciones que solo eran metástasis en
  su nota ("lesiones hepáticas", "nódulos pulmonares"). El clasificador lo compensa en
  buena parte, pero su F1 de metastásico baja igual de 0,969 a 0,932 (−0,037 [−0,074;
  −0,012]).
- **La variante léxica lo reduce mucho, pero no del todo.** Con el clasificador,
  metastásico queda igual (2 falsos positivos en `ambas`, como en `base`). En el híbrido,
  los falsos positivos de metastásico pasan de 3 a 9 y la precisión de 0,969 a 0,917. El
  F1 queda en 0,892 porque el recall sube, y su IC es ancho: [−0,053; +0,046].
- **Con el clasificador, el F1 general sube poco y sin IC que excluya el cero:** +0,006 a
  +0,009. Lo que sí cambia es la precisión. En `ambas`, los falsos positivos bajan de 62 a
  45 (+0,022 [+0,007; +0,039] de precisión) y el recall queda casi igual (0,880 contra
  0,881). Sin DisTEMIST, el recall baja algo más: 0,872, con −0,009 [−0,027; +0,008].
- **Pliegues.** Con el clasificador, `ambas` gana en 3 de 5 pliegues y pierde en 2 (uno
  por 0,0004 y otro por 0,009). Con el híbrido, la variante léxica gana en 4 de 5.
- **Costo.** Con la variante léxica, el índice crece entre 2.500 y 2.700 filas (2,6 %).
  La búsqueda es la misma multiplicación de matrices, un poco más grande.

**Para #154: el clasificador de #151 con `ambas`.** Era la combinación declarada antes
de medir, y es la mejor en el punto en F1 general, tumor maligno y metastásico. Que
DisTEMIST aporte casi nada no es razón para quitarlo después de ver los resultados. La
ganancia en F1 general no es significativa en validación cruzada. Lo que la justifica es
tumor maligno y la precisión, y lo confirmará o no la medición única en desarrollo.
CANTEMIST completo se descarta por lo que le hace a metastásico.

## Medición final en desarrollo (#154)

Lo elegido en entrenamiento se mide **una sola vez** en CodiEsp desarrollo, junto al
híbrido de hoy. Lo elegido es el clasificador de #151 con los sinónimos de DisTEMIST y la
variante léxica de CANTEMIST (`ambas`, declarada en #153 antes de medir). Desarrollo ya se
había mirado en #132 y #133, así que sus cifras son **optimistas**. Prueba no se volvió a
usar: solo se leyeron los nombres de sus archivos, para verificar que ningún sinónimo
saliera de ahí.

El clasificador final se entrena con los 5 pliegues de entrenamiento, cada uno buscado en
un índice sin sus propios sinónimos. La penalización y el umbral salen de la validación
cruzada sobre esos pliegues (penalización 0,01 y umbral 0,40). En desarrollo, el índice
tiene los 3.338 sinónimos de CodiEsp entrenamiento y 2.663 de las fuentes (104.289 filas),
sin ninguno de un documento de desarrollo o de prueba. El código está en
`evaluacion/medicion_final.py`.

    uv run python -m evaluacion final                       # una sola vez, árbol limpio
    uv run python -m evaluacion.costo_normalizacion --final  # costo en Docker

### Resultado (desarrollo, 250 documentos, 333 menciones gold con categoría)

`evaluacion/resultados/normalizacion_final_desarrollo.json`, medido sobre `6a0cbfd` desde
un árbol limpio. El híbrido reproduce el 0,885 de #133 y las mismas 52 menciones perdidas
de #149 (documento y posición).

| Método | P | R | F1 [IC 95 %] | Δ F1 vs híbrido [IC] | F1 errores 10 % | F1 no vistas |
|---|---|---|---|---|---|---|
| Híbrido de hoy | 0,930 | 0,844 | 0,885 [0,855; 0,913] | — | 0,856 | 0,809 |
| **Clasificador + fuentes (elegido)** | **0,937** | 0,892 | **0,914** [0,882; 0,940] | **+0,029** [+0,002; +0,053] | 0,889 | 0,863 |
| *Contexto:* híbrido + fuentes | 0,919 | 0,850 | 0,883 | −0,002 [−0,013; +0,007] | 0,861 | 0,807 |
| *Contexto:* clasificador sin fuentes | 0,930 | 0,913 | 0,921 | +0,036 [+0,012; +0,060] | 0,904 | 0,878 |

"F1 no vistas" mide las menciones cuyo texto no estaba tal cual entre los sinónimos de
CodiEsp entrenamiento (189 con categoría), como en #132. Las dos filas de contexto separan
los dos pasos. No se eligieron ni se pueden elegir con esta corrida.

- **El clasificador gana, en desarrollo y en entrenamiento.** En desarrollo sube el F1
  en 0,029, con un IC pareado que excluye el cero, aunque por poco. Lo que sube es el
  recall (+0,048 [+0,013; +0,082]); la precisión queda igual (+0,006 [−0,025; +0,037]).
  En validación cruzada anidada, en la misma corrida, el elegido supera al híbrido por
  +0,052 [+0,034; +0,071] (0,908 contra 0,856).
- **Las fuentes no se notan en desarrollo.** El híbrido con fuentes da lo mismo que sin
  ellas y pierde algo de precisión (−0,012 [−0,024; −0,002]). El clasificador sin fuentes
  queda 0,007 por encima del elegido en el punto. Es lo opuesto a la validación cruzada
  (+0,009 a favor de las fuentes), y en ninguno de los dos casos el IC excluye el cero. En
  tumor maligno, el elegido sube de 0,821 a 0,876 (+0,055 [−0,005; +0,122]) y el
  clasificador sin fuentes llega a 0,896. La ganancia de tumor maligno en validación
  cruzada no se confirma como efecto de las fuentes.
  Una diferencia de los dos clasificadores finales: sin fuentes, la validación cruzada
  eligió un umbral más bajo (0,30 contra 0,40), y eso da más recall.
- **Robustez.** Con errores al 10 %, el elegido cae 0,025 y el híbrido 0,029, en el
  punto, sin IC de la diferencia de caídas. Sobre el híbrido, el elegido queda entre
  +0,019 y +0,037 en todas las variantes de estrés. Los IC pareados excluyen el cero en
  todas, salvo minúsculas sin tildes ([−0,007; +0,045]).
- **Menciones no vistas:** de 0,809 a 0,863, en el punto. No se calculó el IC pareado en
  ese subconjunto. Todo indica que la segunda etapa ayuda también fuera de lo
  memorizado, cosa que el híbrido no lograba frente a las reglas (#134).
- **Por categoría** (gold de desarrollo entre paréntesis), el elegido contra el híbrido:
  - tumor maligno (112): 0,821 → 0,876;
  - vascular periférica (9): 0,75 → 0,89;
  - hepática severa (7): 0,875 → 1,0;
  - insuficiencia cardiaca (3): 0,67 → 1,0;
  - hepática leve (26): 0,773 → 0,791;
  - cerebrovascular (14): 0,846 → 0,815;
  - las demás quedan igual.

  Ninguna diferencia por categoría tiene un IC que excluya el cero. La de tumor maligno
  queda muy cerca: [−0,005; +0,122].

### Las 52 menciones que perdía el híbrido

| Causa (#149) | Perdidas | Recuperadas |
|---|---|---|
| Categoría correcta de primera, bajo el umbral | 12 | 9 |
| Categoría correcta en el top-5, no de primera | 26 | 11 |
| Categoría correcta entre el 6.º y el 20.º | 11 | 0 |
| Fuera del top-20 y de las reglas | 3 | 0 |
| **Total** | **52** | **20** |

A cambio, pierde 4 menciones que el híbrido acertaba. Agrega 9 falsos positivos y quita
10.

### Costo (Docker, 2 vCPU y 4 GiB, como Cloud Run)

`evaluacion/resultados/costo_normalizacion_final.json`, medido sobre `3834759` desde un
árbol limpio, con las mismas 138 notas de 2.000 caracteres de #134. Las tres imágenes se
midieron en la misma corrida. Las dos con SapBERT siguen el mismo camino por nota:
pipeline, top-20 de cada entidad y la decisión con las exclusiones. En la del híbrido, la
decisión es el primer candidato. En la final, son las reglas y el clasificador, y el índice
incluye las fuentes.

| Imagen | Tamaño | Carga | Mediana por nota | p95 por nota | Memoria pico |
|---|---|---|---|---|---|
| API actual | 856 MB | 2,5 s | 0,35 s | 0,44 s | 363 MiB |
| API + SapBERT, híbrido | 2.020 MB | 5,3 s | 0,49 s | 0,80 s | 1.037 MiB |
| API + SapBERT, clasificador + fuentes | 2.027 MB | 5,3 s | 0,49 s | 0,86 s | 1.045 MiB |

- **La decisión del clasificador cuesta 1 ms por nota en la mediana y 2 ms en el p95.**
  Está muy por debajo de la meta de 0,05 s. La del híbrido mide menos de 1 ms.
- **El p95 de la imagen completa sube 0,07 s, pero no es por la segunda etapa.** La mediana
  es la misma (0,486 s), y sin la decisión la final ya mide 0,861 s contra 0,795 s. La
  final se midió de última. En una corrida de control con el orden invertido (no
  versionada), la final dio 0,795 s y la del híbrido 0,81 s. La diferencia es variación
  entre contenedores, no un costo de la segunda etapa ni del índice un 2,6 % más grande.
- La imagen crece 7 MB y la memoria 8 MiB.
- La latencia total de la imagen con SapBERT, alrededor de 0,8 s en el p95, sigue siendo
  tema del epic de integración (en #134 midió 1,05 s).

### Metas del PRD

| Meta | Resultado |
|---|---|
| Validación cruzada: el IC pareado de la diferencia contra el híbrido no incluye el cero | **Sí**: +0,052 [+0,034; +0,071] |
| Desarrollo: F1 mayor que 0,885 | **Sí**: 0,914 (optimista: desarrollo ya se había mirado) |
| Desarrollo: la precisión no cae más de 0,01 | **Sí**: sube 0,006 |
| Desarrollo: meta orientativa de 0,90 | **Sí**: 0,914 |
| Recuperar al menos la mitad de las 26 del top-5 | **No**: 11 de 26. Recupera también 9 de las 12 bajo el umbral: 20 de 52 en total |
| Con errores al 10 %, no cae más que el híbrido | **Sí**: cae 0,025 contra 0,029 |
| Costo: la segunda etapa suma 0,05 s o menos al p95, y cabe en 4 GiB | **Sí**: la decisión, 0,002 s al p95. La imagen completa da +0,07 s al p95 con la misma mediana, por variación entre contenedores: en orden invertido da −0,015 s. Memoria pico: 1,0 GiB |

### Recomendación

1. **Adoptar el clasificador como segunda etapa.** Es la conclusión que se sostiene en
   entrenamiento y en desarrollo, en F1, con IC pareados que excluyen el cero. En robustez
   y en menciones no vistas va mejor en el punto. Cuesta 2 ms por nota.
2. **Las fuentes, sin evidencia a favor ni en contra.** En validación cruzada suman
   +0,009 y en desarrollo restan 0,007 en el punto, sin IC que excluya el cero en ninguno
   de los dos. La cifra que se reporta es la del elegido (0,914), porque fue lo declarado.
   El 0,921 sin fuentes se vio después y no se puede presentar como resultado. Quitar las
   fuentes es más simple: no hay atribución que revisar ni columna de UMLS en el índice. Pero
   quitarlas porque desarrollo salió mejor sería elegir con desarrollo. Diego decide.
3. **No cambia lo que bloquea la integración:** la licencia de SapBERT y el epic de
   integración siguen pendientes.

**Decisión (Diego, 2026-11-08): se adopta el clasificador, sin las fuentes externas.**

- La segunda etapa es el clasificador de #151 sobre el top-20 del índice con CodiEsp
  entrenamiento (el catálogo más sus sinónimos), sin DisTEMIST ni CANTEMIST.
- Las fuentes se quitan por la falta de efecto en validación cruzada (+0,009, con IC que
  incluye el cero) y por su costo: atribución CC BY que revisar y una columna de DisTEMIST
  derivada de UMLS, con licencia propia. No se quitan por el resultado en desarrollo.
- La cifra de la medición sigue siendo la del método declarado: 0,914. La variante sin
  fuentes, que es la que se adopta, dio 0,921 [0,893; 0,945] en la misma corrida, pero no
  se eligió de antemano: se cita como contexto, no como resultado.
- El código de las fuentes queda en el repo (`sinonimos_publicos`, `fuentes_sinonimos`).
  Si cambia el tipo de nota y crecen las menciones no vistas, se reactivan y se vuelven a
  medir en validación cruzada.

## Integración: el normalizador pasa a la librería (#163, epic #162)

PRD: [`integracion-normalizador`](prds/integracion-normalizador.md). Lo que el normalizador
necesita en ejecución vive ahora en `ner_clinic.normalizacion`, y `evaluacion/` lo importa
de ahí. Así hay un solo código, y lo medido es lo mismo que correrá el pipeline:

| Módulo | Qué tiene | Venía de |
|---|---|---|
| `indice` | SapBERT con revisión fija, `CodificadorHF`, `Indice`, `Normalizador` | `evaluacion.normalizador` |
| `jerarquia` | bloque y capítulo CIE-10, con su tabla versionada | `evaluacion.jerarquia_cie10` |
| `decision` | propuestas, exclusiones, los 39 atributos, la regresión logística (solo inferencia), su JSON y `SegundaEtapa` | `dos_etapas`, `votos`, `clasificador`, `medicion_final` |

- En `evaluacion/` se quedan el entrenamiento, la validación cruzada, el bootstrap, la
  lectura de corpus y la construcción del índice del catálogo. Los resultados versionados
  no se recalcularon.
- `SegundaEtapa.decidir(textos)` codifica todos los textos de una nota en un lote y
  devuelve la categoría, el código, la probabilidad y el candidato del índice que la
  explica. No guarda los textos ni sus vectores entre llamadas.
- Un test comprueba que `SegundaEtapa` decide igual que el método medido en #154
  (`mapeador_de` con el clasificador y las exclusiones).
- Importar `ner_clinic` sigue sin cargar `torch`.

## Los artefactos de la librería (#164)

`uv run python -m evaluacion artefactos` construye lo que la librería carga
(`ner_clinic.normalizacion.artefactos.cargar`). Se corrió una vez, desde un árbol limpio,
sobre el commit 7f7a6d1.

| Artefacto | Dónde | Qué tiene |
|---|---|---|
| Índice final | `data/normalizacion/indice_final.pt` (no se versiona, 323 MB) | 101.626 filas: los 98.288 códigos del catálogo CIE-10-ES 2018 y 3.338 sinónimos de CodiEsp entrenamiento, con sus vectores de SapBERT. Sin DisTEMIST ni CANTEMIST |
| Clasificador | `src/ner_clinic/normalizacion/clasificador.json` (versionado) | Los 39 pesos, media, desviación, sesgo y umbral; el commit, las versiones y la huella del índice. Sin texto de ningún corpus |

- **Reproduce lo elegido en #154.** Con los 5 pliegues de entrenamiento, la validación
  cruzada vuelve a elegir penalización 0,001 y umbral 0,30.
- **Comprobación, no medición.** Con los artefactos cargados por la librería
  (`SegundaEtapa`), las menciones gold de desarrollo dan exactamente los mismos aciertos y
  errores de #154 sin fuentes: 304 verdaderos positivos, 23 falsos positivos y 29 falsos
  negativos (F1 0,9212). Da lo mismo con los vectores en caché de #154 y con SapBERT
  codificando en vivo, que es lo que corre la librería. Desarrollo ya estaba gastado: esto
  solo confirma que el artefacto es el método medido. El JSON guarda la comprobación.
- **Huella.** El clasificador guarda la huella del índice (códigos, textos, origen y
  configuración de SapBERT; los vectores no, porque los fija esa configuración). `cargar`
  la recalcula con las filas cargadas: si el índice cambia, falla y dice cómo
  reconstruirlo, en vez de decidir con un clasificador entrenado sobre otro índice.
- `RUTA_INDICE` es relativa a la raíz del repo; la API y la imagen Docker pasan la suya.
- **Lo que más pesa en este modelo** (pesos sobre atributos estandarizados): el voto con
  temperatura 0,1 (+0,85), que las reglas den esta categoría (+0,78) y los márgenes contra
  otra categoría y contra el primero (+0,74 y +0,71). En contra, que las reglas den otra
  (−0,53). Los atributos están correlacionados: los pesos sirven para ver qué señales
  usa, no para leer cada uno por separado.

## El normalizador en el pipeline (#165)

`analizar(texto, normalizador=artefactos.cargar(), modo_mapeo="normalizador")` usa la
segunda etapa en el paso 6 en lugar de las reglas. Sin `normalizador`, todo sigue igual que
antes.

- **Qué se decide:** las confirmadas y las posibles, como con las reglas. Las negadas y
  las familiares no se mapean.
- **Qué texto:** primero el nombre de la entidad; si no da categoría, la primera mención
  que la dé, como `_mapear` (#113). Con dos salvaguardas más, porque el normalizador da
  categoría a textos que las reglas no conocen: solo cuentan las menciones que son una
  variante del nombre ("EPOC" para "eppoc"), no las contenidas en él ("insuficiencia
  renal" no le da renal crónica a "insuficiencia renal aguda"); y la mención no vale si la
  exclusión de su categoría quita el nombre, tal como está escrito o corregido ("pre
  diabetis"). #154 decidió menciones gold sueltas: este respaldo es nuevo y lo mide #167.
  Los nombres y las menciones de una nota van en un solo lote.
- **Reemplaza a las reglas** (modo `normalizador`): si la segunda etapa no da categoría,
  la entidad queda sin ella. No se cae a las reglas, porque ese es el método medido (las
  reglas ya entran como atributos del clasificador). Desde #180 el modo por defecto es el
  híbrido (abajo).
- **La salida:** `mapeo` dice de dónde salió la categoría y `candidato` trae el texto del
  índice que la explica, su código, su origen y su similitud. `cie10` es el código
  CIE-10-ES del candidato leído para Charlson (un padre sin categoría pasa a su hijo
  `.9`: "E11" da "E11.9"); si la categoría solo la dieron las reglas dentro del
  clasificador, no hay candidato y `cie10` es el de las reglas. `correcciones` solo
  aparece si las reglas dan la misma categoría corrigiendo la ortografía (aunque el
  normalizador no haya necesitado corregir).
- **Privacidad:** la segunda etapa no guarda los textos de la nota entre llamadas (test).

En una nota sintética con NER y SapBERT reales, las cuatro comorbilidades que las reglas
mapeaban (tres confirmadas y una posible) salen con la misma categoría y el índice de Charlson es el mismo. Cambian los
códigos (los de la CIE-10-ES, más específicos: `N18.9`, `I63.9`) y aparece el candidato
("DM2 insulinorrequiriente" → "DM tipo 2"; "ACV isquémico" → "ictus isquémico"). El
efecto en muchas notas se mide en #167.

## El normalizador en la API (#166)

*El valor por defecto cambió en #168 a `sapbert` y en #182 a `hibrido` (ver el final).*

- `NER_NORMALIZADOR=sapbert` carga al arrancar el índice, el clasificador y SapBERT
  (`artefactos.cargar`); `NER_INDICE_NORMALIZADOR` cambia la ruta del índice. Por defecto
  sigue `reglas`. Si falta un artefacto, la app no arranca y el error dice cómo
  construirlo.
- La nota de calentamiento también pasa por el normalizador, así que la primera petición
  no paga la carga.
- `GET /api/salud` y `meta` de cada análisis dicen el `mapeo` activo. El OpenAPI
  (`/api/openapi.json`) documenta la respuesta con `mapeo` y `candidato`
  (`ner_clinic_api/esquema.py`); un test valida una respuesta real contra ese esquema.
- Los logs siguen sin texto de la nota (test con el centinela).
- En #166 la imagen Docker todavía no traía SapBERT ni el índice; desde #189, sí.
- A mano, con los modelos reales en el portátil: la app arranca en 6,6 s y una nota corta
  tarda 0,13 s. El costo en Docker con 2 vCPU se mide en #167.

## De punta a punta y costo (#167)

Hasta #154 todo se midió sobre las menciones gold de CodiEsp: el texto ya cortado por un
anotador. Aquí corre el pipeline completo, con el NER, una vez con las reglas y otra con
el normalizador, sobre el commit cdaad27 y desde un árbol limpio. Nada se ajustó con estos
resultados y CodiEsp prueba no se tocó.

**Cada método juega de local en un conjunto distinto.** El índice y el clasificador del
normalizador salen de CodiEsp entrenamiento (mismos anotadores y estilo que desarrollo).
El set sintético lo escribimos conociendo las reglas, y las reglas se corrigieron después
de ver sus errores (v2 a v4, [`evaluacion.md`](evaluacion.md)). Ninguno de los dos es una
nota real de una IPS colombiana.

### CodiEsp desarrollo, por documento

`evaluacion/resultados/extremo_codiesp_desarrollo.json` (`python -m evaluacion extremo`).
Las categorías de las entidades **confirmadas** de cada nota contra las de sus códigos gold
(250 casos, 123 con alguna categoría, 197 pares documento-categoría). Supuestos que
afectan a los dos métodos por igual: las posibles no cuentan (la norma de codificación
española codifica las sospechas como confirmadas, así que puede costar recall a los dos);
la jerarquía leve/severa no se aplica antes de contar P/R/F1; y los 127 casos sin
categoría gold suman al "índice exacto" de los dos. Desarrollo ya se miró: es optimista.

| Método | P | R | F1 [IC 95 %] | Índice exacto | Índice ±1 | Error medio del índice | Sesgo medio del índice |
|---|---|---|---|---|---|---|---|
| Reglas | 0,833 | 0,635 | 0,721 [0,669; 0,775] | 0,720 | 0,852 | 0,70 | −0,50 |
| Normalizador | 0,796 | 0,731 | 0,762 [0,709; 0,809] | 0,732 | 0,812 | 0,84 | −0,16 |

- **Δ F1 normalizador − reglas: +0,041 [−0,002; +0,090]**, pareado por documento. El IC
  toca el cero.
- Sube el recall (+0,10 [+0,05; +0,15]) y baja la precisión (−0,04 [−0,11; +0,03]).
- Gana en tumor maligno (48 aciertos contra 40), cerebrovascular (9 contra 5) y hepática
  leve (8 contra 5), y quita los 8 falsos positivos de insuficiencia cardiaca de las
  reglas. Agrega falsos positivos en tumor metastásico (12 contra 2) y tumor maligno (11
  contra 3).
- Índice de Charlson: el exacto casi no cambia (0,732 contra 0,720). Las reglas lo
  subestiman más (sesgo −0,50 contra −0,16), pero el normalizador se equivoca por más
  cuando falla (±1: 0,812 contra 0,852; error medio 0,84 contra 0,70).
- Sobre menciones gold de desarrollo la diferencia era mucho mayor (0,921 contra 0,759).
  Las unidades no son las mismas (mención contra par documento-categoría), pero el patrón
  es claro: de punta a punta las reglas casi no cambian (0,759 → 0,721) y el normalizador
  baja más (0,921 → 0,762). Parte de su ventaja venía de recibir la mención ya cortada.
- **Respaldo a las menciones** (#165): una sola entidad recibió su categoría por una
  mención y no por su nombre, y estaba en el gold.

### Set sintético (100 notas escritas con estilo colombiano)

`sintetico_v5_reglas.json` y `sintetico_v5_normalizador.json`, con los artefactos de #164
(huella del índice 010bb894). El set es reservado y ya se había medido varias veces (v1 a
v4); las reglas reproducen exactamente v4. Son unos 140 pares gold y no hay IC pareado.

| Método | P | R | F1 categoría | Índice exacto | Índice ±1 |
|---|---|---|---|---|---|
| Reglas | 0,848 | 0,879 | **0,863** | **0,92** | **0,96** |
| Normalizador | 0,864 | 0,771 | 0,815 | 0,79 | 0,93 |

Con el normalizador, los errores de categoría pasan de 26 a 37: arregla 6 de las reglas
(entre ellos los falsos positivos de demencia en "HTA" e "hipotiroidismo" y una
cardiopatía isquémica dada como insuficiencia cardiaca), comete 17 nuevos y cambia 3 por
otro error. La precisión sube (0,848 → 0,864) y el recall baja (0,879 → 0,771). De los 17
nuevos, 15 dejan sin categoría lo que las reglas daban bien. Revisados uno por uno:

- **Siglas que CodiEsp no trae.** "ICC" (insuficiencia cardiaca congestiva, 5 casos)
  queda más cerca de "HCC" (hepatocarcinoma) que de cualquier insuficiencia cardiaca. Las
  reglas la conocen, pero el clasificador le da 0,16 de probabilidad, bajo el umbral de
  0,30. Este hallazgo no depende del sesgo del set: es una sigla común en Colombia que el
  índice no tiene.
- **Menciones largas del NER.** "demencia mixta + HTA + hipotiroidismo", "demencia tipo
  Alzheimer en estadio moderado" o "ACV isquemico agudo cardioembolico en territorio de
  ACM izquierda" quedan cerca de otra cosa ("Hipotiroidismo congénito", "Discapacidad
  intelectual moderada"). En CodiEsp el clasificador vio menciones gold, cortas.
- **Matices que las reglas sí tienen.** "cirrosis hepática por alcohol con várices
  esofágicas" sale hepática leve (las reglas: severa, que es lo correcto en Charlson).

### Costo en Docker con 2 vCPU y 4 GiB

`evaluacion/resultados/costo_integrado.json` (`python -m evaluacion.costo_integrado`): la
API por HTTP, con la imagen de hoy y con una que suma SapBERT y el índice. El control es la
imagen con SapBERT usando las reglas.

| Corrida | Imagen | Arranque en frío | Mediana nota 2.000 car. | p95 nota 2.000 car. | p95 DisTEMIST | Memoria pico |
|---|---|---|---|---|---|---|
| Reglas | 856 MB | 5,0 s | 0,346 s | 0,443 s | 1,07 s | 449 MiB |
| Normalizador | 2.021 MB | 7,0 s | 0,479 s | 0,710 s | 1,40 s | 1.137 MiB |
| Control (imagen con SapBERT, reglas) | 2.021 MB | 4,5 s | 0,347 s | 0,430 s | 1,10 s | 465 MiB |

- El normalizador suma 0,27 s al p95 de una nota de 2.000 caracteres (meta del PRD: 0,5 s
  o menos) y 0,7 GiB de memoria (pico 1,1 GiB, holgado en 4 GiB). La imagen crece 1,2 GB y
  el arranque en frío, unos 2 s. El control descarta que sea la imagen o el orden.
- Cumple las metas de costo del PRD.

### Qué significa

- **En CodiEsp, su terreno, el normalizador sube el recall** (sobre todo en tumores), pero
  de punta a punta la mejora en F1 ya no es clara (IC que toca el cero), y en el índice de
  Charlson gana en sesgo y pierde en error medio.
- **En el set sintético, escrito conociendo las reglas, pierde** (F1 0,815 contra 0,863).
  Ese set favorece a las reglas, así que no prueba que el normalizador sea peor en notas
  colombianas reales. Lo que sí muestra, caso por caso, son dos debilidades concretas:
  siglas locales que el índice no tiene ("ICC") y menciones largas del NER.
- **No hay evidencia para activarlo por defecto**: ninguna de las dos mediciones muestra
  una mejora clara de punta a punta.
- El costo no es el problema: cabe de sobra.
- La decisión de si la API lo activa por defecto es de Diego (#168, abajo).

## Decisión: el normalizador por defecto en la API (#168)

*Reemplazada en #182: el valor por defecto ahora es `hibrido` (ver el final).*

**Decisión (Diego, 2026-11-08): la API usa el normalizador por defecto**
(`NER_NORMALIZADOR=sapbert`), con los resultados de #167 a la vista. Con
`NER_NORMALIZADOR=reglas` vuelve a las reglas.

- **La librería no cambia:** `analizar()` sigue mapeando con reglas salvo que se le pase
  `normalizador`, porque cargarlo implícito arrastraría SapBERT y el índice a cualquier
  uso.
- **La imagen Docker fija `NER_NORMALIZADOR=reglas`** mientras no traiga SapBERT y el
  índice. Llevarlos a la imagen (unos 1,2 GB más) y a la demo pública es del epic de
  despliegue, y depende de la licencia de SapBERT ([`datos.md`](datos.md#el-normalizador-integrado-sapbert-y-artefactos-162)).
- **Lo que hay que vigilar** con este defecto, según #167: las siglas de uso colombiano
  que el índice no tiene ("ICC") y las menciones largas del NER, que el normalizador deja
  sin categoría donde las reglas acertaban; y más falsos positivos de tumor metastásico.

## Después de quitar la fusión de entidades (#55)

La fusión de entidades cercanas unía enfermedades distintas ("demencia mixta + HTA +
hipotiroidismo"), una de las causas de los errores del normalizador en #167. Sin ella
(detalle en [`evaluacion.md`](evaluacion.md#sin-fusión-55)):

| | Reglas | Normalizador |
|---|---|---|
| Set sintético, F1 de categoría (antes → sin fusión) | 0,863 → **0,915** | 0,815 → **0,876** |
| Set sintético, índice exacto | 0,92 → **0,96** | 0,79 → **0,82** |
| CodiEsp desarrollo de punta a punta, F1 por documento | 0,721 → 0,724 | 0,762 → 0,764 |

- El normalizador mejora más en el set sintético, pero la diferencia con las reglas se
  mantiene: siguen pesando las siglas locales ("ICC") y las menciones largas de una sola
  enfermedad. En CodiEsp casi nada cambia (Δ +0,040 [−0,004; +0,088]).
- Cifras en `sintetico_v6_sin_fusion.json`, `sintetico_v6_normalizador.json` y
  `extremo_codiesp_desarrollo_v2.json` (sobre c0fd035, a91c8c1 y c69f31f, desde árboles
  limpios).

## Modo híbrido: reglas primero, normalizador de respaldo (#180, epic #179)

`analizar(texto, normalizador=artefactos.cargar())` usa por defecto `modo_mapeo="hibrido"`:

- Cada confirmada o posible se mapea primero con las reglas, como sin normalizador
  (nombre, menciones y corrección ortográfica). Si dan categoría, sale con
  `mapeo="reglas"`, aunque el normalizador dijera otra cosa.
- Solo las que quedan sin categoría pasan por la segunda etapa, en un lote y con las
  mismas salvaguardas del modo `normalizador`. Lo que una regla excluye a propósito
  ("diabetes gestacional") tampoco lo da el normalizador.
- Los diagnósticos registrados (#176) se mapean igual: reglas y, para los que no mapean,
  el normalizador.

Es lo contrario del respaldo que midió #133 (normalizador primero, reglas de respaldo). La
razón: el normalizador pierde donde las reglas son fuertes (siglas locales como "ICC",
tramos largos del NER) y gana donde no tienen regla ("falla renal crónica", formas poco
comunes). Se mide en #181.

## Un solo camino en la API y en Docker (#182)

`NER_NORMALIZADOR` acepta `hibrido` (por defecto), `sapbert` y `reglas`:

| Valor | Mapeo | Sin los artefactos del normalizador |
|---|---|---|
| `hibrido` | Reglas y, para lo que no mapean, el normalizador | Si no está el índice, arranca con las reglas, lo registra (sin rutas ni texto) y `/api/salud` dice `mapeo: "reglas"`. Si el índice está pero algo falla (corrupto, huella distinta, SapBERT ausente sin red), no arranca |
| `sapbert` | Solo el normalizador (#168) | No arranca |
| `reglas` | Solo las reglas | No los necesita |

- `meta.mapeo` de cada análisis y `/api/salud` dicen el mapeo efectivo (`reglas`,
  `normalizador` o `hibrido`); cada entidad dice el suyo en `mapeo`.
- El Dockerfile ya no fija `NER_NORMALIZADOR=reglas`. La etapa `base` no trae SapBERT y
  corre con las reglas; la etapa `completa`, la de por defecto desde #189, los trae y usa
  el híbrido, con la misma configuración.
- La decisión de dejar `hibrido` por defecto sale de la medición de #181 (`docs/evaluacion.md`).

## SapBERT y el índice en la imagen (#189)

El `Dockerfile` tiene dos imágenes finales: `completa` (por defecto: SapBERT, su tokenizador
y el índice, con revisiones fijas) y `base` (`--target base`, sin ellos; la usa CI, donde
no está el índice, y corre con las reglas). Si `data/normalizacion/indice_final.pt` no
existe, `docker build` falla: no sale una imagen grande que solo usa reglas.

Medido con la imagen real, 2 vCPU y 4 GiB (`latencia_docker_completa.json`):

| | Imagen completa, híbrido |
|---|---|
| Tamaño de la imagen | 2,0 GB (la `base`: 0,9 GB) |
| Arranque en frío | 13,8 s |
| Nota de 2.000 caracteres, mediana / p95 | 0,48 s / 0,76 s |
| Nota de 5.000 caracteres | 1,6 a 1,7 s |
| DisTEMIST prueba, mediana / p95 | 0,68 s / 1,28 s |
| Memoria pico | 1,1 GiB |

Una prueba en humo sin red (`--network none`) da `mapeo: "hibrido"`, y "falla renal crónica"
sale del normalizador. El arranque es de una sola corrida: en #167 otra imagen equivalente
arrancó en 7 s con el mismo método, así que varía.

**Licencia:** incluirlo en la imagen fue decisión de Diego (2026-11-08). Publicar esa imagen
o desplegar la demo con ella redistribuye o sirve pesos sin licencia declarada: sigue
pendiente ([`datos.md`](datos.md#el-normalizador-integrado-sapbert-y-artefactos-162)).

## Opciones de código para el experto (#230)

Cada entidad confirmada o posible trae `alternativas_cie10`: hasta 5 códigos distintos del
índice (catálogo CIE-10-ES más sinónimos de CodiEsp entrenamiento), de mayor a menor
similitud, con la descripción del catálogo. Debajo de 0,6 de similitud no salen: eran
ruido ("EPOC" daba "E. coli" con 0,59). En la demo, el experto elige el código que
corresponde y la fila lo muestra como elegido; no se guarda nada y la categoría de
Charlson no cambia. Sin normalizador (modo solo reglas) la lista va vacía.

En CodiEsp prueba, el primer candidato acierta los 3 primeros caracteres del código en el
80,7 % de las menciones con categoría y alguno de los cinco primeros en el 90,7 %
(sección 8 de `evaluacion.md`). El piso de 0,6 puede dejar fuera algún acierto de baja
similitud: no se volvió a medir con él.

