# Evaluación de ner-clinic-es

Este reporte dice qué tan bien funciona el pipeline: con corpus públicos anotados por
terceros donde los hay, y con un set sintético reservado donde no los hay. Todas las cifras
salen de los archivos de [`evaluacion/resultados/`](../evaluacion/resultados/) o de los
issues que se citan, y unos pocos comandos las reproducen (ver [Cómo reproducirlo](#cómo-reproducirlo)). PRD:
[`docs/prds/evaluacion.md`](prds/evaluacion.md); epic #39.

## Resumen

| Qué se mide | Datos | Resultado |
|---|---|---|
| Detectar enfermedades (NER), modelo solo | DisTEMIST, 250 casos clínicos de prueba | F1 **0,728** estricto (span exacto) · **0,877** laxo (se solapa) |
| Detectar enfermedades, pipeline completo | DisTEMIST, mismos 250 casos | F1 **0,683** estricto · **0,860** laxo |
| Negación en texto clínico real, reglas de la tesis | NUBes, 29.682 frases de historias clínicas | F1 de `negada` 0,799 (P 0,849, R 0,754) |
| Negación con las reglas corregidas (epic #71) | NUBes, partición reservada: 14.817 frases que no se usaron para ajustar | F1 de `negada` **0,864** (P 0,885, R 0,844); antes, 0,802 |
| Sospechas: estado `posible` (epic #86) | NUBes, misma partición reservada (segunda vez que se usa) | F1 de `posible` **0,858** (P 0,919, R 0,805); antes el estado no existía. `negada` queda en **0,870** (P 0,899, R 0,842) |
| Estado, Charlson y subregistro | Set sintético reservado, 100 notas, medición ciega (#45) | Estado correcto 94,3 % · índice exacto 86 % (a 1 punto o menos, 95 %) |
| Lo mismo, segunda medición (después de ver los errores) | Mismas 100 notas | Estado correcto 99,4 % · índice exacto 91 % (a 1 punto o menos, 96 %) |
| Lo mismo, tercera medición (después del epic #86) | Mismas 100 notas | Estado correcto 99,7 % · índice exacto 92 % (a 1 punto o menos, 96 %) |
| Notas mal escritas (epic #111) | Set sintético (no ciego) con errores en el 10 % de las palabras de 4 letras o más, media de 5 semillas que no se usaron para ajustar · DisTEMIST con la nota en minúscula | Índice exacto 79,8 → **85,6**; en minúscula, 80 → **89** · Siglas encontradas en minúscula: 25 → 28 de 90 (la lista de siglas no cubre las de DisTEMIST) |
| Detectar enfermedades, pipeline con el filtro de longitud mínimo (#56) | DisTEMIST, mismos 250 casos | F1 **0,691** estricto · **0,868** laxo · siglas encontradas: 46 → **84** de 90 |
| Detectar enfermedades, pipeline sin fusión de entidades cercanas (#55) | DisTEMIST, mismos 250 casos | F1 **0,734** estricto · **0,878** laxo; el pipeline ya no queda por debajo del modelo solo |
| Estado, Charlson y subregistro, cuarta medición (sin fusión, #55) | Mismas 100 notas | Estado correcto 100 % · índice exacto **96 %** (a 1 punto o menos, 98 %) · F1 de categoría 0,863 → **0,915** |
| Subregistro con el cruce por categoría para la diabetes (#176) | Mismas 100 notas | `registrado` correcto 64 → **66 de 68** (97,1 %) |
| Huecos de las reglas de Charlson (#123, #120, #63) | Mismas 100 notas (no ciegas) | Índice exacto 96 → **98 %** · F1 de categoría 0,915 → **0,926** |
| Mapeo híbrido: reglas primero, normalizador de respaldo (epic #179) | CodiEsp desarrollo de punta a punta · set sintético | F1 por documento: reglas 0,731 · normalizador 0,764 · **híbrido 0,759**, con los filtros de #185 **0,769** · índice exacto sintético 98 % · 83 % · 97 %, con filtros **98 %**: el híbrido es el defecto de la API (#182) |
| Categoría de Charlson desde la mención (epic #128) | CodiEsp, 250 casos de prueba con CIE-10, una sola medición | Reglas: F1 **0,792** · SapBERT con sinónimos + exclusiones: F1 **0,874**; en las menciones que no estaban en entrenamiento, 0,755 contra 0,757 |
| Velocidad | DisTEMIST, CPU de portátil | 0,85 s por nota (mediana), 1,48 s (p95) |

Lo principal que dejó la evaluación:

1. **El modelo es sólido.** Casi siempre encuentra la enfermedad. La mayoría de sus errores
   estrictos son de límite: sobra o falta una palabra.
2. **El post-proceso heredado de la tesis empeoraba el NER en DisTEMIST**, sobre todo por
   la fusión de entidades cercanas. Se quitó en #55, decidido con DisTEMIST entrenamiento
   y el set sintético: el F1 estricto en prueba pasa de 0,691 a 0,734.
3. **La negación era el punto débil**: con las reglas de la tesis, en texto real dejaba sin
   negar una de cada cuatro enfermedades negadas. Se corrigió en el epic #71 (posición,
   enumeraciones, "no presenta", claves que faltan y alcance) y se midió en una partición
   de NUBes que no se usó para ajustar: el recall de `negada` pasa de 0,764 a 0,844 y la
   precisión, de 0,845 a 0,885. Ahora deja sin negar una de cada seis o siete. Ver
   [Negación: antes y después](#5-negación-antes-y-después).
4. **Las sospechas tienen su propio estado.** Antes, "sospecha de TEP" salía negada y
   "probable neumonía", confirmada. Desde el epic #86 salen `posible`: no suman al índice
   de Charlson, pero el resultado trae aparte el índice si se confirmaran. En la partición
   reservada de NUBes, de 617 menciones que el gold anota inciertas, 497 salen `posible`
   (antes, 520 salían confirmadas y 94 negadas). Ver
   [Incertidumbre](#6-incertidumbre-estado-posible).
5. **Las notas mal escritas hunden menos el índice** (epic #111): las siglas de la lista
   en minúscula y los errores de una letra se corrigen, con la corrección visible en la
   demo (con errores al 10 %, índice exacto de 79,8 a 85,6; con el texto bien escrito,
   92). Pero una lista hecha a mano solo cubre lo previsto, como muestra DisTEMIST. Ver
   [Robustez](#7-robustez-notas-mal-escritas).
6. **Normalizar con embeddings gana en CodiEsp, pero por memoria** (epic #128). SapBERT
   con las menciones de entrenamiento como sinónimos sube el F1 de categoría de 0,792 a
   0,874 y resuelve "ictus" y "accidente cerebrovascular" sin una regla. En las menciones
   que no estaban en entrenamiento queda igual que las reglas. Además, la licencia del
   modelo no está declarada y la latencia pasa de 1 s. Las reglas se quedan; ver
   [Normalización a CIE-10](#8-normalización-a-cie-10-codiesp).
7. Familiares, Charlson y subregistro funcionan bien en el set sintético. Pero esas notas
   las escribimos nosotros, así que esas cifras son una **cota optimista**.

## Qué se mide y qué no

- **Sí:** si el sistema encuentra las enfermedades de un texto clínico, si distingue cuáles
  están negadas, cuáles son sospechas (`posible`), cuáles son de un familiar, la categoría de Charlson, el índice y si una
  comorbilidad ya estaba entre los diagnósticos registrados.
- **No:** el código CIE-10 sugerido contra un gold (ningún corpus público en español lo
  tiene), los síntomas (el modelo solo busca enfermedades) ni el desempeño en notas reales
  de una IPS colombiana (no hay corpus público de ese tipo).
- **Nadie con formación clínica validó las anotaciones del proyecto** (el set sintético y la
  clasificación de errores). Las de DisTEMIST y NUBes son de sus autores.

## Datos

| Corpus | Qué es | Anota | Licencia | Uso aquí |
|---|---|---|---|---|
| [DisTEMIST](https://doi.org/10.5281/zenodo.7614764) (Miranda-Escalada et al., 2022) | 1.000 casos clínicos publicados en revistas, en español | Menciones de enfermedad, incluidas las negadas | CC BY 4.0 | NER: 250 de prueba (2.598 menciones) y 750 de entrenamiento (8.065) |
| [NUBes](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus) (Lima López et al., LREC 2020) | 29.682 frases de historias clínicas reales de hospitales de España, anonimizadas | Claves y alcance de la negación y la incertidumbre | CC BY-SA 3.0 ES | Negación e incertidumbre |
| [Set sintético reservado](../evaluacion/sintetico/README.md) | 100 notas con estilo de IPS colombiana, escritas para el proyecto | Entidades, estado, categoría de Charlson, índice, `registrado` | (del proyecto) | Familiares, Charlson, índice, subregistro |

Los corpus no están en el repo. Se descargan con el comando de evaluación, con verificación
de checksum. En el repo solo quedan métricas y fragmentos cortos, de 120 caracteres o menos,
citados con atribución.

## 1. Detección de enfermedades (DisTEMIST)

**Método.** Se compara cada mención predicha con las del gold, en caracteres sobre el
texto original:

- **Estricto:** mismo inicio y mismo fin.
- **Laxo:** se solapan; cada mención del gold empareja con una sola predicción.

P, R y F1 son micro. El intervalo del 95 % sale de un bootstrap sobre documentos, con 1.000
réplicas y semilla fija. En el pipeline, las predicciones son todas las menciones de todas
las entidades, con cualquier estado, porque DisTEMIST también anota las enfermedades
negadas (se verificó, #40). Resultados: `distemist_test.json`.

| | P estricta | R estricta | F1 estricta (IC 95 %) | P laxa | R laxa | F1 laxa (IC 95 %) |
|---|---|---|---|---|---|---|
| Modelo crudo | 0,720 | 0,736 | **0,728** (0,709–0,746) | 0,868 | 0,887 | **0,877** (0,867–0,888) |
| Pipeline completo | 0,717 | 0,652 | **0,683** (0,658–0,706) | 0,903 | 0,820 | **0,860** (0,846–0,872) |

La ficha del modelo reporta F1 0,757. Aquí da 0,728. Lo más probable es la forma de medir:
la ficha usa etiquetas por palabra y aquí se miden spans de caracteres sobre el texto
completo, así que un corte a media palabra cuenta como error estricto.

### ¿El modelo vio la partición de prueba?

No, según la evidencia ([`contaminacion.md`](../evaluacion/resultados/contaminacion.md)).
El modelo crudo da F1 0,864 (0,854–0,874) en entrenamiento y 0,728 en prueba: 13,6 puntos
menos, con intervalos que no se tocan. Si la prueba hubiera entrado al entrenamiento, las
dos cifras serían casi iguales y estarían por encima de la ficha, no por debajo.

### Ablación: qué aporta cada decisión de la tesis

Cada fila es el pipeline completo con **un solo** cambio. Los datos están en
`distemist_test.json`.

| Variante | F1 estricta (IC 95 %) | F1 laxa | Conclusión |
|---|---|---|---|
| Pipeline completo (fusión a 18, lista blanca) | 0,683 (0,658–0,706) | 0,860 | referencia |
| Sin fusión (= fusión a 0) | **0,722** (0,702–0,741) | 0,867 | La fusión resta 3,9 puntos de F1 estricto y une enfermedades distintas → **#55** |
| Fusión a 10 | 0,699 | 0,864 | Cuanta más distancia, peor |
| Fusión a 30 | 0,673 | 0,857 | |
| Sin expansión de palabras | 0,683 | 0,859 | Sin efecto (y mete comillas en el span → #53) |
| Filtro de longitud de la tesis (sin lista blanca) | 0,680 | 0,857 | La lista blanca de siglas ayuda un poco |
| Sin filtro de longitud | 0,691 | 0,867 | El filtro descarta siglas y enfermedades cortas que el modelo acierta → **#56** |
| Sin lista negra | 0,685 | 0,862 | Sin efecto claro → #57 |
| Sin descarte de sufijos truncados | 0,683 | 0,860 | Sin efecto → #57 |
| Sin deduplicación interna | 0,683 | 0,860 | Sin efecto a nivel de mención → #57 |

**Lectura.** La fusión se diseñó en la tesis para notas de IPS, donde el modelo parte las
entidades y el texto está sucio. DisTEMIST son casos publicados, con texto limpio. Que
estorbe aquí no prueba que estorbe en notas reales. Por eso **ningún paso se quitó en este
subproyecto**: la decisión se toma en #55 a #57 con evidencia de los dos tipos de texto. La
fusión también tiene un lado bueno: sube la precisión laxa de 0,868 a 0,903, porque une
pedazos que el modelo deja sueltos.

### Sin fusión (#55)

**Decisión: el pipeline ya no fusiona entidades cercanas.** La fusión unía enfermedades
distintas separadas por un conector ("demencia mixta + HTA + hipotiroidismo", "fiebre ni
disnea"), y #167 mostró que eso confunde además al normalizador.

**Cómo se decidió** (regla declarada en #55 antes de medir, como en #56): con DisTEMIST
**entrenamiento** y el set sintético; la más simple que no fuera peor que la mejor por más
de 0,005 de F1 estricto, sin bajar el índice de Charlson del set sintético. Se compararon
tres variantes: la fusión de la tesis, la misma con barreras ampliadas (" ni ", " con ",
": ", " y/o ", paréntesis y "+") y sin fusión.

| Variante | DisTEMIST entrenamiento, F1 estricto | F1 laxo | Set sintético, F1 categoría | Índice exacto |
|---|---|---|---|---|
| Fusión de la tesis (antes) | 0,809 | 0,918 | 0,863 | 0,92 |
| Fusión con barreras ampliadas | 0,847 | 0,930 | 0,898 | 0,95 |
| **Sin fusión** | **0,860** (+0,051 [+0,044; +0,058]) | **0,933** | **0,915** | **0,96** |

Sin fusión es la más simple y además la mejor en todo. Con la decisión tomada, DisTEMIST
**prueba** se midió una vez (`distemist_test_sin_fusion.json`, sobre c69f31f):

| Variante | P estricta | R estricta | F1 estricta (IC 95 %) | F1 laxa |
|---|---|---|---|---|
| Modelo solo | 0,720 | 0,736 | 0,728 (0,709–0,746) | 0,877 |
| Pipeline con la fusión de la tesis | 0,718 | 0,667 | 0,691 (0,668–0,714) | 0,868 |
| Fusión con barreras ampliadas | 0,736 | 0,711 | 0,723 (0,703–0,742) | 0,878 |
| **Pipeline sin fusión** | 0,733 | 0,734 | **0,734** (0,715–0,751) | **0,878** |

- Quitar la fusión sube el F1 estricto 0,042 [0,029; 0,055] (pareado) y el laxo 0,010.
  El pipeline queda igual o apenas por encima del modelo solo (0,734 contra 0,728, con IC
  que se solapan); antes quedaba por debajo.
- La línea base del set sintético con fusión es `sintetico_v5_reglas.json` (cdaad27); entre
  ese commit y este no cambió nada de la extracción.
- En el set sintético el estado sigue en 100 %.
- En CodiEsp desarrollo de punta a punta casi no cambia: reglas 0,721 → 0,724,
  normalizador 0,762 → 0,764 (`extremo_codiesp_desarrollo_v2.json`).
- **Lo que se pierde:**
  - `registrado` pasa de 65 a 64 de 68: arregla un caso (sin021, "DM2 que ingresa por
    IVU") y empeoran dos. "DM2 con retinopatía diabética" y "DM2 con pie diabético" salían
    como una entidad que coincidía con el diagnóstico registrado "Diabetes mellitus no
    insulinodependiente con complicaciones oftálmicas" (o "circulatorias periféricas").
    Separadas, "retinopatía diabética" y "pie diabético" ya no coinciden por texto con él
    y salen como posible subregistro. Es un límite del cruce con los registrados, que
    compara textos y no categorías ni códigos.
  - Cuando el modelo parte una misma enfermedad en dos pedazos ("Artritis" y
    "seropositiva"), ya no se une. En DisTEMIST pesa menos que lo que se gana; si en notas
    reales de una IPS aparece, la fusión sigue en `ConfiguracionExtraccion` para volver a
    medirla.
- **#54**, la fusión que unía renglones distintos cuando el modelo metía el salto de línea
  en la entidad ("TB\n"), ya no pasa en el pipeline (no fusiona). En la fusión que queda
  para la ablación se corrigió: la barrera se busca también en los bordes de las
  entidades. Las filas de fusión de `distemist_test_sin_fusion.json` son de antes de esa
  corrección.
- Primero se corrió prueba con la fila de barreras ampliadas mal configurada (después del
  cambio de valor por defecto, la variante ya no fusionaba); esa corrida se descartó y se
  repitió con la fila corregida. La cifra del pipeline es la misma en las dos.

### Análisis de errores

Muestra con semilla fija de 50 falsos positivos y 50 falsos negativos estrictos del
pipeline completo, de una población de 667 FP y 905 FN. Los clasificó el autor, sin
validación clínica. Datos en [`errores.json`](../evaluacion/resultados/errores.json).

| Categoría | FP | FN | Total |
|---|---|---|---|
| Límite del span (sobra o falta texto) | 21 | 20 | 41 |
| Fusión indebida | 16 | 12 | 28 |
| Enfermedad no detectada | 0 | 11 | 11 |
| Síntoma o hallazgo que el gold no anota | 8 | 0 | 8 |
| Otra (menciones anidadas del gold, descarte por filtro) | 0 | 4 | 4 |
| Error del gold | 2 | 1 | 3 |
| Ruido | 3 | 0 | 3 |
| Sigla | 0 | 2 | 2 |

- 39 de los 50 FP son aciertos en modo laxo.
- 33 de los 100 errores los mete el post-proceso y no el modelo, la mayoría por fusión. En
  todos esos casos, el modelo solo acertaba.

## 2. Negación en texto clínico real (NUBes)

**Método** (#59). Cada frase de NUBes se analiza sola con `analizar()`, porque agrupar
frases cambia lo que detecta el modelo y mezcla la negación de una frase con la de otra.
Cada entidad detectada se cruza con los alcances anotados en el gold:

- Si al menos la mitad de la mención cae dentro de un alcance de **negación**, el estado
  esperado es `negada`.
- Si no cae en ninguno, el esperado es `confirmada`.
- Si cae en un alcance de **incertidumbre**, se reporta aparte.

Se evaluaron 12.378 menciones en 9.474 frases del corpus completo. Datos en
[`nubes.json`](../evaluacion/resultados/nubes.json).

> Esta sección es la medición **con las reglas de la tesis**, antes del epic #71. Lo que
> cambió al corregirlas, medido en la partición reservada, está en
> [Negación: antes y después](#5-negación-antes-y-después).

| Esperado \ obtenido | confirmada | negada | familiar |
|---|---|---|---|
| negada | 282 | **871** | 2 |
| incierta | 1.086 | 199 | 3 |
| afirmada | **9.766** | 155 | 14 |

| | Precisión | Recall | F1 (IC 95 %) |
|---|---|---|---|
| `negada` | 0,849 | 0,754 | **0,799** (0,779–0,817) |
| `negada`, solo la muestra con doble anotación (SAMPLE-001) | 0,859 | 0,840 | 0,849 |

- **Cuando el sistema dice "negada", acierta el 85 % de las veces, pero deja sin negar una
  de cada cuatro negadas.** Una negada que sale confirmada infla el índice de Charlson y
  oculta un posible subregistro. Este es el error que más conviene corregir.
- En una muestra de 30 discrepancias, las causas fueron:
  - 16, alcance mal cortado: 11 se alargan más allá de la cláusula, porque no cortan en
    ";", ":" ni en paréntesis (#66), y 5 se quedan cortos ("no presenta" #61,
    enumeraciones #62, una coma);
  - 6, formas de "descartar" que el pipeline no reconoce ("descartó") → #65;
  - 3, claves que no tiene ("nunca", "ninguna") → #65;
  - 2, falsos disparadores ("seronegativa") → #66;
  - 2, del gold;
  - 1, otra causa.

  Según el conteo completo que se documenta en los issues (no en `nubes.json`), 165 de las
  284 negadas perdidas tienen una clave que el pipeline no reconoce (#65). NUBes también
  confirma las enumeraciones (#62, 48 casos) y "no presenta" (#61, 18 casos), según el
  comentario de cierre de #59.
- **Incertidumbre:** de 1.288 menciones dentro de un alcance de incertidumbre, 199 salen
  negadas, porque "sospecha de" se trata como negación desde la tesis, 3 familiares y el
  resto (1.086) confirmadas. El pipeline no tenía un estado para las sospechas. Era un
  límite, no un error, y lo resolvió el epic #86 con el estado `posible` (ver
  [Incertidumbre](#6-incertidumbre-estado-posible)).
- **Contexto por texto y no por posición** (límite de la tesis, corregido en #73): con
  frases sueltas cambiaba el estado de 23 menciones. Ahora cada mención se clasifica por
  su posición; el efecto de ese cambio está en la tabla de pasos de
  [Negación: antes y después](#5-negación-antes-y-después).

## 3. Familiares, Charlson y subregistro (set sintético reservado)

**El set.** 100 notas sintéticas con estilo de IPS colombiana: 339 entidades (217
confirmadas, 71 negadas, 51 familiares), las 17 categorías de Charlson, 115 siglas y 18
notas con diagnósticos registrados. Las reglas del set:

- **Se midió una sola vez a ciegas** (#45), sin cambiar nada del pipeline ni de las notas
  después. Las cifras de esta sección son esa medición. Después se corrigió la negación
  mirando, entre otras cosas, los errores de este set; la segunda medición
  (`sintetico_v2.json`) va aparte, marcada como tal, en
  [Negación: antes y después](#5-negación-antes-y-después).
- **Validación:** un segundo anotador independiente las anotó a ciegas. El acuerdo fue F1
  0,985 en entidades, kappa 1,0 en el estado e índice igual en 98 de 100 notas. Las 10
  diferencias se adjudicaron con reglas escritas
  ([README del set](../evaluacion/sintetico/README.md)).
- **Límite:** los dos anotadores son instancias del mismo modelo con los mismos criterios.
  El acuerdo prueba que las anotaciones son consistentes, no que un clínico las valide.

Datos en [`sintetico.json`](../evaluacion/resultados/sintetico.json).

| Métrica | Valor |
|---|---|
| Recall de entidades esperadas | 97,6 % (331 de 339) |
| Estado correcto | 94,3 % (312 de 331) |
| — confirmadas / negadas / familiares | 99,5 % / **74,2 %** / 98,0 % |
| Categoría de Charlson (confirmadas): P / R / F1 | 0,847 / 0,878 / 0,862 |
| Índice de Charlson exacto | 86 de 100 notas |
| Índice con 1 punto o menos de error | 95 de 100 notas |
| `registrado` correcto (subregistro) | 94,1 % (64 de 68) |

- Los **familiares** salen bien: el 98 %.
- Las **negadas** dan 74,2 %, una cifra parecida al 75,4 % de recall en NUBes (no es la
  misma métrica: aquí es el estado correcto entre las negadas encontradas). Que el set
  sintético y el texto real se parezcan le da credibilidad al set. 10 de los 17 errores son
  enumeraciones ("niega HTA, DM, EPOC": solo se niega la primera, #62) y 7 son "no presenta"
  (#61).
- De los 14 errores del **índice**:
  - 6 vienen de la negación (#61, #62);
  - 4 de la fusión, que une dos enfermedades y deja una sola categoría (#55);
  - 2 de huecos en las reglas: fibrosis pulmonar e insuficiencia arterial crónica (#63);
  - 2 de otras causas: una entidad no detectada y una hemiparesia de ACV agudo contada como
    hemiplejía.

  La cirrosis Child-Pugh C sale leve (#63), pero en esa nota no cambió el índice.

Por categoría de Charlson (confirmadas):

| Categoría | P | R | VP | FP | FN |
|---|---|---|---|---|---|
| `infarto_miocardio` | 1,00 | 0,86 | 6 | 0 | 1 |
| `insuficiencia_cardiaca` | 0,70 | 1,00 | 7 | 3 | 0 |
| `vascular_periferica` | 1,00 | 0,80 | 4 | 0 | 1 |
| `cerebrovascular` | 0,86 | 1,00 | 6 | 1 | 0 |
| `demencia` | 0,78 | 1,00 | 7 | 2 | 0 |
| `pulmonar_cronica` | 1,00 | 0,92 | 11 | 0 | 1 |
| `reumatologica` | 0,89 | 1,00 | 8 | 1 | 0 |
| `ulcera_peptica` | 0,88 | 1,00 | 7 | 1 | 0 |
| `hepatica_leve` | 0,73 | 0,80 | 8 | 3 | 2 |
| `diabetes` | 0,92 | 0,86 | 12 | 1 | 2 |
| `diabetes_complicada` | 0,67 | 0,86 | 6 | 3 | 1 |
| `hemiplejia` | 1,00 | 0,75 | 3 | 0 | 1 |
| `renal_cronica` | 0,90 | 0,90 | 9 | 1 | 1 |
| `tumor_maligno` | 0,88 | 0,64 | 7 | 1 | 4 |
| `hepatica_severa` | 1,00 | 0,70 | 7 | 0 | 3 |
| `tumor_metastasico` | 0,67 | 1,00 | 8 | 4 | 0 |
| `vih_sida` | 0,86 | 1,00 | 6 | 1 | 0 |

Con 4 a 14 casos por categoría, cada cifra tiene mucha incertidumbre. Sirven para ubicar
huecos, no para comparar categorías entre sí.

### Cruce con los registrados por categoría (#176)

Desde #55, "DM2 con retinopatía diabética" sale como dos entidades, y "retinopatía
diabética" sola no comparte palabras con el registrado "Diabetes mellitus no
insulinodependiente con complicaciones oftálmicas": salía como posible subregistro.

**El cambio:** si una confirmada no coincide por texto, se cruza también por categoría de
Charlson (con el mismo mapeo de las entidades), **solo en la diabetes**: "retinopatía
diabética" coincide con "Diabetes … con complicaciones X"; "DM2" coincide con "Diabetes …
con complicaciones" (la diabetes está registrada); pero "retinopatía diabética" no
coincide con "Diabetes … sin complicación" (la complicación no está registrada). La
entidad dice cómo coincidió: `coincide_por` es `"texto"` o `"categoria"`.

**Cómo se llegó a solo diabetes** (con el set sintético, que es el único con
`registrado`, y con las reglas, no con el normalizador; queda dicho porque se eligió
mirándolo):

| Cruce | `registrado` correcto | Arregla | Rompe |
|---|---|---|---|
| Solo por texto (#55) | 64 de 68 | — | — |
| Por categoría, salvo tumores y reumatológica (`sintetico_v7_cruce_amplio.json`) | 64 de 68 | retinopatía y pie diabético | "cardiopatía isquémica" (las reglas la daban como insuficiencia cardiaca; corregido en #201) y "hepatitis B crónica" (cruzada con una cirrosis registrada) |
| **Solo diabetes** (`sintetico_v7.json`, 15bc353) | **66 de 68 (97,1 %)** | retinopatía y pie diabético | nada |

- La diabetes es el caso en que la CIE-10 registra la complicación dentro del diagnóstico
  ("Diabetes … con complicaciones X"). En las demás categorías, una categoría agrupa
  enfermedades distintas o arrastra los errores del mapeo, y un falso `registrado`
  esconde un subregistro, que es lo que el auditor busca.
- El índice exacto (96 %), la categoría (F1 0,915) y el estado (100 %) no cambian.
- **Límite:** por categoría no se distingue la complicación. Una "nefropatía diabética"
  coincidiría con "Diabetes … con complicaciones oftálmicas" aunque la nefropatía no esté
  registrada. Los códigos no sirven para separarlas: las reglas le dan a la descripción
  registrada un código genérico (E10.8). Para Charlson es la misma categoría; para el
  auditor que busca subregistro de códigos, no. En el set sintético no aparece.
- Con el normalizador (el mapeo por defecto de la API), las categorías de los registrados
  salen de SapBERT; no se midió aparte.
- Quedan 2 errores: "neumonía broncoaspirativa" registrada como "Neumonitis debida a
  aspiración" (no comparten palabras ni es de Charlson) y una "HTA" que coincide con un
  registrado que no debería.

### Huecos de las reglas (#123, #120, #63)

Las reglas no conocían varias formas comunes. Se agregaron, cada una con su test
(`tests/test_charlson_correcciones.py`):

- **Cerebrovascular (#123):** "accidente/enfermedad/evento cerebrovascular", "evento
  vascular cerebral", "ictus", "derrame cerebral", "apoplejía" y "embolia cerebral"; el
  ACV "hemorrágico" va a I61.
- **Vascular periférica (#123, #63):** "arteriopatía periférica (obliterante)" e
  "insuficiencia arterial crónica de miembros".
- **Pulmonar crónica (#63):** "fibrosis pulmonar" y la enfermedad pulmonar intersticial
  (J84). Quan no incluye J84; se sigue el Charlson original y los criterios del set (ver
  `docs/datos.md`).
- **Tumor (#120):** un tumor "sin metástasis" (o "sin evidencia de", "libre de") vuelve a
  ser tumor maligno; antes la palabra "metástasis" lo excluía aunque estuviera negada.
- **Hepática (#63):** cirrosis con Child-Pugh C, descompensada o con ascitis es severa;
  Child-Pugh A, compensada o "sin ascitis" sigue leve. La hepatitis alcohólica es leve
  (K70.1, como en Quan).

Set sintético (`sintetico_v8.json`; `sintetico_v9.json` repite la medición tras la revisión
del PR #183, que agregó negaciones, con las mismas cifras; las mismas 100 notas, que ya no
son ciegas: estos huecos salieron de sus errores):

| | Antes (`sintetico_v7.json`) | Después |
|---|---|---|
| Índice exacto | 96 % | **98 %** |
| F1 de categoría | 0,915 | **0,926** (recall 0,915 → 0,936) |
| `registrado` correcto | 66 de 68 | 66 de 68 |

Los dos errores del índice que quedan (sin007 y sin038) no son de las reglas. La
hepatopatía severa sigue con recall 0,70: en sus tres pérdidas la entidad dice solo
"cirrosis hepática" y la severidad está en otra parte de la nota. Las reglas mapean la
entidad, no la nota; en esas notas el índice sale bien porque el marcador severo sale
como entidad aparte.

## 4. Rendimiento

Pipeline completo sin caché, nota por nota, sobre los 250 casos de prueba de DisTEMIST
(mediana de 2.138 caracteres). Datos en `distemist_test.json`.

| | Valor |
|---|---|
| Tiempo por nota: mediana / p95 / máximo | 0,85 s / 1,48 s / 2,57 s |
| Tiempo por cada 1.000 caracteres | 359 ms |
| Memoria pico del proceso | 1.045 MiB |
| CPU | Intel Core Ultra 7 255U (14 hilos), sin GPU |

**Para el despliegue** (Cloud Run, 2 vCPU y 4 GiB):

- La memoria cabe con holgura.
- Con 2 vCPU la inferencia va a ser varias veces más lenta que en esta CPU de 14 hilos. Hay
  que medirlo en el subproyecto de la API antes de fijar el timeout y el tamaño máximo de
  nota.

## 5. Negación: antes y después

El epic #71 corrigió la negación heredada de la tesis (PRD
[`docs/prds/negacion.md`](prds/negacion.md)). Para que el antes y el después sea honesto,
NUBes se partió en dos (#72):

- **Desarrollo** (muestras pares, 14.865 frases): la única que se miró mientras se
  ajustaban las reglas.
- **Reservada** (muestras impares, 14.817 frases, incluida SAMPLE-001, la de doble
  anotación): medida una vez con las reglas de la tesis (línea base) y **una sola vez**
  con el código final (#74). Es la cifra con peso.

El método es el de la [sección 2](#2-negación-en-texto-clínico-real-nubes). Antes y después
se evalúan las mismas menciones (5.954 en la reservada), porque el contexto no cambia lo
que detecta el NER.

### Paso a paso, en la partición de desarrollo

**Ojo:** esta es la partición con la que se ajustaron las reglas. Cada paso se eligió
porque subía estas cifras, así que la mejora que muestra es optimista. La primera y la
última fila salen de
[`nubes_linea_base_desarrollo.json`](../evaluacion/resultados/nubes_linea_base_desarrollo.json)
y [`nubes_despues_desarrollo.json`](../evaluacion/resultados/nubes_despues_desarrollo.json);
las intermedias, de la medición de cada PR, documentada en el docstring de
`src/ner_clinic/contexto.py`.

| Paso | Issue | P | R | F1 de `negada` |
|---|---|---|---|---|
| Reglas de la tesis (línea base) | #72 | 0,853 | 0,746 | 0,795 |
| Clasificar cada mención por su posición, no por su texto | #73 | 0,854 | 0,775 | 0,813 |
| Enumeraciones ("niega HTA, DM y EPOC") | #62 | 0,845 | 0,792 | 0,817 |
| "No presenta X" | #61 | 0,845 | 0,802 | 0,823 |
| Claves que faltaban ("descartó", "tampoco", "ningún"…) | #65 | 0,849 | 0,855 | 0,852 |
| Alcance: corte en ";", ":" y paréntesis; hacia atrás, 4 palabras | #66 | **0,903** | **0,851** | **0,877** |

Las claves (#65) son las que más recall suman, y los cortes del alcance (#66), los que
más precisión.

### Partición reservada: la cifra con peso

Datos en
[`nubes_linea_base_reservada.json`](../evaluacion/resultados/nubes_linea_base_reservada.json)
(reglas de la tesis, commit `272d12f`) y
[`nubes_despues_reservada.json`](../evaluacion/resultados/nubes_despues_reservada.json)
(código final, commit `9ad2002`, árbol limpio). IC del 95 % por bootstrap.

| `negada` | Precisión (IC 95 %) | Recall (IC 95 %) | F1 (IC 95 %) | VP | FP | FN |
|---|---|---|---|---|---|---|
| Antes (reglas de la tesis) | 0,845 (0,814–0,875) | 0,764 (0,725–0,802) | 0,802 (0,774–0,829) | 420 | 77 | 130 |
| Después (reglas corregidas) | **0,885** (0,858–0,911) | **0,844** (0,811–0,875) | **0,864** (0,841–0,886) | 464 | 60 | 86 |

Matriz de confusión (esperado en filas, obtenido en columnas):

| Esperado \ obtenido | Antes: confirmada | Antes: negada | Antes: familiar | Después: confirmada | Después: negada | Después: familiar |
|---|---|---|---|---|---|---|
| negada (550) | 128 | 420 | 2 | **84** | **464** | 2 |
| incierta (617) | 518 | 96 | 3 | 520 | 94 | 3 |
| afirmada (4.787) | 4.700 | 77 | 10 | 4.718 | **60** | 9 |

- **Las negadas que salen confirmadas bajan de 128 a 84** (un 34 % menos). Es el error más
  caro para un auditor: suma al índice de Charlson y oculta un posible subregistro.
- **Las afirmadas que salen negadas bajan de 77 a 60**: el recall no se pagó con más
  falsas negaciones, al contrario.
- El F1 sube en las cinco muestras. En SAMPLE-001, la de doble anotación, pasa de 0,849 a
  0,900.
- Los intervalos del F1 de antes (0,774–0,829) y de después (0,841–0,886) no se tocan.
- En la reservada el F1 queda en 0,864, por debajo del 0,877 de desarrollo. Es lo esperable
  al ajustar mirando una partición: parte de la mejora en desarrollo era ajuste a esas
  frases. Por eso la cifra que se reporta es la de la reservada.

### Set sintético: segunda medición, después de ver los errores

**No es ciega.** Los errores de la primera medición (#45) se usaron para encontrar las
causas que corrigió el epic, así que esta cifra es todavía más optimista que la primera.
Se reporta para ver si las correcciones llegaron a las formas colombianas y al índice de
Charlson, no como prueba de que el sistema mejoró. La primera medición (`sintetico.json`)
no se tocó. Datos en [`sintetico_v2.json`](../evaluacion/resultados/sintetico_v2.json)
(commit `9ad2002`).

| Métrica | v1, ciega (#45) | v2, segunda medición |
|---|---|---|
| Estado correcto | 94,3 % (312 de 331) | 99,4 % (329 de 331) |
| — confirmadas | 99,5 % (213 de 214) | 99,5 % (213 de 214) |
| — negadas | 74,2 % (49 de 66) | **100 %** (66 de 66) |
| — familiares | 98,0 % (50 de 51) | 98,0 % (50 de 51) |
| Categoría de Charlson (confirmadas): P / R / F1 | 0,847 / 0,878 / 0,862 | 0,847 / 0,878 / 0,862 |
| Índice de Charlson exacto | 86 de 100 | **91 de 100** |
| Índice con 1 punto o menos de error | 95 de 100 | 96 de 100 |
| `registrado` correcto (subregistro) | 94,1 % (64 de 68) | 95,6 % (65 de 68) |

- Los 17 errores de negadas de la v1 (10 enumeraciones y 7 "no presenta") desaparecen.
- Salen 6 notas de la lista de errores del índice (sin013, sin016, sin025, sin049, sin081
  y sin100) y entra una nueva, sin086: en "ACV … sin secuelas motoras, demencia vascular
  leve", la regla de enumeraciones (#62) sigue la negación del "sin" a través de la coma y
  niega la demencia. Es el tipo de FP que se aceptó al medir la lista en desarrollo, a
  cambio del recall que suma. Se corrigió después, en #84 (ver la
  [sección 6](#6-incertidumbre-estado-posible)).
- La confirmada que salía como familiar en la v1 (la ICC de sin095) ahora sale bien, por la
  clasificación por posición (#73), y por eso sube `registrado`. El error de confirmadas de
  la v2 es otro: la demencia de sin086.
- La categoría de Charlson queda igual en conjunto: la ICC de sin095 suma un VP a
  insuficiencia cardiaca y la demencia de sin086 le quita uno a demencia.

### ¿Se cumplieron las metas del PRD?

| Meta | Resultado | ¿Cumple? |
|---|---|---|
| Recall de `negada` en la reservada ≥ 0,85 | 0,844 (IC 0,811–0,875) | **No, por poco**: faltaron 4 de 550 negadas |
| … sin que la precisión baje de 0,80 | 0,885 | Sí |
| `negada` en el sintético (2.ª medición) ≥ 0,90 | 1,000 | Sí, con la salvedad de que no es ciega |
| Índice exacto en el sintético ≥ 90 | 91 | Sí, con la misma salvedad |
| Sin regresiones de más de 1 punto en confirmadas y familiares del sintético | 99,5 % → 99,5 %; 98,0 % → 98,0 % | Sí |

**Lectura honesta.** La mejora es real: en frases que no se usaron para ajustar, el recall
sube 8 puntos y la precisión 4, y los intervalos del F1 de antes y de después no se tocan.
La meta de recall (0,85) no se alcanzó por 0,006. Está dentro del ruido (el intervalo la
incluye), pero no se puede decir que se cumplió. La meta se fijó frente a la línea base
del corpus completo (0,754); la de la reservada era 0,764. Las metas del set sintético se
cumplen, pero ese set ya no es ciego para estas reglas: la cifra que vale es la de NUBes.

**DisTEMIST no cambia.** Con la caché, el pipeline final da las mismas métricas que
`distemist_test.json` (F1 estricto 0,683, laxo 0,860) y las mismas 2.360 menciones
predichas, en las mismas posiciones: el contexto no toca los spans. Solo cambia el orden
en que salen algunas, porque una entidad con menciones en estados distintos se separa en
dos (#73).

### Lo que queda

- **Incertidumbre.** Al cerrar este epic, "sospecha de" seguía contando como negación:
  en la reservada, 94 de 617 menciones dentro de un alcance de incertidumbre salían
  negadas (96 antes) y 520, confirmadas. Además, "no se descarta TEP" salía negada,
  porque "descarta" (#65) es disparador. Lo resolvió el epic #86 con el estado `posible`
  (#83, #88); ver [Incertidumbre](#6-incertidumbre-estado-posible).
- **Resueltas.** "cede", "desaparición de" y "se resuelve" no niegan, a propósito: para un
  auditor, una enfermedad resuelta no es una negada (PRD, FR5). Un estado "resuelta"
  también quedó fuera.
- **Las 86 negadas que aún salen confirmadas** en la reservada no se clasificaron, para no
  ajustar reglas mirando esa partición. Si se ajusta algo más, hace falta otra partición
  reservada o aceptar que esta deja de serlo.
- **Un modelo de negación** (*assertion classification*) frente a estas reglas, entrenado en
  desarrollo y medido en la reservada: experimento #77, de prioridad baja. Las reglas
  tienen a su favor que se le pueden explicar a un auditor.

## 6. Incertidumbre (estado `posible`)

Una sospecha ("sospecha de TEP", "probable neumonía", "no se descarta IAM") no es un
diagnóstico ni un descarte. Antes del epic #86 tenía que caer en otro estado: "sospecha
de" salía negada (herencia de la tesis) y "probable" o "compatible con", confirmadas. El
epic agregó un cuarto estado, **`posible`** (PRD
[`docs/prds/incertidumbre.md`](prds/incertidumbre.md)):

- Se detecta con reglas, con la misma mecánica que la negación (#83): una clave, un
  alcance hacia adelante con cortes y, para unas pocas claves, las 4 palabras de antes.
- No suma a `indice_charlson`. El resultado trae aparte `indice_charlson_si_se_confirman`
  (#88).
- La prioridad es familiar > posible > negada > confirmada: en "no se descarta X", el "no"
  abre una negación, pero X queda posible.
- Además, se corrigió la lista de #62, que negaba una enfermedad de la cláusula siguiente
  (#84).

**Método.** El de la [sección 2](#2-negación-en-texto-clínico-real-nubes), con la
matriz en 4 columnas. Las métricas de `negada` no cambian de definición: se calculan
sobre las menciones que el gold anota negadas o afirmadas, así que una incierta que deja
de salir negada no les suma. Las de `posible` se calculan sobre todas las menciones, y lo
esperado es que el gold las ponga en un alcance de incertidumbre. Las particiones son las
de #72 (sección 5).

### Partición reservada: la cifra con peso

Es la **segunda vez** que se usa la reservada: la primera fue la medición final de la
negación (#74). Entre las dos, las reglas de este epic se ajustaron solo con la partición
de desarrollo. Las menciones evaluadas son las mismas 5.954, porque el contexto no cambia
lo que detecta el NER.

Datos en
[`nubes_despues_reservada.json`](../evaluacion/resultados/nubes_despues_reservada.json)
(antes: código final del epic #71, commit `9ad2002`) y
[`nubes_incertidumbre_reservada.json`](../evaluacion/resultados/nubes_incertidumbre_reservada.json)
(después: commit `990ca2b`, árbol limpio). IC del 95 % por bootstrap.

| Estado | | Precisión (IC 95 %) | Recall (IC 95 %) | F1 (IC 95 %) | VP | FP | FN |
|---|---|---|---|---|---|---|---|
| `negada` | Antes (#74) | 0,885 (0,858–0,911) | 0,844 (0,811–0,875) | 0,864 (0,841–0,886) | 464 | 60 | 86 |
| `negada` | Después | **0,899** (0,873–0,924) | 0,842 (0,810–0,875) | **0,870** (0,847–0,892) | 463 | 52 | 87 |
| `posible` | Antes | (el estado no existía) | 0 | 0 | 0 | 0 | 617 |
| `posible` | Después | **0,919** (0,896–0,941) | **0,805** (0,772–0,836) | **0,858** (0,835–0,879) | 497 | 44 | 120 |

Matriz de confusión, antes → después (esperado en filas, obtenido en columnas):

| Esperado \ obtenido | confirmada | negada | posible | familiar |
|---|---|---|---|---|
| negada (550) | 84 → 84 | 464 → 463 | 0 → 1 | 2 → 2 |
| incierta (617) | 520 → **91** | 94 → **26** | 0 → **497** | 3 → 3 |
| afirmada (4.787) | 4.718 → 4.683 | 60 → 52 | 0 → 43 | 9 → 9 |

- **De 617 sospechas, 497 salen `posible`.** Antes, 520 salían confirmadas (sumaban al
  índice como si el paciente las tuviera) y 94 negadas (se veían igual que "niega X").
  Ahora quedan 91 confirmadas y 26 negadas.
- **`negada` mejora poco**: el F1 pasa de 0,864 a 0,870, por 8 FP menos, con los
  intervalos solapados. Es lo esperado: lo que más cambia para la negación son las
  inciertas que dejan de salir negadas (de 94 a 26), y esas no entran en sus métricas.
  El recall queda igual (una negada sale ahora posible). En SAMPLE-001, la de doble
  anotación, el F1 de negada pasa de 0,900 a 0,910.
- **El costo:** 43 de 4.787 menciones afirmadas (0,9 %) salen posibles. En neto, 35
  confirmadas y 8 negadas menos. Una afirmada que sale posible deja de sumar a
  `indice_charlson`, aunque sí suma al índice si se confirman.
- En la reservada, el F1 de `posible` (0,858) queda por encima del de desarrollo (0,846,
  abajo): el recall es mayor (0,805 frente a 0,768) y la precisión, menor (0,919 frente a
  0,941). Los intervalos se solapan, así que no se puede decir que funcione mejor en la
  reservada, pero sí que no hay señales de que las reglas se ajustaran de más a
  desarrollo.
- Las 120 sospechas que no salen posibles y los 44 FP no se clasificaron, para no ajustar
  reglas mirando la reservada.

### Partición de desarrollo: la de ajuste

**Ojo:** con esta partición se eligieron las claves, así que su cifra es optimista.
Datos en
[`nubes_incertidumbre_linea_base_desarrollo.json`](../evaluacion/resultados/nubes_incertidumbre_linea_base_desarrollo.json)
(línea base del epic, commit `c12ab1e`, antes de #84 y #83) y
[`nubes_incertidumbre_desarrollo.json`](../evaluacion/resultados/nubes_incertidumbre_desarrollo.json)
(final, commit `990ca2b`).

| Desarrollo (6.424 menciones) | `negada`: P / R / F1 | `posible`: P / R / F1 (IC 95 % del F1) | `posible`: VP / FP / FN |
|---|---|---|---|
| Línea base | 0,903 / 0,851 / 0,877 | 0 / 0 / 0 | 0 / 0 / 671 |
| Final | 0,913 / 0,850 / **0,880** | 0,941 / 0,768 / **0,846** (0,820–0,868) | 515 / 32 / 156 |

Cada grupo de claves entró solo si subía el F1 de `posible` en desarrollo, no bajaba el
de `negada` más de 0,002 y sus aciertos nuevos superaban a sus errores nuevos. Las cifras
de cada grupo están en el docstring de `src/ner_clinic/contexto.py` (medidas en cada PR,
no en un JSON de este reporte). Las principales, como lo que pierde el F1 de `posible`
al quitar el grupo del conjunto final (aciertos y FP que aporta):

| Clave | Aporte al F1 de `posible` | Aciertos | FP |
|---|---|---|---|
| "probable(s)", "probablemente" | +0,126 | 123 | 6 |
| "compatible(s) con" | +0,093 | 93 | 3 |
| "sospecha de", "sospechoso de", "sospechar" | +0,082 | 84 | 5 |
| "posible(s)", "posiblemente" | +0,054 | 56 | 2 |
| "sugestivo/a(s) de" | +0,038 | 39 | 1 |
| "orientación diagnóstica:" | +0,011 | 11 | 0 |
| "parece(n)", "no parece(n)" | +0,010 | 12 | 2 |

Entraron también, con menos de 0,01 cada una: "sugiere", "podría", "impresiona (de)", "no
se descarta" y sus variantes, "puede", "se interpreta como", "sin clara", "posibilidad
de", "dudoso", "aparente", "vs", "?" hacia atrás, "se plantea", "pudiera", "a valorar" y
"al parecer".

Se **descartaron**, porque bajaban el F1 de `posible` al agregarlas:

| Clave | Cambio en el F1 de `posible` | Por qué |
|---|---|---|
| "probable" hacia atrás ("neumonía probable") | −0,015 | 6 aciertos y 31 FP ("lo más probable", "como probable causa") |
| "a descartar", "para descartar" | −0,012 | 0 aciertos y 18 FP: dicen qué se busca, no qué se sospecha |
| "posible" hacia atrás | −0,008 | 0 aciertos y 12 FP |
| "valorar" suelto | −0,0006 | 3 aciertos y 5 FP ("valorar amigdalectomía") |

### Set sintético: tercera medición

**No es ciega**, igual que la segunda. Sirve para comprobar que el epic no empeoró
negadas, familiares ni el índice frente a la v2, y que sin086 (#84) quedó corregida. El
set no anota `posible` y no se reanotó: una entidad del gold que saliera posible contaría
como error de estado. Datos en
[`sintetico_v3.json`](../evaluacion/resultados/sintetico_v3.json) (commit `990ca2b`); la v1
y la v2 no se tocaron.

| Métrica | v2 (#74) | v3 (#89) |
|---|---|---|
| Estado correcto | 99,4 % (329 de 331) | **99,7 %** (330 de 331) |
| — confirmadas | 99,5 % (213 de 214) | **100 %** (214 de 214) |
| — negadas | 100 % (66 de 66) | 100 % (66 de 66) |
| — familiares | 98,0 % (50 de 51) | 98,0 % (50 de 51) |
| Entidades del gold que salen `posible` | (el estado no existía) | **0** de 331 |
| Categoría de Charlson (confirmadas): P / R / F1 | 0,847 / 0,878 / 0,862 | 0,848 / 0,879 / 0,863 |
| Índice de Charlson exacto | 91 de 100 | **92 de 100** |
| Índice con 1 punto o menos de error | 96 de 100 | 96 de 100 |
| `registrado` correcto (subregistro) | 95,6 % (65 de 68) | 95,6 % (65 de 68) |

- **sin086 quedó corregida**: la demencia vascular de "sin secuelas motoras, demencia
  vascular leve" sale confirmada, suma un VP a demencia y la nota sale de los errores del
  índice.
- **Ninguna entidad del gold sale `posible`.** Sí sale posible una de las 38 entidades que
  el gold no anota: "síndrome hereditario" en sin047 ("se sospecha síndrome
  hereditario"), que es en efecto una sospecha y no tiene categoría de Charlson.
- El único error de estado que queda es el de la v1 y la v2: la "AR" de un familiar en
  sin094, que sale confirmada.

### ¿Se cumplieron las metas del PRD?

| Meta | Resultado | ¿Cumple? |
|---|---|---|
| F1 de `posible` en la reservada ≥ 0,70 | 0,858 (IC 0,835–0,879) | Sí, con holgura: el IC entero está por encima |
| F1 de `negada` en la reservada ≥ 0,85 | 0,870 (IC 0,847–0,892) | Sí; el extremo bajo del IC queda apenas por debajo |
| Sintético v3 sin regresiones frente a la v2 en negadas, familiares e índice exacto | 100 % → 100 %; 98,0 % → 98,0 %; 91 → 92 | Sí, con la salvedad de que no es ciego |
| sin086 (#84) corregida | Sale confirmada | Sí |
| Los ejemplos de US1 y US2 son tests | `tests/test_contexto.py`, `tests/test_pipeline.py` | Sí |

**Lectura honesta.** La meta de `posible` (0,70) se fijó pensando en que la incertidumbre
es más difusa que la negación, y se supera con margen en frases que no se usaron para
ajustar. Pero hay que leerla con tres salvedades:

1. **NUBes anota la incertidumbre con criterio de lingüistas sobre texto de España.**
   "Compatible con" es la segunda clave que más aporta, y en radiología colombiana a veces
   se lee como casi una confirmación. En notas colombianas la cifra puede ser otra, y no
   hay un corpus público para medirlo.
2. **La negación casi no cambia.** El F1 sube 0,006, dentro del ruido. Lo que mejora para
   el auditor es que las sospechas ya no se ven como descartes, y eso no lo mide el F1 de
   `negada`. La meta de recall de negada del epic #71 (0,85) sigue sin alcanzarse: 0,842.
3. **La reservada ya se usó dos veces.** Las cifras siguen siendo de frases que no se
   miraron para ajustar, pero si se ajusta algo más mirando sus errores deja de ser
   reservada.

**DisTEMIST no cambia.** Con la caché, el pipeline final (commit `990ca2b`) da las mismas
métricas que `distemist_test.json`, con los mismos intervalos: F1 estricto 0,683, laxo
0,860 y 2.360 menciones predichas (VP + FP). El contexto no toca los spans.

### Lo que queda

- **Grados de certeza.** "Probable" y "posible" salen igual. NUBes no los distingue y el
  PRD los dejó fuera.
- **Resueltas.** "cede" o "se resolvió" siguen sin estado propio (ni negadas ni posibles),
  como se decidió en el epic #71.
- **"Compatible con" en radiología colombiana.** Se decidió con los datos de NUBes. Si un
  clínico colombiano lo lee como confirmación, se cambia la clave; hace falta su criterio
  o notas colombianas anotadas.
- **"bajo riesgo de" y "riesgo bajo para"** siguen negando: no aparecen en desarrollo y
  no hay datos para pasarlas a `posible` (docstring de `contexto`).
- **Un modelo de negación e incertidumbre** frente a estas reglas: experimento #77, de
  prioridad baja.

### Ajustes posteriores: "no claro" y resultados en la lista (#136, #137)

- **"No claro de X"** ("antecedente no claro de EPOC") pasa a ser clave de `posible`: antes
  el "no" negaba. "Poco claro" no entró: dice que la causa es dudosa, no la enfermedad.
- **Un resultado no sigue la lista negada.** En "no déficit motor o sensitivo, reflejos
  osteotendinosos +++/+++" los reflejos salían negados. Un token hecho solo de signos
  (`+++/+++`) no es parte de un elemento, y tras un elemento largo con otro corto unido por
  conjunción ("motor o sensitivo") la coma ya no sigue con otro largo. "VIH+", "L4/5" y las
  listas unidas por comas no cambian.

Regla declarada antes de medir (comentarios en los issues), con NUBes **desarrollo**
(`nubes_136_137_desarrollo.json`, a3fd62e). La reservada no se volvió a medir.

| | `negada` F1 | `posible` F1 | Estado correcto, sintético |
|---|---|---|---|
| Antes (e5f17ce) | 0,892 | 0,860 | 100 % |
| Después | 0,893 | 0,861 | 100 % |

Cambian 3 menciones, todas por "no claro": 2 inciertas pasan de negada a posible y 1
afirmada pasa de falso positivo de `negada` a falso positivo de `posible`. El cambio de la
lista no altera ninguna mención de NUBes desarrollo, y el set sintético no anota `posible`:
#137 lo respaldan los tests con frases sintéticas, no una cifra.

## 7. Robustez: notas mal escritas

Las notas reales se escriben rápido: en minúscula, sin tildes, con errores de digitación.
DisTEMIST y el set sintético están bien escritos, así que las secciones anteriores miden
un caso favorable. El epic #111 (PRD [`docs/prds/robustez.md`](prds/robustez.md)) agregó
una **prueba de estrés** y tres correcciones:

- **Siglas en minúscula (#114).** El modelo distingue mayúsculas en las siglas: "pte con
  EPOC" la detecta y "pte con epoc" no. Antes del modelo, 30 siglas conocidas se pasan a
  mayúscula en una copia que solo ve él; la nota y las posiciones no cambian.
- **Errores de ortografía en Charlson (#115).** El modelo tolera "diabetis" o "eppoc",
  pero las reglas de Charlson son regex. Si ninguna regla aplica, las palabras de 5
  letras o más que están a **una edición** (Damerau-Levenshtein) de una de 64 palabras
  clave se corrigen y se vuelve a intentar. La entidad trae lo que se corrigió y la demo
  lo muestra ("Corregido: diabetis → diabetes", #116). Las palabras reales que
  cambiarían de sentido ("séptica", "isquemia", "señal") se protegen; salen de barrer
  95.278 palabras de diccionarios de español y, aparte, las 18.227 del texto de DisTEMIST
  **entrenamiento**.
- **Fusión con un nombre mal escrito (#113).** "EPOC, epoc y eppoc" se fusionaba como
  "eppoc" y perdía la categoría; ahora se prueba con cada mención.

**Método.** Las variantes del texto conservan el largo carácter por carácter, así que los
spans de DisTEMIST siguen sirviendo (`evaluacion/robustez.py`, #112): `minusculas`,
`minusculas_sin_tildes` y `errores_5` / `errores_10` (en el 5 % o el 10 % de las palabras
de 4 letras o más, una letra cambiada o dos vecinas intercambiadas, nunca la primera).
Los errores son uniformes: **no son la distribución real de los errores de un médico**,
sino una prueba de estrés. Se mide:

- **DisTEMIST prueba**, la cifra con peso: no se usó para ajustar nada de este epic. F1 del
  modelo crudo y del pipeline y, aparte, el **recall en las 90 menciones del gold que son
  siglas**, donde se concentra el efecto de las minúsculas (en el F1 global no se ve: solo
  el 3,5 % de las menciones de DisTEMIST son siglas).
- **Set sintético**: índice de Charlson exacto y categorías de las confirmadas, por nota.
  Los umbrales se ajustaron con la semilla de errores 0; el reporte usa las semillas 1 a 5,
  que no se miraron. El set no es ciego para lo demás (cuarta medición).

Datos en
[`robustez_linea_base.json`](../evaluacion/resultados/robustez_linea_base.json) (antes:
commit `2b8f991`, ya con #113) y
[`robustez_despues.json`](../evaluacion/resultados/robustez_despues.json) (después: commit
`6acd4d6`, árbol limpio). IC del 95 % por bootstrap sobre documentos. En DisTEMIST, las
variantes con errores usan una sola semilla (la 1).

### DisTEMIST prueba: la cifra con peso

F1 del pipeline completo, laxo y estricto (IC 95 % de después; los de antes están en
`robustez_linea_base.json` y son casi iguales, porque las cifras casi no se mueven):

| Variante | F1 laxo | F1 estricto |
|---|---|---|
| Original | 0,860 → 0,860 (0,846–0,872) | 0,683 → 0,683 (0,658–0,706) |
| Minúsculas | 0,862 → 0,863 (0,851–0,875) | 0,695 → 0,695 (0,671–0,719) |
| Minúsculas sin tildes | 0,842 → 0,843 (0,828–0,857) | 0,6671 → 0,6665 (0,642–0,691) |
| Errores al 5 % | 0,857 → 0,857 (0,844–0,870) | 0,680 → 0,680 (0,654–0,703) |
| Errores al 10 % | 0,848 → 0,848 (0,835–0,861) | 0,658 → 0,658 (0,633–0,683) |

Siglas del gold encontradas (emparejamiento laxo), de 90. El IC es del recall del
pipeline:

| Variante | Modelo crudo | Pipeline | IC del recall antes → después |
|---|---|---|---|
| Original | 85 → 85 | 46 → 46 | 0,395–0,625 → 0,395–0,625 |
| Minúsculas | 51 → **56** | 25 → **28** | 0,181–0,374 → 0,207–0,422 |
| Minúsculas sin tildes | 47 → 50 | 25 → 29 | 0,194–0,375 → 0,221–0,431 |
| Errores al 5 % | 82 → 82 | 46 → 46 | |
| Errores al 10 % | 82 → 82 | 44 → 44 | |

- **Con el texto original no cambia nada**: mismas métricas, estrictas y laxas, que
  `distemist_test.json`.
- **En minúscula se recupera poco.** De las 21 siglas que el pipeline pierde, recupera 3,
  y los IC de antes y después se solapan. La razón principal: solo 13 de las 90 siglas
  de DisTEMIST están en la lista de 30 (HTA y EPOC, 4 cada una; TVP, DM, IAM, DMNID y
  FA, 1). El resto son siglas que una lista hecha a mano no prevé (LNH, SIADH, UDVP…). La lista no se amplió mirando estos errores, porque es la
  partición de prueba.
- **El filtro de longitud pesa más que las minúsculas.** Con el texto original, el modelo
  encuentra 85 siglas y el pipeline se queda con 46: el filtro de 4 caracteres o menos
  descartaba las que no están en su lista blanca. Desde #56 el filtro es mínimo (ver
  [`normalizacion.md`](normalizacion.md)); el antes y el después en prueba se mide en #134.
- **Los errores de digitación casi no afectan la detección**: el F1 laxo del pipeline
  pasa de 0,860 a 0,848 con el 10 % de las palabras mal escritas (una sola semilla). Es
  coherente con que el modelo tokenice por subpalabras, pero eso no se midió aparte.
- **Falsos positivos de la corrección.** En DisTEMIST prueba con el texto original, solo
  2 entidades reciben categoría por corrección. Las dos son correctas: "linfomas" →
  "linfoma", un plural que la regla no aceptaba. Revisadas una por una
  (`--correcciones`); el listado tiene texto del corpus y queda fuera del repo.

### Set sintético: índice de Charlson exacto

| Variante | Antes | Después | A 1 punto o menos (después) |
|---|---|---|---|
| Original | 92 | 92 | 96 |
| Minúsculas | 80 | **89** | 94 |
| Minúsculas sin tildes | 80 | **90** | 95 |
| Errores al 5 %, semilla 0 (ajuste) | 86 | 91 | 95 |
| Errores al 5 %, semillas 1–5 (reporte) | 86,2 (85–89) | **88,4** (87–89) | |
| Errores al 10 %, semilla 0 (ajuste) | 80 | 83 | 92 |
| Errores al 10 %, semillas 1–5 (reporte) | 79,8 (77–83) | **85,6** (82–88) | |

En las semillas de reporte, el índice exacto sube en las cinco con errores al 10 % (de 4 a
9 notas cada una) y las categorías de sobra no cambian, nota por nota. En la semilla 0
sobra una más (de 9 a 10): `sin038`, una hemiplejia que ya sobraba con el texto original
(un error de mapeo de las reglas); el error inyectado la ocultaba y la corrección la
devuelve. De las que sobran con errores al 10 % en la semilla 0, la mayoría vienen de
errores en la palabra que niega o que marca un familiar ("niegh HTA, DM, EPOC", "majre
con DM2"): la enfermedad sale confirmada y suma (#126). Las de `sin038` y `sin098`
(hemiplejia) son errores de mapeo, no de negación.

### ¿Se cumplieron las metas del PRD?

| Meta | Resultado | ¿Cumple? |
|---|---|---|
| DisTEMIST en minúscula: recuperar al menos la mitad de las siglas perdidas | 3 de 21 (25 → 28 de 90) | **No.** La lista de siglas cubre 13 de las 90 de DisTEMIST |
| DisTEMIST con el texto original: F1 sin bajar más de 0,005 | Idéntico | Sí |
| Set sintético en minúscula: índice exacto ≥ 88 | 89 | Sí, con la salvedad de que no es ciego |
| Set sintético con errores al 10 %, semillas 1–5: +3 puntos o más | +5,8 (79,8 → 85,6) | Sí |
| Set sintético con el texto original sin regresiones | 92 → 92 | Sí |
| Falsos positivos de la corrección < 10 % | 0 de 2 en DisTEMIST | Sí, con muy pocos casos |
| US3: F1 laxo y estricto de DisTEMIST con IC, por variante | Tablas de arriba | Sí |
| El tiempo por nota no sube más de un 5 % | Con caché, el texto original pasa de 0,79 a 1,23 s en 250 documentos: unos 2 ms por nota, frente a unos 850 ms del modelo | Sí |
| Los ejemplos de US1 y US2 son tests | `tests/test_pipeline.py`, `tests/test_charlson_correccion.py`, `tests/test_extraccion.py` | Sí |

**Lectura honesta.** Las correcciones funcionan donde se diseñaron: las siglas de la
lista (HTA, DM, EPOC, IRC…) y los errores de una letra en enfermedades de Charlson. Eso
es lo que mide el set sintético, y ahí la mejora es clara y se mantiene en semillas que
no se miraron. No borran el problema: con errores al 10 % el índice exacto queda en
85,6, lejos de 92. Y DisTEMIST muestra el límite del enfoque:
**una lista hecha a mano solo cubre lo que alguien previó**. Lo mismo pasa con las reglas
de Charlson, que no reconocen "accidente cerebrovascular" escrito completo (#123) ni un
tumor "sin metástasis" (#120). Por eso el siguiente paso es probar una normalización por
similitud semántica: llevar cada mención al código CIE-10 más parecido con embeddings
entrenados para términos médicos, y de ahí a Charlson con la tabla publicada de Quan et
al. (2005). En la tesis ya se probaron embeddings para el mapeo CIE-10 y no dieron
buenos resultados, así que el experimento tiene que mostrar con datos qué cambia: un
modelo entrenado para enlazar sinónimos médicos y un corpus con CIE-10 para medirlo. El
filtro de longitud (#56) entra en ese experimento, porque cambia qué llega
a la normalización.

### Lo que queda

- **Negación con la clave mal escrita** (#126): "niegh", "se descarra", "majre". Desde
  #115 esas enfermedades sí suman al índice.
- **Siglas fuera de la lista** y **filtro de longitud** (#56): pasan al experimento de
  normalización.
- **Un error en una palabra de una exclusión** ("diabetes gestacinal") no se corrige: la
  inclusión aplica sin corregir y la exclusión no se ve. Pasaba igual antes.
- **Errores que no son de una letra**: palabras pegadas ("insuficienciacardiaca"),
  abreviaturas propias de cada médico. Fuera de alcance.


### Claves de negación, sospecha y familiar mal escritas (#126)

Las enfermedades mal escritas ya se corregían (#115), pero la palabra que las niega no: con
"niegh HTA, DM, EPOC" las tres salían confirmadas y sumaban al índice. Ahora `contexto`
corrige a distancia 1 de Damerau-Levenshtein unas pocas claves largas (niega, descarta,
sospecha, probable, posible, madre, padre, hermano/a, abuelo/a), con la primera letra bien,
sin ambigüedad y con una lista de palabras reales que no se tocan ("nieta", "parte",
"podre"). Las posiciones de las menciones siguen siendo las del texto original.

Regla declarada antes de medir (#126), sobre 6d59ff8 (la primera versión, a89512d, daba las mismas cifras, pero corregía "descarga" a "descarta" y la negaba; se corrigió y se repitió la medición):

| | Antes | Después |
|---|---|---|
| NUBes desarrollo, `negada` F1 (`nubes_126_desarrollo.json`) | 0,893 | 0,893 |
| NUBes desarrollo, `posible` F1 | 0,861 | 0,860 |
| Set sintético, texto original (`sintetico_v14.json`) | índice 98 %, estado 100 % | igual |
| Errores al 10 %, categorías sobrantes, semilla 0 (`robustez_claves.json`) | 11 | **4** |
| Errores al 10 %, índice exacto, semillas 1–5 | 90,6 | **93,0** |
| Errores al 5 %, índice exacto, semillas 1–5 | 94,4 | **95,4** |

El set sintético ya no es ciego y sus errores son simulados. Claves cortas como "tío" o
"hijo" no se corrigen: a distancia 1 hay demasiadas palabras reales.

**Sobre las particiones reservadas.** Durante la revisión, un script exploratorio contó las
palabras parecidas a estas claves en todos los textos, incluidas la partición reservada de
NUBes y CodiEsp prueba, sin que se pidiera. Solo produjo conteos de esas palabras, sin
frases ni métricas. La lista de palabras protegidas se rehízo con particiones permitidas
(NUBes desarrollo, entrenamiento y desarrollo de CodiEsp, entrenamiento de DisTEMIST y
CANTEMIST) y un diccionario, que dieron los mismos casos. No se midió nada en esas
particiones, pero conviene saberlo si se vuelven a usar.

## 8. Normalización a CIE-10 (CodiEsp)

Hasta aquí, la categoría de Charlson salía de 17 reglas regex y el código CIE-10 sugerido
no se había medido contra un gold. El epic #128 (PRD
[`docs/prds/normalizacion.md`](prds/normalizacion.md)) probó otra forma: convertir la
mención en un vector y buscar el código CIE-10 más parecido en el catálogo. La bitácora
completa, con todas las cifras de desarrollo, está en [`normalizacion.md`](normalizacion.md).

**Datos.** CodiEsp (CC BY 4.0) son 1.000 casos clínicos de España codificados a mano con
CIE-10-ES, con el fragmento de texto de cada código. La categoría de Charlson del gold sale
del código con la tabla de Quan. Se mide **por mención**: el texto de cada mención de
diagnóstico del gold pasa por cada método.
- Entrenamiento (500) dio los sinónimos y el umbral.
- Desarrollo (250) sirvió para comparar métodos.
- Prueba (250) se usó **una sola vez**, al final, desde un árbol limpio y con todo fijado
  antes.

CodiEsp anota también los diagnósticos negados y los de familiares, así que no se compara
el índice de Charlson por documento: esta sección mide solo el paso de mención a categoría.

**Métodos.**
- **Reglas + corrector**: el método actual (#115).
- **SapBERT con sinónimos**: SapBERT multilingüe busca en el catálogo CIE-10-ES (98.288
  códigos) más 3.338 menciones de CodiEsp entrenamiento, cada una con su código. El
  umbral, 0,69, se eligió por validación cruzada en entrenamiento (#131, #132).
- **Híbrido con exclusiones**: SapBERT y, después, las exclusiones de las reglas
  ("mioma", "adenoma"), con el arreglo de #144 (#133).
- **Híbrido con respaldo**: lo anterior y, cuando no queda categoría, las reglas. Se midió
  como alternativa que favorece el recall; no se adoptó en desarrollo.

### CodiEsp prueba: la cifra con peso

Son 250 documentos y 353 menciones gold con categoría. IC del 95 % por bootstrap sobre
documentos. Datos en
[`codiesp_prueba.json`](../evaluacion/resultados/codiesp_prueba.json), commit `d707867`,
árbol limpio.

| Método | P | R | F1 [IC 95 %] | F1 errores 10 % | F1 no vistas [IC 95 %] | Código CIE-10 a 3 caracteres |
|---|---|---|---|---|---|---|
| Reglas + corrector | 0,893 | 0,711 | 0,792 [0,739; 0,835] | 0,785 | 0,755 [0,687; 0,814] | 0,501 |
| SapBERT con sinónimos | 0,912 | 0,827 | 0,868 [0,830; 0,899] | 0,847 | 0,746 [0,684; 0,800] | 0,782 |
| **Híbrido con exclusiones** | **0,927** | 0,827 | **0,874** [0,837; 0,904] | 0,853 | 0,757 [0,695; 0,810] | 0,782 |
| Híbrido con respaldo | 0,859 | **0,878** | 0,868 [0,833; 0,899] | **0,866** | **0,793** [0,729; 0,845] | **0,802** |

"Código CIE-10 a 3 caracteres" es la fracción de las 353 menciones con categoría gold
cuyo código sugerido, con el umbral aplicado, coincide con el gold en sus 3 primeros
caracteres.

- **El híbrido supera a las reglas en 0,08 de F1.** Los intervalos apenas no se solapan
  (0,837 contra 0,835). Pierde 61 menciones con categoría, contra 102 de las reglas, y
  tiene menos falsos positivos (23 contra 30). En minúscula y sin tildes da casi lo mismo.
- **La ganancia es memoria.** "F1 no vistas" mide solo las 1.385 menciones cuyo texto no
  estaba entre los sinónimos (192 con categoría). Ahí el híbrido y las reglas empatan:
  0,757 contra 0,755. Lo que gana sale de las menciones que ya estaban, tal cual, en
  CodiEsp entrenamiento. El mismo corpus y los mismos anotadores hacen eso fácil; en notas
  colombianas, el vocabulario es otro.
- **El respaldo de reglas** es el mejor en menciones no vistas (0,793) y el más robusto a
  errores ortográficos (pierde 0,003). Su recall es el más alto (0,878), a cambio de
  precisión (0,859).
- **Por categoría**, el modelo gana donde las reglas tienen huecos:
  - cerebrovascular: de 0,27 a 0,84 (9 menciones gold);
  - hepática leve: de 0,44 a 0,79 (25);
  - renal crónica: de 0,72 a 0,97 (31);
  - reumatológica: de 0,56 a 0,87 (12).

  Empata o pierde en tumor maligno (0,80 contra 0,80, 114 menciones), tumor metastásico
  (0,87 contra 0,91, 55) e insuficiencia cardiaca (0,44 contra 0,56, 6).
- **Los casos de los issues** ("accidente cerebrovascular", "ictus" y "linfoma de Hodgkin
  sin metástasis", #120 y #123): el modelo resuelve los 3 sin una regla escrita para
  ellos; las reglas, ninguno.
- **El código CIE-10.** En las menciones con categoría, el código sugerido por las reglas
  acierta los 3 primeros caracteres en el 50,1 % de los casos (#130 ya lo había medido en
  desarrollo) y el de SapBERT en el 78,2 %. Sin umbral, el primer candidato de SapBERT
  acierta en el 80,7 % y alguno de los cinco primeros en el 90,7 %. Sobre todas las
  menciones de diagnóstico, también las que no son de Charlson, da 86,8 % y 92,3 %.

### Costo de despliegue

La imagen candidata es la de la API más SapBERT, su tokenizador y el índice con sinónimos
(101.626 filas, 314 MB). Las dos se midieron con 2 vCPU y 4 GiB, como Cloud Run. Las notas
son las de prueba de DisTEMIST de más de 2.000 caracteres, recortadas a 2.000; en la
candidata, cada nota pasa por el pipeline y luego todas sus entidades se normalizan en un
lote. Datos en [`costo_normalizacion.json`](../evaluacion/resultados/costo_normalizacion.json)
(`python -m evaluacion.costo_normalizacion`).

| Imagen | Tamaño | Carga | Mediana por nota | p95 por nota | Memoria pico |
|---|---|---|---|---|---|
| API actual | 856 MB | 2,5 s | 0,37 s | 0,66 s | 369 MiB |
| API + SapBERT | 2.020 MB | 5,5 s | 0,53 s | 1,05 s | 1.026 MiB |

La normalización suma unos 0,16 s por nota en la mediana y 0,39 s en el p95. La memoria
cabe con holgura en 4 GiB. La imagen crece a unos 2 GB sin comprimir.

La latencia de la base varía entre corridas. Con la misma imagen, las mismas 138 notas y
los mismos 2 vCPU, `latencia_docker.json` midió un p95 de 0,48 s (por HTTP), y esta
corrida 0,66 s. Sumando los 0,39 s a esa base saldrían unos 0,87 s. El p95 de 1,05 s está
en el límite del NFR, no claramente por encima.

### Licencias

| Recurso | Licencia |
|---|---|
| CodiEsp y su catálogo CIE-10-ES 2018 (Zenodo 3837305 y 3706838) | CC BY 4.0 |
| Tabla de Quan (Quan et al., *Med Care* 2005) | Algoritmo publicado; aquí está reescrito como código |
| Modelo NER `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` | CC BY 4.0 |
| SapBERT `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | **No declarada** en el modelo; el código del repositorio es MIT. Se entrenó con UMLS 2020AB, que tiene licencia propia de la NLM |
| Tokenizador `FacebookAI/xlm-roberta-base` | MIT |

### ¿Se cumplieron las metas del PRD?

| Meta | Resultado |
|---|---|
| El mejor método supera a las reglas en F1 de categoría, con IC que no se solapan | **Sí, por poco**: 0,874 [0,837; 0,904] contra 0,792 [0,739; 0,835] |
| Con errores al 10 %, cae menos que las reglas con el corrector | **No**: el híbrido cae 0,021 y las reglas 0,007. En valor absoluto, con errores sigue por encima (0,853 contra 0,785). El respaldo de reglas sí cumple (cae 0,003) |
| "accidente cerebrovascular", "ictus" y "linfoma de Hodgkin sin metástasis" sin una regla escrita | **Sí**: 3 de 3; las reglas, 0 de 3 |
| Costo: imagen de unos 2 GB comprimida, memoria dentro de 4 GiB y p95 ≤ 1 s en una nota de 2.000 caracteres | **Casi**: imagen ✓ (2,0 GB sin comprimir), memoria ✓ (1,0 GiB), p95 en el límite (1,05 s en esta corrida; unos 0,87 s con la base de `latencia_docker`) |
| El NER y la prueba de estrés no empeoran con la decisión de #56 | **Sí, mejoran**: DisTEMIST prueba 0,683 → 0,691 estricto y 0,860 → 0,868 laxo; siglas 46 → 84 de 90 (39 de 90 en minúscula). El set sintético da lo mismo (índice exacto 92) |

### Decisión

**No se integra todavía; las reglas se quedan en el pipeline mientras tanto** (Diego,
2026-11-06). El normalizador se sigue mejorando en un epic nuevo, que busca subir el F1 de
la categoría: cada comorbilidad que se muestre tiene que serlo, y tienen que salir casi
todas. La licencia de SapBERT se resuelve aparte, antes de integrarlo. Por qué no se
integra hoy:
- Los pesos no declaran licencia y vienen de UMLS: no se pueden redistribuir en la imagen
  pública sin resolverlo.
- La ganancia en CodiEsp viene de los sinónimos, no del modelo: en las menciones no vistas
  empata con las reglas.
- No cumple la meta de robustez del PRD: con errores ortográficos cae más que las reglas,
  aunque en valor absoluto sigue por encima.
- La latencia p95 queda en el límite de 1 s.

Los candidatos con licencia clara se midieron en entrenamiento + desarrollo (#131) y
quedaron muy por debajo: multilingual-e5-base (MIT), F1 0,48; bsc-bio-ehr-es (Apache-2.0), 0,35. Ver
[`normalizacion.md`](normalizacion.md#embeddings-sin-sinónimos-131).

Lo que sí deja el epic:
- **El filtro de longitud mínimo** (#56), ya integrado. Es la mejora más grande para el
  usuario: las siglas encontradas en DisTEMIST pasan de 46 a 84 de 90.
- **El código CIE-10 medido** contra un gold en prueba: el de las reglas acierta los 3
  primeros caracteres en el 50 % de las menciones con categoría y el de SapBERT en el
  78 %. Y la evidencia de dónde fallan las reglas: cerebrovascular, hepática leve,
  renal y reumatológica.
- **Por dónde mejorar**, que es el epic nuevo. En desarrollo, la mitad de las menciones
  perdidas tienen la categoría correcta entre los cinco primeros candidatos, y otra
  cuarta parte la tienen de primera, pero bajo el umbral. Se explora una segunda etapa
  que decida entre los candidatos, el umbral, el respaldo de reglas y más ejemplos
  públicos (CANTEMIST). Ya no hay una partición de prueba de CodiEsp limpia: esas cifras
  serán de desarrollo y se dirá así.

### Segunda etapa: el clasificador sobre el top-k (epic #148)

El epic #148 dejó de decidir con el primer candidato solo. El índice propone 20 candidatos
y un clasificador pequeño, una regresión logística con 39 atributos (similitudes, votos,
jerarquía CIE-10, reglas, la categoría), elige la categoría o ninguna. Se eligió todo con validación
cruzada anidada en CodiEsp entrenamiento. Se sumaron sinónimos de DisTEMIST y de
CANTEMIST. Se midió **una sola vez en desarrollo**, que ya se había mirado, así que la
cifra es optimista. Prueba no se volvió a usar. El detalle está en
[`normalizacion.md`](normalizacion.md#medición-final-en-desarrollo-154).

| Desarrollo (333 menciones con categoría) | P | R | F1 [IC 95 %] | Δ F1 pareado [IC] | F1 errores 10 % | F1 no vistas |
|---|---|---|---|---|---|---|
| Híbrido con exclusiones | 0,930 | 0,844 | 0,885 [0,855; 0,913] | — | 0,856 | 0,809 |
| **Clasificador + DisTEMIST y CANTEMIST** | **0,937** | **0,892** | **0,914** [0,882; 0,940] | **+0,029** [+0,002; +0,053] | **0,889** | **0,863** |

- Recupera 20 de las 52 menciones que perdía el híbrido y pierde 4 que acertaba. La
  precisión no baja.
- En validación cruzada, la ganancia es de +0,052 [+0,034; +0,071].
- En el punto, cae menos con errores ortográficos (0,025 contra 0,029) y mejora en las
  menciones no vistas (0,863 contra 0,809).
- La decisión del clasificador suma 2 ms al p95 por nota en Docker con 2 vCPU. El p95
  de la imagen completa varía unos 0,07 s entre contenedores en uno y otro sentido.
- Los sinónimos de DisTEMIST y CANTEMIST no muestran efecto claro: en validación cruzada
  suman +0,009 y en desarrollo restan 0,007 en el punto, sin IC que excluya el cero.
- **Decisión (Diego, 2026-11-08): clasificador sin las fuentes externas**, por la falta
  de efecto en validación cruzada y el costo de licencias; sin fuentes dio 0,921 en la
  misma corrida, visto después y citado solo como contexto. La integración al pipeline sigue esperando la
  licencia de SapBERT y su propio epic.

### Mapeo híbrido: reglas primero, normalizador de respaldo (epic #179)

Hasta aquí la categoría salía de las reglas **o** del normalizador. El epic #179 los
combina (#180): primero las reglas y, para lo que no mapean, el normalizador. Se midió
(#181) con las reglas ya corregidas (#123, #120, #63), desde un árbol limpio: CodiEsp y
el normalizador y el híbrido del set sintético sobre 8dc916a; las reglas del set sintético
son `sintetico_v9.json` (2472b36, antes de #180, que no cambia el camino de las reglas:
sus errores de categoría son los mismos que los del híbrido).

**Regla de decisión, declarada antes de medir** (`docs/prds/mapeo-hibrido.md`): el
híbrido queda por defecto en la API si (1) en CodiEsp desarrollo de punta a punta su F1
supera al de las reglas y (2) en el set sintético su índice exacto no cae más de 1 punto
frente a las reglas.

CodiEsp desarrollo, de punta a punta (`extremo_codiesp_desarrollo_v3.json`, 250 casos):

| Método | P | R | F1 [IC 95 %] | Índice exacto | Índice ±1 | Error medio del índice |
|---|---|---|---|---|---|---|
| Reglas | **0,837** | 0,650 | 0,731 [0,680; 0,782] | 0,724 | **0,856** | **0,68** |
| Normalizador | 0,789 | 0,741 | **0,764** [0,710; 0,813] | **0,732** | 0,816 | 0,84 |
| Híbrido | 0,743 | **0,777** | 0,759 [0,710; 0,805] | 0,712 | 0,840 | 0,77 |

Δ F1 pareado contra las reglas: normalizador +0,033 [−0,010; +0,078]; híbrido +0,028
[−0,010; +0,065].

Set sintético (las mismas 100 notas, no ciegas):

| Método | F1 de categoría | Índice exacto | Índice ±1 | `registrado` |
|---|---|---|---|---|
| Reglas (`sintetico_v9.json`) | **0,926** | **98 %** | 98 % | 66/68 |
| Normalizador (`sintetico_v10_normalizador.json`) | 0,880 | 83 % | 97 % | 65/68 |
| Híbrido (`sintetico_v10_hibrido.json`) | **0,926** | 97 % | 98 % | 66/68 |

**Decisión: se cumplen las dos condiciones** (F1 0,759 > 0,731; índice exacto 97 % contra
98 %, justo en el límite de 1 punto). La API usa el híbrido por defecto desde #182.

Lo que hay que leer junto a la decisión:

- **Ninguna diferencia de F1 en CodiEsp tiene un IC que excluya el cero.** La regla pedía
  superar a las reglas en el punto, no con significancia. Las de recall y precisión del
  híbrido sí lo excluyen: recall +0,127 [+0,085; +0,173], precisión −0,094 [−0,144;
  −0,051].
- **El híbrido gana en recall y paga en precisión.** Recupera lo que las reglas no conocen
  (renal crónica 9 → 13 aciertos, tumor maligno 39 → 49, cerebrovascular 6 → 9, úlcera
  péptica 2 → 4) y suma los falsos positivos del normalizador en lo que las reglas no
  mapean: de 25 a 53, sobre todo de tumor (20 de los 28 nuevos: metastásico 2 → 12,
  maligno 3 → 13; el resto, cerebrovascular, hepática y otras). **En CodiEsp el índice del
  híbrido no mejora al de las reglas**: exacto 0,712 contra 0,724, ±1 0,840 contra 0,856,
  error medio 0,77 contra 0,68; probablemente por esos falsos positivos de peso alto. Solo
  mejora el sesgo medio (+0,004, contra −0,488 de las reglas, que subestiman). La regla de
  decisión no lo pedía, pero es lo que hay que mirar en #185.
- **En el set sintético el híbrido conserva lo de las reglas** (siglas como "ICC", tramos
  largos), que es donde el normalizador solo perdía 15 puntos de índice exacto. El único
  error nuevo (sin097) es un falso positivo del normalizador: "lesiones hepáticas" sale
  como hepatopatía leve. El F1 de categoría idéntico (0,926) no quiere decir que no haya
  mapeos nuevos: los hay en entidades sin categoría anotada, como esa y una oclusión de la
  arteria cerebral media (sin098) que sale cerebrovascular sin cambiar el índice.
- Los falsos positivos de tumor del normalizador quedan en #185; no se
  ajustan aquí, porque desarrollo es con lo que se decidió.

### Filtros del respaldo del híbrido (#185)

La medición anterior mostró el costo del híbrido: los falsos positivos del normalizador en
lo que las reglas no mapean. Los filtros se diseñaron con CodiEsp entrenamiento, aunque la
idea de buscarlos nació de los conteos de falsos positivos en desarrollo (#181). Solo
aplican al respaldo (`_respaldo_valido`); las reglas y el modo `normalizador` no cambian:

1. **Siglas:** una sigla solo vale si el candidato es la misma sigla. SapBERT acerca siglas
   distintas por su forma ("SB" → "LB", linfoma; "NM" → "DM"). Cuesta recall: una sigla
   correcta cuyo candidato es el nombre completo también se pierde. Una palabra en
   mayúsculas ("GLIOMA") no cuenta como sigla, ni una estadificación TNM.
2. **Tumor metastásico:** solo con una señal de metástasis en la mención (metástasis, mts,
   secundario, diseminado, infiltración, carcinomatosis, siembra, implantes, M1, estadio
   IV). Sin ella, "proliferación neoplásica" o "lesión tumoral" sumaban 6 puntos. La
   señal se busca en la mención y no en la nota; la decisión se descarta, no baja a tumor
   maligno. Es un filtro simple: "infección secundaria" cuenta como señal, pero solo
   importa si el normalizador ya dijo metastásico.
3. **"Lesión hepática" y "LOE"** no reciben hepatopatía leve del respaldo: una lesión focal
   no es una enfermedad hepática crónica. "Cirrosis con LOE" la siguen mapeando las reglas.

En entrenamiento (optimista para el normalizador: sus menciones están en el índice), F1
0,757 → 0,777, falsos positivos 102 → 77. Son cifras de una corrida local, sin JSON.

**Regla declarada antes de medir en desarrollo** (comentario en #185): F1 ≥ 0,754, precisión
> 0,743 e índice exacto sintético ≥ 97 %. Se cumplen las tres. La primera versión (v4, v11)
también las cumplía, pero tenía un error: su exclusión estaba en la regla y le quitaba la
hepatopatía a "cirrosis con LOE". Se corrigió y se repitió la medición con los mismos
umbrales; el híbrido dio las mismas cifras.

| CodiEsp desarrollo (`extremo_codiesp_desarrollo_v5.json`, b242e40) | P | R | F1 [IC 95 %] | Índice exacto | Índice ±1 | Error medio |
|---|---|---|---|---|---|---|
| Reglas | 0,837 | 0,650 | 0,731 [0,680; 0,782] | 0,724 | 0,856 | 0,68 |
| Normalizador | 0,789 | 0,741 | 0,764 [0,710; 0,813] | 0,732 | 0,816 | 0,84 |
| Híbrido sin filtros (#181) | 0,743 | 0,777 | 0,759 [0,710; 0,805] | 0,712 | 0,840 | 0,77 |
| **Híbrido con filtros** | **0,777** | 0,761 | **0,769** [0,719; 0,812] | 0,728 | **0,864** | **0,64** |

- Δ F1 híbrido − reglas: +0,038 [+0,004; +0,070]. El límite inferior está pegado al cero y
  desarrollo ya se había mirado. Normalizador − reglas sigue con un IC que incluye el cero
  (+0,033 [−0,010; +0,078]), y la diferencia entre híbrido y normalizador no se calculó.
- Tumor metastásico: falsos positivos 12 → 3, a cambio de 2 aciertos (17 → 15).
- En CodiEsp, el índice exacto del híbrido es el de las reglas más un documento (0,728
  contra 0,724 en 250, sin IC); mejoran el ±1 y el error medio.
- Set sintético (`sintetico_v12_hibrido.json`): índice exacto 97 → **98 %**, igual que las
  reglas; sin097 ya no falla. Normalizador solo (`sintetico_v12_normalizador.json`): 83 %.

## Límites

- **DisTEMIST no es una nota de IPS.** Son casos publicados en revistas: texto limpio, pocas
  siglas, sin folios ni errores de digitación. Las reglas de limpieza y la fusión,
  pensadas para notas reales, casi no se ejercitan o se ejercitan en el texto equivocado.
- **NUBes es de hospitales de España**, no de Colombia, y no anota familiares. Su
  criterio de incertidumbre es el de sus anotadores ("compatible con" es incertidumbre),
  que puede no coincidir con el de una nota colombiana.
- **El set sintético lo escribimos nosotros**, que conocemos las reglas. Sus cifras son una
  cota optimista. La primera medición de negación se parecía a la de NUBes; la segunda y
  la tercera ya no son ciegas, porque las reglas se corrigieron mirando sus errores. El
  set no anota sospechas, así que no mide el estado `posible`.
- **Nadie con formación clínica validó las anotaciones del proyecto.** La doble anotación
  ciega mide consistencia, no criterio clínico.
- **CodiEsp no es una nota de IPS.** El código CIE-10 se midió contra CodiEsp (casos
  clínicos de España, con CIE-10-ES). Colombia usa la CIE-10 de la OMS y escribe distinto.
  La partición de prueba de CodiEsp ya se usó (#134).

## Mejoras abiertas

Todas tienen evidencia numérica de esta evaluación. Las cuatro de negación (#61, #62, #65
y #66) ya se corrigieron en el epic #71 (ver la [sección 5](#5-negación-antes-y-después)),
y las sospechas (#83) y la lista que negaba de más (#84), en el epic #86 (ver la
[sección 6](#6-incertidumbre-estado-posible)). Estas siguen abiertas:

| Issue | Tipo | Qué |
|---|---|---|
| #56 | mejora | El filtro de longitud descarta siglas y enfermedades cortas (resuelto: filtro mínimo) |
| #63 | mejora | Huecos en las reglas de Charlson (resuelto en el epic #179) |
| #53 | bug | Comillas en el span |
| #57 | mejora | Decidir los pasos sin efecto en DisTEMIST |
| #77 | experimento | Un modelo de negación e incertidumbre frente a las reglas |
| #120 | bug | Un tumor "sin metástasis" pierde la categoría de Charlson (resuelto en el epic #179) |
| #123 | bug | La regla cerebrovascular no reconoce "accidente cerebrovascular" ni "ictus" (resuelto en el epic #179) |
| #126 | mejora | Negación, sospecha y familiar con la clave mal escrita (resuelto: ver sección 7) |
| #185 | mejora | Falsos positivos del normalizador en el respaldo del híbrido (resuelto: ver "Filtros del respaldo") |

En la tercera medición del set sintético quedaban 8 errores del índice: los 4 de la fusión
(#55, resuelta: sin fusión, el índice exacto sube a 96 %), los 2 de huecos en las reglas
(#63) y los 2 de otras causas, que ya estaban en la primera. El de la negación (sin086) se corrigió en #84.

## Cómo reproducirlo

```bash
uv sync
uv run python -m evaluacion descargar                         # DisTEMIST (MD5 verificado)
uv run python -m evaluacion distemist --particion test        # NER + ablación + tiempos
uv run python -m evaluacion distemist --particion train       # chequeo de contaminación
uv run python -m evaluacion distemist --particion train --variantes pipeline fusion_barreras sin_fusion --sin-tiempos   # elección de #55 (antes del cambio)
uv run python -m evaluacion sintetico --variante sin_fusion --salida evaluacion/resultados/sintetico_v6_sin_fusion.json   # (#55)
uv run python -m evaluacion nubes                             # negación (descarga NUBes)
uv run python -m evaluacion nubes --particion desarrollo --salida evaluacion/resultados/otra.json  # partición de ajuste
uv run python -m evaluacion nubes --particion reservada --salida evaluacion/resultados/otra.json   # medición final (#74, #89)
uv run python -m evaluacion sintetico --salida otra_medicion.json   # set reservado
uv run python -m evaluacion.errores --help                    # muestra de errores (#46)
uv run python -m evaluacion robustez --salida otra.json       # prueba de estrés (#112, #117)
uv run python -m evaluacion robustez --correcciones           # entidades con categoría por corrección (texto en data/)
uv run python -m evaluacion descargar --corpus codiesp      # CodiEsp y su catálogo CIE-10 (#129)
uv run python -m evaluacion codiesp                           # línea base de las reglas en CodiEsp desarrollo (#130)
uv run python -m evaluacion normalizacion --modelo sapbert   # embeddings en CodiEsp desarrollo (#131); --tabla compara
uv run python -m evaluacion sinonimos                       # sinónimos de CodiEsp entrenamiento, medidos en desarrollo (#132)
uv run python -m evaluacion prueba                          # CodiEsp prueba, una sola vez y con el árbol limpio (#134)
uv run python -m evaluacion.costo_normalizacion             # costo en Docker, con la imagen ya construida (#134)
uv run python -m evaluacion final                           # segunda etapa en CodiEsp desarrollo, una sola vez (#154)
uv run python -m evaluacion.costo_normalizacion --final     # costo con la segunda etapa (#154)
uv run python -m evaluacion artefactos                      # índice y clasificador del normalizador para la librería (#164)
uv run python -m evaluacion extremo                         # reglas contra normalizador de punta a punta en CodiEsp desarrollo (#167)
uv run python -m evaluacion sintetico --normalizador --salida evaluacion/resultados/sintetico_v5_normalizador.json   # (#167)
uv run python -m evaluacion.costo_integrado                # costo de la API con el normalizador en Docker (#167)
```

- El modelo corre una sola vez por fragmento, gracias a una caché en `data/cache/`. Con la
  caché llena, todas las corridas toman segundos.
- Volver a correr da las mismas métricas (verificado en #46).
- Cada corrida de NUBes guarda las menciones con su texto en `data/evaluacion/`, fuera del
  repo. Con `--detalle` se elige el archivo, para no pisar el de otra corrida.
- Cada JSON de métricas (DisTEMIST, NUBes y sintético) registra el commit, las versiones (`transformers` 5.21.0, `torch` 2.15.0 CPU,
  Python 3.12.14) y la CPU.

## Atribución

- **CodiEsp:** Miranda-Escalada A, Gonzalez-Agirre A, Armengol-Estapé J, Krallinger M.
  *Overview of automatic clinical coding: annotations, guidelines, and solutions for
  non-English clinical cases at CodiEsp track of CLEF eHealth 2020.* CEUR Workshop
  Proceedings 2696, 2020. CC BY 4.0.
- **SapBERT:** Liu F, Vulić I, Korhonen A, Collier N. *Learning Domain-Specialised
  Representations for Cross-Lingual Biomedical Entity Linking.* ACL-IJCNLP 2021.
- **Tabla de Quan:** Quan H, Sundararajan V, Halfon P, et al. *Coding algorithms for
  defining comorbidities in ICD-9-CM and ICD-10 administrative data.* Med Care
  2005;43(11):1130-9.
- **DisTEMIST:** Miranda-Escalada A, Gascó L, Lima-López S, et al. *Overview of DisTEMIST
  at BioASQ: Automatic detection and normalization of diseases from clinical texts.* CEUR
  Workshop Proceedings 3180, 2022. CC BY 4.0.
- **NUBes:** Lima López S, Pérez N, Cuadros M, Rigau G. *NUBes: A Corpus of Negation and
  Uncertainty in Spanish Clinical Texts.* LREC 2020, pp. 5772–5781. CC BY-SA 3.0 ES.
- **Modelo:** `BSC-NLP4BIA/bsc-bio-ehr-es-distemist`, de BSC-NLP4BIA, CC BY 4.0.
