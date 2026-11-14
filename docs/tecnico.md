# Detalle técnico

Lo que el [README](../README.md) resume: qué trae cada resultado, el normalizador por
embeddings, el cruce con los diagnósticos registrados, cómo correr la API y la tabla
completa de métricas. La evaluación paso a paso está en [`evaluacion.md`](evaluacion.md);
el normalizador, en [`normalizacion.md`](normalizacion.md); el despliegue, en
[`despliegue.md`](despliegue.md); los datos y sus licencias, en [`datos.md`](datos.md).

## Usar la librería

```python
from ner_clinic import analizar

nota = (
    "Paciente femenina de 67 años con antecedente de HTA y diabetes mellitus tipo 2 en "
    "manejo con metformina. Niega EPOC. Madre con cáncer de mama."
)
resultado = analizar(nota, diagnosticos_registrados=["Diabetes mellitus no insulinodependiente"])
resultado.to_dict()  # listo para JSON
```

## Qué trae cada resultado

Cada entidad trae en `to_dict()` su `texto`, `inicio` y `fin` (posiciones en la nota tal
como se envió: `nota[inicio:fin]` es lo que se detectó), todas sus `menciones` si aparece
varias veces, `estado`, `categoria_charlson`, `cie10`, `registrado`, `coincide_con`,
`coincide_por`, `categoria_por_correccion`, `correcciones`, `mapeo` (de dónde salió la categoría:
`"reglas"` o `"normalizador"`) y `candidato` (con el normalizador, el texto del índice
CIE-10 que la explica). El `estado` es uno de cuatro:

| `estado` | Qué significa | Charlson y CIE-10 | Suma a `indice_charlson` | Suma a `indice_charlson_si_se_confirman` |
|---|---|---|---|---|
| `confirmada` | El paciente la tiene | Sí | Sí | Sí |
| `negada` | La nota dice que no la tiene ("niega EPOC") | No | No | No |
| `posible` | Se sospecha, sin afirmarla ni negarla ("sospecha de TEP") | Sí, para mostrarla | No | Sí |
| `familiar` | Es de un familiar ("madre con cáncer de mama") | No | No | No |

Además de `entidades` y `descartes`, el resultado trae dos índices:
`indice_charlson`, con lo confirmado, e `indice_charlson_si_se_confirman`, con lo
confirmado más las sospechas y la misma regla de forma leve y severa. En "Paciente con
DM2. Sospecha de cáncer metastásico." el primero es 1 y el segundo 7: el riesgo actual y
el que habría si se confirma la sospecha, sin mezclarlos. Sin sospechas, los dos son
iguales. Las sospechas se detectan con reglas, como la negación: "sospecha de",
"probable", "compatible con", "no se descarta", "X vs Y" y otras claves medidas una a una
en NUBes (docstring de `contexto`).
El estado se decide por la posición de cada mención, así que **una enfermedad puede salir
dos veces con estados distintos**: en "Niega IAM previo. Ingresa por IAM con elevación del
ST" el IAM sale negado (con la mención de los antecedentes) y confirmado (con la del
ingreso), y solo el confirmado suma al índice.
Las notas se escriben rápido, así que el pipeline aguanta lo más común: las siglas en
minúscula ("pte con epoc") se le pasan al modelo en mayúscula, y una enfermedad mal
escrita que ninguna regla de Charlson reconoce se corrige si está a una letra de una
palabra clave ("diabetis mellitus" da diabetes). Esa entidad sale con
`categoria_por_correccion: true` y `correcciones: [{"escrita": "diabetis", "corregida":
"diabetes"}]`, para que se vea que la categoría no salió de una coincidencia exacta.
`descartes` lista lo que se detectó y se excluyó (basura, duplicado interno o diagnóstico
ya registrado) con el motivo, para que el auditor vea por qué.

## Normalizador por embeddings

En la librería, `analizar()` mapea con reglas regex salvo que se le pase el normalizador;
con él, por defecto, en modo **híbrido**: primero las reglas y, para lo que no mapean,
SapBERT y un clasificador sobre los 20 códigos CIE-10 más parecidos
([`normalizacion.md`](normalizacion.md)). El híbrido cumplió la regla de
decisión del experimento y es el defecto de la API.

| | Reglas | Normalizador | Híbrido |
|---|---|---|---|
| CodiEsp desarrollo, menciones anotadas (F1 de categoría) | 0,759¹ | 0,921² | — |
| CodiEsp desarrollo, de punta a punta con el NER (F1 por documento) | 0,731 | 0,764 | 0,769 |
| Set sintético de estilo colombiano (F1 de categoría)³ | 0,926 | 0,880 | 0,926 |
| Set sintético, índice de Charlson exacto³ | 98 % | 83 % | 98 % |

¹ Con una versión anterior de las reglas. ² Variante sin fuentes externas, la que se
adoptó. Se miró después de medir, así que se cita como contexto: la cifra declarada fue
0,914. Desarrollo ya se había mirado: es optimista. ³ El set se escribió conociendo las
reglas y favorece a las reglas; CodiEsp, de donde sale el índice, favorece al normalizador.
El híbrido incluye los filtros del respaldo. Frente a las reglas, en CodiEsp gana
0,038 de F1 ([+0,004; +0,070], límite pegado al cero y en una partición ya mirada): más
recall (0,650 → 0,761) a cambio de precisión (0,837 → 0,777). El IC del normalizador solo
incluye el cero (+0,033 [−0,010; +0,078]). Detalle
en [`evaluacion.md`](evaluacion.md#mapeo-híbrido-reglas-primero-normalizador-de-respaldo-epic-179).

Las filas de punta a punta son sin la fusión de entidades cercanas. El normalizador
solo gana claramente sobre las menciones ya cortadas; de punta a punta, no hay una mejora
clara, y en notas de estilo colombiano tiene dos debilidades: siglas que CodiEsp no trae
("ICC" queda cerca de "HCC") y menciones largas del NER. El híbrido las evita, porque esas
las mapean las reglas. El normalizador suma 0,27 s al p95 de una nota de 2.000 caracteres
y 0,7 GiB de memoria; el híbrido carga lo mismo y su latencia no se midió aparte. Se
activa así, después de construir
los artefactos con `uv run python -m evaluacion artefactos`:

```python
from ner_clinic import analizar
from ner_clinic.normalizacion import artefactos

normalizador = artefactos.cargar()  # índice CIE-10 + clasificador; carga SapBERT
nota = "Paciente con falla renal crónica y ACV isquémico. Antecedente de ICC."
resultado = analizar(nota, normalizador=normalizador)  # modo_mapeo="hibrido"
```

Por defecto el modo es **híbrido**: primero las reglas y, para lo que no mapean,
el normalizador. En la nota de arriba, "ACV isquémico" e "ICC" salen de las reglas y
"falla renal crónica", que las reglas no conocen, del normalizador. Con
`modo_mapeo="normalizador"` decide solo el normalizador, y ahí "ICC" se pierde.

Cada entidad que mapea el normalizador sale con `mapeo: "normalizador"` y su `candidato`,
p. ej. `{"texto": "insuficiencia renal crónica", "codigo": "N18.9", "origen": "mención de
CodiEsp entrenamiento", "similitud": 0.943}`: el texto es del índice (catálogo CIE-10-ES o
sinónimos de CodiEsp, CC BY 4.0), nunca de la nota. **Licencia de SapBERT:** los autores declaran
el proyecto MIT, pesos incluidos, pero se entrenó con UMLS: sirve para la demo y, para un
cliente comercial, conviene revisarlo (ver
[`datos.md`](datos.md#el-normalizador-integrado-sapbert-y-artefactos-162)).

## Diagnósticos registrados y subregistro

`diagnosticos_registrados` es opcional: la lista de diagnósticos codificados del egreso
(principal y secundarios), como texto. Si se envía, cada comorbilidad **confirmada** sale
con:

- `registrado: true` y `coincide_con`: el diagnóstico registrado con el que coincide.
  `coincide_por` dice cómo: `"texto"` (comparten una palabra) o `"categoria"` (solo en la
  diabetes: "retinopatía diabética" coincide con "Diabetes mellitus … con complicaciones
  oftálmicas"). Por categoría no se distingue la complicación: una nefropatía diabética
  también coincidiría con ese registrado.
- `registrado: false`: está en la nota pero no en lo registrado. Es un **posible
  subregistro** (o una glosa por venir), lo que más le interesa a un auditor.

Si no se envía, o la entidad no es confirmada (una sospecha tampoco es un subregistro),
`registrado` queda vacío. Con
`descartar_registrados=True` las que ya están registradas se descartan en vez de
marcarse, como hacía la tesis.

## Correr la API en local

```bash
uv sync                                                # incluye el extra `api` (FastAPI y uvicorn)
uv run python -m ner_clinic_api                        # http://127.0.0.1:8000, sin access log
```

`NER_NORMALIZADOR` elige el mapeo a Charlson:

- `hibrido` (por defecto): primero las reglas y, para lo que no mapean, el normalizador.
  Si su índice no está (ver abajo), la API arranca igual con las reglas y `/api/salud` lo
  dice (`mapeo: "reglas"`). La ruta del índice es relativa al directorio actual: hay que
  arrancar desde la raíz del repo. Si el índice está, la primera vez se descarga SapBERT
  (cerca de 1,1 GB) y la app usa unos 0,7 GiB más de memoria; si algo del normalizador
  falla con el índice presente, no arranca. La imagen Docker usa este mismo valor y trae
  SapBERT y el índice.
- `sapbert`: solo el normalizador; sin sus artefactos, no arranca.
- `reglas`: solo las reglas.

Con uvicorn directamente: `uv run uvicorn ner_clinic_api.app:app --no-access-log`. Al
arrancar, el modelo se carga y calienta en segundo plano: `/api/salud` dice
`cargando` hasta que está listo (`?esperar=25` espera la carga) y `/api/analizar`
responde 503 mientras tanto. La primera vez el modelo se descarga de Hugging Face.

```bash
curl -s http://127.0.0.1:8000/api/analizar -H 'content-type: application/json' \
  -d '{"texto": "Paciente con HTA y DM2. Niega EPOC.", "diagnosticos_registrados": []}'
```

Responde `to_dict()` más `meta` (`version`, `segundos`, `restantes_hoy`, `mapeo`); el
esquema está en `/api/openapi.json`. Los límites se cambian con variables de entorno:
`MAX_CARACTERES` (5000), `TIMEOUT_S` (30), `TOPE_IP_DIA` (20), `TOPE_GLOBAL_DIA` (200),
`PROXIES_CONFIABLES` (1) y `SAL_IP`. El texto recibido no se guarda ni se registra en logs.

**Por defecto la API usa el mapeo híbrido** (`NER_NORMALIZADOR=hibrido`): reglas y,
para lo que no mapean, el normalizador por embeddings. El normalizador necesita sus
artefactos (`uv run python -m evaluacion artefactos`); `NER_INDICE_NORMALIZADOR` cambia la
ruta del índice. Se construyen una vez, desde la raíz del repo y con CodiEsp descargado
(`uv run python -m evaluacion descargar --corpus codiesp`); el índice pesa 323 MB y su ruta
es relativa a la raíz. La app los carga al arrancar (en Docker, de 2 a unos 9 s más según la corrida). Si falta el
índice, en `hibrido` arranca con las reglas; con `sapbert` no arranca. `GET /api/salud` dice qué
`mapeo` está activo.

La imagen Docker por defecto (etapa `completa`) trae SapBERT y el índice: hay que
construir el índice antes (`uv run python -m evaluacion artefactos`), o `docker build`
falla. La etapa `base` (`docker build --target base`) no los trae y corre con las reglas.
La demo en Cloud Run usa esta imagen desde un registro privado ([`despliegue.md`](despliegue.md)); la imagen no se publica en un registro público, por el
entrenamiento con UMLS de SapBERT
([`datos.md`](datos.md#el-normalizador-integrado-sapbert-y-artefactos-162)).

## Métricas completas

Evaluado con corpus públicos anotados por terceros y con un set sintético reservado.
Las cifras de Charlson de esta tabla son con las reglas; las del normalizador están en
[su sección](#normalizador-por-embeddings).

| Qué | Datos | Resultado |
|---|---|---|
| Encontrar las enfermedades de la nota | [DisTEMIST](https://doi.org/10.5281/zenodo.7614764), 250 casos clínicos | F1 **0,878** (P 0,877 · R 0,878) |
| Saber si están negadas, en historias clínicas reales | [NUBes](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus), 14.817 frases reservadas (no se usaron para ajustar) | F1 **0,870** (P 0,899 · R 0,842); con las reglas de la tesis, 0,802 |
| Saber si son sospechas (estado `posible`) | NUBes, las mismas 14.817 frases reservadas | F1 **0,858** (P 0,919 · R 0,805); antes no había estado para las sospechas |
| Estado, índice de Charlson y subregistro | 100 notas sintéticas reservadas | Medición ciega: estado 94,3 % · índice exacto 86 % · `registrado` 94,1 %. Mediciones posteriores, ya no ciegas: 99,4 % · 91 % · 95,6 % (tras corregir la negación), 99,7 % · 92 % · 95,6 % (tras agregar las sospechas), 100 % · **96 %** · 94,1 % (sin la fusión de entidades) y `registrado` **97,1 %** con el cruce por categoría para la diabetes |
| Notas mal escritas | Las 100 notas sintéticas (medición no ciega) con errores en el 10 % de las palabras de 4 letras o más, o en minúscula | Índice exacto 79,8 → **85,6** con errores (media de 5 semillas); 80 → **89** en minúscula; 92 con el texto bien escrito ([sección 7](evaluacion.md#7-robustez-notas-mal-escritas)) |
| Sugerir el código CIE-10 | [CodiEsp](https://doi.org/10.5281/zenodo.3837305) prueba, casos clínicos de España, medido una sola vez | El sugerido acierta los 3 primeros caracteres en el **78 %** de las menciones con categoría; alguno de los 5 primeros, en el **91 %** (sin el piso de similitud que usa la demo) |
| Velocidad (CPU, sin GPU) | DisTEMIST | 0,85 s por nota (mediana) |

**Cómo leer la primera fila:** cuenta como acierto que el sistema encuentre la enfermedad y
la ubique en la nota, aunque el resaltado tenga una palabra de más o de menos (medida
*laxa*). Es lo que importa para el uso: listar las comorbilidades y llevar al lector al
lugar de la nota donde aparecen. Para compararse con la literatura, la medida *estricta*
(el span exacto, carácter por carácter) da F1 0,734 en el pipeline y 0,728 con el modelo
solo. Antes de quitar la fusión de entidades cercanas, el pipeline daba 0,691.

El punto débil medido era la **negación**: con las reglas de la tesis, en texto real
quedaba sin negar una de cada cuatro enfermedades negadas, por claves que faltaban
("descartó", "tampoco"), alcance mal cortado y enumeraciones como "niega HTA, DM, EPOC",
donde solo se negaba la primera. Se corrigió y se midió en una parte de NUBes que no se
miró al ajustar las reglas: ahora queda sin negar una de cada seis o siete (recall 0,842,
cerca de la meta de 0,85 sin llegar a ella), y la precisión sube de 0,845 a 0,899. Las
sospechas ("sospecha de", "probable", "no se descarta") ya no cuentan como negadas ni
confirmadas: salen `posible`. En esa misma parte de NUBes, de 617 sospechas anotadas, 497
salen posibles; antes, 520 salían confirmadas y sumaban al índice. NUBes es de España: en
notas colombianas, claves como "compatible con" pueden leerse distinto. Las mediciones
del set sintético después de la primera no son ciegas, porque las reglas se corrigieron
mirando sus errores. Lo que falta está en las [limitaciones](../README.md#limitaciones).
