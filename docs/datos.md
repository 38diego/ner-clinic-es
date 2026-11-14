# Datos: fuentes, versiones y licencias

Todos los datos del proyecto son públicos o sintéticos. Los corpus y el catálogo no se
commitean: los comandos de `evaluacion` los bajan a `data/` (en `.gitignore`), con
verificación de MD5 o SHA-256. La tabla de Quan es la excepción: es corta y está escrita
en el código. Ninguno contiene historias clínicas de pacientes de una IPS.

| Datos | Qué es | Versión | Licencia | Se baja con |
|---|---|---|---|---|
| [DisTEMIST](https://doi.org/10.5281/zenodo.7614764) | 1.000 casos clínicos publicados, con las enfermedades anotadas | Zenodo 7614764 | CC BY 4.0 | `python -m evaluacion descargar` |
| [NUBes](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus) | Frases de historias de hospitales de España, anonimizadas, con negación e incertidumbre | commit `91c81c1` | CC BY-SA 3.0 ES | `python -m evaluacion nubes` |
| [CodiEsp](https://doi.org/10.5281/zenodo.3837305) | 1.000 casos clínicos publicados, codificados a mano con CIE-10-ES | Zenodo 3837305, v1.4 | CC BY 4.0 | `python -m evaluacion descargar --corpus codiesp` |
| [Catálogo de CodiEsp](https://doi.org/10.5281/zenodo.3706838) | CIE-10-ES Diagnósticos 2018: 98.288 códigos con su descripción en español | Zenodo 3706838, v1.1 | CC BY 4.0 | `python -m evaluacion descargar --corpus codiesp` |
| [CANTEMIST](https://doi.org/10.5281/zenodo.3978041) | 1.301 casos clínicos de oncología, con las menciones de tumor codificadas en CIE-O-3 | Zenodo 3978041, v1.6 | CC BY 4.0 | `python -m evaluacion descargar --corpus cantemist` |
| Tabla de Quan (2005) | CIE-10 de cada categoría de Charlson | Tabla 1 del artículo | Publicación científica; se cita | Está en el código: `src/ner_clinic/charlson/quan.py` |
| Modelos de embeddings (#131) | SapBERT multilingüe, multilingual-e5 base y small, bsc-bio-ehr-es | Revisión fija en `evaluacion/normalizador.py` | SapBERT: **no declarada**; e5: MIT; bsc-bio-ehr-es: Apache-2.0 | Hugging Face, al correr `python -m evaluacion normalizacion`. Detalle en [`normalizacion.md`](normalizacion.md) |
| Normalizador integrado (epic #162) | SapBERT, el índice final (catálogo + sinónimos de CodiEsp entrenamiento) y el clasificador | SapBERT con la revisión de `ner_clinic/normalizacion/indice.py`; clasificador versionado | Ver [abajo](#el-normalizador-integrado-sapbert-y-artefactos-162): SapBERT **pendiente** | `python -m evaluacion artefactos` |

## CodiEsp

Miranda-Escalada A, Gonzalez-Agirre A, Armengol-Estapé J, Krallinger M. Overview of
automatic clinical coding: annotations, guidelines, and solutions for non-English
clinical cases at CodiEsp track of CLEF eHealth 2020. CEUR Workshop Proceedings, 2020.

| Partición | Casos | Códigos de diagnóstico por documento (D) | Menciones de diagnóstico con posición (X) | De ellas, con categoría de Charlson (Quan) |
|---|---|---|---|---|
| Entrenamiento (`train`) | 500 | 5.639 | 7.207 | 767 |
| Desarrollo (`dev`) | 250 | 2.677 | 3.430 | 333 |
| Prueba (`test`) | 250 | 2.842 | 3.664 | 353 |

- **Uso.** Entrenamiento y desarrollo sirven para ajustar umbrales y sinónimos.
  La partición de prueba queda solo para la medición final (#134).
- **Menciones excluidas.** 8 menciones (4, 3 y 1 por partición, contando diagnósticos y
  procedimientos; 2, 1 y 1 son de diagnóstico) se excluyen al leer porque sus tramos no
  reproducen su texto: les sobra o les falta un carácter en el borde (un punto, un
  paréntesis, un "%", un espacio). El lector lo advierte y falla si pasan del 1 %.
- **Lector.** `evaluacion/codiesp.py`.
- **Menciones discontinuas.** Sus tramos van en el orden del texto de la mención, que no
  siempre es el de la nota: en 65, 33 y 37 menciones por partición, un tramo está antes que
  el anterior. `inicio` y `fin` son los límites de todos los tramos.

### Qué anota CodiEsp (#130)

La [guía de anotación](https://doi.org/10.5281/zenodo.4121205) (v1.3) pide anotar cada
mención de un diagnóstico cada vez que aparece (regla P2), **también las de otras personas**
(regla P4: "anotar independientemente del sujeto", por ejemplo los antecedentes
familiares). No dice nada de la negación, y una muestra de desarrollo muestra que las
menciones negadas **sí se anotan**: "sin tos", "no presentaba hematuria", "no se observan
adenopatías" llevan su código.

Con nuestro módulo de contexto (una estimación, no un gold), de las 1.100 menciones de
diagnóstico de entrenamiento y desarrollo con categoría de Charlson, 937 salen confirmadas,
76 negadas, 73 posibles y 14 de familiares.

Qué implica:

- **Normalización por mención** (#130 a #133): se compara el texto de cada mención con su
  código, sin importar su estado. Es lo que hacen las reglas y los embeddings, que no miran
  el contexto.
- **Índice por documento** (#134): CodiEsp-D no es el conjunto de comorbilidades del
  paciente, porque incluye diagnósticos negados y de familiares. No se puede comparar
  directo con el índice del pipeline, que solo suma las confirmadas.

## Catálogo CIE-10

Se eligió el **catálogo que acompaña a CodiEsp**, por tres razones:

- tiene licencia clara (CC BY 4.0);
- trae descripciones en español;
- es la terminología con la que está codificado el gold, así que todo código del gold
  está en el catálogo (0 de 14.301 fuera).

Las alternativas que se miraron:

- **CIE-10-ES del Ministerio de Sanidad de España.** Es la misma terminología. El catálogo
  de CodiEsp ya es su versión 2018 en un TSV.
- **Tabla de referencia CIE-10 de MinSalud Colombia (SISPRO).** Colombia usa la CIE-10 de
  la OMS (Resolución 1895 de 2001) y está en transición a la **CIE-11** (Resolución 1442
  de 2024). No se encontró una descarga de la tabla de SISPRO con licencia explícita.
  Para un cliente colombiano habría que revisarla, y la CIE-11 queda fuera de este
  experimento.

**CIE-10-ES frente a la CIE-10 de la OMS.** La CIE-10-ES es la versión española de la
CIE-10-CM de Estados Unidos. Comparte con la OMS la mayoría de los códigos de 3 y 4
caracteres y agrega subcategorías de 5 a 7 caracteres ("E11.21"), que heredan la
categoría de su prefijo. Pero no todos: estos prefijos de la tabla de Quan **no tienen
ningún código** en la CIE-10-ES 2018, así que nunca se activan en CodiEsp:

- B21, B22, B24 (en la CIE-10-CM el VIH con enfermedad es solo B20);
- C97;
- E10.0, E10.7, E11.1, E11.7, E13.7, todo E12 y todo E14 (la CIE-10-CM reorganizó las
  subcategorías de la diabetes: no tiene "con complicaciones múltiples", la diabetes
  ligada a desnutrición ni la no especificada, y el coma y la cetoacidosis van en otras
  subcategorías según el tipo);
- F00 y F05.1;
- I64, I79.2, I85.9, I98.2 y J46;
- Z49.1 y Z49.2.

Y hay códigos de la CIE-10-CM que la tabla de la OMS trata distinto de lo que se
esperaría:

| Códigos | Qué son | Categoría con Quan |
|---|---|---|
| E08, E09 | Diabetes por otra afección y diabetes inducida por fármacos | Ninguna: la tabla cubre E10 a E14 |
| C7A, C7B, C4A | Tumores neuroendocrinos y carcinoma de Merkel | Ninguna: la tabla solo lista códigos C con dos dígitos |
| I85.1x | Várices esofágicas secundarias (el equivalente de I98.2) | Ninguna: se pierde la hepática severa |
| Z49.3x | Contacto para prueba de adecuación de diálisis | Ninguna: se pierde la renal crónica |
| E11.6x | "Con otras complicaciones especificadas": úlcera de pie diabético, hipoglucemia, hiperglucemia | Diabetes sin complicación, como en la OMS |
| C80.1 | Neoplasia maligna **primaria** no especificada | Tumor metastásico (peso 6): Quan toma todo C80 como metastásico, y en la CIE-10-ES C80.1 no lo es |

En el gold de CodiEsp no aparece ningún E08, E09, C7A, C7B ni C4A. C80 sí: 5 veces C80.0
(diseminada) y 2 veces C80.1 en entrenamiento y desarrollo. Para el experimento se
mantiene la tabla de Quan tal cual, y estas diferencias se reportan junto con los
resultados.

## Tabla de Quan (2005)

Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, et al. Coding algorithms
for defining comorbidities in ICD-9-CM and ICD-10 administrative data. Med Care.
2005;43(11):1130-1139. doi:10.1097/01.mlr.0000182534.20362.5d

- **Dónde está.** `TABLA_QUAN` en `src/ner_clinic/charlson/quan.py` copia la columna
  CIE-10 de la tabla 1, con la notación del artículo ("I42.5-I42.9").
- **Cómo se usa.** `categoria_de(codigo)` da la categoría de un código, con o sin punto.
- **Cruce.** `tests/test_charlson_quan.py` la compara con la implementación del paquete R
  [`coder`](https://github.com/ropensci/coder) de rOpenSci (columna `regex_icd10` de
  `data-raw/classcodes.xlsx`, commit `0826da0`). Coinciden en los 28.600 códigos posibles
  de 3 y 4 caracteres, de A00 a Z99.9.

### Diferencias con las reglas actuales

Las 17 reglas (`charlson/reglas.py` y `charlson/reglas_parte2.py`) asignan la
categoría por el texto y después un código CIE-10. En 13 categorías, todos los códigos que pueden dar caen en la misma
categoría con Quan. Hay 4 diferencias, y las cuatro cambian el peso en el índice:

| Código que dan las reglas | Cuándo | Categoría de la regla | Categoría con Quan |
|---|---|---|---|
| E10.8, E11.8 | Diabetes "complicada" sin un sistema reconocible | Diabetes complicada (peso 2) | Diabetes sin complicación (peso 1) |
| N03.9 | Glomerulonefritis crónica sin especificar | Renal crónica (peso 2) | Ninguna: Quan solo toma N03.2 a N03.7 |
| C80 | Tumor maligno con un sitio que no está en la lista de la regla (el código por defecto) | Tumor maligno (peso 2) | Tumor metastásico (peso 6) |
| J84 | Fibrosis pulmonar y enfermedad pulmonar intersticial (#63) | Pulmonar crónica (peso 1) | Ninguna: Quan no incluye J84; de las intersticiales solo toma las neumoconiosis (J60–J67), J68.4, J70.1 y J70.3. Se sigue el Charlson original, que cuenta la enfermedad pulmonar crónica, y los criterios del set sintético |

En el caso de C80, la regla decide bien la categoría; el problema es el código. Un
"cáncer de vulva" o un "carcinoma de vesícula" salen como tumor maligno (correcto), pero
con C80 en vez de C51 o C23, porque esos sitios no están en la lista de la regla. Si la
categoría se derivara del código, pasarían a metastásico. Hay además códigos aproximados:
el recto entra en la entrada del colon y sale C18 en vez de C20.

**Códigos de las reglas que no existen en la CIE-10-ES.** Las reglas dan 15 códigos de la
CIE-10 de la OMS que no están en el catálogo de CodiEsp: B22, B24, E10.0, E10.7, E11.1,
E11.7, E14.0, E14.1, E14.6, E14.9, F00, F05.1, I64, I85.9 e I98.2. Al comparar el código
de las reglas con el gold de CodiEsp (#130), esos nunca coinciden de forma exacta. Por
eso la comparación principal es por categoría de Charlson, no por código.

**Códigos de Quan que las reglas nunca dan.** Las reglas igual pueden asignar la
categoría por el texto, pero con otro código:

- infarto: I22 e I25.2 (infarto antiguo);
- insuficiencia cardiaca: I42.8;
- vascular periférica: I73.8, I79.0, I79.2, K55.1, K55.8 y Z95.9;
- cerebrovascular: G46 e I68;
- demencia: G30 (el Alzheimer sale como F00);
- pulmonar: I27.8, J40, J41, J46, J60 a J63, J65 a J67, J68.4, J70.1 y J70.3;
- reumatológica: M36.0;
- hepática leve: K70.2, K71.3 a K71.5, K71.7, K76.2 a K76.4 y K76.8;
- diabetes: E12, E13 y E14 complicada (E14.2 a E14.5, E14.7) y E14.8 (la regla
  complicada solo arma E10 y E11);
- hemiplejia: G11.4;
- renal: I13.1, N03.2 a N03.7, N05.2 a N05.7 y Z49;
- VIH: B21;
- tumor: la mayoría de los sitios (C00 a C13, C17, C19 a C21, C23, C24, C26, C30, C31,
  C33, C37 a C40, C45, C47, C48, C51, C52, C55, C57, C58, C60, C63, C65, C66, C68 a C70,
  C72, C74 a C76, C82 a C84, C88, C93, C94, C96 y C97): todo lo que no está en la lista
  de sitios va a C80.

`tests/test_charlson_quan.py` fija las 3 diferencias de categoría. Saca los códigos de
cada regla de su código fuente, incluidos los que la diabetes complicada arma con
f-strings: si un cambio en una regla agrega o quita una diferencia de categoría, el test
falla. Las otras dos listas (códigos fuera del catálogo y códigos de Quan que las reglas
nunca dan) se calcularon con la misma función, pero ningún test las fija: hay que
recalcularlas si cambian las reglas.

## Jerarquía de la CIE-10: bloques y capítulos (#150)

El catálogo de CodiEsp no trae los bloques ("I60-I69") ni los capítulos. La CIE-10-ES 2018
es la traducción de la CIE-10-CM de Estados Unidos y comparte su estructura, así que la
jerarquía sale de la tabla tabular de la **CIE-10-CM 2018 del CDC/NCHS**:

- **Archivo:** `ICD-10-CM-Codes-Tables-and-Index-2018.zip`, con MD5
  `b6186cc0a1d6d2503bb5b3957e3a9a32`, en `ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2018/`.
- **Licencia:** los archivos del CDC son obra del gobierno de Estados Unidos y de uso
  libre. La estructura de base de la CIE-10 es de la OMS. Se versionan solo los códigos y
  los rangos de bloque, sin descripciones.
- **Lo que se versiona:** solo la tabla derivada, en
  `src/ner_clinic/normalizacion/jerarquia_cie10cm_2018.tsv`. Trae 1.910 códigos de 3
  caracteres con su bloque (284 bloques) y su capítulo en romanos, sin descripciones. Cubre todos los
  códigos de 3 caracteres del catálogo de CodiEsp. Las 20 cabeceras de capítulo del
  catálogo (`CAP.02`, …) toman su capítulo del número, y como bloque, su propio código.
- **Para regenerarla:** `uv run python -m evaluacion.jerarquia_cie10`.

Para Colombia, la CIE-10 de la OMS tiene casi todos los mismos bloques, pero no todos.
Algunas diferencias:
- **Diabetes:** E10-E14 en la OMS, E08-E13 en la CM.
- **Hipertensión:** I10-I15 en la OMS, I10-I16 en la CM.
- **Trastornos mentales orgánicos:** F00-F09 en la OMS, F01-F09 en la CM.
- **Neoplasias:** C00-D48 en la OMS, C00-D49 en la CM.
- **Códigos propios de la CM:** por ejemplo, O9A.
- **Capítulo XXII (códigos U):** está en la OMS y no en la CM de 2018.

Con notas colombianas habría que regenerar la tabla desde la CIE-10 de la OMS.

## Sinónimos de CANTEMIST y DisTEMIST (#153)

Dos corpus públicos más, solo como sinónimos del índice del normalizador. Nunca se usan
para medir. El lector está en `evaluacion/sinonimos_publicos.py`.

### CANTEMIST

Miranda-Escalada A, Farré E, Krallinger M. Named Entity Recognition, Concept
Normalization and Clinical Coding: Overview of the Cantemist Track for Cancer Text Mining
in Spanish, Corpus, Guidelines, Methods and Results. IberLEF 2020.

- **Versión:** Zenodo 3978041, v1.6 (la última), `cantemist.zip` con MD5
  `a219fcca9078f19490471c920a3b5816`. **Licencia:** CC BY 4.0.
- **Qué trae:** 1.301 casos clínicos de oncología y 16.032 menciones de tumor en
  `cantemist-norm`. Quitadas las TNM quedan 15.429, de 1.300 casos. Se usan sus cuatro particiones, porque no
  se mide nada en CANTEMIST.
- **El código es solo morfología** (`8140/3`, CIE-O-3), sin topografía. La categoría sale
  del comportamiento, el primer dígito tras la barra (conteos antes de quitar las TNM):

| Comportamiento CIE-O | Menciones | Código CIE-10-ES que se le da | Categoría de Quan |
|---|---|---|---|
| `/6` metastásico | 6.699 | C79.9 | Tumor metastásico |
| `/9` maligno, incierto si primario o metastásico ("carcinomatosis") | 148 | C80.0 | Tumor metastásico |
| `/3` maligno | 5.394 | Según la morfología: linfoma de Hodgkin (9650-9667) C81.9, otros linfomas (9590-9729) C85.9, mieloma (9730-9739) C90.0, leucemias (9800-9949) C95.9, melanoma (8720-8799) C43.9, el resto C76.8 | Tumor maligno |
| `/3` basocelular (8090-8098) | (dentro de las `/3`) | C44.91, piel | Ninguna: Quan excluye C44 |
| `/3` mielodisplásicos y mieloproliferativos (9950-9989) | (dentro de las `/3`) | D47.9: la CIE-10 los pone en D45-D47 | Ninguna |
| `/2` *in situ* | 73 | D09.9 | Ninguna |
| `/1` incierto | 3.326 | D48.9 | Ninguna |
| `/0` benigno | 392 | D36.9 | Ninguna |

- Las menciones sin categoría también entran: enseñan qué **no** es un tumor maligno
  ("tumoración", "neoplasia" sin más llevan `/1`).
- El segundo dígito (grado de diferenciación) y el sufijo `/H` se ignoran.
- **Se descartan las etiquetas TNM**, simples o compuestas ("M0", "pT2N0M0"): 603.
  CANTEMIST las anota como `/6` aunque "M0" dice que **no** hay metástasis. Una mención con
  tumor y estadio ("adenocarcinoma rectal T3N1M1") sí entra.
- **Sin topografía no se distingue la piel.** Un carcinoma epidermoide de piel (C44, sin
  categoría) cae en tumor maligno como cualquier otro epidermoide. Solo el basocelular,
  que en la práctica siempre es de piel, va a C44.91.
- **Mención en contexto.** CANTEMIST anota "lesiones hepáticas", "nódulos pulmonares" o
  "adenopatía supraclavicular" con `/6` porque en esa nota eran metástasis. Por eso se mide
  también la variante `cantemist_lexico`, declarada antes de medir: solo los textos con una
  raíz oncológica (neopl-, tumor, cáncer, carcin-, sarcom-, linfom-, leuc-, mielom-,
  melanom-, blastom-, metást-, malign-, -oma). Deja 11.523 de las 15.429 menciones.
- **Casos compartidos con CodiEsp:** ninguno. Se compararon los textos de CANTEMIST con
  los de CodiEsp entrenamiento y desarrollo por 8-gramas de palabras. La mayor fracción
  compartida fue de 1,8 %, lejos del 20 % que marcaría el mismo caso. CodiEsp prueba no se
  lee, y como ya se gastó (#134), no se vuelve a medir en ella.

### DisTEMIST: la columna CIE-10 de sus `cross-mappings`

- **Versión:** la misma de la tabla de arriba (Zenodo 7614764, CC BY 4.0).
- **Qué se usa:** los archivos `cross-mappings/*.tsv` traen, para las enfermedades de
  entrenamiento enlazadas a SNOMED CT, el código en otras terminologías. Los organizadores
  los sacaron del metatesauro UMLS, a partir del enlace manual a SNOMED CT. Solo se lee la
  columna `icd10_code` (CIE-10 de la OMS).
- **SNOMED CT no se usa.** No se usan sus códigos ni una tabla SNOMED → CIE-10 propia.
  Para la evaluación, el índice vive en `data/`, que no se versiona, y no se
  redistribuye. **Antes de desplegar** hay que revisarlo de nuevo: si estos sinónimos
  entran a la imagen Docker publicada, hay que citar CANTEMIST y DisTEMIST (CC BY 4.0) y
  aclarar que la columna CIE-10 de DisTEMIST viene de UMLS, y confirmar que eso no
  arrastra condiciones de la licencia de UMLS. El enlace a SNOMED de las notas sin
  `cross-mappings` no se usa. Esas notas son de CodiEsp de todos modos (ver abajo), y
  menos de 80 serían de entrenamiento.
- **Qué se descarta:** 3.567 sin código CIE-10; 69 con relación `COMPOSITE` (la mención
  junta varios conceptos, como los 7 códigos compuestos "N17.0+R82.1"); 130 rangos ("I10-I15.9"); y 30 cuyos 3 primeros caracteres no son un código de la CIE-10-CM (25
  son "E14", la diabetes sin especificar de la OMS). Quedan 1.340 menciones de 467 notas.
- **Todas las notas son de CodiEsp.** DisTEMIST es CodiEsp con el prefijo `es-` en el id.
  Por id normalizado, de las 467 notas, 116 son de CodiEsp desarrollo (369 menciones) y
  113 de prueba (328 menciones). Esas se quitan siempre. Las 238 restantes son de
  entrenamiento (643 menciones). En la validación cruzada, las del pliegue que se evalúa
  tampoco entran, y `verificar_origen` lo comprueba con el id normalizado.

### Choques

Si un texto ya es sinónimo de CodiEsp en ese índice, gana el de CodiEsp. Entre las
fuentes externas, un texto queda con su código más frecuente, como en #132.

## El normalizador integrado: SapBERT y artefactos (#162)

El normalizador por embeddings es opcional en la librería (`normalizador=`) y el mapeo por
respaldo de las reglas en el mapeo por defecto de la API (`NER_NORMALIZADOR=hibrido`, #182;
antes `sapbert`, #168). Usa tres cosas, cada una con su licencia:

| Pieza | Qué es | Licencia | Dónde vive |
|---|---|---|---|
| [SapBERT multilingüe](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR) | XLM-R base afinado con los sinónimos de UMLS 2020AB (Liu et al., 2021) | Código ([cambridgeltl/sapbert](https://github.com/cambridgeltl/sapbert)): MIT. Pesos: **no declarada** | Se baja de Hugging Face con revisión fija; no se versiona |
| Tokenizador | El de `FacebookAI/xlm-roberta-base` (idéntico al de SapBERT) | MIT | Hugging Face, revisión fija |
| Índice final | Las 98.288 descripciones del catálogo CIE-10-ES 2018 y 3.338 menciones de CodiEsp entrenamiento, con sus vectores | Catálogo y CodiEsp: CC BY 4.0 (citar Zenodo 3706838 y 3837305) | `data/normalizacion/indice_final.pt`; no se versiona; va en la imagen `completa`, así que publicarla lo redistribuye (CC BY 4.0, exige atribución) |
| Clasificador | 39 pesos, media, desviación y umbral, entrenados con CodiEsp entrenamiento; sin texto | Derivado de CodiEsp (CC BY 4.0) | `src/ner_clinic/normalizacion/clasificador.json`, versionado |

**Licencia de los pesos (revisada el 2026-11-09, #102):**

- La tarjeta de Hugging Face no declara licencia. Pero el README de
  [cambridgeltl/sapbert](https://github.com/cambridgeltl/sapbert) dice que el repositorio
  "holds code, data, and pretrained weights", enlaza a este modelo exacto y cierra con
  "SapBERT is MIT licensed". Se toma como MIT, declarado por los autores.
- Queda una zona gris: se entrenó con UMLS 2020AB, cuya licencia restringe redistribuir su
  contenido (por eso el repo no publica los datos de entrenamiento). Los pesos no son
  contenido de UMLS, pero un cliente comercial debería revisarlo con su área legal.
- Decisión de Diego (2026-11-09): para la demo, que no es un producto ni se cobra, basta. La
  demo corre en Cloud Run desde un registro privado: el visitante usa la API y nunca recibe
  los pesos, así que no hay redistribución. Publicar la imagen `completa` en un registro
  público sí lo sería; no se hace por ahora.
- Para un cliente: confirmar con los autores (issue en `cambridgeltl/sapbert`) o afinar un
  modelo propio sin UMLS (p. ej. bsc-bio-ehr-es, Apache-2.0, con el catálogo y CodiEsp) y
  volver a medir.
- La imagen Docker por defecto trae SapBERT y el índice (#189); la etapa `base` no.
- Si el índice o los resultados se muestran, o se publica la imagen `completa`, se cita el catálogo CIE-10-ES de CodiEsp y
  CodiEsp (CC BY 4.0). El campo `candidato` de la API trae texto de los dos.
