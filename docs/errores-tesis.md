# Errores heredados de la tesis: corrección y efecto

Al portar el pipeline de la tesis a `ner_clinic` se respetó su lógica tal cual (PRD
`pipeline-core`). Las revisiones de código de cada módulo encontraron errores en esas reglas.
Este documento lista los que se corrigieron en el issue #24 (PR #34, #35 y #36, más este),
cómo se corrigió cada grupo y el efecto medido.

Todas las notas usadas aquí son **sintéticas**, escritas para el proyecto.

## Qué se corrigió

Cada corrección tiene un test que fallaba antes del cambio y pasa después. El detalle caso a
caso (ejemplo, resultado antes y después) está en el cuerpo de cada PR.

### Reglas de Charlson / CIE-10 (PR #34)

| Tipo de error | Ejemplo | Antes | Después |
|---|---|---|---|
| Negación sin `\b`: el "no", "ni" o "sin" dentro de otra palabra se tomaba como negación | "tumor maligno metastásico de colon" ("malig**no**") | sin categoría | tumor metastásico (C80) |
| Lo mismo en las exclusiones de las 12 categorías restantes | "ecografía de este a**ño** se evidencia cirrosis" | sin categoría | hepática leve (K74) |
| `ñ` en regex que nunca coincidía porque el texto se normaliza a `n` | "infarto agudo de miocardio en cuña", "diabetes sin daño renal" | I21 / E11.8 | sin categoría / E14.9 |
| Subclasificación del ACV: siglas sin `\b` y orden I63 antes que I65/I66 | "placa de ateroma carotídea", "estenosis basilar" | I63 | I65 |
| `\b` mal ubicado en la ICC | "falla cardiaca con FEVI y diámetros normales" | I43 | I50 |
| Plurales y variantes que no entraban | "bronquiectasias", "aneurisma de aorta", "metástasis óseas", "l.e.s. activo" | sin categoría o código errado | J47, I71.9, C79, M32 |
| Monoplejía con `ms\|mi` sin `\b` | "monoparesia secundaria a mielopatía" | G83.1 | G83.3 |
| Diabetes sin tipo especificado con coma o cetoacidosis | "cetoacidosis diabética" | E11.1 | E14.1 |
| Erratas | `gestacvional`, `diab[eé]tical?` | — | corregidas |

### Contexto, depuración y siglas (PR #36)

| Error | Ejemplo | Antes | Después |
|---|---|---|---|
| **Siglas frente a diagnósticos registrados**: una sigla nunca coincidía, así que salía como falso subregistro | "HTA" con registrado "Hipertensión esencial (primaria)" | `registrado=False` | `registrado=True` (expansión de siglas en `siglas.py`) |
| "descartado" antes de la entidad no negaba | "se considera descartado TEP" | confirmada | negada |
| "sin embargo" abría una negación nueva | "niega disnea sin embargo cefalea" | cefalea negada | cefalea confirmada |
| Jaro-Winkler sobre la frase completa fusionaba enfermedades distintas con el mismo comienzo | "insuficiencia cardiaca" / "insuficiencia renal crónica"; "cáncer de mama" / "cáncer de próstata" | se fusionaban (se perdía una) | se conservan las dos |
| Una palabra común no específica marcaba otra enfermedad como registrada | "insuficiencia renal crónica" con registrado "Insuficiencia cardiaca congestiva" | registrada / descartada | no registrada |
| Entre variantes de igual longitud quedaba la primera | "hipertension" / "hipertensión" | sin tilde | con tilde |

Consecuencia del error de Jaro-Winkler: la "enfermedad de Alzheimer" de un familiar se fusionaba
con la "enfermedad renal crónica" del paciente y desaparecía de la salida.

### Extracción y limpieza (PR #35)

| Error | Ejemplo | Antes | Después |
|---|---|---|---|
| Saltos de línea de Windows (`\r\n`) no se colapsaban | texto pegado desde Windows | saltos repetidos | normalizados, conservando el mapa de posiciones |
| Siglas que son palabras comunes | "se **les** explica", "vive con su **tia**" | pasaban como LES / TIA | solo cuentan en mayúsculas |
| Palabras que no son siglas dentro de `SIGLAS_CLINICAS` | "ASMA", "GOTA" | mezcladas | `PALABRAS_CORTAS_CLINICAS` |

Las correcciones de "y" como barrera de fusión y de la ventana deslizante de 512 tokens se
hicieron antes, en el PR #28, y no entran en esta medición.

## Efecto medido

Método: `experimentos/errores_tesis/medir.py` corre `analizar()` con el modelo real (CPU)
sobre notas sintéticas anotadas a mano, con el código **antes** del #24 (commit `8bcb699`) y
**después**. Resultados completos en `experimentos/errores_tesis/resultados/`.

**Casos de error** (`experimentos/errores_tesis/notas.jsonl`, 15 notas escritas con los
patrones de arriba, 26 entidades esperadas):

| Métrica | Antes | Después |
|---|---|---|
| Recall de entidades | 76,9 % | **92,3 %** |
| Estado correcto (confirmada / negada / familiar) | 90,0 % | **95,8 %** |
| Diagnóstico registrado correcto | 2 de 4 | **4 de 4** |
| Índice de Charlson exacto | 10 de 15 notas | **14 de 15** |

**No regresión** (`experimentos/siglas/notas.jsonl`, las 40 notas del experimento de siglas,
152 entidades esperadas): mismos resultados antes y después (recall 83,6 %, estado correcto
97,6 %). Esas notas no contienen los patrones corregidos; sirven para comprobar que las
correcciones no rompieron nada.

**Límite de esta medición:** las 15 notas de casos de error se escribieron para contener
justamente esos errores, así que muestran que la corrección funciona, no cuánto mejora el
sistema en texto clínico real. La medición independiente, sobre DisTEMIST, NUBes y un set
sintético reservado, está en [`docs/evaluacion.md`](evaluacion.md).

## Decisiones clínicas

- **Diabetes sin tipo especificado** con coma o cetoacidosis: E14.0 / E14.1 (diabetes no
  especificada en CIE-10), no E11.x como en la tesis.
- **ACV**: I63 solo si el texto habla de infarto o isquemia; estenosis u oclusión sin
  infarto van a I65 (arterias precerebrales, incluida la basilar) o I66 (cerebrales).
- **VIH sin SIDA**: se mantiene el peso 6 de Charlson (1987), como la tesis. En versiones
  posteriores del índice el VIH sin SIDA pesa menos; separar la categoría queda pendiente.

## Lo que queda

- **Contexto por texto, no por posición** (diseño de la tesis): si la misma palabra aparece
  en un bloque familiar y como diagnóstico del paciente, gana el familiar. En los casos de
  error, "Vive con su tia materna. Antecedente de TIA" deja el TIA como familiar.
  **Corregido en #73**: cada mención se clasifica por su posición y una enfermedad puede
  salir dos veces, con estados distintos.
- Ruido que pasa todos los filtros ("yugular", "exfumador"): sin una regla general que lo
  justifique; se mide en la evaluación.
- El cruce con diagnósticos registrados acepta una sola palabra en común: "TEP" expandido a
  "embolia pulmonar" podría coincidir con un registrado de EPOC por "pulmonar".
- Síntomas ("disnea", "cefalea") no se extraen: el modelo fue entrenado con DisTEMIST, que
  solo anota enfermedades. Es el alcance del modelo, no un error.
