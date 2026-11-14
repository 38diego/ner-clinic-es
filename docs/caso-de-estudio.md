# Caso de estudio: de una tesis a una herramienta para leer historias clínicas

**Diego Galindo** · NLP clínico en español ·
[Demo](https://ner-clinic-es-355389257185.us-central1.run.app) ·
[Código](https://github.com/38diego/ner-clinic-es)

## En una frase

Un sistema que lee una nota clínica en español, encuentra las enfermedades que acompañan al
paciente, dice si las tiene, las niega, se sospechan o son de un familiar, y las lleva al
índice de Charlson y a un código CIE-10 que el experto confirma. Nació en una tesis con
historias reales de un hospital y terminó como una demo pública, medida con datos de
terceros.

## 1. El problema: pacientes que en el registro parecen más sanos

En nuestro trabajo de grado (lo hicimos entre tres) estudiamos el perfil de morbimortalidad
y la calidad del dato clínico en un hospital de Bogotá. El hallazgo que dio origen a este proyecto fue simple: el
diagnóstico principal queda codificado, pero **las comorbilidades se quedan en el texto
libre**. La diabetes, la cirrosis o la EPOC aparecen en la nota de evolución y no en los
campos estructurados.

Al pasar el pipeline por los 42.983 egresos del estudio, **el 84,9 % de los pacientes
tenía al menos una comorbilidad que solo estaba en el texto libre**, sin código en el
registro estructurado.

Eso tiene consecuencias concretas:

- **El paciente queda registrado como más sano de lo que está.** Su índice de Charlson,
  que estima el riesgo según las enfermedades que lo acompañan, se calcula con lo
  codificado y sale bajo.
- **Los indicadores de la institución se distorsionan**: los pacientes complejos parecen
  sencillos y los perfiles de morbimortalidad quedan incompletos.
- **Las glosas y auditorías se vuelven lentas**: para comprobar la complejidad de una
  atención, alguien tiene que leer historia por historia.

Leerlo todo a mano no escala. Un hospital genera miles de egresos al año, y cada historia
puede tener decenas de notas.

## 2. La primera versión: la tesis

En la tesis el pipeline era un notebook:

1. Un modelo de reconocimiento de entidades (NER) entrenado en español biomédico,
   [`bsc-bio-ehr-es-distemist`](https://huggingface.co/BSC-NLP4BIA/bsc-bio-ehr-es-distemist),
   para encontrar las enfermedades en el texto.
2. Reglas de contexto para separar lo confirmado de lo negado y de los antecedentes
   familiares.
3. Una depuración: duplicados, errores de ortografía, términos demasiado cortos.
4. Reglas para llevar cada enfermedad a su categoría de Charlson y a un código CIE-10.

Corrió sobre todos los egresos del estudio:

| | |
|---|---|
| Egresos procesados | **42.983** |
| Entidades detectadas (antes de depurar) | **834.414** |
| Comorbilidades confirmadas, tras depurar | **395.711** |
| Enfermedades negadas o descartadas | **206.276** |
| Antecedentes familiares | **10.900** |
| Pacientes con al menos una comorbilidad sin codificar | **84,9 %** |
| Tiempo de procesamiento | unas **10 horas** en una GPU GTX 1650 |

Funcionó, pero era un notebook: solo se podía correr con esos datos y en esa máquina. En
la tesis también se probó mapear a CIE-10 con embeddings, y no dio mejores resultados que las
reglas, así que se quedaron las reglas.

## 3. Lo que había que resolver para que alguien más lo usara

Pasar de la tesis a algo que un auditor o un médico pueda usar exigía cuatro cosas.

### Que se pudiera confiar en los números

Las cifras de la tesis venían de una auditoría manual sobre los datos del hospital, que no
se pueden publicar. Así que todo se volvió a medir con **corpus públicos anotados por
terceros**, separando desde el principio una parte que nunca se miró al ajustar el sistema:

- **DisTEMIST** (casos clínicos anotados) para saber si encuentra las enfermedades.
- **NUBes** (frases de historias clínicas reales de España) para la negación y la sospecha:
  14.817 frases reservadas que se usaron una sola vez.
- **CodiEsp** para el código CIE-10, también medido una sola vez en su partición de prueba.
- **100 notas sintéticas** con estilo de IPS colombiana, escritas para el proyecto, para el
  índice de Charlson y el cruce con los diagnósticos registrados.

Medir así fue lo que más cambió el proyecto, porque mostró errores que en la tesis no se
veían.

### Corregir lo que la medición encontró

- **La negación.** Con las reglas de la tesis, en texto real quedaba sin negar **una de cada
  cuatro** enfermedades negadas. Faltaban claves como "descartó" o "tampoco", el alcance se
  cortaba mal y en "niega HTA, DM, EPOC" solo se negaba la primera. Corregido y medido en la
  parte reservada de NUBes: F1 de 0,80 a **0,87**.
- **Las sospechas.** "Probable EPOC" contaba como confirmada y sumaba al índice. Se agregó un
  estado `posible`: se muestra, pero no suma hasta que alguien la confirme.
- **Errores en las reglas heredadas**: un "no" dentro de "maligno" se tomaba como negación,
  la "ñ" nunca coincidía, "insuficiencia cardiaca" e "insuficiencia renal" se fusionaban por
  empezar igual. Cada error quedó con un test y su efecto medido
  ([`errores-tesis.md`](errores-tesis.md)).

### Un código CIE-10 mejor, sin perder lo que ya funcionaba

Volví a probar embeddings, esta vez con **SapBERT**, un modelo hecho para emparejar términos
médicos, contra el catálogo CIE-10-ES y los sinónimos de CodiEsp. Solo, era mejor en los
casos de España y peor en las siglas colombianas ("ICC" quedaba cerca de "HCC"). La
solución fue **híbrida**: primero las reglas y, para lo que las reglas no reconocen, el
normalizador. Y como el código exacto a veces es discutible, la demo muestra los **5 más
parecidos del catálogo** y el experto elige.

### Un producto, no un notebook

- **Una librería y una API** (FastAPI) con tests, que cualquiera instala y corre.
- **Una demo web** pensada con la idea de no dar "listas ciegas": cada comorbilidad lleva
  a su lugar en la nota, para que quien lee vea el contexto antes de creerle al sistema.
- **Privacidad por diseño**: la demo no guarda ni registra el texto que recibe.
- **Costo con techo**: corre en CPU en Google Cloud Run, con una instancia máxima y topes de
  análisis por persona y por día, para quedarse en la capa gratuita.

## 4. Resultados

Todo medido con datos que no se usaron para ajustar el sistema:

| Qué | Resultado |
|---|---|
| Encuentra la enfermedad y la ubica en la nota (DisTEMIST) | **88 %** (F1 0,878) |
| Distingue las enfermedades negadas (NUBes, reservado) | F1 **0,87** (con las reglas de la tesis, 0,80) |
| Distingue las sospechas (NUBes, reservado) | F1 **0,86** (antes no existía ese estado) |
| Índice de Charlson exacto (notas sintéticas) | **96 %** de las notas (86 % en la primera medición, la única ciega) |
| Código CIE-10: acierta la enfermedad a 3 caracteres (CodiEsp prueba) | **78 %**; está entre los 5 que se muestran en el **91 %** |
| Tiempo por nota, en CPU | **menos de 1 segundo** |

Las 10 horas en GPU de la tesis no se pueden comparar directo con esta cifra, porque las
historias reales son mucho más largas que una nota. Pero el sistema ya no necesita GPU.

## 5. Lo que no resuelve (todavía)

- **Es una ayuda, no un codificador automático.** Puede omitir una enfermedad o equivocarse
  en su estado; por eso la demo siempre lleva al texto.
- **El vocabulario de evaluación es de España.** En notas colombianas cambian siglas y
  expresiones. Para un cliente, lo primero sería medir con sus notas.
- **El catálogo es la CIE-10-ES.** Colombia usa la CIE-10 de la OMS (casi igual a 3
  caracteres) y está en transición a la CIE-11.
- **La negación no es perfecta**: queda sin negar una de cada seis o siete enfermedades
  negadas.

El detalle de cada límite está en [`evaluacion.md`](evaluacion.md).

## 6. Qué aprendí

- **Medir con datos que no miraste cambia todo.** En la tesis las reglas de negación no se
  midieron por separado; medidas en datos reservados, dejaban pasar una de cada cuatro
  enfermedades negadas.
- **Lo que se descarta también enseña.** Los embeddings no sirvieron en la tesis y sí aquí,
  pero solo como respaldo de las reglas, no como reemplazo.
- **La interfaz es parte de la solución.** Ningún modelo acierta siempre; si cada resultado
  lleva a su lugar en el texto, quien lee detecta el error en segundos en vez de heredarlo.

## Qué haría para una IPS o clínica

1. Medir con 50 a 100 de sus notas, anotadas por su equipo, para saber cómo rinde con su
   forma de escribir.
2. Pasar los códigos a la CIE-10 que usa en Colombia (tabla de SISPRO) o a la CIE-11.
3. Conectarlo a sus egresos para marcar, antes de facturar, las comorbilidades que están en
   la nota y no en el registro.
