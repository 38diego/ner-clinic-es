"""Clasificación de las menciones según el contexto en que aparecen en la nota.

Porta las reglas de `categorizar_comorbilidades` de la tesis, pero mira la posición:

1. `intervalos` busca en la nota (limpia) los bloques negados, los posibles y los de
   antecedentes familiares con expresiones regulares, y devuelve sus posiciones con su
   estado.
2. `clasificar_mencion` mira en qué intervalos cae cada mención: está dentro de uno si
   su **inicio** cae dentro de él (`intervalo.inicio <= mencion.inicio < intervalo.fin`).
3. Prioridad: familiar, luego posible, luego negada; si no cae en ningún intervalo,
   queda confirmada.

Estado `posible` (#88, PRD `incertidumbre`): lo ponen las claves de incertidumbre de
#83 ("sospecha de", "probable", "no se descarta"...), descritas al final. Posible gana
sobre negada porque varias claves de incertidumbre contienen una de negación: en "no se
descarta X" el "no" abre un alcance negado, pero X es una sospecha. Familiar gana sobre
posible porque la sospecha en un familiar sigue sin ser del paciente.

Por qué el inicio, y no el solapamiento ni la contención (medido en la partición de
desarrollo de NUBes, #73):

- Con solapamiento, una mención larga que contiene un disparador ("IAM no Q", "fractura
  no desplazada") queda negada aunque empiece antes del alcance: F1 0,801, P 0,810.
- Con contención, "niega diabetes con nefropatía" no queda negada, porque la barrera
  "con" corta el alcance a mitad de la mención: F1 0,807, P 0,857.
- Con el inicio dentro, la negación cubre lo que empieza en su alcance y no lo que
  empieza antes: F1 0,813, P 0,854, el mejor de los tres.

Las posiciones son las del texto donde corren las regex (el limpio, en el pipeline): las
menciones tienen que venir en esas mismas coordenadas.

Limitación heredada, a revisar después de la evaluación:

- "riesgo bajo para" y "bajo riesgo de" cuentan como negación. No aparecen en la
  partición de desarrollo de NUBes, así que no hay datos para pasarlas a `posible`
  (#83): siguen negando, que no suma al índice, como posible.

Corregido respecto a la tesis (#24):

- "descartado", "se descarta" y "negativo para" también niegan lo que va después
  ("descartado TEP"), no solo lo que va antes ("TEP descartado").
- "sin embargo" y "no obstante" cortan la negación sin abrir otra: en la tesis el
  "sin" de "sin embargo" volvía a negar lo que seguía.

Corregido respecto a la tesis (#62): enumeraciones. En la tesis el alcance terminaba en
la primera coma y en la "y", así que "niega HTA, DM y EPOC" solo negaba la HTA. Ahora,
tras el disparador, el alcance sigue por una **lista**, y dentro de ella la "y" ya no es
barrera:

- Elementos de hasta 3 palabras, unidos por coma, "y", "e", "o", "u" o "ni".
- Ninguna palabra de un elemento es barrera: "niega fiebre y presenta diabetes" y
  "niega fiebre, refiere diabetes" no son listas, y la diabetes queda confirmada.
- "salvo", "excepto", "sí", "que", "por" y "donde" no abren un elemento nuevo.
- Cada elemento termina entero (en coma, punto, salto, conjunción o barrera). Si uno es
  más largo, la lista acaba en el anterior.
- La lista es un patrón aparte del tramo de siempre (hasta la primera coma, punto,
  salto, "y" u otra barrera), que no cambia: el alcance es la unión de los dos. Si fuera
  una alternativa del mismo patrón, en "sin tos ni fiebre o signos de dificultad
  respiratoria" ganaría la lista corta ("tos ni fiebre") y dejaría confirmada la
  dificultad respiratoria, que el tramo sí niega.

Por qué elementos cortos y no seguir por todas las comas (medido en la partición de
desarrollo de NUBes): seguir por toda coma y toda "y" sube el recall de 0,775 a 0,820
pero baja la precisión de 0,854 a 0,729 (F1 0,772), porque la prosa clínica encadena
cláusulas con comas ("sin gran esfuerzo, 3 síncopes"). Con las reglas de arriba, el F1
es 0,818 con elementos de 2 palabras (P 0,850, R 0,788), 0,817 con 3 (P 0,845,
R 0,792) y 0,816 con 4 (P 0,838, R 0,795): la diferencia es de una o dos menciones. Se
eligió 3 porque cubre elementos como "antecedentes de IAM" o "sangrado de encías", que
con 2 se pierden. Frente a la lista como alternativa del tramo (F1 0,818), la unión gana
un acierto y suma dos FP, pero nunca niega menos que antes.

Corregido respecto a la tesis (#61): "no presenta X" no negaba X. El disparador "no"
abría el bloque, pero "presenta" es barrera y lo dejaba vacío. Ahora "no presenta" es
un disparador propio, como "no refiere":

- "No presenta EPOC", "no presenta antecedente de IAM" y "no presenta HTA, DM ni EPOC"
  quedan negadas.
- "presenta" sigue siendo barrera cuando no va justo después del "no": en "niega fiebre
  pero presenta diabetes" y "niega fiebre y presenta diabetes" la diabetes queda
  confirmada.
- "no presentó", "no presentaba", "no presentan" y "no ha presentado" ya negaban: la
  barrera es solo la palabra "presenta".

Medido en la partición de desarrollo de NUBes: el F1 de negada pasa de 0,817 a 0,823
(P 0,845 → 0,845, R 0,792 → 0,802). Cambian 7 menciones: 6 aciertos y un FP, en el que
el alcance de "no presenta" sigue después de un ";" (el corte en ";" es de #66).

Corregido respecto a la tesis (#65): claves de negación que faltaban.

- Todas las formas del verbo "descartar" niegan lo que va después: "se descartó TEP",
  "descarta complicaciones", "descartándose TVP", "tras descartar TVP". Antes solo
  "descartado/a(s)" y "se descarta". Hacia atrás sigue negando solo el participio ("TEP
  descartado"); el alcance hacia atrás es de #66.
- "para descartar" y "a descartar" no niegan: dicen qué se busca ("remitido para
  descartar TVP"). Con ellas, el F1 era 0,847 (P 0,833, R 0,863): 5 aciertos más, pero
  13 FP y una incierta más que sin ellas.
- "tampoco" (y "tampoco refiere/presenta"), "nunca", "ningún/ninguna/ninguno" (con
  ellas "en ningún momento") y "no se evidencia(n)". "No se evidencia X" quedaba
  confirmada por lo mismo que "no presenta X" (#61): la barrera "se evidencia" vaciaba
  el bloque del "no".
- "cede", "desaparición de" y "se resuelve" no se agregan: una enfermedad resuelta no
  es una negada para el auditor (PRD de negación, FR5).
- Como todas las claves, tienen que ser palabras enteras: "descartable", "ninguneo" o
  "anunca" no disparan, y "sin embargo" y "no obstante" siguen sin negar.

Medido en la partición de desarrollo de NUBes: el F1 de negada pasa de 0,823 a 0,852
(P 0,845 → 0,849, R 0,802 → 0,855). Cambian 37 menciones: 32 aciertos, 3 FP (dos por
el alcance de la lista de #62 tras "descartándose" y "ninguna", uno por "y descartar")
y 2 inciertas (una tras "habría que descartar", otra en un alcance largo tras
"descartando", que no se corta hasta el punto). Cada grupo por separado, frente a 0,823:
"descartar" 0,846 (25 aciertos, 2 FP, 2 inciertas), "tampoco" 0,827 (4 aciertos),
"ningún" 0,824 (2 aciertos, 1 FP); "nunca" y "no se evidencia" no cambian ninguna
mención de desarrollo y se dejan por el PRD y los tests.

Corregido respecto a la tesis (#66): la negación se alargaba más allá de la cláusula.

- El alcance también se corta en ";", ":", paréntesis y " - " (guion entre espacios),
  además de coma, punto y salto de línea: en "sin dolor ) TAC: fractura costal" la
  fractura queda confirmada. Vale para el tramo, para los elementos de una lista y para
  el patrón hacia atrás.
- Hacia atrás ("TEP descartado", "cultivo negativo") la clave tiene que ser una palabra
  entera ("poliartritis seronegativa" no niega) y solo niega las 4 palabras de antes,
  no toda la cláusula hasta la coma anterior. "coagulasa negativo" y "gram negativos"
  nombran un germen, no un resultado: no niegan. "VIH: negativo" sigue negando, aunque
  los dos puntos cortan el alcance hacia adelante.
- "ya que" y "dado/a(s)" son barreras: en "no anticoagulado dada la hemorragia" la
  hemorragia queda confirmada. Ninguna aparece dentro de un nombre de enfermedad: de
  las 6.839 menciones distintas de DisTEMIST, ninguna tiene "ya que" y solo una tiene
  "dada", que es una cláusula entera y no un nombre.
- "por" **no** es barrera: es parte del nombre de muchas enfermedades ("infección por
  VIH", "neumonía por COVID-19", "cirrosis por virus C"; 230 de las 6.839 menciones de
  DisTEMIST). Con ella, "niega infección por VIH" dejaba el VIH confirmado, lo que suma
  6 puntos al índice de Charlson: el error más caro posible.

Medido cada cambio por separado en la partición de desarrollo de NUBes, frente a F1
0,852 (P 0,849, R 0,855, VP 517, FP 92, FN 88):

- Cortes en ";", ":", paréntesis y " - ": F1 0,874 (P 0,896, R 0,853). Cambian 43
  menciones: 34 FP quitados, 2 FP nuevos (la lista de #62 ahora puede terminar en un
  paréntesis y llega a un elemento más), 1 FN nuevo (el gold niega tras un paréntesis
  de cierre) y 6 inciertas.
- `\\b` antes de la clave hacia atrás, la excepción de "coagulasa" y "gram" y los dos
  puntos antes de la clave: no cambian ninguna mención de desarrollo; se dejan por el
  PRD y los tests.
- Hacia atrás limitado a N palabras: F1 0,860 con 2 (P 0,871, R 0,850), 0,861 con 3,
  0,861 con 4 (P 0,869, R 0,853) y 0,859 con 6. Con los cortes, 3 da 0,874 y 4 y 5 dan
  0,875 (P 0,900, R 0,851): se eligió 4. Que esas palabras no puedan ser barrera
  quita 2 FP pero pierde 2 aciertos (F1 0,875 igual): no se agrega.
- Barreras, cada una sobre lo anterior (F1 0,875): "ya que" 0,876 (1 FP quitado),
  "dado/a(s)" 0,876 (1 FP quitado); "dado que" no cambia nada. Se descartan "que"
  (0,871: 13 aciertos más, 11 FN nuevos), "debido a" (0,874: un FN nuevo) y "por"
  (0,877, pero con 6 FN nuevos, una ganancia dentro del ruido y el riesgo de arriba).

Con todo, F1 0,877 (P 0,903, R 0,851; VP 515, FP 55, FN 90). Cambian 51 menciones:
39 FP quitados, 2 FP nuevos, 2 FN nuevos (uno tras un paréntesis de cierre, otro por
el límite de 4 palabras) y 8 inciertas que dejan de salir negadas.

El patrón hacia atrás era `[^,\\.\\n]+?(?:descartad…|negativ…)`: desde cada posición
recorría el texto hasta la siguiente coma, así que un texto largo sin comas tardaba en
proporción al cuadrado de su largo (unos 28 s con 5.000 palabras). Ahora cada intento
empieza en una palabra y mira 4 como mucho, y el texto de 5.000 palabras tarda
centésimas. Por lo mismo, el bloque "Antecedentes familiares:" ya no empieza con `\\s*`:
recorría entera cada racha larga de espacios desde cada una de sus posiciones.

Corregido (#84): la lista de #62 saltaba a la cláusula siguiente. En "sin secuelas
motoras, demencia vascular leve" la coma separa dos ideas, pero los dos lados caben
como elementos de hasta 3 palabras y la demencia salía negada. Ahora:

- Tras un elemento de varias palabras, la lista solo sigue por la coma con un elemento
  de una palabra (o sigla): "niega antecedentes de IAM, ACV o enfermedad renal" niega
  las tres, "sin secuelas motoras, demencia vascular leve" deja la demencia confirmada.
- Dos elementos de varias palabras seguidos sí son lista si los une una conjunción
  ("y", ", y", "o", "ni"...): "niega úlcera péptica y enfermedad hepática".
- La lista sigue exigiendo al menos dos elementos. Sin ese mínimo el F1 sube a 0,880
  (un FP menos), pero por un efecto lateral: en "no se evidencian LOE, atrofia
  corticosubcortical" el disparador "no" se queda con un elemento suelto ("se
  evidencian LOE") y ya no se prueba "no se evidencian". Un elemento suelto no
  agrega nada al tramo de siempre: no se quita el mínimo.

Medido en la partición de desarrollo de NUBes, frente a F1 0,877 (P 0,903, R 0,851;
VP 515, FP 55, FN 90). Sin la lista de #62, el F1 es 0,875 (P 0,920, R 0,835): la
lista aporta 10 aciertos, 11 FP y 3 inciertas. Variantes:

- Tras un elemento largo, el siguiente es de una palabra, también tras conjunción:
  F1 0,880 (P 0,910, R 0,851). 4 FP y 2 inciertas menos, ningún acierto perdido. Pero
  "niega úlcera péptica y enfermedad hepática" deja la enfermedad hepática confirmada.
- La elegida (lo mismo, salvo dos largos unidos por conjunción): F1 0,879 (P 0,908,
  R 0,851; VP 515, FP 52, FN 90). Cambian 5 menciones: 3 FP quitados ("sin gran
  esfuerzo, 3 síncopes", "sin evidencia de sangrado activo, leve hipertiroidismo" y un
  hallazgo de TAC) y 2 inciertas. Frente a la anterior, solo vuelve un FP: "no se pudo
  revascularizar y FA crónica".
- Lista uniforme (todos los elementos de una palabra o todos de varias): F1 0,878
  (P 0,913, R 0,846). Quita 6 FP, pero pierde 3 aciertos (entre ellos "sangrado de
  encías" tras elementos cortos y "asimetrías faciales" tras "cefalea").
- Elemento siguiente no más largo que el anterior: F1 0,876 (un acierto perdido).
- Que un elemento largo terminado en plural ("motoras", "previas") corte la lista:
  F1 0,877, un solo FP quitado.

Corregido respecto a la tesis (#73): la tesis buscaba el **texto** de la entidad dentro
de los bloques, así que si la palabra aparecía negada en una parte de la nota quedaba
negada también donde estaba confirmada. Ahora cada mención se clasifica por su posición,
y el pipeline separa en una entidad por estado la que tiene menciones en estados
distintos. Las reglas (patrones, barreras y prioridad) son las mismas.

Incertidumbre (#83, PRD `incertidumbre`, FR4): las claves de `CLAVES_POSIBLE` ponen en
`posible` lo que va después, con la misma mecánica que la negación: el tramo hasta el
primer corte o barrera, la lista de #62 y, para unas pocas (`CLAVES_POSIBLE_ATRAS`), las
4 palabras de antes ("foco aparente", "AIT vs ITU", "síncope ?"). "sospecha de" sale de
los disparadores de negación. Frente a la negación hay tres reglas:

1. Una negación que empieza dentro de una clave es parte de ella: el "no" y el
   "descarta" de "no se descarta", el "sin" de "sin clara". Sin esta regla, en "sin
   clara imagen sugestiva de X" el "sin" negaría la clave "sugestiva de".
2. Una clave hacia adelante dentro de un alcance negado está negada ella misma y su
   alcance niega: "sin datos sugestivos de neumonía", "no se evidencian imágenes
   compatibles con neumonía". Hacia atrás no: en "sin foco aparente" la clave matiza la
   negación y el gold de NUBes la anota incierta.
3. El alcance posible se corta donde empieza una negación: en "probable neumonía sin
   derrame" el derrame queda negado.

Medido en la partición de desarrollo de NUBes, frente a main (negada F1 0,879: P 0,908,
R 0,851; VP 515, FP 52, FN 90; posible 0, FN 671). Con todo: negada F1 0,880 (P 0,913,
R 0,850; VP 514, FP 49, FN 91) y posible F1 0,846 (P 0,941, R 0,768; VP 515, FP 32,
FN 156). Sin las reglas, solo con la prioridad: negada 0,853 (28 negadas menos,
casi todas tras "sin datos sugestivos de") y posible 0,839. Cada regla
quitada por separado: sin la 1, posible 0,843 (3 aciertos menos); sin la 2, negada
0,857 y posible 0,842; sin la 3, negada 0,876 y posible 0,843.

Un grupo de claves entra si sube el F1 de `posible`, no baja el de `negada` más de
0,002 y sus aciertos nuevos superan a sus errores nuevos. Lo que aporta cada grupo
aceptado, quitándolo del conjunto final (F1 de posible; aciertos y FP de posible):

- "probable(s)", "probablemente": +0,126 (123 aciertos, 6 FP).
- "compatible(s) con": +0,093 (93, 3). Para NUBes es incertidumbre; en radiología
  colombiana a veces se lee como casi confirmación (PRD, Constraints). Se decide con los
  datos y la duda queda aquí.
- "sospecha (clínica) de", "sospechoso/a(s) de", "sospechar": +0,082 (84, 5); en
  negada +0,001 (3 FP menos, un acierto menos).
- "posible(s)", "posiblemente": +0,054 (56, 2).
- "sugestivo/a(s) de": +0,038 (39, 1).
- "orientación diagnóstica:": +0,011 (11, 0). Es el encabezado de la conclusión de los
  informes de radiología españoles.
- "parece(n)", "no parece(n)": +0,010 (12, 2). "no parece" es una clave entera (regla
  1): para NUBes es incertidumbre, no negación.
- "sugiere(n)", "sugiriendo": +0,007 (8, 1).
- "podría(n)": +0,006 (6, 0); "impresiona (de)": +0,006 (6, 0); "no se descarta", "no
  se puede descartar", "no descartamos", "sin (que se pueda) descartar": +0,006 (6, 0).
- "puede(n)", "pueda(n)": +0,006 (8, 3).
- "se interpreta/interpretó como", "interpretado como", "interpretamos": +0,005 (5, 0);
  "sin clara(s)": +0,005 (5, 0); "posibilidad(es) de": +0,005 (5, 0).
- "dudoso/a(s)", también hacia atrás: +0,004 (5, 1); "aparente(s)", "aparentemente",
  hacia atrás "aparente(s)": +0,004 (4, 0); "vs", "vs.", "versus", también hacia atrás:
  +0,004 (4, 0).
- "?" hacia atrás: +0,002 (4, 2).
- "(se) plantea", "planteando": +0,002 (2, 0); "pudiera(n)", "pudiendo": +0,002 (3, 2);
  "a valorar": +0,002 (3, 2); "al parecer": +0,001 (1, 0).

En `negada`, solo "sospecha" cambia algo. Sacar "sospecha de" de la negación no cambia
nada con las claves puestas (la regla 1 ya la absorbe); se saca por el PRD, para que la
negación no dependa de la incertidumbre.

Descartadas, agregándolas al conjunto final:

- "probable(s)" hacia atrás ("neumonía probable"): -0,015 (6 aciertos, 31 FP: "lo más
  probable", "como probable causa").
- "a descartar" y "para descartar": -0,012 (0 aciertos, 18 FP). Dicen qué se busca, no
  qué se sospecha (#65), y siguen sin negar. "descartar" al inicio de la frase (plan):
  -0,001 en posible y -0,002 en negada (2 negadas se pierden).
- "posible(s)" hacia atrás: -0,008 (0 aciertos, 12 FP).
- "valorar" suelto: -0,0006 (3 aciertos, 5 FP: "valorar amigdalectomía", "valorar
  remitir"); con "a valorar" ya puesta. "justificaría(n)": -0,0007 (un FP). "¿" hacia
  adelante: no cambia nada (el "?" ya cubre las preguntas).
- Erratas ("probablememte"): +0,001 (un acierto), pero no se agregan erratas sueltas.

El "?" y el patrón hacia atrás no cuestan más con el largo: cada intento mira 4 palabras
como mucho, y las reglas frente a la negación usan búsqueda binaria, así que un texto con
miles de claves y negaciones se clasifica en centésimas.
"""

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from itertools import accumulate
from typing import NamedTuple

from rapidfuzz.distance import DamerauLevenshtein

from ner_clinic.tipos import Estado, Mencion

# Palabras que cortan un bloque negado: en "niega fiebre pero presenta diabetes",
# la negación cubre "fiebre" y no "diabetes". "ya que" y "dado/a" (#66) abren otra
# cláusula: "no anticoagular dada la hemorragia". "por" no es barrera: es parte del
# nombre de muchas enfermedades ("infección por VIH", "neumonía por COVID-19").
_BARRERAS = (
    r"\b(?:pero|aunque|sin embargo|no obstante|actualmente|y|refiere|presenta|con"
    r"|se evidencia|ya que|dad[oa]s?)\b"
)

# Signos que cortan cualquier alcance negado (#66): coma, punto y salto de línea, como
# en la tesis, y además punto y coma, dos puntos y paréntesis. Va dentro de una clase de
# caracteres. El guion entre espacios (" - ") también corta, con `_GUION`.
_CORTES = r",;:()\.\n"
_GUION = r"[ \t]-(?=\s|$)"

# Claves que faltaban (#65). Las formas del verbo "descartar" niegan lo que va después:
# "se descartó TEP", "descarta complicaciones", "descartándose neumonía". Hacia atrás
# ("TEP descartado") ya negaba el participio, en el patrón de abajo. "para descartar" y
# "a descartar" dicen qué se busca, no qué se encontró: no disparan.
_DESCARTAR = r"(?<!\bpara )(?<!\ba )descart(?:a|an|ó|o|aron|ando|ándose|andose|ar|e|en)"
# "tampoco refiere" y "tampoco presenta" son disparadores propios, como "no refiere" y
# "no presenta": con "tampoco" solo, la barrera dejaría el bloque vacío. "en ningún
# momento" va por "ningún". "no se evidencia" es disparador porque "se evidencia" es
# barrera y dejaba vacío el bloque del "no".
_OTRAS_CLAVES = r"tampoco(?:\s+(?:refiere|presenta))?|nunca|ning[uú]n[oa]?|no\s+se\s+evidencian?"

# "sin" y "no" no disparan cuando son el inicio de "sin embargo" o "no obstante".
# "no presenta" es un disparador propio (#61): con "no" solo, la barrera "presenta"
# dejaba el bloque vacío. Fuera de ese caso, "presenta" sigue cortando.
_DISPARADORES_NEGACION = (
    r"no(?!\s+obstante\b)|no\s+presenta|ni|niega|negando|neg[aá]ndose(?!\s+a\b)|neg[oó](?!\s+a\b)|sin(?!\s+embargo\b)"
    r"|ausencia de"
    r"|no refiere|libre de"
    r"|riesgo bajo para|no signos de|sin evidencia de|no clinica de"
    r"|bajo riesgo de|descartad[oa]s?|negativ[oa]s? para"
    rf"|{_DESCARTAR}|{_OTRAS_CLAVES}"
)

# Enumeraciones (#62): "niega HTA, DM y EPOC" niega las tres. Tras el disparador, un
# segundo patrón sigue por una lista de elementos cortos (hasta `_PALABRAS_POR_ELEMENTO`
# palabras) unidos por coma, "y", "e", "o", "u" o "ni". El tramo de siempre (hasta la
# primera coma, "y" u otra barrera) se mantiene como patrón aparte, así el alcance es la
# unión de los dos y una lista corta no recorta un tramo más largo.
_PALABRAS_POR_ELEMENTO = 3
_CONJUNCION = r"(?:y|e|o|u|ni)"
# Una palabra de un elemento no es barrera, conjunción ni un guion suelto: "niega fiebre
# y presenta diabetes" no es una lista, porque "presenta" no puede abrir el segundo
# elemento.
# Tampoco un resultado hecho solo de signos ("+++/+++"), como token entero: un hallazgo
# con su medida no es un elemento de la lista negada (#137). "VIH+", "L4/5" o "NIC 2/3" sí.
_RESULTADO = rf"\++(?:/\++)?(?![^\s{_CORTES}])"
_PALABRA = rf"(?!{_BARRERAS}|{_CONJUNCION}\b|-(?:\s|$)|{_RESULTADO})[^\s{_CORTES}]+"
# Elementos de una palabra ("HTA", "cefalea") y de varias, hasta el máximo
# ("úlcera péptica", "antecedentes de IAM").
_CORTO = _PALABRA
_LARGO = rf"{_PALABRA}(?:[ \t]+{_PALABRA}){{1,{_PALABRAS_POR_ELEMENTO - 1}}}"
# Palabras que no abren un elemento nuevo: "sin antecedentes, salvo gastritis" o
# "no DBT, sí HTA" cambian de tema, no siguen la lista.
_NO_ABREN_ELEMENTO = r"(?:salvo|excepto|s[ií]|que|por|donde)\b"
_SEPARADOR = (
    rf"(?:[ \t]*,[ \t]*(?:{_CONJUNCION}[ \t]+)?|[ \t]+{_CONJUNCION}[ \t]+)(?!{_NO_ABREN_ELEMENTO})"
)
_SOLO_COMA = rf"[ \t]*,[ \t]*(?!{_CONJUNCION}[ \t]|{_NO_ABREN_ELEMENTO})"
# Separador con conjunción ("y", ", y", "ni"...), sin coma sola.
_CON_CONJUNCION = rf"(?:[ \t]*,)?[ \t]+{_CONJUNCION}[ \t]+(?!{_NO_ABREN_ELEMENTO})"
# Tras un elemento de varias palabras, la lista solo sigue por la coma con uno de una
# palabra (#84): en "sin secuelas motoras, demencia vascular leve" la coma separa dos
# cláusulas y la demencia queda confirmada. Dos elementos largos sí van seguidos si los
# une una conjunción: "niega úlcera péptica y enfermedad hepática".
_LARGOS = rf"{_LARGO}(?:{_CON_CONJUNCION}{_LARGO})*"
# El elemento tiene que terminar entero: en un corte (coma, punto, salto de línea, punto
# y coma, dos puntos, paréntesis o guion), fin del texto, conjunción o barrera. Si es
# más largo que el máximo, la lista acaba en el anterior.
_FIN_DE_ELEMENTO = rf"(?=\s*(?:[{_CORTES}]|$)|{_GUION}|[ \t]+{_CONJUNCION}\b|\s+{_BARRERAS})"
# Al menos dos elementos: un corto y unos largos, unos largos (o un corto) y un corto
# (con el límite de #137 si los une una conjunción), o
# dos largos unidos por conjunción. Después, cortos, cada uno seguido o no de largos.
_LISTA = (
    rf"(?:{_CORTO}{_SEPARADOR}{_LARGOS}"
    rf"|{_CORTO}{_SEPARADOR}{_CORTO}(?:{_SEPARADOR}{_LARGOS})?"
    rf"|{_LARGOS}{_SOLO_COMA}{_CORTO}(?:{_SEPARADOR}{_LARGOS})?"
    # Si el corto entró por conjunción ("déficit motor o sensitivo"), la coma no sigue con
    # un largo: es otra cláusula (#137).
    rf"|{_LARGOS}{_CON_CONJUNCION}{_CORTO}(?:{_CON_CONJUNCION}{_LARGOS})?"
    rf"|{_LARGO}(?:{_CON_CONJUNCION}{_LARGO})+)"
    rf"(?:{_SEPARADOR}{_CORTO}(?:{_SEPARADOR}{_LARGOS})?)*{_FIN_DE_ELEMENTO}"
)

# Hacia atrás (#66): "TEP descartado" y "cultivo negativo" niegan solo las
# `_PALABRAS_ATRAS` palabras de antes, no toda la cláusula. La clave es una palabra
# entera ("seronegativa" no niega) y no va tras "coagulasa" ni "gram", que nombran un
# germen y no un resultado. Puede ir tras dos puntos ("VIH: negativo"), aunque los dos
# puntos cortan el alcance hacia adelante. El patrón empieza en el inicio de una
# palabra: así cada intento recorre unas pocas palabras y el costo crece con el largo
# del texto, no con su cuadrado.
_PALABRAS_ATRAS = 4
_PALABRA_ATRAS = rf"(?<![^\s{_CORTES}])(?!-(?:\s|$))[^\s{_CORTES}]+"
_ATRAS = (
    rf"{_PALABRA_ATRAS}(?:[ \t]+{_PALABRA_ATRAS}){{0,{_PALABRAS_ATRAS - 1}}}"
    r"(?:[ \t]*:[ \t]*|[ \t]+)(?<!coagulasa )(?<!gram )\b(?:descartad[oa]s?|negativ[oa]s?)\b"
)

# Modificador de modo tras el disparador (#218): "se niega de forma categórica y
# explícita haber presentado …". Va con el disparador, así su "y" no corta el alcance.
# Solo con "de forma/manera/modo": "rotunda y categóricamente" también atraparía "niega
# HTA y posteriormente se diagnostica DM".
_MODO = r"(?:\s+de\s+(?:forma|manera|modo)\s+[^\W\d_]+(?:\s+y\s+[^\W\d_]+)?)?"

_PATRONES_NEGACION = [
    # El tramo de siempre: hasta el primer corte, guion o barrera.
    re.compile(
        rf"\b(?:{_DISPARADORES_NEGACION})\b{_MODO}\s+(?:(?!{_BARRERAS}|{_GUION})[^{_CORTES}])+",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:{_DISPARADORES_NEGACION})\b{_MODO}\s+{_LISTA}", re.IGNORECASE),
    re.compile(_ATRAS, re.IGNORECASE),
    re.compile(
        r"(?:Alérgicos|Alergia al látex|Antecedentes familiares anestésicos|Toxicológicos"
        r"|Alérgias):\s*no refiere",
        re.IGNORECASE,
    ),
]

_PATRONES_FAMILIARES = [
    # Bloque "Antecedentes familiares: ..." hasta el fin de línea. Empieza en el guion o
    # en la palabra, no en los espacios de antes (#66): con un `\s*` al inicio, cada
    # posición de una racha larga de espacios la recorría entera (costo cuadrático).
    re.compile(r"(?:-\s*)?(?:Antecedentes\s*)?Familiares\s*:.*?(?=\n|$)", re.IGNORECASE),
    # Frase que empieza con un parentesco, hasta la siguiente coma, punto o salto de línea.
    re.compile(
        r"\b(?:madre|padre|herman[oa]s?|tí[oa]s?|ti[oa]s?|abuel[oa]s?|hij[oa]s?"
        r"|primos?|primas?|familiar(?:es)?)\b[^,\.\n]*",
        re.IGNORECASE,
    ),
]


# Incertidumbre (#83): claves que ponen en `posible` lo que va después ("sospecha de
# cáncer", "probable neumonía", "no se descarta IAM"), por grupos. Cada grupo entró
# porque sube el F1 de `posible` en la partición de desarrollo de NUBes sin bajar el de
# `negada` (cifras en el docstring del módulo). Todas son palabras enteras.
CLAVES_POSIBLE: dict[str, str] = {
    # #202: también el pasado y el participio ("se sospechó Parkinson", "sospechada").
    "sospecha": (r"sospech(?:a|an|ar|[oó]|aron|ad[oa]s?|os[oa]s?)(?:\s+cl[ií]nica)?(?:\s+de)?"),
    "probable": r"probables?|probablemente",
    "posible": r"posibles?|posiblemente",
    "compatible con": r"compatibles?\s+con",
    "sugestivo de": r"sugestiv[oa]s?\s+de",
    "orientación diagnóstica": r"orientaci[oó]n\s+diagn[oó]stica\s*:",
    "parece": r"(?:no\s+)?parecen?",
    "sugiere": r"sugieren?|sugiriendo",
    "puede": r"pueden?|puedan?",
    "podría": r"podr[ií]an?",
    "impresiona": r"impresiona(?:\s+de)?",
    "se interpreta": r"se\s+interpret[aóo]\s+(?:como|de)|interpretado\s+como|interpretamos",
    "sin clara": r"sin\s+clar[oa]s?",
    # #136: "antecedente no claro de EPOC" expresa duda. Como clave, su "no" no niega.
    "no claro": r"no\s+clar[oa]s?(?:\s+de)?",
    "posibilidad de": r"posibilidad(?:es)?\s+de",
    "aparente": r"aparentes?|aparentemente",
    "vs": r"vs\.?|versus",
    "dudoso": r"dudos[oa]s?",
    "se plantea": r"(?:se\s+)?plantea|planteando",
    "pudiera": r"pudieran?|pudiendo",
    "al parecer": r"al\s+parecer",
    "a valorar": r"a\s+valorar",
    # #202: "para confirmar o descartar neuropatía" dice qué se busca, como "para
    # descartar"; posible gana sobre el "descartar" que niega.
    "confirmar o descartar": r"confirmar\s+o\s+descartar|descartar\s+o\s+confirmar",
    # Descartar con incertidumbre: el "no" y el "descarta" que tienen dentro no niegan.
    "no se descarta": (
        r"no\s+se\s+(?:puede\s+)?descartar?|no\s+descartamos"
        r"|sin\s+(?:que\s+se\s+pueda\s+)?descartar"
    ),
}
# Hacia atrás, como "TEP descartado": "foco aparente", "AIT vs ITU", "neuralgia ?". El
# signo de interrogación va pegado o no a la palabra ("¿TEP?" y "TEP ?").
CLAVES_POSIBLE_ATRAS: dict[str, str] = {
    "aparente": r"aparentes?",
    "vs": r"vs\.?|versus",
    "dudoso": r"dudos[oa]s?",
    "?": r"\?",
}


def patrones_posible(claves: Iterable[str], claves_atras: Iterable[str]) -> list[re.Pattern[str]]:
    """Los patrones de incertidumbre para unas claves hacia adelante y otras hacia atrás.

    Con las de `CLAVES_POSIBLE` y `CLAVES_POSIBLE_ATRAS` dan `_PATRONES_POSIBLE`; se
    exponen para medir cada grupo por separado.

    Dos claves seguidas abren cada una su alcance: en "pudiera ser compatible con un
    infarto" el tramo de "pudiera" se corta en la barrera "con", y si se comiera
    "compatible" la búsqueda seguiría después sin probar "compatible con". Por eso el
    tramo termina antes de otra clave, y la lista y el patrón hacia atrás van dentro de
    un lookahead (grupo `alcance`), que deja solaparse los alcances. El tramo no va en un
    lookahead: sin cortes, cada clave recorrería el texto hasta el final (costo
    cuadrático); la lista y el patrón hacia atrás miran unas pocas palabras.
    """
    claves, claves_atras = list(claves), list(claves_atras)
    patrones = []
    if claves:
        una_clave = rf"\b(?:{'|'.join(claves)})(?!\w)"
        clave = rf"(?P<clave>{una_clave})"
        patrones += [
            # El tramo y la lista, como en la negación.
            re.compile(
                rf"(?P<alcance>{clave}[ \t]*"
                rf"(?:(?!{_BARRERAS}|{_GUION}|{una_clave})[^{_CORTES}])+)",
                re.IGNORECASE,
            ),
            re.compile(rf"(?=(?P<alcance>{clave}[ \t]*{_LISTA}))", re.IGNORECASE),
        ]
    # "?" no empieza en palabra: las demás claves hacia atrás sí.
    palabras = [c for c in claves_atras if c != r"\?"]
    clave_atras = ([rf"\b(?:{'|'.join(palabras)})(?!\w)"] if palabras else []) + (
        [r"\?"] if r"\?" in claves_atras else []
    )
    if clave_atras:
        patrones.append(
            re.compile(
                rf"(?=(?P<alcance>{_PALABRA_ATRAS}(?:[ \t]+{_PALABRA_ATRAS})"
                rf"{{0,{_PALABRAS_ATRAS - 1}}}[ \t]*(?P<clave>{'|'.join(clave_atras)})))",
                re.IGNORECASE,
            )
        )
    return patrones


_PATRONES_POSIBLE = patrones_posible(CLAVES_POSIBLE.values(), CLAVES_POSIBLE_ATRAS.values())


PRIORIDAD: tuple[Estado, ...] = (Estado.FAMILIAR, Estado.POSIBLE, Estado.NEGADA)
"""Estados que puede dar un intervalo, del que más manda al que menos."""


class Intervalo(NamedTuple):
    """Tramo `[inicio, fin)` del texto que pone a sus menciones en `estado`."""

    inicio: int
    fin: int
    estado: Estado


def _tramos(patrones: list[re.Pattern[str]], texto: str, estado: Estado) -> list[Intervalo]:
    return [
        Intervalo(m.start(), m.end(), estado)
        for patron in patrones
        for m in patron.finditer(texto)
        if m.end() > m.start()
    ]


class _Tramos:
    """Tramos `[inicio, fin)` ordenados por inicio, para preguntar por posición.

    Con búsqueda binaria: un texto con miles de claves y negaciones no compara cada
    clave con cada negación.
    """

    def __init__(self, tramos: Iterable[tuple[int, int]]) -> None:
        ordenados = sorted(tramos)
        self.inicios = [inicio for inicio, _ in ordenados]
        # El fin más lejano entre los tramos hasta cada posición de la lista.
        self.fin_maximo = list(accumulate((fin for _, fin in ordenados), max))

    def cubren(self, posicion: int, desde_antes: bool = False) -> bool:
        """Si algún tramo contiene `posicion`; con `desde_antes`, uno que empiece antes."""
        hasta = (bisect_left if desde_antes else bisect_right)(self.inicios, posicion)
        return hasta > 0 and self.fin_maximo[hasta - 1] > posicion

    def primer_inicio(self, desde: int, hasta: int) -> int | None:
        """El primer inicio de tramo en `[desde, hasta)`, o `None`."""
        i = bisect_left(self.inicios, desde)
        return self.inicios[i] if i < len(self.inicios) and self.inicios[i] < hasta else None


def _posibles(alcances: list[re.Match[str]], negados: list[Intervalo]) -> list[Intervalo]:
    """Los alcances de las claves de incertidumbre, con las reglas frente a la negación
    (ver el docstring del módulo)."""
    tramos_negados = _Tramos((n.inicio, n.fin) for n in negados)
    salida = []
    for m in alcances:
        inicio, fin = m.span("alcance")
        clave_inicio, clave_fin = m.span("clave")
        # Una clave hacia adelante dentro de un alcance negado está negada ella misma:
        # "sin datos sugestivos de X" niega X. Hacia atrás no: en "sin foco aparente" la
        # clave matiza la negación.
        estado = Estado.POSIBLE
        if inicio == clave_inicio and tramos_negados.cubren(clave_inicio, desde_antes=True):
            estado = Estado.NEGADA
        # El alcance se corta donde empieza una negación: en "probable neumonía sin
        # derrame" el derrame queda negado.
        corte = tramos_negados.primer_inicio(clave_fin, fin)
        fin = fin if corte is None else corte
        if fin > inicio:
            salida.append(Intervalo(inicio, fin, estado))
    return salida


# Claves mal escritas (#126): "niegh HTA", "majre con DM2". Solo palabras largas, con la
# primera letra bien y a distancia 1 de Damerau-Levenshtein de una clave; las palabras
# reales que quedan cerca ("nieta", "parte", "descarga") no se tocan. Sin ambigüedad: si
# queda cerca de dos claves, no se corrige.
_CLAVES_ESCRITAS = frozenset(
    {
        "niega", "niegan", "descarta", "descartan", "descartado", "descartada",
        "sospecha", "probable", "posible", "madre", "padre",
        "hermano", "hermana", "abuelo", "abuela",
    }
)  # fmt: skip
_PALABRAS_REALES = frozenset(
    {
        "nieta", "niego", "niegue", "nieva", "siega", "riega",
        "descarte", "descartes", "descartar", "descarto", "descartando", "descartaron",
        "sospechar", "sospecho", "sospechas", "sospeche",
        "probables", "posibles", "posibilidad",
        "madres", "padres", "pardo", "parda", "parte", "parto", "pare", "podre",
        "mare", "made", "madera", "madure", "pudre", "marre", "maure",
        "descarga", "descargan", "descargado", "descargada", "descarada", "descarado",
        "descarnada", "descastada", "descastado", "descartaba", "descartara",
        "aquela", "aquelo", "azuela", "asuelo", "hermanal", "hermanar", "pasible",
        "hermanos", "hermanas", "abuelos", "abuelas",
    }
)  # fmt: skip
_MISMA_CLAVE = {
    "niegan": "niega",
    "descartan": "descarta",
    "descartada": "descartado",
    "hermana": "hermano",
    "abuela": "abuelo",
}
_PALABRA_DE_CLAVE = re.compile(r"(?<![^\W\d_])[^\W\d_]{4,11}(?![^\W\d_])")


def _clave_corregida(palabra: str) -> str | None:
    """La clave que `palabra` escribe mal, o `None`. Sin caché: no se retienen palabras de
    las notas entre análisis."""
    limpia = "".join(
        c for c in unicodedata.normalize("NFD", palabra.lower()) if unicodedata.category(c) != "Mn"
    )
    if limpia in _CLAVES_ESCRITAS or limpia in _PALABRAS_REALES:
        return None
    cercanas = [
        clave
        for clave in _CLAVES_ESCRITAS
        if clave[0] == limpia[0]
        and abs(len(clave) - len(limpia)) <= 1
        and DamerauLevenshtein.distance(limpia, clave, score_cutoff=1) <= 1
    ]
    # Variantes de la misma clave ("hermanp" está cerca de hermano y hermana) no son ambiguas.
    canonicas = {_MISMA_CLAVE.get(c, c) for c in cercanas}
    return canonicas.pop() if len(canonicas) == 1 else None


def _corregir_claves(texto: str) -> tuple[str, list[int] | None]:
    """`texto` con las claves mal escritas corregidas y, si cambió, el mapa de cada posición
    del texto corregido a la del original (con una posición de más, para los finales)."""
    partes: list[str] = []
    mapa: list[int] = []
    previo = 0
    hubo = False
    for m in _PALABRA_DE_CLAVE.finditer(texto):
        clave = _clave_corregida(m.group())
        if clave is None:
            continue
        hubo = True
        partes.append(texto[previo : m.start()])
        mapa.extend(range(previo, m.start()))
        partes.append(clave)
        largo = len(m.group())
        mapa.extend(m.start() + min(i, largo - 1) for i in range(len(clave)))
        previo = m.end()
    if not hubo:
        return texto, None
    partes.append(texto[previo:])
    mapa.extend(range(previo, len(texto) + 1))
    return "".join(partes), mapa


def intervalos(texto: str) -> list[Intervalo]:
    """Los bloques de familiares y los alcances posibles y negados de `texto`, con su estado.

    Primero los familiares, después los alcances de las claves de incertidumbre (posibles,
    o negados si la clave está negada) y al final los negados, cada grupo en el orden de
    los patrones. Los intervalos pueden solaparse; la prioridad la resuelve
    `clasificar_mencion`. Una negación que empieza dentro de una clave de incertidumbre
    ("no se descarta", "sin clara") es parte de la clave y no se devuelve. Las claves mal
    escritas se corrigen antes (#126), y las posiciones son siempre las de `texto`.
    """
    corregido, mapa = _corregir_claves(texto)
    tramos = _intervalos(corregido)
    if mapa is None:
        return tramos
    return [Intervalo(mapa[i.inicio], mapa[i.fin], i.estado) for i in tramos]


def _intervalos(texto: str) -> list[Intervalo]:
    alcances = [m for patron in _PATRONES_POSIBLE for m in patron.finditer(texto)]
    claves = _Tramos(m.span("clave") for m in alcances)
    negados = [
        n for n in _tramos(_PATRONES_NEGACION, texto, Estado.NEGADA) if not claves.cubren(n.inicio)
    ]
    return (
        _tramos(_PATRONES_FAMILIARES, texto, Estado.FAMILIAR)
        + _posibles(alcances, negados)
        + negados
    )


def _dentro(mencion: Mencion, intervalo: Intervalo) -> bool:
    # Una mención vacía no tiene nada que clasificar, aunque caiga dentro.
    if mencion.inicio == mencion.fin:
        return False
    return intervalo.inicio <= mencion.inicio < intervalo.fin


def clasificar_mencion(mencion: Mencion, intervalos_texto: Iterable[Intervalo]) -> Estado:
    """Estado de la mención según los intervalos donde cae.

    Prioridad: familiar > posible > negada > confirmada.

    La mención cae en un intervalo si su inicio está dentro de él, aunque termine
    después (ver el docstring del módulo). Una mención vacía no cae en ninguno y queda
    confirmada. `mencion` e `intervalos_texto` tienen que estar en posiciones del mismo
    texto.
    """
    tocados = {i.estado for i in intervalos_texto if _dentro(mencion, i)}
    for estado in PRIORIDAD:
        if estado in tocados:
            return estado
    return Estado.CONFIRMADA
