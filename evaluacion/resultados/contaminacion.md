# Chequeo de contaminación (FR6)

Pregunta: ¿el modelo `BSC-NLP4BIA/bsc-bio-ehr-es-distemist` vio la partición de prueba de
DisTEMIST durante su entrenamiento? Si la vio, las cifras de la prueba serían optimistas.

## Cifras

Modelo crudo (`aggregation_strategy="simple"`, ventana deslizante, sin pipeline), micro a
nivel de mención, con IC del 95 % por bootstrap sobre documentos. Datos en
`distemist_test.json` y `distemist_train.json`.

| Conjunto | Documentos | Menciones gold | P estricta | R estricta | F1 estricta (IC 95 %) | F1 laxa (IC 95 %) |
|---|---|---|---|---|---|---|
| Ficha del modelo (DisTEMIST, partición no dicha) | — | — | 0,754 | 0,759 | 0,757 | — |
| DisTEMIST prueba (`test_annotated`) | 250 | 2.598 | 0,720 | 0,736 | 0,728 (0,709–0,746) | 0,877 (0,867–0,888) |
| DisTEMIST entrenamiento (`training`) | 750 | 8.065 | 0,860 | 0,869 | 0,864 (0,854–0,874) | 0,942 (0,938–0,947) |

## Qué dicen las fuentes

- La ficha de Hugging Face dice que el modelo es "a fine-tuned version of the
  bsc-bio-ehr-es model on the DisTEMIST corpus (original Spanish Gold Standard)" y reporta
  P 0,754, R 0,759 y F1 0,757 sobre "DisTEMIST", sin decir qué particiones usó para
  entrenar ni para evaluar
  (<https://huggingface.co/BSC-NLP4BIA/bsc-bio-ehr-es-distemist>). Como referencia de la
  tarea cita a Miranda-Escalada et al., *Overview of DisTEMIST at BioASQ*, CEUR Workshop
  Proceedings 3180, 2022.
- La copia del IIC con el mismo nombre (`IIC/bsc-bio-ehr-es-distemist`) declara
  `bigbio/distemist` como datos de entrenamiento y un F1 de 0,759 etiquetado como
  "test set" (<https://huggingface.co/IIC/bsc-bio-ehr-es-distemist>; García Subies et
  al., *A comparative analysis of Spanish Clinical encoder-based models on NER and
  classification tasks*, JAMIA 2024, doi:10.1093/jamia/ocae054).
- Ninguna de las dos fuentes dice explícitamente que la prueba quedó fuera del
  entrenamiento, pero las dos reportan una cifra de prueba del orden de 0,76.

## Conclusión

**Se asume que el modelo no vio la partición de prueba.** El F1 estricto en la prueba
(0,728) está 13,6 puntos por debajo del de entrenamiento (0,864), y los intervalos no se
tocan. El patrón es el de un modelo ajustado sobre `training` y evaluado sobre una
partición que no vio. Si la prueba hubiera entrado al entrenamiento, las dos cifras
serían casi iguales.

La prueba queda 2,9 puntos por debajo de la ficha (0,728 contra 0,757), fuera de nuestro
IC. Lo más probable es una diferencia de método y no de datos. La ficha no dice cómo midió
(lo usual es `seqeval` sobre etiquetas BIO con el texto ya tokenizado en palabras, como
recomienda la misma ficha). Aquí medimos spans de caracteres sobre el texto completo, con
ventana deslizante y `aggregation_strategy="simple"`, y un span que el modelo corta a
media palabra cuenta como error estricto.
Esta diferencia no cambia la conclusión, porque una prueba contaminada daría más que la
ficha, no menos.

Con esto, las cifras de prueba de `distemist_test.json` se pueden reportar como una
estimación honesta para textos del mismo tipo (casos clínicos publicados), y las
conclusiones de la ablación son comparaciones relativas sobre la misma partición.
