"""Correcciones a las reglas de Charlson heredadas de la tesis (issue #24, parte A).

Cada caso reproduce un error de las regex portadas sin cambios y fija el resultado
correcto. Todos los textos son sintéticos.
"""

import pytest

from ner_clinic.charlson.reglas import mapear
from ner_clinic.tipos import CategoriaCharlson as C

CASOS = [
    # `ñ` en la regex: tras normalizar el texto ya no la tiene
    pytest.param("infarto agudo de miocardio en cuña", None, id="iam-cuna"),
    pytest.param("acuñamiento vertebral", None, id="acv-acunamiento"),
    pytest.param("diabetes sin daño renal", (C.DIABETES, "E14.9"), id="dm-sin-dano"),
    # Negación sin `\b` inicial: el "no" de "maligno", el "ni" de "mansoni"
    pytest.param(
        "tumor maligno metastasico de colon",
        (C.TUMOR_METASTASICO, "C80"),
        id="metastasis-maligno",
    ),
    pytest.param(
        "infeccion por schistosoma mansoni con hipertension portal",
        (C.HEPATICA_SEVERA, "K76.6"),
        id="hepatica-severa-mansoni",
    ),
    pytest.param(
        "coinfeccion por schistosoma mansoni y vih",
        (C.VIH_SIDA, "B24"),
        id="vih-mansoni",
    ),
    # ACV: siglas de arterias sin `\b` y orden I63 frente a I65/I66
    pytest.param("placa de ateroma carotidea", (C.CEREBROVASCULAR, "I65"), id="acv-placa"),
    pytest.param("estenosis de acm izquierda", (C.CEREBROVASCULAR, "I66"), id="acv-estenosis-acm"),
    pytest.param("oclusion de aca", (C.CEREBROVASCULAR, "I66"), id="acv-oclusion-aca"),
    # ICC: `les\b` coincidía con "normales"
    pytest.param(
        "falla cardiaca con fevi y diametros normales",
        (C.INSUFICIENCIA_CARDIACA, "I50"),
        id="icc-normales",
    ),
    # Pulmonar: plural y la hipertensión pulmonar severa de la subclasificación
    pytest.param("bronquiectasias", (C.PULMONAR_CRONICA, "J47"), id="pulmonar-plural"),
    pytest.param(
        "hipertension pulmonar severa", (C.PULMONAR_CRONICA, "I27.9"), id="pulmonar-hp-severa"
    ),
    # Reumatológica: `l\.e\.s\.` seguido de `\b`
    pytest.param("l.e.s. activo", (C.REUMATOLOGICA, "M32"), id="reumato-les"),
    # EVP: "aneurisma de aorta" sin "la"
    pytest.param("aneurisma de aorta", (C.VASCULAR_PERIFERICA, "I71.9"), id="evp-aneurisma"),
    # Metástasis óseas en femenino plural
    pytest.param("metastasis oseas", (C.TUMOR_METASTASICO, "C79"), id="metastasis-oseas"),
    # Monoplejía: `ms|mi` sin `\b` coincidían dentro de "mielopatia"
    pytest.param(
        "monoparesia secundaria a mielopatia", (C.HEMIPLEJIA, "G83.3"), id="monoparesia-mi"
    ),
    pytest.param("monoparesia de msd", (C.HEMIPLEJIA, "G83.2"), id="monoparesia-msd"),
    pytest.param("monoplejia de mii", (C.HEMIPLEJIA, "G83.1"), id="monoplejia-mii"),
    # Diabetes sin tipo con coma o cetoacidosis: E14.0 / E14.1 (no E11)
    pytest.param("diabetes en coma", (C.DIABETES, "E14.0"), id="dm-coma"),
    pytest.param("cetoacidosis diabetica", (C.DIABETES, "E14.1"), id="dm-cetoacidosis"),
    # Errata `diab[eé]tical?`: debía ser `diab[eé]tic[oa]s?`
    pytest.param(
        "neuropatias diabeticas", (C.DIABETES_COMPLICADA, "E11.4"), id="dm-complicada-plural"
    ),
]


@pytest.mark.parametrize(("texto", "esperado"), CASOS)
def test_correccion(texto, esperado):
    assert mapear(texto) == esperado


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        # Las correcciones no rompen los casos que ya funcionaban
        ("infarto en territorio de acm", (C.CEREBROVASCULAR, "I63")),
        ("estenosis carotidea con infarto cerebral", (C.CEREBROVASCULAR, "I63")),
        ("estenosis carotidea", (C.CEREBROVASCULAR, "I65")),
        ("lupus eritematoso sistemico", (C.REUMATOLOGICA, "M32")),
        ("sin metastasis", None),
        ("vacuna posterior a infarto agudo de miocardio", (C.INFARTO_MIOCARDIO, "I21")),
        ("diabetes tipo 2 con coma", (C.DIABETES, "E11.0")),
    ],
)
def test_sin_regresion(texto, esperado):
    assert mapear(texto) == esperado


# --- Negaciones sin `\b` inicial en las demás reglas --------------------------
# Todas las negaciones van seguidas de `\s+`, así que solo fallan cuando una palabra
# termina en la negación: "año" queda como "ano" al normalizar y su "no" se leía como
# "no se evidencia"; el "no" de "daño" borraba la etiología de la ICC.

NEGACION_DENTRO_DE_PALABRA = [
    pytest.param(
        "ecocardiograma de este año evidencia falla cardiaca",
        (C.INSUFICIENCIA_CARDIACA, "I50"),
        id="icc",
    ),
    pytest.param(
        "insuficiencia cardiaca por daño isquemico",
        (C.INSUFICIENCIA_CARDIACA, "I25.5"),
        id="icc-subclasificacion",
    ),
    pytest.param(
        "este año se evidencia ateromatosis aortica", (C.VASCULAR_PERIFERICA, "I70.9"), id="evp"
    ),
    pytest.param(
        "tac de este año se evidencia infarto cerebral", (C.CEREBROVASCULAR, "I63"), id="acv"
    ),
    pytest.param(
        "espirometria de este año se evidencia epoc", (C.PULMONAR_CRONICA, "J44"), id="pulmonar"
    ),
    pytest.param(
        "consulta de este año se evidencia lupus", (C.REUMATOLOGICA, "M32"), id="reumatologica"
    ),
    pytest.param(
        "endoscopia de este año se evidencia ulcera gastrica",
        (C.ULCERA_PEPTICA, "K25"),
        id="ulcera",
    ),
    pytest.param(
        "ecografia de este año se evidencia cirrosis", (C.HEPATICA_LEVE, "K74"), id="hepatica-leve"
    ),
    pytest.param(
        "tamizaje de este año se evidencia diabetes mellitus", (C.DIABETES, "E14.9"), id="diabetes"
    ),
    pytest.param(
        "control de este año se evidencia nefropatia diabetica",
        (C.DIABETES_COMPLICADA, "E11.2"),
        id="diabetes-complicada",
    ),
    pytest.param(
        "examen de este año se evidencia hemiparesia derecha",
        (C.HEMIPLEJIA, "G81"),
        id="hemiplejia",
    ),
    pytest.param(
        "control de este año se evidencia erc estadio 3", (C.RENAL_CRONICA, "N18"), id="renal"
    ),
    pytest.param(
        "biopsia de este año se evidencia adenocarcinoma de colon",
        (C.TUMOR_MALIGNO, "C18"),
        id="tumor",
    ),
]


@pytest.mark.parametrize(("texto", "esperado"), NEGACION_DENTRO_DE_PALABRA)
def test_negacion_dentro_de_otra_palabra_no_excluye(texto, esperado):
    assert mapear(texto) == esperado


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("no se evidencia falla cardiaca", None),
        ("insuficiencia cardiaca no isquemica", (C.INSUFICIENCIA_CARDIACA, "I50")),
        ("no se evidencia ateromatosis aortica", None),
        ("sin infarto cerebral", None),
        ("sin epoc", None),
        ("sin lupus", None),
        ("sin ulcera gastrica", None),
        ("sin cirrosis", None),
        ("niega diabetes", None),
        ("dm2 sin complicacion renal", (C.DIABETES, "E11.9")),
        ("sin hemiparesia", None),
        ("niega erc", None),
        ("no hay adenocarcinoma de colon", None),
    ],
)
def test_negacion_real_sigue_excluyendo(texto, esperado):
    assert mapear(texto) == esperado


# --- Localización ósea del cáncer y estenosis basilar --------------------------


def test_neoplasia_osea_da_c41():
    assert mapear("neoplasia maligna osea") == (C.TUMOR_MALIGNO, "C41")


def test_estenosis_basilar_es_precerebral():
    assert mapear("estenosis basilar") == (C.CEREBROVASCULAR, "I65")


# --- Huecos de las reglas (#123, #120, #63) -----------------------------------

HUECOS = [
    # #123: cerebrovascular escrito completo
    pytest.param("accidente cerebrovascular", (C.CEREBROVASCULAR, "I64"), id="acv-completo"),
    pytest.param(
        "accidente cerebrovascular isquémico", (C.CEREBROVASCULAR, "I63"), id="acv-isquemico"
    ),
    pytest.param("enfermedad cerebrovascular", (C.CEREBROVASCULAR, "I64"), id="ecv-completo"),
    pytest.param("evento cerebrovascular", (C.CEREBROVASCULAR, "I64"), id="evento-cv"),
    pytest.param("evento vascular cerebral", (C.CEREBROVASCULAR, "I64"), id="evento-vascular"),
    pytest.param("ictus", (C.CEREBROVASCULAR, "I64"), id="ictus"),
    pytest.param("derrame cerebral", (C.CEREBROVASCULAR, "I64"), id="derrame-cerebral"),
    pytest.param("apoplejía", (C.CEREBROVASCULAR, "I64"), id="apoplejia"),
    pytest.param("ictus isquémico", (C.CEREBROVASCULAR, "I63"), id="ictus-isquemico"),
    pytest.param("ACV hemorrágico", (C.CEREBROVASCULAR, "I61"), id="acv-hemorragico"),
    pytest.param(
        "secuelas de accidente cerebrovascular", (C.CEREBROVASCULAR, "I69"), id="acv-secuela"
    ),
    # #123 y #63: vascular periférica
    pytest.param(
        "arteriopatía periférica obliterante",
        (C.VASCULAR_PERIFERICA, "I73.9"),
        id="arteriopatia-periferica",
    ),
    pytest.param(
        "insuficiencia arterial crónica de miembros inferiores",
        (C.VASCULAR_PERIFERICA, "I73.9"),
        id="insuficiencia-arterial-cronica",
    ),
    # #123 y #63: pulmonar crónica
    pytest.param("fibrosis pulmonar idiopática", (C.PULMONAR_CRONICA, "J84"), id="fibrosis"),
    pytest.param("enfermedad pulmonar intersticial difusa", (C.PULMONAR_CRONICA, "J84"), id="epid"),
    # #120: tumor sin metástasis
    pytest.param(
        "Linfoma de Hodgkin sin metástasis", (C.TUMOR_MALIGNO, "C81"), id="linfoma-sin-mtx"
    ),
    pytest.param(
        "cáncer de mama sin evidencia de metástasis",
        (C.TUMOR_MALIGNO, "C50"),
        id="mama-sin-evidencia-mtx",
    ),
    pytest.param("cáncer de próstata metastásico", (C.TUMOR_METASTASICO, "C80"), id="prostata-mtx"),
    pytest.param(
        "metástasis hepáticas de cáncer de colon",
        (C.TUMOR_METASTASICO, "C78"),
        id="colon-mtx-hepaticas",
    ),
    # #63: hepatopatía severa y leve
    pytest.param(
        "cirrosis hepática Child-Pugh C", (C.HEPATICA_SEVERA, "K72.1"), id="cirrosis-child-c"
    ),
    pytest.param(
        "cirrosis hepática con ascitis refractaria",
        (C.HEPATICA_SEVERA, "K76.6"),
        id="cirrosis-ascitis",
    ),
    pytest.param("cirrosis descompensada", (C.HEPATICA_SEVERA, "K72.1"), id="cirrosis-descomp"),
    pytest.param("cirrosis hepática Child-Pugh A", (C.HEPATICA_LEVE, "K74"), id="cirrosis-child-a"),
    pytest.param("cirrosis sin ascitis", (C.HEPATICA_LEVE, "K74"), id="cirrosis-sin-ascitis"),
    pytest.param("cirrosis compensada", (C.HEPATICA_LEVE, "K74"), id="cirrosis-compensada"),
    pytest.param("hepatitis alcohólica aguda", (C.HEPATICA_LEVE, "K70.1"), id="hepatitis-oh"),
]


@pytest.mark.parametrize(("texto", "esperado"), HUECOS)
def test_huecos_de_las_reglas(texto, esperado):
    assert mapear(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "sin antecedente de accidente cerebrovascular",
        "niega ictus",
        "sin metastasis",
        "riesgo cardiovascular alto",
        "insuficiencia venosa crónica de miembros inferiores",
        "fibrosis hepática",
    ],
)
def test_huecos_sin_falsos_positivos(texto):
    resultado = mapear(texto)
    assert resultado is None or resultado[0] not in (
        C.CEREBROVASCULAR,
        C.VASCULAR_PERIFERICA,
        C.PULMONAR_CRONICA,
        C.TUMOR_MALIGNO,
    )


# Revisión del PR #183: lo que la primera versión rompía o mapeaba de más
REVISION_183 = [
    # La negación de la ascitis no tumba los marcadores severos
    ("cirrosis con varices esofagicas sin ascitis", (C.HEPATICA_SEVERA, "I85.9")),
    ("cirrosis descompensada por varices esofagicas sin ascitis", (C.HEPATICA_SEVERA, "I85.9")),
    ("hipertension portal sin ascitis", (C.HEPATICA_SEVERA, "K76.6")),
    ("encefalopatia hepatica sin ascitis", (C.HEPATICA_SEVERA, "K72.1")),
    ("insuficiencia hepatica, no descompensada actualmente", (C.HEPATICA_SEVERA, "K72.9")),
    ("cirrosis no descompensada", (C.HEPATICA_LEVE, "K74")),
    # "descompensada" de otro órgano
    (
        "insuficiencia cardiaca descompensada con antecedente de cirrosis",
        (C.INSUFICIENCIA_CARDIACA, "I50"),
    ),
    # ACV isquémico con transformación hemorrágica sigue siendo I63
    ("infarto cerebral con transformacion hemorragica", (C.CEREBROVASCULAR, "I63")),
    (
        "accidente cerebrovascular isquemico con transformacion hemorragica",
        (C.CEREBROVASCULAR, "I63"),
    ),
    ("evento cerebrovascular isquemico, no hemorragico", (C.CEREBROVASCULAR, "I63")),
    # Metástasis negada después o con más palabras
    ("carcinoma de mama, metastasis descartadas", (C.TUMOR_MALIGNO, "C50")),
    (
        "carcinoma de prostata sin datos clinicos ni radiologicos de metastasis",
        (C.TUMOR_MALIGNO, "C61"),
    ),
    ("carcinoma de mama sin micrometastasis", (C.TUMOR_MALIGNO, "C50")),
    ("gammagrafia negativa para metastasis", None),
]


@pytest.mark.parametrize(("texto", "esperado"), REVISION_183)
def test_revision_del_pr_183(texto, esperado):
    assert mapear(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "apoplejia hipofisaria",
        "enfermedad intersticial renal",
        "nefritis intersticial",
        "enfermedad intersticial",
        "fibrosis pulmonar descartada",
        "sin fibrosis pulmonar",
        "riesgo de evento cerebrovascular",
        "profilaxis de ictus",
        "no ictus",
        "ictus descartado",
        "codigo ictus negativo",
        "embolia cerebral descartada",
        "paciente con ascitis, se descarta cirrosis",
    ],
)
def test_revision_del_pr_183_sin_categoria_afirmada(texto):
    resultado = mapear(texto)
    assert resultado is None or resultado[0] not in (
        C.CEREBROVASCULAR,
        C.PULMONAR_CRONICA,
        C.HEPATICA_SEVERA,
    )
