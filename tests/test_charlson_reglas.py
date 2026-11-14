"""Tests de las reglas de Charlson, parte 1 (IAM a enfermedad hepática leve).

Todos los textos son sintéticos.
"""

import pytest

from ner_clinic.charlson import reglas
from ner_clinic.charlson.reglas import REGLAS, REGLAS_PARTE1, Regla, mapear, normalizar
from ner_clinic.tipos import CategoriaCharlson as C


def _regla(categoria: C) -> Regla:
    return next(r for r in REGLAS if r.categoria == categoria)


def _no_es(texto: str, categoria: C) -> bool:
    """El texto coincide con la inclusión de `categoria`, pero su exclusión lo descarta.

    Otra regla posterior puede quedarse con el texto; la de `categoria` no.
    """
    regla = _regla(categoria)
    texto_normalizado = normalizar(texto)
    assert regla.inclusion.search(texto_normalizado), "el texto no coincide con la inclusión"
    assert regla.exclusion.search(texto_normalizado), "el texto no coincide con la exclusión"
    resultado = mapear(texto)
    return resultado is None or resultado[0] != categoria


# --- Estructura común ---------------------------------------------------------


def test_normalizar_quita_tildes_mayusculas_control_y_espacios():
    assert normalizar("  Infarto\tAGUDO \x07de  Miocárdio\n") == "infarto agudo de miocardio"


def test_normalizar_convierte_enie_en_n():
    assert normalizar("Daño hepático") == "dano hepatico"


def test_parte1_son_las_9_primeras_categorias_en_orden_de_la_tesis():
    assert [r.categoria for r in REGLAS_PARTE1] == list(C)[:9]


def test_reglas_empieza_por_la_parte1():
    assert REGLAS[:9] == REGLAS_PARTE1


@pytest.mark.parametrize("texto", ["", "   ", "hipotiroidismo", "dolor abdominal"])
def test_sin_coincidencia_devuelve_none(texto):
    assert mapear(texto) is None


def test_regla_aplica_exige_inclusion_y_no_exclusion():
    iam = _regla(C.INFARTO_MIOCARDIO)
    assert iam.aplica("infarto agudo de miocardio")
    assert not iam.aplica("infarto cerebral agudo")
    assert not iam.aplica("hipertension arterial")


def test_gana_la_primera_regla_en_orden():
    # "infarto cerebral agudo" coincide con la inclusión del IAM, pero su exclusión
    # lo deja pasar a la regla cerebrovascular, que va después.
    assert mapear("infarto cerebral agudo") == (C.CEREBROVASCULAR, "I63")


# --- Infarto de miocardio -----------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    ["infarto agudo de miocardio", "IAMCEST anterior", "sindrome coronario agudo", "IAM"],
)
def test_iam_positivo(texto):
    assert mapear(texto) == (C.INFARTO_MIOCARDIO, "I21")


@pytest.mark.parametrize("texto", ["infarto pulmonar agudo", "infarto renal agudo"])
def test_iam_exclusion(texto):
    assert _no_es(texto, C.INFARTO_MIOCARDIO)


# --- Insuficiencia cardiaca ---------------------------------------------------


def test_icc_positivo():
    assert mapear("insuficiencia cardiaca congestiva") == (C.INSUFICIENCIA_CARDIACA, "I50")


@pytest.mark.parametrize(
    "texto", ["sin evidencia de falla cardiaca", "descartar insuficiencia cardiaca"]
)
def test_icc_exclusion(texto):
    assert _no_es(texto, C.INSUFICIENCIA_CARDIACA)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("insuficiencia cardiaca neonatal", "P29.0"),
        ("insuficiencia cardiaca hipertensiva con enfermedad renal terminal", "I13.2"),
        ("insuficiencia cardiaca hipertensiva con nefropatia", "I13.0"),
        ("miocardiopatia por quimioterapia", "I42.7"),
        ("miocardiopatia isquemica", "I25.5"),
        ("cardiopatia isquemica dilatada", "I25.5"),
        ("insuficiencia cardiaca por cardiopatia hipertensiva", "I11.0"),
        ("miocardiopatia chagasica", "I43"),
        ("miocardiopatia reumatica", "I09.9"),
        ("miocardiopatia alcoholica", "I42.6"),
        ("miocardiopatia dilatada", "I42.0"),
        ("miocardiopatia restrictiva", "I42.5"),
        ("miocardiopatia", "I42.9"),
        ("falla cardiaca", "I50"),
        # La negación se borra antes de subclasificar: no cae en I25.5
        ("insuficiencia cardiaca no isquemica", "I50"),
    ],
)
def test_icc_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.INSUFICIENCIA_CARDIACA, cie10)


@pytest.mark.parametrize(
    "texto",
    [
        "cardiopatia isquemica",
        "cardiopatia isquemica cronica asintomatica",
        "cardiopatia hipertensiva",
    ],
)
def test_cardiopatia_isquemica_o_hipertensiva_sin_falla_no_es_icc(texto):
    # Quan: I25.9 e I11.9 no cuentan; sí I25.5 e I11.0 (#201).
    assert mapear(texto) is None


# --- Enfermedad vascular periférica -------------------------------------------


def test_evp_positivo():
    assert mapear("enfermedad arterial periferica") == (C.VASCULAR_PERIFERICA, "I73.9")


@pytest.mark.parametrize("texto", ["aterosclerosis coronaria", "ateromatosis carotidea"])
def test_evp_exclusion(texto):
    assert _no_es(texto, C.VASCULAR_PERIFERICA)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("isquemia mesenterica", "K55.9"),
        ("aneurisma aortico abdominal", "I71.9"),
        ("tromboangeitis obliterante", "I73.1"),
        ("bypass vascular femoropopliteo", "Z95.8"),
        ("aterosclerosis aortica", "I70.9"),
        ("claudicacion intermitente", "I73.9"),
        ("estenosis arterial femoral", "I77.1"),
        ("enfermedad arterial periferica", "I73.9"),
    ],
)
def test_evp_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.VASCULAR_PERIFERICA, cie10)


# --- Enfermedad cerebrovascular -----------------------------------------------


def test_acv_positivo():
    assert mapear("acv isquemico") == (C.CEREBROVASCULAR, "I63")


@pytest.mark.parametrize("texto", ["fractura vertebral", "sin evidencia de acv"])
def test_acv_exclusion(texto):
    assert _no_es(texto, C.CEREBROVASCULAR)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("secuelas de acv", "I69"),
        ("hemorragia subaracnoidea (hsa)", "I60"),
        ("hemorragia intraparenquimatosa cerebral", "I61"),
        ("hematoma subdural hemisferico", "I62"),
        ("oclusion de arteria central retiniana", "H34.0"),
        ("ait", "G45"),
        ("leucoencefalopatia periventricular", "I67"),
        ("infarto cerebral", "I63"),
        ("estenosis carotidea", "I65"),
        ("estenosis arterial cerebral", "I66"),
        ("acv", "I64"),
    ],
)
def test_acv_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.CEREBROVASCULAR, cie10)


# --- Demencia -----------------------------------------------------------------


def test_demencia_positivo():
    assert mapear("demencia") == (C.DEMENCIA, "F03")


@pytest.mark.parametrize(
    "texto", ["deterioro cognitivo con alzas termicas", "trastorno neurocognitivo e hipotonia"]
)
def test_demencia_exclusion(texto):
    assert _no_es(texto, C.DEMENCIA)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("delirium sobre demencia", "F05.1"),
        ("enfermedad de alzheimer", "F00"),
        ("demencia vascular", "F01"),
        ("demencia por cuerpos de lewy", "F02"),
        ("degeneracion senil", "G31.1"),
        ("deterioro cognitivo", "F03"),
    ],
)
def test_demencia_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.DEMENCIA, cie10)


# --- Enfermedad pulmonar crónica ----------------------------------------------


def test_pulmonar_positivo():
    assert mapear("epoc") == (C.PULMONAR_CRONICA, "J44")


@pytest.mark.parametrize("texto", ["neumonia en paciente con epoc", "sin asma"])
def test_pulmonar_exclusion(texto):
    assert _no_es(texto, C.PULMONAR_CRONICA)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("cor pulmonale", "I27.9"),
        ("bronquiectasia", "J47"),
        ("epoc oxigenorequiriente", "J44"),
        ("asma", "J45"),
        ("enfisema", "J43"),
        ("silicosis", "J64"),
        ("bronquitis cronica", "J42"),
        ("enfermedad pulmonar obstructiva", "J44"),
    ],
)
def test_pulmonar_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.PULMONAR_CRONICA, cie10)


# --- Enfermedad reumatológica -------------------------------------------------


def test_reumatologica_positivo():
    assert mapear("artritis reumatoide") == (C.REUMATOLOGICA, "M06")


@pytest.mark.parametrize("texto", ["madre con lupus", "artrosis y artritis reumatoide"])
def test_reumatologica_exclusion(texto):
    assert _no_es(texto, C.REUMATOLOGICA)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("lupus eritematoso sistemico", "M32"),
        ("artritis reumatoide seropositiva", "M05"),
        ("artritis reumatoide", "M06"),
        ("esclerodermia", "M34"),
        ("dermatomiositis", "M33"),
        ("arteritis de celulas gigantes", "M31.5"),
        ("polimialgia reumatica", "M35.3"),
        ("enfermedad mixta del tejido conectivo", "M35.1"),
    ],
)
def test_reumatologica_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.REUMATOLOGICA, cie10)


def test_reumatologica_subclasificacion_por_defecto():
    # Ningún texto que pase la inclusión llega aquí; se prueba la rama directamente.
    assert _regla(C.REUMATOLOGICA).subclasificar("otra enfermedad") == "M06"


# --- Úlcera péptica -----------------------------------------------------------


def test_ulcera_positivo():
    assert mapear("ulcera gastrica") == (C.ULCERA_PEPTICA, "K25")


@pytest.mark.parametrize(
    "texto", ["adenocarcinoma ulcerado gastrico", "sin ulcera gastrica ni duodenal"]
)
def test_ulcera_exclusion(texto):
    assert _no_es(texto, C.ULCERA_PEPTICA)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("ulcera yeyunal", "K28"),
        ("ulcera duodenal", "K26"),
        ("ulcera gastrica", "K25"),
        ("ulcera peptica", "K27"),
    ],
)
def test_ulcera_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.ULCERA_PEPTICA, cie10)


# --- Enfermedad hepática leve -------------------------------------------------


def test_hepatica_positivo():
    assert mapear("cirrosis hepatica") == (C.HEPATICA_LEVE, "K74")


@pytest.mark.parametrize(
    "texto", ["hepatitis b negativo", "sin cirrosis", "hepatitis b y hepatitis a"]
)
def test_hepatica_exclusion(texto):
    assert _no_es(texto, C.HEPATICA_LEVE)


@pytest.mark.parametrize(
    ("texto", "cie10"),
    [
        ("trasplante hepatico", "Z94.4"),
        ("cirrosis alcoholica", "K70.3"),
        ("esteatohepatitis alcoholica", "K70.1"),
        ("higado graso alcoholico", "K70.0"),
        ("hepatopatia cronica etilica", "K70.9"),
        # "no alcohólica" no entra en K70
        ("esteatohepatitis no alcoholica", "K76.0"),
        ("fibrosis hepatica", "K74"),
        ("hepatitis c cronica", "B18"),
        ("hepatitis autoinmune", "K73"),
        ("higado graso", "K76.0"),
        ("hepatopatia cronica", "K76.9"),
        ("dhc", "K76.9"),
    ],
)
def test_hepatica_subclasificacion(texto, cie10):
    assert mapear(texto) == (C.HEPATICA_LEVE, cie10)


# --- Tildes y mayúsculas ------------------------------------------------------


@pytest.mark.parametrize(
    ("con_tildes", "sin_tildes"),
    [
        ("Infarto Agudo de MIOCARDIO", "infarto agudo de miocardio"),
        ("Insuficiencia Cardíaca Hipertensiva", "insuficiencia cardiaca hipertensiva"),
        ("Claudicación intermitente", "claudicacion intermitente"),
        ("ACV isquémico", "acv isquemico"),
        ("Enfermedad de ALZHEIMER", "enfermedad de alzheimer"),
        ("EPOC oxígeno dependiente", "epoc oxigeno dependiente"),
        ("Lupus Eritematoso Sistémico", "lupus eritematoso sistemico"),
        ("Úlcera Gástrica", "ulcera gastrica"),
        ("Cirrosis hepática alcohólica", "cirrosis hepatica alcoholica"),
    ],
)
def test_tildes_y_mayusculas_mapean_igual(con_tildes, sin_tildes):
    assert mapear(con_tildes) == mapear(sin_tildes)
    assert mapear(con_tildes) is not None


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("EPOC", True),
        ("diabetes gestacional", True),
        ("carcinoma basocelular", True),
        ("prediabetes", True),
        ("eppoc", False),
        ("cefalea tensional", False),
    ],
)
def test_reconoce_distingue_desconocido_de_excluido(texto, esperado):
    assert reglas.reconoce(texto) is esperado


def test_reconoce_con_categoria_solo_mira_esa_regla():
    texto = "eppoc sin insuficiencia respiratoria"
    assert reglas.reconoce(texto)  # la exclusión de insuficiencia cardiaca lo conoce
    assert not reglas.reconoce(texto, C.PULMONAR_CRONICA)
    assert reglas.reconoce("prediabetes", C.DIABETES)
