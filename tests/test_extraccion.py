"""Tests de la extracción NER: fusión, expansión, filtros y lista blanca de siglas.

Todas las notas son sintéticas. Salvo el test marcado `modelo`, usan `ExtractorFalso`,
que devuelve salidas fijas con el formato de `pipeline("ner", aggregation_strategy=
"simple")`.
"""

import pytest
import torch

from ner_clinic import extraccion
from ner_clinic.contexto import clasificar_mencion, intervalos
from ner_clinic.extraccion import (
    CONFIGURACION_POR_DEFECTO,
    MAX_DISTANCIA_FUSION,
    PALABRAS_CORTAS_CLINICAS,
    SIGLAS_CLINICAS,
    SIGLAS_SENSIBLES_A_MAYUSCULAS,
    ConfiguracionExtraccion,
    ExtractorFalso,
    ExtractorHF,
    extraer_candidatos,
    menciones_crudas,
)
from ner_clinic.segmentacion import segmentar
from ner_clinic.tipos import Descarte, Estado, Mencion, MotivoDescarte


def ent(texto: str, palabra: str, desde: int = 0) -> dict:
    """Entidad con el formato de Hugging Face para `palabra` dentro de `texto`."""
    inicio = texto.index(palabra, desde)
    fin = inicio + len(palabra)
    return {
        "entity_group": "ENFERMEDAD",
        "score": 0.99,
        "word": " " + palabra,
        "start": inicio,
        "end": fin,
    }


def extraer(texto: str, *palabras: str, **kwargs):
    """Extrae de `texto` (un solo fragmento) marcando cada palabra como entidad."""
    falso = ExtractorFalso({texto: [ent(texto, p) for p in palabras]})
    return extraer_candidatos(texto, falso, **kwargs)


# --- Flujo básico ----------------------------------------------------------------------


def test_texto_vacio_no_llama_al_modelo():
    falso = ExtractorFalso({})

    assert extraer_candidatos("", falso) == ([], [])
    assert falso.llamadas == []


def test_extractor_falso_devuelve_vacio_para_fragmentos_desconocidos():
    falso = ExtractorFalso({})

    assert falso.extraer(["texto cualquiera", "otro"]) == [[], []]
    assert falso.llamadas == [["texto cualquiera", "otro"]]


def test_candidato_con_mencion_en_su_posicion():
    texto = "Paciente con cirrosis hepatica en seguimiento."

    candidatos, descartes = extraer(texto, "cirrosis hepatica")

    assert descartes == []
    assert len(candidatos) == 1
    assert candidatos[0].texto == "cirrosis hepatica"
    assert candidatos[0].menciones == [Mencion("cirrosis hepatica", 13, 30)]


def test_posiciones_relativas_al_texto_limpio_con_varios_fragmentos():
    # Dos párrafos que no caben en un fragmento de 40 caracteres.
    texto = "Antecedente de cirrosis hepatica.\nAhora con insuficiencia renal."
    segundo = "Ahora con insuficiencia renal."
    falso = ExtractorFalso(
        {
            "Antecedente de cirrosis hepatica.": [
                ent("Antecedente de cirrosis hepatica.", "cirrosis hepatica")
            ],
            segundo: [ent(segundo, "insuficiencia renal")],
        }
    )

    candidatos, _ = extraer_candidatos(texto, falso, max_caracteres=40)

    assert len(falso.llamadas) == 1
    assert len(falso.llamadas[0]) == 2
    mencion = candidatos[1].menciones[0]
    assert mencion == Mencion("insuficiencia renal", 44, 63)
    assert texto[mencion.inicio : mencion.fin] == mencion.texto


def test_candidato_conserva_el_texto_original_sin_colapsar_espacios():
    texto = "Dx: Diabetes  Mellitus tipo 2 descompensada."

    candidatos, _ = extraer(texto, "Diabetes  Mellitus tipo 2")

    assert candidatos[0].texto == "Diabetes  Mellitus tipo 2"
    assert candidatos[0].menciones[0].texto == "Diabetes  Mellitus tipo 2"


def test_menciones_con_espacios_distintos_se_agrupan():
    texto = "Diabetes  mellitus en control. Ahora diabetes mellitus sin lesion de organo."
    falso = ExtractorFalso(
        {texto: [ent(texto, "Diabetes  mellitus"), ent(texto, "diabetes mellitus")]}
    )

    candidatos, _ = extraer_candidatos(texto, falso)

    assert len(candidatos) == 1
    assert candidatos[0].texto == "Diabetes  mellitus"
    assert len(candidatos[0].menciones) == 2


def test_candidato_con_espacios_dobles_se_clasifica_por_contexto():
    # La limpieza no colapsa espacios dobles; la mención queda en su posición literal.
    texto = "Niega diabetes  mellitus."

    candidatos, _ = extraer(texto, "diabetes  mellitus")
    (mencion,) = candidatos[0].menciones

    assert candidatos[0].texto == "diabetes  mellitus"
    assert clasificar_mencion(mencion, intervalos(texto)) is Estado.NEGADA


def test_menciones_del_mismo_texto_normalizado_se_agrupan():
    texto = "Cirrosis hepatica compensada; control de la cirrosis hepatica."
    falso = ExtractorFalso(
        {
            texto: [
                ent(texto, "Cirrosis hepatica"),
                ent(texto, "cirrosis hepatica"),
            ]
        }
    )

    candidatos, _ = extraer_candidatos(texto, falso)

    assert len(candidatos) == 1
    assert candidatos[0].texto == "Cirrosis hepatica"
    assert [m.texto for m in candidatos[0].menciones] == [
        "Cirrosis hepatica",
        "cirrosis hepatica",
    ]


def test_no_deduplica_subcadenas_entre_candidatos():
    # Absorber "cirrosis" en "cirrosis hepatica alcoholica" es tarea de depuracion.py.
    texto = "Cirrosis hepatica alcoholica. La cirrosis requiere control."
    candidatos, _ = extraer(texto, "Cirrosis hepatica alcoholica", "cirrosis")

    assert [c.texto for c in candidatos] == ["Cirrosis hepatica alcoholica", "cirrosis"]


# --- Fusión de entidades cercanas -----------------------------------------------------

# La fusión de la tesis, que el pipeline ya no usa por defecto (#55): estos tests prueban
# la función, que sigue en la ablación.
FUSION_TESIS = ConfiguracionExtraccion(max_distancia_fusion=MAX_DISTANCIA_FUSION)


def test_fusiona_entidades_cercanas_sin_barrera():
    texto = "Insuficiencia cardiaca de origen isquemico en estudio."

    candidatos, _ = extraer(
        texto, "Insuficiencia cardiaca", "isquemico", configuracion=FUSION_TESIS
    )

    assert [c.texto for c in candidatos] == ["Insuficiencia cardiaca de origen isquemico"]
    assert candidatos[0].menciones == [Mencion("Insuficiencia cardiaca de origen isquemico", 0, 42)]


@pytest.mark.parametrize(("relleno", "esperados"), [(16, 1), (17, 2)])
def test_fusion_hasta_18_caracteres_de_distancia(relleno, esperados):
    # Distancia = espacio + relleno + espacio: 18 se fusiona, 19 no.
    texto = "cardiopatia " + "y" * relleno + " isquemica"

    candidatos, _ = extraer(texto, "cardiopatia", "isquemica", configuracion=FUSION_TESIS)

    assert len(candidatos) == esperados


@pytest.mark.parametrize("intermedio", ["\n", ", ", ". ", " - ", " 2) ", " y 3."])
def test_barreras_impiden_la_fusion(intermedio):
    texto = "gastritis cronica" + intermedio + "ulcera peptica"

    candidatos, _ = extraer(
        texto, "gastritis cronica", "ulcera peptica", configuracion=FUSION_TESIS
    )

    assert [c.texto for c in candidatos] == ["gastritis cronica", "ulcera peptica"]


@pytest.mark.parametrize(
    ("texto", "primera", "segunda"),
    [
        ("HTA y diabetes mellitus tipo 2", "HTA", "diabetes mellitus tipo 2"),
        ("EPOC GOLD 3 y FA permanente", "EPOC GOLD 3", "FA permanente"),
        (
            "demencia tipo Alzheimer y enfermedad de Parkinson",
            "demencia tipo Alzheimer",
            "enfermedad de Parkinson",
        ),
        ("anemia ferropenica e hipotiroidismo", "anemia ferropenica", "hipotiroidismo"),
        ("Cirrosis hepatica O hepatitis cronica", "Cirrosis hepatica", "hepatitis cronica"),
    ],
)
def test_conjunciones_aisladas_impiden_la_fusion(texto, primera, segunda):
    # Desviación intencional frente a la tesis: el modelo las separa y la fusión las unía.
    candidatos, _ = extraer(texto, primera, segunda, configuracion=FUSION_TESIS)

    assert [c.texto for c in candidatos] == [primera, segunda]


@pytest.mark.parametrize(
    ("texto", "partes", "fusionada"),
    [
        (
            "IC de origen isquemico.",
            ("IC", "isquemico"),
            "IC de origen isquemico",
        ),
        (
            "Enfermedad renal cronica estadio 3 por nefropatia diabetica.",
            ("Enfermedad renal cronica", "nefropatia diabetica"),
            "Enfermedad renal cronica estadio 3 por nefropatia diabetica",
        ),
        (
            "Cardiopatia hipertensiva y/o isquemica.",
            ("Cardiopatia hipertensiva", "isquemica"),
            "Cardiopatia hipertensiva y/o isquemica",
        ),
    ],
)
def test_fusion_sin_conjuncion_aislada_sigue_ocurriendo(texto, partes, fusionada):
    candidatos, _ = extraer(texto, *partes, configuracion=FUSION_TESIS)

    assert [c.texto for c in candidatos] == [fusionada]


def test_fusion_ordena_las_entidades_por_posicion():
    texto = "Artritis reumatoide seropositiva."
    falso = ExtractorFalso({texto: [ent(texto, "seropositiva"), ent(texto, "Artritis")]})

    candidatos, _ = extraer_candidatos(texto, falso, configuracion=FUSION_TESIS)

    assert [c.texto for c in candidatos] == ["Artritis reumatoide seropositiva"]


# --- Expansión de palabras cortadas ---------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "cortada", "completa"),
    [
        ("Cursa con hipertension arterial.", "hipertens", "hipertension"),
        ("Cursa con hipertension arterial.", "ertension arterial", "hipertension arterial"),
        ("Dx (hipertension) estable.", "pertensi", "hipertension"),
        ("Sin cambios;cirrosis;estable", "irros", "cirrosis"),
    ],
)
def test_expande_palabras_cortadas(texto, cortada, completa):
    candidatos, _ = extraer(texto, cortada)

    assert [c.texto for c in candidatos] == [completa]
    mencion = candidatos[0].menciones[0]
    assert texto[mencion.inicio : mencion.fin] == completa


# --- Limpieza de bordes y viñetas -----------------------------------------------------


def test_quita_puntuacion_de_los_bordes_y_ajusta_la_posicion():
    texto = "Dx: (cirrosis hepatica), en control."

    candidatos, _ = extraer(texto, "(cirrosis hepatica),")

    assert candidatos[0].menciones == [Mencion("cirrosis hepatica", 5, 22)]


@pytest.mark.parametrize("vineta", ["1. ", "2- ", "3) ", "12.", "4.  "])
def test_quita_vinetas_numericas_iniciales(vineta):
    texto = "Antecedentes:\n" + vineta + "Diabetes mellitus"

    candidatos, descartes = extraer(texto, vineta + "Diabetes mellitus")

    assert descartes == []
    assert [c.texto for c in candidatos] == ["Diabetes mellitus"]
    mencion = candidatos[0].menciones[0]
    assert texto[mencion.inicio : mencion.fin] == "Diabetes mellitus"


@pytest.mark.parametrize(
    ("entidad", "esperado"),
    [
        ("1.- HTA", "HTA"),
        ("2.-Diabetes mellitus", "Diabetes mellitus"),
        ("1. (HTA)", "HTA"),
        ("12.5 hipertension", "12.5 hipertension"),
    ],
)
def test_vinetas_combinadas_con_puntuacion(entidad, esperado):
    texto = f"Antecedentes:\n{entidad}\nSin otros."

    candidatos, descartes = extraer(texto, entidad)

    assert descartes == []
    assert [c.texto for c in candidatos] == [esperado]
    mencion = candidatos[0].menciones[0]
    assert mencion.texto == esperado
    assert texto[mencion.inicio : mencion.fin] == mencion.texto


# --- Filtros de basura ----------------------------------------------------------------


def test_lista_blanca_descarta_entidades_de_4_caracteres_o_menos():
    texto = "Refiere tos seca. Ahora con gripa."

    candidatos, descartes = extraer(texto, "tos", "gripa", filtro_longitud="lista_blanca")

    assert [c.texto for c in candidatos] == ["gripa"]
    assert descartes == [Descarte("tos", MotivoDescarte.BASURA, "longitud de 4 caracteres o menos")]


def test_vineta_se_quita_antes_del_filtro_de_longitud():
    texto = "1. tos"

    candidatos, descartes = extraer(texto, "1. tos", filtro_longitud="lista_blanca")

    assert candidatos == []
    assert [d.texto for d in descartes] == ["tos"]


def test_siglas_minimas_en_la_lista_blanca():
    minimas = {"HTA", "EPOC", "IAM", "ACV", "ICC", "ERC", "IRC", "DM", "DM2", "VIH", "FA", "TEP"}

    assert minimas <= SIGLAS_CLINICAS


def test_lista_blanca_incluye_lo_que_el_experimento_de_siglas_recupero():
    # Detectadas por el modelo real y perdidas por el filtro de longitud (#8,
    # docs/experimentos/siglas.md): equivalentes de Charlson y otras comorbilidades.
    charlson = {"ECV", "SCA", "IMA", "TIA", "HIV", "EVP"}
    otras = {"TVP", "LES", "IVU", "TBC", "ERGE", "HPB"}

    assert charlson | otras <= SIGLAS_CLINICAS
    assert {"ASMA", "GOTA"} <= PALABRAS_CORTAS_CLINICAS


def test_palabras_cortas_no_estan_entre_las_siglas():
    # "asma" y "gota" son palabras, no siglas (#24): van en su propia constante.
    assert not PALABRAS_CORTAS_CLINICAS & SIGLAS_CLINICAS


@pytest.mark.parametrize("palabra", ["asma", "Asma", "GOTA", "gota"])
def test_palabras_cortas_clinicas_no_se_descartan(palabra):
    texto = f"Antecedente de {palabra} en control."

    candidatos, descartes = extraer(texto, palabra)

    assert descartes == []
    assert [c.texto for c in candidatos] == [palabra]


def test_siglas_ambiguas_son_siglas_de_la_lista_blanca():
    assert {"LES", "TIA"} <= SIGLAS_SENSIBLES_A_MAYUSCULAS <= SIGLAS_CLINICAS


@pytest.mark.parametrize(
    ("texto", "palabra"),
    [
        ("Se les explica a los familiares.", "les"),
        ("Les explica el plan de manejo.", "Les"),
        ("Vive con su tia materna.", "tia"),
        ("Vive con su tía materna.", "tía"),  # con tilde, Charlson la llevaría a ACV
        ("Tía paterna con diabetes.", "Tía"),
    ],
)
def test_palabra_comun_igual_a_una_sigla_ambigua_se_descarta(texto, palabra):
    # Antes de #24 la lista blanca no distinguía mayúsculas: el pronombre "les" pasaba
    # como LES (lupus) si el modelo lo marcaba.
    candidatos, descartes = extraer(texto, palabra)

    assert candidatos == []
    assert descartes == [
        Descarte(palabra, MotivoDescarte.BASURA, "longitud de 4 caracteres o menos")
    ]


@pytest.mark.parametrize(
    ("texto", "sigla"), [("LES activo.", "LES"), ("Antecedente de TIA.", "TIA")]
)
def test_sigla_ambigua_en_mayusculas_se_conserva(texto, sigla):
    candidatos, descartes = extraer(texto, sigla)

    assert descartes == []
    assert [c.texto for c in candidatos] == [sigla]


@pytest.mark.parametrize("sigla", sorted(SIGLAS_CLINICAS))
def test_siglas_clinicas_no_se_descartan(sigla):
    texto = f"Antecedente de {sigla} en control."

    candidatos, descartes = extraer(texto, sigla)

    assert descartes == []
    assert [c.texto for c in candidatos] == [sigla]


@pytest.mark.parametrize("sigla", ["hta", "Dm2", "Epoc"])
def test_siglas_sin_distinguir_mayusculas(sigla):
    texto = f"Antecedente de {sigla}."

    candidatos, _ = extraer(texto, sigla)

    assert [c.texto for c in candidatos] == [sigla]


# --- Variantes del filtro de longitud (experimento de siglas, #8) ---------------------


def test_filtro_tesis_descarta_siglas_de_la_lista_blanca():
    texto = "Antecedente de HTA y EPOC."

    candidatos, descartes = extraer(texto, "HTA", "EPOC", filtro_longitud="tesis")

    assert candidatos == []
    assert [d.texto for d in descartes] == ["HTA", "EPOC"]
    assert {d.detalle for d in descartes} == {"longitud de 4 caracteres o menos"}


def test_filtro_tesis_conserva_entidades_de_mas_de_4_caracteres():
    texto = "Antecedente de gastritis."

    candidatos, descartes = extraer(texto, "gastritis", filtro_longitud="tesis")

    assert [c.texto for c in candidatos] == ["gastritis"]
    assert descartes == []


def test_sin_filtro_de_longitud_conserva_entidades_cortas():
    texto = "Refiere tos y HTA."

    candidatos, descartes = extraer(texto, "tos", "HTA", filtro_longitud="ninguno")

    assert [c.texto for c in candidatos] == ["tos", "HTA"]
    assert descartes == []


def test_sin_filtro_de_longitud_mantiene_los_demas_filtros():
    texto = "Diagnostico. Lesion per"

    candidatos, descartes = extraer(texto, "Diagnostico", "Lesion per", filtro_longitud="ninguno")

    assert candidatos == []
    assert [d.detalle for d in descartes] == ["lista negra administrativa", "termina en ' per'"]


def test_filtro_minimo_es_el_de_defecto():
    texto = "Refiere tos y HTA."

    por_defecto = extraer(texto, "tos", "HTA")
    explicito = extraer(texto, "tos", "HTA", filtro_longitud="minimo")

    assert por_defecto == explicito
    assert [c.texto for c in por_defecto[0]] == ["tos", "HTA"]
    # La lista blanca, el de defecto hasta #56, descarta "tos"
    lista_blanca = extraer(texto, "tos", "HTA", filtro_longitud="lista_blanca")
    assert [c.texto for c in lista_blanca[0]] == ["HTA"]


@pytest.mark.parametrize(
    ("corta", "pasa"),
    [
        ("lúes", True),  # palabra de 4 letras
        ("coma", True),
        ("EC", True),  # sigla en mayúscula, aunque no esté en la lista
        ("GIST", True),
        ("M0", True),  # una letra, pero en mayúscula
        ("dm", True),  # de la lista blanca, en minúscula
        ("1", False),  # sin letras
        ("de", False),  # minúscula de 2 letras, fuera de la lista
        ("en", False),
        ("Mx", True),  # metástasis: con mayúscula inicial pasa
        ("y 2", False),
    ],
)
def test_filtro_minimo(corta, pasa):
    texto = f"Paciente con {corta} hoy."

    candidatos, descartes = extraer(texto, corta, filtro_longitud="minimo")

    assert [c.texto for c in candidatos] == ([corta] if pasa else [])
    assert len(descartes) == (0 if pasa else 1)


def test_filtro_de_longitud_desconocido_falla():
    with pytest.raises(ValueError, match="filtro_longitud"):
        extraer("Refiere tos.", "tos", filtro_longitud="otro")


@pytest.mark.parametrize("sufijo", ["per", "end", "inf"])
def test_descarta_terminaciones_truncadas(sufijo):
    texto = f"Insuficiencia renal {sufijo} control."

    candidatos, descartes = extraer(texto, f"Insuficiencia renal {sufijo}")

    assert candidatos == []
    assert descartes == [
        Descarte(
            f"Insuficiencia renal {sufijo}",
            MotivoDescarte.BASURA,
            f"termina en ' {sufijo}'",
        )
    ]


@pytest.mark.parametrize(
    "palabra",
    ["Diagnostico", "Diagnóstico", "ANTECEDENTES", "paciente", "Historia  clinica", "lesión"],
)
def test_descarta_lista_negra_administrativa(palabra):
    texto = f"{palabra}: sin cambios."

    candidatos, descartes = extraer(texto, palabra)

    assert candidatos == []
    assert len(descartes) == 1
    assert descartes[0].motivo is MotivoDescarte.BASURA
    assert descartes[0].detalle == "lista negra administrativa"


def test_un_descarte_por_texto_y_regla():
    texto = "Paciente estable. Paciente sin fiebre."
    falso = ExtractorFalso({texto: [ent(texto, "Paciente"), ent(texto, "Paciente", 5)]})

    _, descartes = extraer_candidatos(texto, falso)

    assert len(descartes) == 1


def test_entidad_solo_de_puntuacion_se_ignora():
    texto = "Cirrosis hepatica - estable."

    assert extraer(texto, "-") == ([], [])


# --- Configuración para la ablación (#42) ----------------------------------------------

# Nota sintética que ejercita todas las decisiones a la vez: fusión, expansión, viñeta,
# filtro de longitud, sigla, sufijo truncado y lista negra.
NOTA_ABLACION = (
    "Diagnostico.\n1. Insuficiencia cardiaca de origen isquemico, hipertens arterial, "
    "HTA y tos. Insuficiencia renal per control."
)
PARTES_ABLACION = (
    "Diagnostico",
    "1. Insuficiencia cardiaca",
    "isquemico",
    "hipertens",
    "HTA",
    "tos",
    "Insuficiencia renal per",
)


def extraer_ablacion(**kwargs):
    return extraer(NOTA_ABLACION, *PARTES_ABLACION, **kwargs)


def textos(candidatos) -> list[str]:
    return [c.texto for c in candidatos]


def test_configuracion_por_defecto_es_la_del_pipeline():
    sin_configuracion = extraer_ablacion()

    assert extraer_ablacion(configuracion=ConfiguracionExtraccion()) == sin_configuracion
    assert extraer_ablacion(configuracion=CONFIGURACION_POR_DEFECTO) == sin_configuracion
    candidatos, descartes = sin_configuracion
    # "tos" pasa con el filtro mínimo (#56); con la lista blanca se descartaba. Sin la
    # fusión (#55), "Insuficiencia cardiaca" e "isquemico" quedan separadas.
    assert textos(candidatos) == [
        "Insuficiencia cardiaca",
        "isquemico",
        "hipertens",
        "HTA",
        "tos",
    ]
    assert [d.detalle for d in descartes] == ["lista negra administrativa", "termina en ' per'"]


def test_valores_por_defecto_de_la_configuracion():
    assert ConfiguracionExtraccion() == ConfiguracionExtraccion(
        max_distancia_fusion=None,  # sin fusión desde #55
        expandir_palabras=True,
        filtro_longitud="minimo",  # desde #56; la tesis descartaba todo lo corto
        usar_lista_negra=True,
        descartar_sufijos_truncados=True,
    )


def test_configuracion_es_inmutable():
    with pytest.raises(AttributeError):
        CONFIGURACION_POR_DEFECTO.expandir_palabras = False  # type: ignore[misc]


def test_sin_fusion_cada_entidad_queda_sola():
    configuracion = ConfiguracionExtraccion(max_distancia_fusion=None)

    candidatos, _ = extraer_ablacion(configuracion=configuracion)

    assert textos(candidatos)[:2] == ["Insuficiencia cardiaca", "isquemico"]


@pytest.mark.parametrize(("distancia", "esperados"), [(0, 2), (10, 2), (11, 1), (30, 1)])
def test_fusion_a_otra_distancia(distancia, esperados):
    # "Insuficiencia cardiaca" e "isquemico" están a 11 caracteres (" de origen ").
    texto = "Insuficiencia cardiaca de origen isquemico."
    configuracion = ConfiguracionExtraccion(max_distancia_fusion=distancia)

    candidatos, _ = extraer(
        texto, "Insuficiencia cardiaca", "isquemico", configuracion=configuracion
    )

    assert len(candidatos) == esperados


def test_fusion_a_distancia_cero_une_entidades_contiguas():
    texto = "Hipotiroidismo primario."
    falso = ExtractorFalso({texto: [ent(texto, "Hipotiroid"), ent(texto, "ismo primario")]})
    configuracion = ConfiguracionExtraccion(max_distancia_fusion=0)

    candidatos, _ = extraer_candidatos(texto, falso, configuracion=configuracion)

    assert textos(candidatos) == ["Hipotiroidismo primario"]


def test_distancia_de_fusion_negativa_falla():
    with pytest.raises(ValueError, match="max_distancia_fusion"):
        ConfiguracionExtraccion(max_distancia_fusion=-1)


def test_sin_expansion_no_completa_palabras_cortadas():
    configuracion = ConfiguracionExtraccion(expandir_palabras=False)
    texto = "Cursa con hipertension arterial."

    candidatos, _ = extraer(texto, "ertension arterial", configuracion=configuracion)

    assert textos(candidatos) == ["ertension arterial"]


def test_expansion_y_fusion_son_independientes():
    # Sin fusión, cada entidad se sigue expandiendo por su lado.
    texto = "Cursa con hipertension arterial."
    configuracion = ConfiguracionExtraccion(max_distancia_fusion=None)

    candidatos, _ = extraer(texto, "pertens", "arteri", configuracion=configuracion)

    assert textos(candidatos) == ["hipertension", "arterial"]


@pytest.mark.parametrize(
    ("filtro", "cortas"),
    [
        ("minimo", ["HTA", "tos"]),
        ("lista_blanca", ["HTA"]),
        ("tesis", []),
        ("ninguno", ["HTA", "tos"]),
    ],
)
def test_filtro_de_longitud_desde_la_configuracion(filtro, cortas):
    configuracion = ConfiguracionExtraccion(filtro_longitud=filtro)

    candidatos, _ = extraer_ablacion(configuracion=configuracion)

    assert [t for t in textos(candidatos) if len(t) <= 4] == cortas


def test_filtro_longitud_explicito_reemplaza_el_de_la_configuracion():
    configuracion = ConfiguracionExtraccion(filtro_longitud="ninguno", usar_lista_negra=False)

    candidatos, _ = extraer_ablacion(filtro_longitud="tesis", configuracion=configuracion)

    # Filtro "tesis" (sin HTA ni tos), pero la lista negra sigue apagada.
    assert "HTA" not in textos(candidatos)
    assert "tos" not in textos(candidatos)
    assert "Diagnostico" in textos(candidatos)


def test_configuracion_con_filtro_desconocido_falla():
    with pytest.raises(ValueError, match="filtro_longitud"):
        ConfiguracionExtraccion(filtro_longitud="otro")  # type: ignore[arg-type]


def test_sin_lista_negra_conserva_el_texto_administrativo():
    configuracion = ConfiguracionExtraccion(usar_lista_negra=False)

    candidatos, descartes = extraer_ablacion(configuracion=configuracion)

    assert textos(candidatos)[0] == "Diagnostico"
    assert "lista negra administrativa" not in [d.detalle for d in descartes]


def test_sin_descarte_de_sufijos_conserva_las_entidades_truncadas():
    configuracion = ConfiguracionExtraccion(descartar_sufijos_truncados=False)

    candidatos, descartes = extraer_ablacion(configuracion=configuracion)

    assert textos(candidatos)[-1] == "Insuficiencia renal per"
    assert not any(d.detalle.startswith("termina en") for d in descartes)


def test_todo_apagado_solo_recorta_bordes_y_vinetas():
    configuracion = ConfiguracionExtraccion(
        max_distancia_fusion=None,
        expandir_palabras=False,
        filtro_longitud="ninguno",
        usar_lista_negra=False,
        descartar_sufijos_truncados=False,
    )

    candidatos, descartes = extraer_ablacion(configuracion=configuracion)

    assert descartes == []
    assert textos(candidatos) == [
        "Diagnostico",
        "Insuficiencia cardiaca",
        "isquemico",
        "hipertens",
        "HTA",
        "tos",
        "Insuficiencia renal per",
    ]


# --- Menciones crudas del modelo -------------------------------------------------------


def test_menciones_crudas_sin_fusion_expansion_recorte_ni_filtros():
    falso = ExtractorFalso({NOTA_ABLACION: [ent(NOTA_ABLACION, p) for p in PARTES_ABLACION]})

    menciones = menciones_crudas(NOTA_ABLACION, falso)

    assert [m.texto for m in menciones] == list(PARTES_ABLACION)
    for mencion in menciones:
        assert NOTA_ABLACION[mencion.inicio : mencion.fin] == mencion.texto
    assert len(falso.llamadas) == 1


def test_menciones_crudas_en_orden_y_con_posiciones_de_varios_fragmentos():
    parrafo = "Paciente estable con HTA en manejo y EPOC sin exacerbaciones recientes. "
    texto = "\n".join(parrafo * 6 for _ in range(6))
    fragmentos = segmentar(texto)
    assert len(fragmentos) > 1
    # Entidades al revés dentro de cada fragmento: la salida igual queda en orden.
    falso = ExtractorFalso(
        {f.texto: [ent(f.texto, "EPOC"), ent(f.texto, "HTA")] for f in fragmentos}
    )

    menciones = menciones_crudas(texto, falso)

    assert [m.inicio for m in menciones] == sorted(m.inicio for m in menciones)
    assert len(menciones) == 2 * len(fragmentos)
    for mencion in menciones:
        assert texto[mencion.inicio : mencion.fin] == mencion.texto


def test_menciones_crudas_omiten_entidades_vacias_y_texto_vacio_no_llama_al_modelo():
    texto = "Refiere tos."
    vacia = {"entity_group": "ENFERMEDAD", "score": 0.5, "word": "", "start": 3, "end": 3}
    falso = ExtractorFalso({texto: [vacia, ent(texto, "tos")]})

    assert menciones_crudas(texto, falso) == [Mencion("tos", 8, 11)]
    vacio = ExtractorFalso({})
    assert menciones_crudas("", vacio) == []
    assert vacio.llamadas == []


# --- ExtractorHF (sin descargar el modelo) --------------------------------------------


class PipelineFalso:
    """Imita el pipeline de Hugging Face con una lista de textos: una lista por texto."""

    def __init__(self):
        self.llamadas = []

    def __call__(self, entradas, batch_size):
        self.llamadas.append((entradas, batch_size))
        salida = {"entity_group": "ENFERMEDAD", "score": 0.9, "word": " x", "start": 0, "end": 1}
        return [[dict(salida)] for _ in entradas]


@pytest.fixture
def fabrica(monkeypatch):
    creados = []

    def crear(nombre_modelo, dispositivo, revision=None):
        creados.append((nombre_modelo, dispositivo, revision))
        return PipelineFalso()

    monkeypatch.setattr(extraccion, "_crear_pipeline", crear)
    return creados


def test_extractor_hf_carga_perezosa_una_sola_vez(fabrica):
    extractor = ExtractorHF()
    assert fabrica == []

    extractor.extraer(["uno", "dos"])
    extractor.extraer(["tres"])

    assert len(fabrica) == 1
    assert fabrica[0][0] == "BSC-NLP4BIA/bsc-bio-ehr-es-distemist"


def test_extractor_hf_procesa_por_lotes_con_batch_size(fabrica):
    extractor = ExtractorHF()
    salidas = extractor.extraer(["uno", "dos", "tres"])

    assert len(salidas) == 3
    assert extractor._pipeline.llamadas == [(["uno", "dos", "tres"], 8)]

    otro = ExtractorHF(batch_size=2)
    otro.extraer(["uno"])
    assert otro._pipeline.llamadas[0][1] == 2


def test_extractor_hf_un_fragmento_devuelve_lista_de_listas(fabrica):
    salidas = ExtractorHF().extraer(["uno"])

    assert len(salidas) == 1
    assert salidas[0][0]["entity_group"] == "ENFERMEDAD"


def test_extractor_hf_lista_vacia_no_carga_el_modelo(fabrica):
    assert ExtractorHF().extraer([]) == []
    assert fabrica == []


@pytest.mark.parametrize(("hay_gpu", "dispositivo"), [(True, 0), (False, -1)])
def test_extractor_hf_usa_gpu_si_hay(fabrica, monkeypatch, hay_gpu, dispositivo):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: hay_gpu)

    ExtractorHF().extraer(["uno"])

    assert fabrica[0][1] == dispositivo


def test_pipeline_usa_ventana_deslizante():
    argumentos = extraccion._argumentos_pipeline("modelo-x", -1)

    assert argumentos == {
        "task": "ner",
        "model": "modelo-x",
        "aggregation_strategy": "simple",
        "stride": extraccion.STRIDE_TOKENS,
        "device": -1,
    }
    assert 0 < extraccion.STRIDE_TOKENS < 512


def test_pipeline_pasa_la_revision_solo_si_se_da():
    argumentos = extraccion._argumentos_pipeline("modelo-x", -1, "abc123")

    assert argumentos["revision"] == "abc123"
    assert "revision" not in extraccion._argumentos_pipeline("modelo-x", -1, None)


def test_extractor_hf_sin_revision_usa_la_rama_por_defecto(fabrica, monkeypatch):
    monkeypatch.delenv(extraccion.VARIABLE_REVISION, raising=False)

    ExtractorHF().extraer(["uno"])

    assert fabrica[0][2] is None


def test_extractor_hf_lee_la_revision_del_entorno(fabrica, monkeypatch):
    monkeypatch.setenv(extraccion.VARIABLE_REVISION, "abc123")

    ExtractorHF().extraer(["uno"])
    ExtractorHF(revision="def456").extraer(["uno"])

    assert [creado[2] for creado in fabrica] == ["abc123", "def456"]


# --- Modelo real ----------------------------------------------------------------------


# Nota sintética densa: signos vitales y laboratorios en un solo párrafo, con muchos
# tokens por carácter. Cabe en un fragmento de 2.000 caracteres pero pasa de 512 tokens.
_NOTA_DENSA = (
    "Ingresa por control. " + "TA 128/76 mmHg, FC 88 lpm, FR 18 rpm, SatO2 94% FiO2 21%, T 36,8 C; "
    "Hb 11,2 g/dL Hto 34% Leu 9.800/mm3 Plq 245.000/mm3; Cr 1,3 mg/dL BUN 22 mg/dL; "
    "Na 138 K 4,1 Cl 102 mEq/L; PCR 0,8 mg/dL; INR 1,1 TP 12,4 s TPT 31 s. "
    * 6
    + "Se documenta cirrosis hepatica en seguimiento."
)


@pytest.mark.modelo
def test_modelo_real_procesa_fragmentos_de_mas_de_512_tokens():
    from transformers import AutoTokenizer

    tokenizador = AutoTokenizer.from_pretrained(extraccion.NOMBRE_MODELO)
    assert len(segmentar(_NOTA_DENSA)) == 1
    assert len(tokenizador(_NOTA_DENSA)["input_ids"]) > 512

    candidatos, _ = extraer_candidatos(_NOTA_DENSA, ExtractorHF())

    textos = {c.texto.lower() for c in candidatos}
    assert "cirrosis hepatica" in textos
    for candidato in candidatos:
        for mencion in candidato.menciones:
            assert _NOTA_DENSA[mencion.inicio : mencion.fin] == mencion.texto


@pytest.mark.modelo
def test_modelo_real_sobre_nota_sintetica():
    nota = (
        "Paciente masculino de 67 años con antecedente de HTA y diabetes mellitus tipo 2 "
        "en manejo con metformina.\n"
        "Niega EPOC.\n"
        "Madre con cancer de mama."
    )

    candidatos, _ = extraer_candidatos(nota, ExtractorHF())

    textos = {c.texto.lower() for c in candidatos}
    # La conjunción " y " separa las dos comorbilidades (no se fusionan).
    assert "hta" in textos
    assert "diabetes mellitus tipo 2" in textos
    assert not any(" y " in t for t in textos)
    for candidato in candidatos:
        for mencion in candidato.menciones:
            assert nota[mencion.inicio : mencion.fin] == mencion.texto


# --- Siglas en mayúscula para el modelo (#114) ------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("pte con epoc", "pte con EPOC"),
        ("hta, dm, epoc", "HTA, DM, EPOC"),
        ("dm2 y Epoc", "DM2 y EPOC"),
        ("irc estadio 3; fa paroxística", "IRC estadio 3; FA paroxística"),
        # Palabras comunes iguales a una sigla y palabras que la contienen no cambian.
        ("se les explica a su tia", "se les explica a su tia"),
        ("artritis ar", "artritis ar"),
        ("epocal, dmso, fatiga", "epocal, dmso, fatiga"),
        ("Paciente con hipertensión.", "Paciente con hipertensión."),
    ],
)
def test_siglas_para_el_modelo(texto, esperado):
    assert extraccion.siglas_para_el_modelo(texto) == esperado


def test_siglas_para_el_modelo_conserva_el_largo():
    texto = "pte con epoc, hta, dm2, irc, ic, tep, tvp, vih, sida y erge. Ñandú cardíaca."
    assert len(extraccion.siglas_para_el_modelo(texto)) == len(texto)


def test_siglas_para_el_modelo_lista_fija():
    """Fija la lista: agregar o quitar una sigla es una decisión que se justifica (#114)."""
    assert {
        "ACV", "AIT", "DM", "DM1", "DM2", "DMID", "DMNID", "ECV", "EPOC", "ERC", "ERGE",
        "EVP", "FA", "HBP", "HIV", "HPB", "HTA", "IAM", "IC", "ICC", "IMA", "IRC", "IVU",
        "SAHOS", "SCA", "SIDA", "TBC", "TEP", "TVP", "VIH",
    } == extraccion.SIGLAS_PARA_EL_MODELO  # fmt: skip
    assert {"LES", "AR", "TIA", "EAP"}.isdisjoint(extraccion.SIGLAS_PARA_EL_MODELO)


def test_eap_en_minuscula_no_se_sube():
    """ "eap" suele ser edema agudo de pulmón, no enfermedad arterial periférica."""
    texto = "eap: edema agudo de pulmón descartado"
    assert extraccion.siglas_para_el_modelo(texto) == texto


def test_extractor_falso_con_claves_que_el_modelo_ve_iguales():
    with pytest.raises(ValueError, match="mayúsculas de una sigla"):
        ExtractorFalso({"Pte con epoc.": [], "Pte con EPOC.": []})


class _SoloMayusculas:
    """Extractor que solo ve "EPOC" en mayúscula, como el modelo real."""

    def __init__(self) -> None:
        self.llamadas: list[list[str]] = []

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        self.llamadas.append(list(fragmentos))
        return [
            [
                {
                    "entity_group": "ENFERMEDAD",
                    "score": 0.99,
                    "word": "EPOC",
                    "start": i,
                    "end": i + 4,
                }
            ]
            if (i := f.find("EPOC")) != -1
            else []
            for f in fragmentos
        ]


def test_el_modelo_ve_la_sigla_en_mayuscula_y_la_mencion_es_la_de_la_nota():
    extractor = _SoloMayusculas()
    candidatos, _ = extraccion.extraer_candidatos("Pte con epoc en manejo.", extractor)

    assert extractor.llamadas == [["Pte con EPOC en manejo."]]
    assert [(m.texto, m.inicio, m.fin) for c in candidatos for m in c.menciones] == [
        ("epoc", 8, 12)
    ]


def test_menciones_crudas_tambien_ven_la_sigla_en_mayuscula():
    menciones = extraccion.menciones_crudas("Pte con epoc.", _SoloMayusculas())
    assert [(m.texto, m.inicio, m.fin) for m in menciones] == [("epoc", 8, 12)]


# --- Barreras ampliadas de la fusión (#55) --------------------------------------------

AMPLIADAS = ConfiguracionExtraccion(
    max_distancia_fusion=MAX_DISTANCIA_FUSION, barreras_fusion="ampliadas"
)


@pytest.mark.parametrize(
    ("texto", "partes"),
    [
        ("dx: demencia mixta + HTA + hipotiroidismo", ("demencia mixta", "HTA", "hipotiroidismo")),
        ("Niega diabetes ni HTA", ("diabetes", "HTA")),
        ("Cirrosis hepatica con varices esofagicas", ("Cirrosis hepatica", "varices esofagicas")),
        ("hemorragia subaracnoidea (HSA) aguda", ("hemorragia subaracnoidea", "HSA")),
        ("Antecedentes: asma bronquial: EPOC", ("asma bronquial", "EPOC")),
        ("Cardiopatia hipertensiva y/o isquemica", ("Cardiopatia hipertensiva", "isquemica")),
    ],
)
def test_barreras_ampliadas_no_unen_a_traves_de_conectores(texto, partes):
    candidatos, _ = extraer(texto, *partes, configuracion=AMPLIADAS)
    assert textos(candidatos) == list(partes)
    # Con las de la tesis, las mismas partes se unen
    candidatos, _ = extraer(texto, *partes, configuracion=FUSION_TESIS)
    assert len(candidatos) < len(partes)


def test_barreras_ampliadas_siguen_uniendo_sin_conector():
    candidatos, _ = extraer(
        "Insuficiencia cardiaca de origen isquemico.",
        "Insuficiencia cardiaca",
        "isquemico",
        configuracion=AMPLIADAS,
    )
    assert textos(candidatos) == ["Insuficiencia cardiaca de origen isquemico"]


def test_barreras_de_fusion_desconocidas_fallan():
    with pytest.raises(ValueError, match="barreras_fusion"):
        ConfiguracionExtraccion(barreras_fusion="otras")  # type: ignore[arg-type]


def test_la_barrera_cuenta_aunque_el_modelo_la_meta_en_la_entidad():
    """#54: el modelo incluye el salto de línea en "TB\\n"; la fusión no debe unir renglones
    distintos de una lista."""
    texto = (
        "- TB pulmonar y lesiones cerebrales por toxoplasmosis y/o TB\n"
        "- dermatitis seborreica prácticamente constante"
    )
    tb = texto.index("TB\n")
    entidades = [
        ent(texto, "lesiones cerebrales por toxoplasmosis"),
        {"entity_group": "ENFERMEDAD", "score": 0.9, "word": "TB", "start": tb, "end": tb + 3},
        ent(texto, "dermatitis seborreica"),
    ]
    falso = ExtractorFalso({texto: entidades})
    candidatos, _ = extraer_candidatos(texto, falso, configuracion=FUSION_TESIS)
    assert textos(candidatos) == [
        "lesiones cerebrales por toxoplasmosis y/o TB",
        "dermatitis seborreica",
    ]
    # Sin fusión (por defecto, #55), tampoco
    candidatos, _ = extraer_candidatos(texto, falso)
    assert textos(candidatos) == [
        "lesiones cerebrales por toxoplasmosis",
        "TB",
        "dermatitis seborreica",
    ]


# --- cortar_parentesis (#195) --------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("EPOC) secundaria", "EPOC"),
        ("enfermedad pulmonar (EPOC", "enfermedad pulmonar "),
        (
            "insuficiencia cardiaca (IC) secundaria a HTA",
            "insuficiencia cardiaca (IC) secundaria a HTA",
        ),
        ("(EPOC", "(EPOC"),
        (")EPOC", ")EPOC"),
        ("diabetes mellitus", "diabetes mellitus"),
        ("a (b (c) d", "a "),
        # El espacio de después de la viñeta lo quita `_recortar`.
        ("a) asma bronquial", " asma bronquial"),
        ("iv) diabetes mellitus tipo 2", " diabetes mellitus tipo 2"),
        ("B) EPOC) secundaria", " EPOC"),
    ],
)
def test_cortar_parentesis(texto, esperado):
    inicio, fin = extraccion.cortar_parentesis(0, len(texto), texto)
    assert texto[inicio:fin] == esperado


def test_la_sigla_pegada_a_la_palabra_siguiente_se_corta_en_el_parentesis():
    # El modelo real devuelve "EPOC) secundaria" para esta frase sintética (#195).
    nota = "Sospecha clínica de enfermedad pulmonar obstructiva crónica (EPOC) secundaria a tabaco."
    extractor = ExtractorFalso(
        {
            nota: [
                {"entity_group": "ENFERMEDAD", "score": 0.99, "word": "", "start": s, "end": e}
                for s, e in ((20, 59), (61, 77))
            ]
        }
    )

    candidatos, _ = extraer_candidatos(nota, extractor)
    sin_corte, _ = extraer_candidatos(
        nota, extractor, configuracion=ConfiguracionExtraccion(cortar_parentesis=False)
    )

    assert [c.texto for c in candidatos] == ["enfermedad pulmonar obstructiva crónica", "EPOC"]
    assert [c.texto for c in sin_corte][1] == "EPOC) secundaria"


@pytest.mark.parametrize(
    "marcada", ["**hipertensión arterial**", "_hipertensión arterial_", "# hipertensión arterial"]
)
def test_quita_las_marcas_de_markdown_de_los_bordes(marcada):
    # Texto pegado desde markdown (#211).
    texto = f"Antecedente de {marcada} en manejo."

    candidatos, _ = extraer(texto, marcada)

    assert [c.texto for c in candidatos] == ["hipertensión arterial"]
