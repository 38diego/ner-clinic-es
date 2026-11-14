"""Tests de la depuración: duplicados internos y cruce con los diagnósticos registrados.

Todas las notas y entidades son sintéticas.
"""

import pytest

from ner_clinic.depuracion import (
    CATEGORIAS_CON_CRUCE,
    contenida_por_palabras,
    cruzar_con_registrados,
    cruzar_por_categoria,
    deduplicar_internas,
    filtrar_por_diagnostico,
)
from ner_clinic.tipos import Candidato, Entidad, Estado, Mencion, MotivoDescarte
from ner_clinic.tipos import CategoriaCharlson as C


def candidato(texto: str, inicio: int = 0) -> Candidato:
    return Candidato(texto=texto, menciones=[Mencion(texto, inicio, inicio + len(texto))])


def confirmada(texto: str) -> Entidad:
    return candidato(texto).a_entidad(Estado.CONFIRMADA)


# --- deduplicar_internas -------------------------------------------------------------


def test_sin_candidatos_devuelve_listas_vacias():
    assert deduplicar_internas([]) == ([], [])


def test_candidatos_distintos_se_conservan():
    entrada = [candidato("asma", 0), candidato("cirrosis hepatica", 10)]

    quedan, descartes = deduplicar_internas(entrada)

    assert [c.texto for c in quedan] == ["asma", "cirrosis hepatica"]
    assert descartes == []


def test_variantes_ortograficas_se_fusionan_y_suman_menciones():
    # Nota sintética: "Paciente con hipertension ... controla su hipertensión"
    con_tilde = candidato("hipertensión", 40)
    sin_tilde = candidato("hipertension", 13)

    quedan, descartes = deduplicar_internas([sin_tilde, con_tilde])

    assert len(quedan) == 1
    assert {m.texto for m in quedan[0].menciones} == {"hipertension", "hipertensión"}
    assert len(descartes) == 1
    assert descartes[0].motivo is MotivoDescarte.DUPLICADO_INTERNO
    assert quedan[0].texto in descartes[0].detalle


def test_entidad_contenida_se_absorbe_en_la_mas_larga():
    corta = candidato("diabetes", 50)
    larga = candidato("diabetes mellitus", 5)

    quedan, descartes = deduplicar_internas([corta, larga])

    assert [c.texto for c in quedan] == ["diabetes mellitus"]
    assert quedan[0].menciones == [Mencion("diabetes mellitus", 5, 22), Mencion("diabetes", 50, 58)]
    assert descartes[0].texto == "diabetes"
    assert descartes[0].motivo is MotivoDescarte.DUPLICADO_INTERNO
    assert "diabetes mellitus" in descartes[0].detalle


def test_mencion_dentro_de_otra_del_mismo_lugar_no_se_cuenta_dos_veces():
    """#223: "Fumador" dentro de "Fumador pesado", en el mismo lugar, es una mención."""
    # Nota sintética: "Fumador pesado. Probable EPOC."
    larga = candidato("Fumador pesado", 0)
    corta = candidato("Fumador", 0)

    quedan, descartes = deduplicar_internas([corta, larga])

    assert [c.texto for c in quedan] == ["Fumador pesado"]
    assert quedan[0].menciones == [Mencion("Fumador pesado", 0, 14)]
    assert [d.texto for d in descartes] == ["Fumador"]


def test_contencion_es_por_palabras_completas_no_por_letras():
    # Nota sintética (#194): "Refiere tos crónica. Se sospechó una artropatía por
    # depósitos de microcristales." "tos" está en las letras de "depósitos".
    entrada = [candidato("tos", 8), candidato("artropatía por depósitos de microcristales", 40)]

    quedan, descartes = deduplicar_internas(entrada)

    assert [c.texto for c in quedan] == ["tos", "artropatía por depósitos de microcristales"]
    assert descartes == []


@pytest.mark.parametrize(
    ("corta", "larga", "esperado"),
    [
        ("insuficiencia renal", "insuficiencia renal crónica", True),
        ("epoc", "enfermedad pulmonar obstructiva crónica (epoc)", True),
        ("dm2", "dm2", True),
        ("tos", "artropatía por depósitos", False),
        ("ira", "insuficiencia respiratoria", False),
        ("renal", "suprarrenal", False),
        ("sida", "vih/sida", True),
        ("d.m.", "d.m. tipo 2", True),
        ("ictus", "post-ictus", True),
        ("cáncer", "cáncer de mama", True),
        ("", "asma", False),
    ],
)
def test_contenida_por_palabras(corta, larga, esperado):
    assert contenida_por_palabras(corta, larga) is esperado


def test_el_prefijo_de_una_palabra_se_fusiona_como_variante_no_como_contenido():
    # "DM" no está como palabra en "DM2", pero sigue siendo variante ortográfica.
    quedan, descartes = deduplicar_internas([candidato("DM", 0), candidato("DM2", 10)])

    assert [c.texto for c in quedan] == ["DM2"]
    assert "variante" in descartes[0].detalle


def test_contencion_ignora_mayusculas():
    quedan, _ = deduplicar_internas([candidato("Diabetes"), candidato("diabetes mellitus")])

    assert [c.texto for c in quedan] == ["diabetes mellitus"]


def test_no_modifica_los_candidatos_de_entrada():
    larga = candidato("diabetes mellitus", 5)

    deduplicar_internas([candidato("diabetes", 50), larga])

    assert larga.menciones == [Mencion("diabetes mellitus", 5, 22)]


def test_conserva_el_orden_de_entrada_de_los_que_quedan():
    entrada = [candidato("asma", 0), candidato("enfermedad renal cronica", 10)]

    quedan, _ = deduplicar_internas(entrada)

    assert [c.texto for c in quedan] == ["asma", "enfermedad renal cronica"]


def test_umbral_de_similitud_configurable():
    # Con umbral 1.0 solo fusiona por contención, no por ortografía.
    quedan, descartes = deduplicar_internas(
        [candidato("hipertension"), candidato("hipertensión")], umbral_similitud=1.0
    )

    assert len(quedan) == 2
    assert descartes == []


# --- filtrar_por_diagnostico ---------------------------------------------------------


@pytest.mark.parametrize("diagnostico", [None, "", "   "])
def test_sin_diagnostico_no_filtra(diagnostico):
    entrada = [confirmada("neumonia basal"), confirmada("asma")]

    quedan, descartes = filtrar_por_diagnostico(entrada, diagnostico)

    assert quedan == entrada
    assert descartes == []


def test_typo_frente_al_diagnostico_se_descarta_como_ortografica():
    quedan, descartes = filtrar_por_diagnostico(
        [confirmada("neumonia basal"), confirmada("asma")], "Neumonía no especificada"
    )

    assert [e.texto for e in quedan] == ["asma"]
    assert len(descartes) == 1
    assert descartes[0].texto == "neumonia basal"
    assert descartes[0].motivo is MotivoDescarte.DIAGNOSTICO_REGISTRADO
    assert "ortográfica" in descartes[0].detalle


def test_palabra_igual_al_diagnostico_se_descarta_como_exacta():
    quedan, descartes = filtrar_por_diagnostico(
        [confirmada("Apendicitis aguda")], "Apendicitis aguda, no especificada"
    )

    assert quedan == []
    assert "exacta" in descartes[0].detalle
    assert "apendicitis" in descartes[0].detalle


def test_terminos_genericos_cie10_no_cuentan_como_coincidencia():
    # "otros", "tipo", "debido a", "no especificado" y "clasificado en otra parte"
    # se quitan del diagnóstico antes de comparar.
    entrada = [
        confirmada("otros hallazgos"),
        confirmada("tipo de sangre"),
        confirmada("cefalea no especificada"),
        confirmada("lesiones clasificadas"),
    ]

    quedan, descartes = filtrar_por_diagnostico(
        entrada,
        "Otros trastornos tipo ansioso debido a estrés, no especificados, "
        "clasificados en otra parte",
    )

    assert quedan == entrada
    assert descartes == []


def test_stopwords_y_palabras_cortas_no_cuentan():
    entrada = [confirmada("dolor para orinar"), confirmada("tos seca")]

    quedan, _ = filtrar_por_diagnostico(entrada, "Fiebre para estudio con tos")

    assert quedan == entrada


def test_entidades_no_confirmadas_no_se_filtran():
    negada = candidato("neumonia").a_entidad(Estado.NEGADA)

    quedan, descartes = filtrar_por_diagnostico([negada], "Neumonía")

    assert quedan == [negada]
    assert descartes == []


def test_umbral_del_filtro_configurable():
    quedan, _ = filtrar_por_diagnostico(
        [confirmada("neumonia basal")], "Neumonía", umbral_similitud=1.0
    )

    assert [e.texto for e in quedan] == ["neumonia basal"]


def test_diagnostico_que_no_es_texto_no_filtra():
    entrada = [confirmada("neumonia basal")]

    quedan, descartes = filtrar_por_diagnostico(entrada, float("nan"))

    assert quedan == entrada
    assert descartes == []


# --- cruzar_con_registrados ----------------------------------------------------------


def test_marca_registradas_con_principal_y_secundarios_sin_descartar():
    entrada = [
        confirmada("neumonia basal"),
        confirmada("diabetes mellitus"),
        confirmada("asma"),
    ]

    quedan, descartes = cruzar_con_registrados(
        entrada, ["Neumonía no especificada", "Diabetes mellitus tipo 2"]
    )

    assert descartes == []
    assert [(e.texto, e.registrado, e.coincide_con) for e in quedan] == [
        ("neumonia basal", True, "Neumonía no especificada"),
        ("diabetes mellitus", True, "Diabetes mellitus tipo 2"),
        ("asma", False, None),
    ]


def test_no_modifica_las_entidades_recibidas():
    original = confirmada("neumonia basal")

    cruzar_con_registrados([original], ["Neumonía"])

    assert original.registrado is None
    assert original.coincide_con is None


@pytest.mark.parametrize("registrados", [None, [], ["", "   "], [float("nan"), None]])
def test_sin_diagnosticos_validos_no_marca_nada(registrados):
    entrada = [confirmada("neumonia basal"), confirmada("asma")]

    quedan, descartes = cruzar_con_registrados(entrada, registrados)

    assert quedan == entrada
    assert all(e.registrado is None for e in quedan)
    assert descartes == []


def test_ignora_elementos_que_no_son_texto_o_estan_vacios():
    quedan, _ = cruzar_con_registrados([confirmada("neumonia basal")], [None, "", 42, "Neumonía"])

    assert quedan[0].registrado is True
    assert quedan[0].coincide_con == "Neumonía"


@pytest.mark.parametrize("estado", [Estado.NEGADA, Estado.POSIBLE, Estado.FAMILIAR])
def test_solo_evalua_confirmadas(estado):
    otra = candidato("neumonia").a_entidad(estado)

    quedan, descartes = cruzar_con_registrados([otra], ["Neumonía"])

    assert quedan == [otra]
    assert quedan[0].registrado is None
    assert descartes == []


def test_varias_coincidencias_reporta_la_primera_de_la_lista():
    quedan, _ = cruzar_con_registrados(
        [confirmada("neumonia con derrame")], ["Derrame pleural", "Neumonía"]
    )

    assert quedan[0].coincide_con == "Derrame pleural"


def test_modo_descartar_indica_con_que_diagnostico_coincidio():
    quedan, descartes = cruzar_con_registrados(
        [confirmada("asma"), confirmada("diabetes mellitus")],
        ["Neumonía", "Diabetes mellitus tipo 2"],
        descartar=True,
    )

    assert [e.texto for e in quedan] == ["asma"]
    assert len(descartes) == 1
    assert descartes[0].texto == "diabetes mellitus"
    assert descartes[0].motivo is MotivoDescarte.DIAGNOSTICO_REGISTRADO
    assert "Diabetes mellitus tipo 2" in descartes[0].detalle


def test_umbral_del_cruce_configurable():
    quedan, _ = cruzar_con_registrados(
        [confirmada("neumonia basal")], ["Neumonía"], umbral_similitud=1.0
    )

    assert quedan[0].registrado is False


def test_filtro_por_principal_indica_el_diagnostico_en_el_detalle():
    _, descartes = filtrar_por_diagnostico([confirmada("neumonia basal")], "Neumonía")

    assert "Neumonía" in descartes[0].detalle


# --- Errores heredados de la tesis, corregidos (#24) ----------------------------------


@pytest.mark.parametrize(
    ("larga", "corta"),
    [
        ("insuficiencia renal cronica", "insuficiencia cardiaca"),
        ("hipertension arterial", "hipertension pulmonar"),
        ("cancer de prostata", "cancer de mama"),
        ("enfermedad renal crónica", "enfermedad de Alzheimer"),
    ],
)
def test_no_fusiona_enfermedades_distintas_con_el_mismo_prefijo(larga, corta):
    # En la tesis, Jaro-Winkler sobre la frase completa premiaba el prefijo común y
    # fusionaba estas parejas. Ahora se compara palabra por palabra.
    quedan, descartes = deduplicar_internas([candidato(larga, 0), candidato(corta, 100)])

    assert [c.texto for c in quedan] == [larga, corta]
    assert descartes == []


@pytest.mark.parametrize(
    ("larga", "corta"),
    [
        ("diabetes mellitus", "diabetis melitus"),
        ("hipertensión arterial", "hipertension"),
        ("insuficiencia cardiaca congestiva", "insuficiencia cardiaca congestva"),
    ],
)
def test_sigue_fusionando_variantes_ortograficas_por_palabra(larga, corta):
    quedan, descartes = deduplicar_internas([candidato(larga, 0), candidato(corta, 100)])

    assert [c.texto for c in quedan] == [larga]
    assert "variante ortográfica" in descartes[0].detalle


def test_entre_variantes_de_igual_longitud_queda_la_que_tiene_tilde():
    # En la tesis quedaba la primera que llegaba, aunque fuera la versión sin tilde.
    quedan, descartes = deduplicar_internas(
        [candidato("hipertension", 13), candidato("hipertensión", 40)]
    )

    assert [c.texto for c in quedan] == ["hipertensión"]
    assert descartes[0].texto == "hipertension"
    assert [m.inicio for m in quedan[0].menciones] == [40, 13]


def test_palabra_comun_no_especifica_no_descarta_otra_enfermedad():
    # "insuficiencia" (como "enfermedad", "aguda" o "crónica") no identifica la
    # enfermedad: la tesis descartaba la renal por coincidir con la cardiaca.
    entrada = [confirmada("insuficiencia renal crónica"), confirmada("enfermedad de Alzheimer")]

    quedan, descartes = filtrar_por_diagnostico(entrada, "Insuficiencia cardiaca congestiva")

    assert quedan == entrada
    assert descartes == []
    quedan, _ = cruzar_con_registrados(entrada, ["Enfermedad pulmonar obstructiva crónica"])
    assert [e.registrado for e in quedan] == [False, False]


def test_palabra_nucleo_igual_sigue_coincidiendo():
    quedan, _ = cruzar_con_registrados(
        [confirmada("insuficiencia renal crónica"), confirmada("enfermedad de Alzheimer")],
        ["Enfermedad renal crónica, no especificada", "Enfermedad de Alzheimer, no especificada"],
    )

    assert [e.coincide_con for e in quedan] == [
        "Enfermedad renal crónica, no especificada",
        "Enfermedad de Alzheimer, no especificada",
    ]


@pytest.mark.parametrize(
    ("sigla", "registrado"),
    [
        ("HTA", "Hipertensión esencial (primaria)"),
        ("DM2", "Diabetes mellitus no insulinodependiente"),
        ("EPOC", "Enfermedad pulmonar obstructiva crónica, no especificada"),
        ("IAM", "Infarto agudo de miocardio, sin otra especificación"),
        ("ACV", "Accidente vascular encefálico agudo"),
        ("ERC", "Enfermedad renal crónica, no especificada"),
        ("FA", "Fibrilación y aleteo auricular"),
        ("TEP", "Embolia pulmonar sin mención de corazón pulmonar agudo"),
        ("TVP", "Embolia y trombosis de otras venas"),
        (
            "VIH",
            "Enfermedad por virus de la inmunodeficiencia humana [VIH], sin otra especificación",
        ),
        ("LES", "Lupus eritematoso sistémico"),
    ],
)
def test_sigla_coincide_con_el_diagnostico_registrado(sigla, registrado):
    # En la tesis una sigla nunca coincidía (solo cuentan palabras de más de 3 letras):
    # "HTA" frente a "Hipertensión esencial (primaria)" salía como posible subregistro.
    quedan, _ = cruzar_con_registrados([confirmada(sigla)], [registrado])

    assert quedan[0].registrado is True
    assert quedan[0].coincide_con == registrado


def test_sigla_expandida_queda_en_el_detalle():
    _, descartes = filtrar_por_diagnostico([confirmada("HTA")], "Hipertensión esencial (primaria)")

    assert "hipertensión arterial" in descartes[0].detalle


def test_sigla_no_coincide_con_otra_enfermedad():
    quedan, _ = cruzar_con_registrados(
        [confirmada("HTA"), confirmada("EPOC")], ["Diabetes mellitus no insulinodependiente"]
    )

    assert [e.registrado for e in quedan] == [False, False]


# --- Cruce por categoría de Charlson (#176) --------------------------------------------


DM_COMPLICADA = "Diabetes mellitus no insulinodependiente con complicaciones oftálmicas"
DM_SIMPLE = "Diabetes mellitus no insulinodependiente sin mención de complicación"


def _sin_registrar(texto, categoria, estado=Estado.CONFIRMADA, registrado=False):
    return Entidad(
        texto=texto,
        menciones=[Mencion(texto, 0, len(texto))],
        estado=estado,
        categoria_charlson=categoria,
        registrado=registrado,
    )


def test_la_complicacion_coincide_por_categoria():
    (entidad,), descartes = cruzar_por_categoria(
        [_sin_registrar("retinopatía diabética", C.DIABETES_COMPLICADA)],
        [("Infarto agudo", C.INFARTO_MIOCARDIO), (DM_COMPLICADA, C.DIABETES_COMPLICADA)],
    )
    assert (entidad.registrado, entidad.coincide_con, entidad.coincide_por) == (
        True,
        DM_COMPLICADA,
        "categoria",
    )
    assert descartes == []


def test_jerarquia_la_leve_coincide_con_la_severa_pero_no_al_reves():
    (dm,), _ = cruzar_por_categoria(
        [_sin_registrar("DM2", C.DIABETES)], [(DM_COMPLICADA, C.DIABETES_COMPLICADA)]
    )
    assert dm.registrado is True
    # La complicación no está registrada si solo está la diabetes sin complicaciones
    (retinopatia,), _ = cruzar_por_categoria(
        [_sin_registrar("retinopatía diabética", C.DIABETES_COMPLICADA)], [(DM_SIMPLE, C.DIABETES)]
    )
    assert retinopatia.registrado is False


@pytest.mark.parametrize("categoria", sorted(set(C) - CATEGORIAS_CON_CRUCE))
def test_las_demas_categorias_no_cruzan(categoria):
    """Un cáncer de mama no está registrado porque se registró uno de próstata; una
    hepatitis B no es la cirrosis registrada (#176)."""
    severa = {C.HEPATICA_LEVE: C.HEPATICA_SEVERA, C.TUMOR_MALIGNO: C.TUMOR_METASTASICO}
    registrados = [("Otro diagnóstico", categoria), ("Su versión severa", severa.get(categoria))]
    (entidad,), _ = cruzar_por_categoria([_sin_registrar("enfermedad", categoria)], registrados)
    assert entidad.registrado is False


def test_solo_toca_confirmadas_sin_coincidencia_y_con_categoria():
    registrados = [(DM_COMPLICADA, C.DIABETES_COMPLICADA)]
    entidades = [
        _sin_registrar(
            "pie diabético", C.DIABETES_COMPLICADA, estado=Estado.POSIBLE, registrado=None
        ),
        _sin_registrar("tos", None),
        ya_registrada := _sin_registrar("DM", C.DIABETES, registrado=True),
    ]
    salida, _ = cruzar_por_categoria(entidades, registrados)
    assert salida[0].registrado is None and salida[1].registrado is False
    assert salida[2] is ya_registrada


def test_con_descartar_se_descarta_con_el_detalle():
    entidad = _sin_registrar("retinopatía diabética", C.DIABETES_COMPLICADA, registrado=None)
    quedan, (descarte,) = cruzar_por_categoria(
        [entidad], [(DM_COMPLICADA, C.DIABETES_COMPLICADA)], descartar=True
    )
    assert quedan == []
    assert descarte.motivo is MotivoDescarte.DIAGNOSTICO_REGISTRADO
    assert "diabetes_complicada" in descarte.detalle and DM_COMPLICADA in descarte.detalle


def test_la_coincidencia_por_texto_dice_texto():
    (entidad,), _ = cruzar_con_registrados(
        [_sin_registrar("diabetes mellitus", None, registrado=None)], [DM_SIMPLE]
    )
    assert (entidad.registrado, entidad.coincide_por) == (True, "texto")
