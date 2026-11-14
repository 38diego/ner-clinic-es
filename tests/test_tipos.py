import json

import pytest

from ner_clinic.tipos import (
    Candidato,
    CategoriaCharlson,
    Descarte,
    Entidad,
    Estado,
    Mencion,
    MotivoDescarte,
    Resultado,
)


def _entidad(**cambios):
    datos = dict(
        texto="diabetes mellitus tipo 2",
        menciones=[Mencion("diabetes mellitus tipo 2", 30, 54), Mencion("DM tipo 2", 120, 129)],
        estado=Estado.CONFIRMADA,
        categoria_charlson=CategoriaCharlson.DIABETES,
        cie10="E11.9",
    )
    datos.update(cambios)
    return Entidad(**datos)


def test_inicio_y_fin_son_los_de_la_primera_mencion():
    entidad = _entidad()

    assert (entidad.inicio, entidad.fin) == (30, 54)


def test_entidad_sin_menciones_no_tiene_posicion():
    entidad = _entidad(menciones=[])

    assert entidad.inicio is None
    assert entidad.fin is None


def test_mencion_rechaza_fin_antes_del_inicio():
    with pytest.raises(ValueError):
        Mencion("x", 10, 5)


def test_mencion_rechaza_inicio_negativo():
    with pytest.raises(ValueError):
        Mencion("x", -1, 5)


def test_inicio_y_fin_son_los_de_la_mencion_mas_temprana_aunque_la_lista_este_desordenada():
    entidad = _entidad(menciones=[Mencion("DM2", 120, 123), Mencion("diabetes", 30, 38)])

    assert (entidad.inicio, entidad.fin) == (30, 38)


def test_entidad_fuera_de_charlson_tiene_categoria_y_cie10_vacios():
    entidad = Entidad(
        texto="gastritis", menciones=[Mencion("gastritis", 0, 9)], estado=Estado.CONFIRMADA
    )

    assert entidad.categoria_charlson is None
    assert entidad.cie10 is None


def test_entidad_to_dict_es_plano_y_usa_valores_de_texto():
    assert _entidad().to_dict() == {
        "texto": "diabetes mellitus tipo 2",
        "inicio": 30,
        "fin": 54,
        "menciones": [
            {"texto": "diabetes mellitus tipo 2", "inicio": 30, "fin": 54, "estado": "confirmada"},
            {"texto": "DM tipo 2", "inicio": 120, "fin": 129, "estado": "confirmada"},
        ],
        "estado": "confirmada",
        "otros_estados": [],
        "categoria_charlson": "diabetes",
        "cie10": "E11.9",
        "registrado": None,
        "coincide_con": None,
        "coincide_por": None,
        "categoria_por_correccion": False,
        "correcciones": [],
        "mapeo": None,
        "candidato": None,
        "alternativas_cie10": [],
    }


def test_entidad_to_dict_con_correcciones():
    datos = _entidad(correcciones=(("diabetis", "diabetes"),)).to_dict()

    assert datos["categoria_por_correccion"] is True
    assert datos["correcciones"] == [{"escrita": "diabetis", "corregida": "diabetes"}]


def test_entidad_sin_diagnosticos_registrados_no_esta_marcada():
    entidad = _entidad()

    assert entidad.registrado is None
    assert entidad.coincide_con is None


def test_entidad_to_dict_incluye_el_cruce_con_los_registrados():
    datos = _entidad(registrado=True, coincide_con="Diabetes mellitus tipo 2").to_dict()

    assert datos["registrado"] is True
    assert datos["coincide_con"] == "Diabetes mellitus tipo 2"


def test_motivo_de_descarte_por_diagnostico_registrado():
    assert MotivoDescarte.DIAGNOSTICO_REGISTRADO.value == "diagnostico_registrado"
    assert not hasattr(MotivoDescarte, "DIAGNOSTICO_PRINCIPAL")


def test_descarte_to_dict_usa_el_valor_del_motivo():
    descarte = Descarte(
        texto="hipertension",
        motivo=MotivoDescarte.DUPLICADO_INTERNO,
        detalle="misma ortografía que 'hipertensión'",
    )

    assert descarte.to_dict() == {
        "texto": "hipertension",
        "motivo": "duplicado_interno",
        "detalle": "misma ortografía que 'hipertensión'",
    }


def test_resultado_to_dict_se_puede_serializar_a_json():
    resultado = Resultado(
        entidades=[_entidad(), _entidad(texto="epoc", estado=Estado.NEGADA, menciones=[])],
        descartes=[Descarte("paciente", MotivoDescarte.BASURA, "lista negra")],
        indice_charlson=1,
    )

    datos = json.loads(json.dumps(resultado.to_dict()))

    assert datos["indice_charlson"] == 1
    assert [e["estado"] for e in datos["entidades"]] == ["confirmada", "negada"]
    assert datos["descartes"][0]["motivo"] == "basura"


def test_estado_posible():
    assert Estado.POSIBLE == "posible"
    assert [e.value for e in Estado] == ["confirmada", "negada", "posible", "familiar"]
    entidad = _entidad(estado=Estado.POSIBLE)
    assert entidad.to_dict()["estado"] == "posible"
    assert entidad.to_dict()["categoria_charlson"] == "diabetes"


def test_resultado_to_dict_incluye_el_indice_si_se_confirman():
    resultado = Resultado(indice_charlson=1, indice_charlson_si_se_confirman=7)
    datos = resultado.to_dict()
    assert (datos["indice_charlson"], datos["indice_charlson_si_se_confirman"]) == (1, 7)
    # Compatibilidad (US3): los campos de antes siguen, en el mismo orden, y el nuevo va al final.
    assert list(datos) == [
        "entidades",
        "descartes",
        "indice_charlson",
        "indice_charlson_si_se_confirman",
    ]


def test_resultado_vacio():
    assert Resultado().to_dict() == {
        "entidades": [],
        "descartes": [],
        "indice_charlson": 0,
        "indice_charlson_si_se_confirman": 0,
    }


def test_hay_17_categorias_de_charlson():
    assert len(CategoriaCharlson) == 17


def test_candidato_se_vuelve_entidad_con_estado():
    candidato = Candidato(texto="epoc", menciones=[Mencion("EPOC", 10, 14)])

    entidad = candidato.a_entidad(Estado.NEGADA)

    assert entidad == Entidad(
        texto="epoc", menciones=[Mencion("EPOC", 10, 14)], estado=Estado.NEGADA
    )
