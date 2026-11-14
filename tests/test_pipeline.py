"""Tests del orquestador `analizar()`.

Todas las notas son sintéticas. Salvo los tests marcados `modelo`, usan `ExtractorFalso`:
las salidas se arman sobre el texto limpio, que es lo que recibe el modelo.
"""

import inspect
import json
import subprocess
import sys
import time

import pytest

import ner_clinic
from ner_clinic import pipeline
from ner_clinic.extraccion import ConfiguracionExtraccion, ExtractorFalso, ExtractorHF
from ner_clinic.limpieza import limpiar
from ner_clinic.segmentacion import segmentar
from ner_clinic.tipos import CategoriaCharlson, Estado, MotivoDescarte, Resultado

ENCABEZADO = "Folio: 7 | Fecha: 2024-02-03 10:15:00.0"


def extractor_para(nota: str, *palabras: str) -> ExtractorFalso:
    """`ExtractorFalso` que marca cada aparición de `palabras` en el texto limpio de `nota`.

    Busca cada palabra en cada fragmento del texto limpio y devuelve todas sus
    apariciones, con el formato de `pipeline("ner", aggregation_strategy="simple")`.
    """
    salidas = {}
    for fragmento in segmentar(limpiar(nota).texto):
        entidades = []
        for palabra in palabras:
            desde = 0
            while (inicio := fragmento.texto.find(palabra, desde)) != -1:
                fin = inicio + len(palabra)
                entidades.append(
                    {
                        "entity_group": "ENFERMEDAD",
                        "score": 0.99,
                        "word": palabra,
                        "start": inicio,
                        "end": fin,
                    }
                )
                desde = fin
        salidas[fragmento.texto] = sorted(entidades, key=lambda e: e["start"])
    return ExtractorFalso(salidas)


def por_texto(resultado: Resultado) -> dict:
    return {e.texto: e for e in resultado.entidades}


def verificar_posiciones(nota: str, resultado: Resultado) -> None:
    for entidad in resultado.entidades:
        assert entidad.menciones, entidad
        for mencion in entidad.menciones:
            assert nota[mencion.inicio : mencion.fin] == mencion.texto, mencion
        assert nota[entidad.inicio : entidad.fin] == entidad.primera_mencion.texto


# --- Flujo completo ---------------------------------------------------------------------

NOTA = (
    "Paciente femenina de 67 años con HTA y diabetes mellitus tipo 2 en manejo con "
    "metformina. Hipotiroidismo en suplencia. Insuficiencia renal crónica estadio 3. "
    "Niega EPOC. Madre con cáncer de mama. Refiere tos."
)
PALABRAS = (
    "HTA",
    "diabetes mellitus tipo 2",
    "Hipotiroidismo",
    "Insuficiencia renal crónica",
    "EPOC",
    "cáncer de mama",
    "tos",
)


@pytest.fixture
def resultado():
    return pipeline.analizar(
        NOTA,
        diagnosticos_registrados=["Diabetes mellitus no insulinodependiente"],
        extractor=extractor_para(NOTA, *PALABRAS),
    )


def test_flujo_completo_estados(resultado):
    entidades = por_texto(resultado)

    assert entidades["EPOC"].estado is Estado.NEGADA
    assert entidades["cáncer de mama"].estado is Estado.FAMILIAR
    for texto in ("HTA", "diabetes mellitus tipo 2", "Hipotiroidismo"):
        assert entidades[texto].estado is Estado.CONFIRMADA
    assert entidades["Insuficiencia renal crónica"].estado is Estado.CONFIRMADA


def test_solo_las_confirmadas_en_charlson_tienen_categoria_y_cie10(resultado):
    entidades = por_texto(resultado)

    diabetes = entidades["diabetes mellitus tipo 2"]
    assert diabetes.categoria_charlson is CategoriaCharlson.DIABETES
    assert diabetes.cie10 == "E11.9"
    renal = entidades["Insuficiencia renal crónica"]
    assert renal.categoria_charlson is CategoriaCharlson.RENAL_CRONICA
    # Fuera de Charlson: confirmada, sin categoría ni código.
    for texto in ("HTA", "Hipotiroidismo"):
        assert entidades[texto].categoria_charlson is None
        assert entidades[texto].cie10 is None
    # Negadas y familiares no se mapean aunque estén en una categoría de Charlson.
    for texto in ("EPOC", "cáncer de mama"):
        assert entidades[texto].categoria_charlson is None
        assert entidades[texto].cie10 is None


def test_indice_suma_solo_confirmadas(resultado):
    # Diabetes (1) + renal crónica (2); EPOC negada y cáncer familiar no suman.
    assert resultado.indice_charlson == 3


def test_cruce_con_registrados_marca_sin_descartar(resultado):
    entidades = por_texto(resultado)

    diabetes = entidades["diabetes mellitus tipo 2"]
    assert diabetes.registrado is True
    assert diabetes.coincide_con == "Diabetes mellitus no insulinodependiente"
    assert entidades["Insuficiencia renal crónica"].registrado is False
    assert entidades["HTA"].registrado is False
    # No confirmadas: sin cruce.
    assert entidades["EPOC"].registrado is None
    assert entidades["cáncer de mama"].registrado is None


def test_sigla_coincide_con_su_diagnostico_registrado():
    """Corregido (#24): "HTA" frente a "Hipertensión esencial (primaria)" salía como
    posible subregistro porque las siglas nunca coincidían en el cruce."""
    resultado = pipeline.analizar(
        NOTA,
        diagnosticos_registrados=[
            "Hipertensión esencial (primaria)",
            "Diabetes mellitus no insulinodependiente",
        ],
        extractor=extractor_para(NOTA, *PALABRAS),
    )

    hta = por_texto(resultado)["HTA"]
    assert hta.registrado is True
    assert hta.coincide_con == "Hipertensión esencial (primaria)"
    # La renal crónica no coincide con ninguno: sigue como posible subregistro.
    assert por_texto(resultado)["Insuficiencia renal crónica"].registrado is False


def test_descartes_por_basura_con_motivo():
    nota = "Refiere tos. Control en 1 semana."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "tos", "1"))

    assert [(d.texto, d.motivo) for d in resultado.descartes] == [("1", MotivoDescarte.BASURA)]
    assert "1" not in por_texto(resultado)
    assert "tos" in por_texto(resultado)  # corta, pero pasa el filtro mínimo (#56)


def test_sin_diagnosticos_registrado_queda_en_none():
    resultado = pipeline.analizar(NOTA, extractor=extractor_para(NOTA, *PALABRAS))

    assert all(e.registrado is None for e in resultado.entidades)


def test_descartar_registrados_los_lleva_a_descartes():
    resultado = pipeline.analizar(
        NOTA,
        diagnosticos_registrados=["Diabetes mellitus no insulinodependiente"],
        descartar_registrados=True,
        extractor=extractor_para(NOTA, *PALABRAS),
    )

    assert "diabetes mellitus tipo 2" not in por_texto(resultado)
    registrados = [
        d for d in resultado.descartes if d.motivo is MotivoDescarte.DIAGNOSTICO_REGISTRADO
    ]
    assert [d.texto for d in registrados] == ["diabetes mellitus tipo 2"]
    assert "Diabetes mellitus no insulinodependiente" in registrados[0].detalle
    assert resultado.indice_charlson == 2  # solo renal crónica


def test_duplicados_internos_se_fusionan_y_se_reportan():
    nota = "Diabetes mellitus tipo 2 de larga data. Control de diabetes. Niega EPOC."
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "Diabetes mellitus tipo 2", "diabetes", "EPOC"),
    )

    entidades = por_texto(resultado)
    assert set(entidades) == {"Diabetes mellitus tipo 2", "EPOC"}
    diabetes = entidades["Diabetes mellitus tipo 2"]
    assert [m.texto for m in diabetes.menciones] == ["Diabetes mellitus tipo 2", "diabetes"]
    assert [(d.texto, d.motivo) for d in resultado.descartes] == [
        ("diabetes", MotivoDescarte.DUPLICADO_INTERNO)
    ]
    verificar_posiciones(nota, resultado)


def test_entidades_en_orden_de_aparicion(resultado):
    inicios = [e.inicio for e in resultado.entidades]

    assert inicios == sorted(inicios)


def test_to_dict_es_serializable(resultado):
    datos = json.loads(json.dumps(resultado.to_dict(), ensure_ascii=False))

    assert set(datos) == {
        "entidades",
        "descartes",
        "indice_charlson",
        "indice_charlson_si_se_confirman",
    }
    assert datos["indice_charlson"] == 3
    assert datos["indice_charlson_si_se_confirman"] == 3  # sin posibles, el mismo
    diabetes = next(e for e in datos["entidades"] if e["texto"] == "diabetes mellitus tipo 2")
    assert diabetes["estado"] == "confirmada"
    assert diabetes["categoria_charlson"] == "diabetes"
    assert diabetes["cie10"] == "E11.9"
    assert diabetes["registrado"] is True
    assert datos["descartes"] == []  # "tos" pasa el filtro mínimo (#56)


# --- Contexto por posición (#73) --------------------------------------------------------


def test_negada_y_luego_confirmada_es_una_confirmada_con_su_otro_estado():
    """El IAM negado en los antecedentes y el del ingreso: una sola entidad (#208)."""
    nota = "Niega IAM previo. Ingresa por IAM con elevación del ST."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "IAM"))

    [iam] = resultado.entidades
    assert (iam.texto, iam.estado, iam.otros_estados) == (
        "IAM",
        Estado.CONFIRMADA,
        (Estado.NEGADA,),
    )
    assert [iam.estado_de_mencion(i) for i in range(2)] == [Estado.NEGADA, Estado.CONFIRMADA]
    assert iam.categoria_charlson is CategoriaCharlson.INFARTO_MIOCARDIO
    assert resultado.indice_charlson == 1
    verificar_posiciones(nota, resultado)
    datos = resultado.to_dict()["entidades"][0]
    assert datos["otros_estados"] == ["negada"]
    assert [m["estado"] for m in datos["menciones"]] == ["negada", "confirmada"]


# --- Estado posible (#88) ---------------------------------------------------------------


def test_us2_indice_con_lo_confirmado_y_si_se_confirman():
    """US2 del PRD `incertidumbre`: la sospecha no suma al índice, pero sí al otro."""
    nota = "Paciente con DM2. Sospecha de cáncer metastásico."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "DM2", "cáncer metastásico"))

    assert [(e.texto, e.estado) for e in resultado.entidades] == [
        ("DM2", Estado.CONFIRMADA),
        ("cáncer metastásico", Estado.POSIBLE),
    ]
    assert resultado.indice_charlson == 1
    assert resultado.indice_charlson_si_se_confirman == 7
    datos = resultado.to_dict()
    assert (datos["indice_charlson"], datos["indice_charlson_si_se_confirman"]) == (1, 7)
    verificar_posiciones(nota, resultado)


def test_la_posible_lleva_categoria_y_cie10_pero_no_registrado():
    nota = "Paciente con DM2. Sospecha de cáncer metastásico."
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=["Tumor maligno secundario"],
        extractor=extractor_para(nota, "DM2", "cáncer metastásico"),
    )

    posible = por_texto(resultado)["cáncer metastásico"]
    assert posible.categoria_charlson is CategoriaCharlson.TUMOR_METASTASICO
    assert posible.cie10 == "C80"
    # Como las negadas: una sospecha no es un subregistro.
    assert (posible.registrado, posible.coincide_con) == (None, None)
    assert por_texto(resultado)["DM2"].registrado is False
    assert posible.to_dict()["estado"] == "posible"


def test_jerarquia_entre_confirmada_y_posible():
    # La diabetes confirmada (1) y una complicada posible (2): si se confirma, cuenta 2.
    nota = "Diabetes en manejo. Sospecha de nefropatía diabética."
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "Diabetes", "nefropatía diabética")
    )

    assert [e.estado for e in resultado.entidades] == [Estado.CONFIRMADA, Estado.POSIBLE]
    assert resultado.entidades[1].categoria_charlson is CategoriaCharlson.DIABETES_COMPLICADA
    assert (resultado.indice_charlson, resultado.indice_charlson_si_se_confirman) == (1, 2)


def test_un_descarte_sigue_negando_y_no_suma_a_ningun_indice():
    nota = "Paciente con DM2. Se descartó cáncer metastásico."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "DM2", "cáncer metastásico"))

    assert [e.estado for e in resultado.entidades] == [Estado.CONFIRMADA, Estado.NEGADA]
    assert resultado.indice_charlson == resultado.indice_charlson_si_se_confirman == 1


def test_menciones_en_el_mismo_estado_siguen_en_una_entidad():
    nota = "HTA en manejo. Control de HTA. Niega EPOC."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "HTA", "EPOC"))

    assert [(e.texto, e.estado, len(e.menciones)) for e in resultado.entidades] == [
        ("HTA", Estado.CONFIRMADA, 2),
        ("EPOC", Estado.NEGADA, 1),
    ]


def test_cruce_con_registrados_con_una_mencion_negada_antes():
    nota = "Niega HTA. Paciente con HTA en manejo."
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=["Hipertensión esencial (primaria)"],
        extractor=extractor_para(nota, "HTA"),
    )

    [hta] = resultado.entidades
    assert (hta.estado, hta.registrado) == (Estado.CONFIRMADA, True)
    assert hta.coincide_con == "Hipertensión esencial (primaria)"


def test_clasifica_en_el_texto_limpio_y_traduce_al_original():
    # Los encabezados y guiones corren las posiciones: la clasificación se hace sobre el
    # texto limpio, donde corren las regex, y las menciones se traducen después.
    nota = (
        f"{ENCABEZADO}\nPaciente con EPOC en manejo.\n-------------------\n"
        f"{ENCABEZADO}\nMadre con EPOC. Niega EPOC agudizada."
    )
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "EPOC"))

    # La EPOC del paciente sigue confirmada aunque luego se niegue la agudización; la de
    # la madre es otra entidad (#208).
    assert [(e.estado, e.inicio, e.otros_estados) for e in resultado.entidades] == [
        (Estado.CONFIRMADA, nota.index("EPOC"), (Estado.NEGADA,)),
        (Estado.FAMILIAR, nota.index("Madre con EPOC") + 10, ()),
    ]
    verificar_posiciones(nota, resultado)


def test_una_negacion_posterior_no_baja_una_confirmada():
    # "diabetes" se fusiona con "Diabetes mellitus tipo 2" y su mención está negada
    # ("diabetes gestacional"): la del paciente sigue confirmada (#208).
    nota = "Diabetes mellitus tipo 2 de larga data. Niega diabetes gestacional."
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "Diabetes mellitus tipo 2", "diabetes")
    )

    [dm] = resultado.entidades
    assert (dm.texto, dm.estado, dm.otros_estados) == (
        "Diabetes mellitus tipo 2",
        Estado.CONFIRMADA,
        (Estado.NEGADA,),
    )
    assert resultado.indice_charlson == 1
    verificar_posiciones(nota, resultado)


@pytest.mark.parametrize(
    ("nota", "palabras", "estado", "otros"),
    [
        (
            "Sospecha de migraña con aura. Tras la resonancia se descarta la migraña.",
            ("migraña con aura", "migraña"),
            Estado.NEGADA,
            (Estado.POSIBLE,),
        ),
        (
            "Se descartó tuberculosis. Ahora, sospecha de tuberculosis por la tos.",
            ("tuberculosis",),
            Estado.POSIBLE,
            (Estado.NEGADA,),
        ),
        (
            "Sospecha de neumonía. Se confirma neumonía por radiografía.",
            ("neumonía",),
            Estado.CONFIRMADA,
            (Estado.POSIBLE,),
        ),
    ],
)
def test_entre_sospecha_y_negacion_gana_la_ultima_mencion(nota, palabras, estado, otros):
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, *palabras))

    [entidad] = resultado.entidades
    assert (entidad.estado, entidad.otros_estados) == (estado, otros)


# --- Posiciones en el texto original -----------------------------------------------------


def test_posiciones_coinciden_con_el_texto_original(resultado):
    verificar_posiciones(NOTA, resultado)


def test_posiciones_con_encabezados_de_folio_y_guiones():
    nota = (
        f"{ENCABEZADO}\nEVOLUCION\n\n\nPaciente con HTA.\n"
        "-------------------\n"
        f"{ENCABEZADO}\n"
        "Diabetes ------ mellitus tipo 2 en manejo. Niega EPOC.\n\n"
        f"{ENCABEZADO}   Madre con cáncer de mama. Control de HTA."
    )
    resultado = pipeline.analizar(
        nota,
        # La limpieza deja "Diabetes   mellitus tipo 2" (los guiones pasan a espacios); el
        # modelo la marca entera
        extractor=extractor_para(nota, "HTA", "Diabetes   mellitus tipo 2", "EPOC", "cáncer"),
    )

    entidades = por_texto(resultado)
    verificar_posiciones(nota, resultado)
    # En el original, la mención abarca la línea de guiones que la limpieza quitó.
    diabetes = next(e for e in resultado.entidades if e.texto.startswith("Diabetes"))
    assert diabetes.menciones[0].texto == "Diabetes ------ mellitus tipo 2"
    assert len(entidades["HTA"].menciones) == 2
    assert entidades["cáncer"].estado is Estado.FAMILIAR


def test_posiciones_con_varios_fragmentos():
    parrafo = "Paciente estable con HTA en manejo y EPOC sin exacerbaciones recientes. "
    nota = f"{ENCABEZADO}\n" + "\n".join(parrafo * 6 for _ in range(6))
    assert len(segmentar(limpiar(nota).texto)) > 1
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "HTA", "EPOC"))

    verificar_posiciones(nota, resultado)
    assert len(por_texto(resultado)["HTA"].menciones) == 36


# --- Entradas vacías y extractor por defecto --------------------------------------------


@pytest.mark.parametrize("texto", [None, "", "   \n\n", f"{ENCABEZADO}\n-----\n"])
def test_texto_vacio_devuelve_resultado_vacio_sin_cargar_el_modelo(texto, monkeypatch):
    def no_cargar():
        raise AssertionError("no debía pedir el extractor")

    monkeypatch.setattr(pipeline, "_extractor_compartido", no_cargar)

    assert pipeline.analizar(texto) == Resultado()


def test_sin_entidades_devuelve_resultado_vacio():
    assert pipeline.analizar("Paciente asintomático.", extractor=ExtractorFalso({})) == Resultado()


def test_extractor_por_defecto_es_compartido_y_unico(monkeypatch):
    monkeypatch.setattr(pipeline, "_EXTRACTOR", None)
    usados = []

    def extraer(self, fragmentos):
        usados.append(self)
        return [[] for _ in fragmentos]

    monkeypatch.setattr(ExtractorHF, "extraer", extraer)

    pipeline.analizar("Paciente con HTA.")
    pipeline.analizar("Paciente con EPOC.")

    assert len(usados) == 2
    assert usados[0] is usados[1]
    assert isinstance(usados[0], ExtractorHF)


def test_importar_analizar_no_carga_torch():
    codigo = (
        "import sys\n"
        "from ner_clinic import analizar\n"
        "assert 'torch' not in sys.modules, 'torch cargado al importar'\n"
        "assert 'transformers' not in sys.modules, 'transformers cargado al importar'\n"
    )
    subprocess.run([sys.executable, "-c", codigo], check=True)


def test_api_publica():
    assert ner_clinic.analizar is pipeline.analizar
    assert ner_clinic.Resultado is Resultado


# --- Orquestador interno y menciones crudas (ablación, #42) -----------------------------

NOTA_CON_FOLIOS = (
    f"{ENCABEZADO}\nEVOLUCION\n\n\nPaciente con HTA.\n"
    "-------------------\n"
    f"{ENCABEZADO}\n"
    "Diabetes ------ mellitus tipo 2 en manejo. Niega EPOC.\n\n"
    f"{ENCABEZADO}   Madre con cáncer de mama. Control de HTA. Refiere tos."
)
PALABRAS_CON_FOLIOS = ("HTA", "Diabetes", "mellitus tipo 2", "EPOC", "cáncer", "tos")


def test_firma_publica_de_analizar_no_cambia():
    assert list(inspect.signature(pipeline.analizar).parameters) == [
        "texto",
        "diagnosticos_registrados",
        "descartar_registrados",
        "extractor",
        "normalizador",  # #165, opcional
        "modo_mapeo",  # #180, opcional
    ]


@pytest.mark.parametrize(
    ("nota", "palabras", "registrados"),
    [
        (NOTA, PALABRAS, ["Diabetes mellitus no insulinodependiente"]),
        (NOTA_CON_FOLIOS, PALABRAS_CON_FOLIOS, None),
    ],
)
def test_orquestador_interno_por_defecto_es_analizar(nota, palabras, registrados):
    publico = pipeline.analizar(
        nota, diagnosticos_registrados=registrados, extractor=extractor_para(nota, *palabras)
    )
    interno = pipeline._analizar(
        nota,
        diagnosticos_registrados=registrados,
        extractor=extractor_para(nota, *palabras),
        configuracion=ConfiguracionExtraccion(),
        deduplicar=True,
    )

    assert interno == publico
    assert interno.to_dict() == publico.to_dict()


def test_analizar_llama_al_orquestador_interno_con_los_valores_por_defecto(monkeypatch):
    llamadas = []

    def falso(texto, **kwargs):
        llamadas.append((texto, kwargs))
        return Resultado()

    monkeypatch.setattr(pipeline, "_analizar", falso)
    extractor = ExtractorFalso({})

    pipeline.analizar("Paciente con HTA.", ["HTA"], True, extractor)

    assert llamadas == [
        (
            "Paciente con HTA.",
            {
                "diagnosticos_registrados": ["HTA"],
                "descartar_registrados": True,
                "extractor": extractor,
                "normalizador": None,
                "modo_mapeo": "hibrido",
            },
        )
    ]


def test_orquestador_interno_sin_deduplicar_conserva_las_variantes():
    nota = "Diabetes mellitus tipo 2 de larga data. Control de diabetes. Niega EPOC."
    palabras = ("Diabetes mellitus tipo 2", "diabetes", "EPOC")

    resultado = pipeline._analizar(
        nota, extractor=extractor_para(nota, *palabras), deduplicar=False
    )

    assert set(por_texto(resultado)) == {"Diabetes mellitus tipo 2", "diabetes", "EPOC"}
    assert all(d.motivo is not MotivoDescarte.DUPLICADO_INTERNO for d in resultado.descartes)
    verificar_posiciones(nota, resultado)


def test_orquestador_interno_pasa_la_configuracion_a_la_extraccion():
    configuracion = ConfiguracionExtraccion(max_distancia_fusion=None, filtro_longitud="ninguno")

    resultado = pipeline._analizar(
        NOTA_CON_FOLIOS,
        extractor=extractor_para(NOTA_CON_FOLIOS, *PALABRAS_CON_FOLIOS),
        configuracion=configuracion,
    )

    entidades = por_texto(resultado)
    # Sin fusión, "Diabetes" y "mellitus tipo 2" no se unen a través de los guiones, y sin
    # filtro de longitud "tos" sobrevive.
    assert {"Diabetes", "mellitus tipo 2", "tos"} <= set(entidades)
    verificar_posiciones(NOTA_CON_FOLIOS, resultado)


def test_menciones_del_modelo_en_posiciones_del_texto_original():
    extractor = extractor_para(NOTA_CON_FOLIOS, *PALABRAS_CON_FOLIOS)

    menciones = pipeline.menciones_del_modelo(NOTA_CON_FOLIOS, extractor=extractor)

    # Crudas: sin fusión ("Diabetes" y "mellitus tipo 2" por separado), sin filtros ("tos")
    # y sin agrupar (dos menciones de "HTA").
    assert [m.texto for m in menciones] == [
        "HTA",
        "Diabetes",
        "mellitus tipo 2",
        "EPOC",
        "cáncer",
        "HTA",
        "tos",
    ]
    for mencion in menciones:
        assert NOTA_CON_FOLIOS[mencion.inicio : mencion.fin] == mencion.texto
    assert [m.inicio for m in menciones] == sorted(m.inicio for m in menciones)


def test_menciones_del_modelo_con_varios_fragmentos():
    parrafo = "Paciente estable con HTA en manejo y EPOC sin exacerbaciones recientes. "
    nota = f"{ENCABEZADO}\n" + "\n".join(parrafo * 6 for _ in range(6))

    menciones = pipeline.menciones_del_modelo(nota, extractor=extractor_para(nota, "HTA", "EPOC"))

    assert len(menciones) == 72
    for mencion in menciones:
        assert nota[mencion.inicio : mencion.fin] == mencion.texto


@pytest.mark.parametrize("texto", [None, "", "   \n\n", f"{ENCABEZADO}\n-----\n"])
def test_menciones_del_modelo_texto_vacio_no_carga_el_modelo(texto, monkeypatch):
    def no_cargar():
        raise AssertionError("no debía pedir el extractor")

    monkeypatch.setattr(pipeline, "_extractor_compartido", no_cargar)

    assert pipeline.menciones_del_modelo(texto) == []


def test_menciones_del_modelo_usa_el_extractor_compartido(monkeypatch):
    monkeypatch.setattr(pipeline, "_EXTRACTOR", None)
    usados = []

    def extraer(self, fragmentos):
        usados.append(self)
        return [[] for _ in fragmentos]

    monkeypatch.setattr(ExtractorHF, "extraer", extraer)

    assert pipeline.menciones_del_modelo("Paciente con HTA.") == []
    assert usados == [pipeline._extractor_compartido()]


# --- Modelo real -------------------------------------------------------------------------

EJEMPLO_README = (
    "Paciente femenina de 67 años con antecedente de HTA y diabetes mellitus tipo 2 en "
    "manejo con metformina. Niega EPOC. Madre con cáncer de mama."
)

NOTA_LARGA = (
    "Folio: 1 | Fecha: 2024-05-10 08:00:00.0\n"
    "NOTA DE INGRESO - MEDICINA INTERNA\n"
    "Paciente masculino de 72 años que consulta por disnea de medianos esfuerzos de una "
    "semana de evolución, asociada a edema de miembros inferiores y ortopnea de dos "
    "almohadas. Antecedentes personales: hipertensión arterial de 15 años de evolución en "
    "manejo con losartán, diabetes mellitus tipo 2 con nefropatía diabética, insuficiencia "
    "cardiaca con fracción de eyección reducida, infarto agudo de miocardio en 2019 con "
    "colocación de stent, enfermedad renal crónica estadio 3b y EPOC con oxígeno "
    "domiciliario. Niega hepatopatía. Niega tabaquismo activo, exfumador de 30 paquetes "
    "año. Antecedentes familiares: padre con cáncer de colon, madre con enfermedad de "
    "Alzheimer.\n"
    "----------------------------------------\n"
    "Revisión por sistemas: refiere tos con expectoración blanquecina, sin fiebre. Niega "
    "dolor torácico. Sin hematuria.\n"
    "Examen físico: TA 150/90, FC 98, FR 24, SatO2 88 % al aire ambiente. Ingurgitación "
    "yugular grado II, crépitos bibasales, edema grado II en miembros inferiores.\n"
    "Paraclínicos: creatinina 2,1 mg/dL, BUN 45, hemoglobina 10,8 g/dL, BNP 1.250 pg/mL, "
    "glucemia 210 mg/dL, HbA1c 8,9 %. Radiografía de tórax con cardiomegalia y signos de "
    "congestión pulmonar.\n"
    "Folio: 2 | Fecha: 2024-05-10 09:30:00.0\n"
    "Análisis: paciente con insuficiencia cardiaca descompensada de perfil húmedo y "
    "caliente, probablemente por transgresión hídrica y mala adherencia, en contexto de "
    "enfermedad renal crónica y diabetes mellitus mal controlada. Se descarta síndrome "
    "coronario agudo por troponinas negativas. Se inicia diurético de asa endovenoso, "
    "restricción hídrica y ajuste de insulina. Seguimiento por nefrología y cardiología.\n"
    "Plan: balance hídrico estricto, peso diario, control de electrolitos y función renal "
    "cada 24 horas, ecocardiograma transtorácico, oxígeno por cánula nasal para saturación "
    "mayor de 90 %, tromboprofilaxis con enoxaparina ajustada a función renal y educación "
    "al paciente y su familia sobre signos de alarma."
)


# Entidades que el modelo real marca en NOTA_LARGA (sin el ruido "yugular" y "exfumador").
ENTIDADES_NOTA_LARGA = (
    "hipertensión arterial",
    "diabetes mellitus tipo 2",
    "nefropatía diabética",
    "insuficiencia cardiaca",
    "infarto agudo de miocardio",
    "enfermedad renal crónica",
    "EPOC",
    "hepatopatía",
    "cáncer de colon",
    "enfermedad de Alzheimer",
    "cardiomegalia",
    "síndrome coronario agudo",
)


def test_nota_larga_conserva_enfermedades_distintas_con_el_mismo_prefijo():
    """Corregido (#24): la depuración fusionaba "enfermedad de Alzheimer" (de la madre)
    con "enfermedad renal crónica" por Jaro-Winkler, y desaparecía de la salida."""
    resultado = pipeline.analizar(
        NOTA_LARGA, extractor=extractor_para(NOTA_LARGA, *ENTIDADES_NOTA_LARGA)
    )

    entidades = por_texto(resultado)
    assert entidades["enfermedad de Alzheimer"].estado is Estado.FAMILIAR
    assert entidades["enfermedad renal crónica"].estado is Estado.CONFIRMADA
    assert entidades["síndrome coronario agudo"].estado is Estado.NEGADA
    verificar_posiciones(NOTA_LARGA, resultado)


@pytest.fixture(scope="module")
def extractor_real():
    extractor = ExtractorHF()
    inicio = time.perf_counter()
    extractor.extraer(["Paciente con HTA."])  # carga el modelo fuera de la medición
    print(f"\ncarga del modelo: {time.perf_counter() - inicio:.2f} s")
    return extractor


@pytest.mark.modelo
def test_modelo_real_ejemplo_del_readme(extractor_real):
    inicio = time.perf_counter()
    resultado = pipeline.analizar(EJEMPLO_README, extractor=extractor_real)
    duracion = time.perf_counter() - inicio
    print(f"ejemplo del README: {duracion:.2f} s")
    print(json.dumps(resultado.to_dict(), ensure_ascii=False, indent=2))

    verificar_posiciones(EJEMPLO_README, resultado)
    estados = {e.texto.lower(): e.estado for e in resultado.entidades}
    assert estados["hta"] is Estado.CONFIRMADA
    assert estados["diabetes mellitus tipo 2"] is Estado.CONFIRMADA
    assert estados["epoc"] is Estado.NEGADA
    assert estados["cáncer de mama"] is Estado.FAMILIAR
    assert resultado.indice_charlson == 1
    assert duracion < 5


@pytest.mark.modelo
def test_modelo_real_nota_de_2000_caracteres(extractor_real):
    assert 1800 <= len(NOTA_LARGA) <= 2200
    inicio = time.perf_counter()
    resultado = pipeline.analizar(NOTA_LARGA, extractor=extractor_real)
    duracion = time.perf_counter() - inicio
    print(f"nota de {len(NOTA_LARGA)} caracteres: {duracion:.2f} s")
    for entidad in resultado.entidades:
        print(
            f"  {entidad.texto!r} {entidad.estado.value} {entidad.categoria_charlson} "
            f"{entidad.cie10}"
        )
    print(f"  índice de Charlson: {resultado.indice_charlson}")

    verificar_posiciones(NOTA_LARGA, resultado)
    estados = {e.texto.lower(): e.estado for e in resultado.entidades}
    assert estados["síndrome coronario agudo"] is Estado.NEGADA
    assert estados["cáncer de colon"] is Estado.FAMILIAR
    assert estados["enfermedad de alzheimer"] is Estado.FAMILIAR
    assert resultado.indice_charlson > 0
    assert duracion < 5


def test_entidad_separada_por_estado_toma_el_texto_de_sus_menciones():
    """La parte familiar de una entidad fusionada se nombra con su propia mención."""
    nota = "Paciente con diabetes mellitus tipo 2. Padre con diabetes."
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "diabetes mellitus tipo 2", "diabetes")
    )
    por_estado = {e.estado: e for e in resultado.entidades}
    assert por_estado[Estado.CONFIRMADA].texto == "diabetes mellitus tipo 2"
    assert por_estado[Estado.FAMILIAR].texto == "diabetes"
    assert por_estado[Estado.FAMILIAR].categoria_charlson is None
    assert [m.texto for m in por_estado[Estado.FAMILIAR].menciones] == ["diabetes"]


# --- Categoría de Charlson con todas las menciones (#113) --------------------------------


def test_categoria_con_las_menciones_si_el_nombre_no_mapea():
    """La fusión nombra la entidad "eppoc", que no mapea; "EPOC" sí, y da la categoría."""
    nota = "Paciente con EPOC, epoc y eppoc."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "EPOC", "epoc", "eppoc"))

    (entidad,) = resultado.entidades
    assert len(entidad.menciones) == 3
    assert entidad.categoria_charlson is CategoriaCharlson.PULMONAR_CRONICA
    assert entidad.cie10 == "J44"
    assert resultado.indice_charlson == 1


def test_categoria_del_nombre_si_mapea():
    """Si el nombre mapea, manda el nombre aunque una mención dé otra categoría."""
    nota = "Diabetes mellitus con nefropatía diabética. Control de diabetes."
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "Diabetes mellitus con nefropatía diabética", "diabetes"),
    )

    (entidad,) = resultado.entidades
    assert entidad.texto == "Diabetes mellitus con nefropatía diabética"
    assert entidad.categoria_charlson is CategoriaCharlson.DIABETES_COMPLICADA


def test_sin_categoria_si_ninguna_mencion_mapea():
    nota = "Paciente con cefalea y cefalea tensional."
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "cefalea tensional", "cefalea")
    )

    assert all(e.categoria_charlson is None for e in resultado.entidades)
    assert resultado.indice_charlson == 0


def test_la_diabetes_ya_no_se_absorbe_en_prediabetes():
    # Cambio de #194: antes "diabetes" quedaba absorbida en "Prediabetes" por las letras y
    # se perdía. Ahora es su propia entidad. "Riesgo de diabetes" sale confirmada porque
    # el contexto no trata "riesgo de" como sospecha; eso es otro problema.
    nota = "Prediabetes en control. Hoy diabetes mellitus tipo 2 en tratamiento."
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "Prediabetes", "diabetes mellitus tipo 2")
    )

    assert {e.texto for e in resultado.entidades} == {"Prediabetes", "diabetes mellitus tipo 2"}
    assert resultado.indice_charlson == 1


@pytest.mark.parametrize(
    ("nota", "palabras"),
    [
        (
            "Diabetes gestacional en el tercer trimestre. La diabetes se controla con dieta.",
            ("Diabetes gestacional", "diabetes"),
        ),
        (
            "Diabetes insípida central. Control de diabetes.",
            ("Diabetes insípida central", "diabetes"),
        ),
        (
            "Carcinoma basocelular en ala nasal. El carcinoma no recidivó.",
            ("Carcinoma basocelular", "carcinoma"),
        ),
        # "Pre diabetes" y no "Prediabetes": la absorción es por palabras completas (#194).
        ("Pre diabetes en control. Riesgo de diabetes.", ("Pre diabetes", "diabetes")),
        (
            "Carcinoma in situ de cérvix. Se reseca el carcinoma.",
            ("Carcinoma in situ de cérvix", "carcinoma"),
        ),
    ],
)
def test_la_exclusion_del_nombre_manda_sobre_las_menciones(nota, palabras):
    """Si una regla excluye el nombre a propósito, la mención corta no le da categoría."""
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, *palabras))

    assert all(e.categoria_charlson is None for e in resultado.entidades)
    assert resultado.indice_charlson == 0


@pytest.mark.parametrize(
    ("nota", "palabras", "categoria"),
    [
        (
            "Paciente con EPOC. Cursa con eppoc sin insuficiencia respiratoria.",
            ("EPOC", "eppoc sin insuficiencia respiratoria"),
            CategoriaCharlson.PULMONAR_CRONICA,
        ),
        (
            "Diabetes mellitus tipo 2. Refiere diabetis mellitus tipo 2 sin complicaciones.",
            ("Diabetes mellitus tipo 2", "diabetis mellitus tipo 2 sin complicaciones"),
            CategoriaCharlson.DIABETES,
        ),
    ],
)
def test_la_exclusion_de_otra_regla_no_bloquea_las_menciones(nota, palabras, categoria):
    """El nombre mal escrito trae la exclusión de otra enfermedad: la mención sí vale."""
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, *palabras))

    assert [e.categoria_charlson for e in resultado.entidades] == [categoria]
    assert resultado.indice_charlson == 1


# --- Siglas en minúscula (#114) -------------------------------------------------------


@pytest.mark.modelo
@pytest.mark.parametrize(
    "nota",
    [
        "pte con epoc",
        "hta, dm, epoc",
        "paciente masculino de 70 años con epoc oxigenorequiriente",
    ],
)
def test_modelo_real_siglas_en_minuscula_como_en_mayuscula(extractor_real, nota):
    """US1 de `robustez`: mismas entidades e índice que con las siglas en mayúscula."""
    mayusculas = nota
    for sigla in ("epoc", "hta", "dm"):
        mayusculas = mayusculas.replace(sigla, sigla.upper())
    minuscula = pipeline.analizar(nota, extractor=extractor_real)
    referencia = pipeline.analizar(mayusculas, extractor=extractor_real)

    assert [e.texto.upper() for e in minuscula.entidades] == [
        e.texto.upper() for e in referencia.entidades
    ]
    assert [e.categoria_charlson for e in minuscula.entidades] == [
        e.categoria_charlson for e in referencia.entidades
    ]
    assert minuscula.indice_charlson == referencia.indice_charlson > 0
    verificar_posiciones(nota, minuscula)


# --- Corrección de ortografía en Charlson (#115) ----------------------------------------


def test_entidad_mal_escrita_recibe_categoria_por_correccion():
    nota = "Paciente con diabetis mellitus tipo 2 en manejo."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "diabetis mellitus tipo 2"))

    (entidad,) = resultado.entidades
    assert entidad.categoria_charlson is CategoriaCharlson.DIABETES
    assert entidad.correcciones == (("diabetis", "diabetes"),)
    assert entidad.to_dict()["categoria_por_correccion"] is True
    assert resultado.indice_charlson == 1


def test_mencion_bien_escrita_gana_a_la_correccion():
    nota = "Paciente con EPOC, epoc y eppoc."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "EPOC", "epoc", "eppoc"))

    (entidad,) = resultado.entidades
    assert entidad.categoria_charlson is CategoriaCharlson.PULMONAR_CRONICA
    assert entidad.correcciones == ()


def test_negada_mal_escrita_no_suma():
    nota = "Niega cirrocis."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "cirrocis"))

    (entidad,) = resultado.entidades
    assert entidad.estado is Estado.NEGADA
    assert entidad.categoria_charlson is None and entidad.correcciones == ()
    assert resultado.indice_charlson == 0


def test_la_exclusion_se_busca_en_el_nombre_corregido():
    """ "pre diabetis" corregido es "pre diabetes", que la regla excluye (revisión del #124)."""
    nota = "Paciente con pre diabetis. Control de diabetis."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "pre diabetis", "diabetis"))

    assert all(e.categoria_charlson is None for e in resultado.entidades)
    assert resultado.indice_charlson == 0


# --- Cruce con los registrados por categoría (#176) ------------------------------------


def test_la_complicacion_separada_coincide_con_la_diabetes_complicada_registrada():
    nota = "Paciente con DM2 con retinopatía diabética."
    registrados = (
        d for d in ["Diabetes mellitus no insulinodependiente con complicaciones oftálmicas"]
    )  # un generador: se recorre dos veces
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=registrados,
        extractor=extractor_para(nota, "DM2", "retinopatía diabética"),
    )
    entidades = por_texto(resultado)
    assert (entidades["DM2"].registrado, entidades["DM2"].coincide_por) == (True, "texto")
    retinopatia = entidades["retinopatía diabética"]
    assert (retinopatia.registrado, retinopatia.coincide_por) == (True, "categoria")


def test_un_segundo_tumor_sigue_como_posible_subregistro():
    nota = "Paciente con cáncer de mama."
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=["Tumor maligno de la próstata"],
        extractor=extractor_para(nota, "cáncer de mama"),
    )
    (cancer,) = resultado.entidades
    assert cancer.registrado is False


def test_cruce_por_categoria_con_descartar():
    nota = "Paciente con DM2 con retinopatía diabética."
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=[
            "Diabetes mellitus no insulinodependiente con complicaciones oftálmicas"
        ],
        descartar_registrados=True,
        extractor=extractor_para(nota, "DM2", "retinopatía diabética"),
    )
    assert resultado.entidades == []
    detalles = [d.detalle for d in resultado.descartes]
    assert any("misma categoría de Charlson" in d for d in detalles)


def test_sin_diabetes_por_cruzar_no_mapea_los_registrados(monkeypatch):
    llamadas = []
    original = pipeline._categorias

    def espiar(textos, normalizador):
        llamadas.append(list(textos))
        return original(textos, normalizador)

    monkeypatch.setattr(pipeline, "_categorias", espiar)
    nota = "Paciente con EPOC."
    pipeline.analizar(
        nota, diagnosticos_registrados=["Hipertensión"], extractor=extractor_para(nota, "EPOC")
    )
    assert llamadas == []


# --- La sigla entre paréntesis hereda el estado (#195) -------------------------------


@pytest.mark.parametrize(
    ("nota", "larga", "sigla", "estado"),
    [
        (
            "Sospecha clínica de enfermedad pulmonar obstructiva crónica (EPOC) por tabaco.",
            "enfermedad pulmonar obstructiva crónica",
            "EPOC",
            Estado.POSIBLE,
        ),
        (
            "Antecedente de enfermedad renal crónica (ERC) estadio 3.",
            "enfermedad renal crónica",
            "ERC",
            Estado.CONFIRMADA,
        ),
        (
            "Niega enfermedad pulmonar obstructiva crónica ( EPOC ) y asma.",
            "enfermedad pulmonar obstructiva crónica",
            "EPOC",
            Estado.NEGADA,
        ),
    ],
)
def test_la_sigla_entre_parentesis_hereda_el_estado(nota, larga, sigla, estado):
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, larga, sigla))

    estados = {(e.texto, e.estado) for e in resultado.entidades}
    assert (larga, estado) in estados
    assert all(est is estado for txt, est in estados if txt == sigla)


def test_un_parentesis_con_mas_texto_es_una_explicacion_y_no_hereda():
    nota = "Niega metrorragias (menopausia hace 4 años)."
    extractor = extractor_para(nota, "metrorragias", "menopausia")
    resultado = pipeline.analizar(nota, extractor=extractor)

    assert {(e.texto, e.estado) for e in resultado.entidades} == {
        ("metrorragias", Estado.NEGADA),
        ("menopausia", Estado.CONFIRMADA),
    }


def test_en_una_cadena_de_parentesis_solo_hereda_el_primero():
    nota = "Sospecha de neumonía adquirida (NAC) (COVID)."
    extractor = extractor_para(nota, "neumonía adquirida", "NAC", "COVID")
    resultado = pipeline.analizar(nota, extractor=extractor)

    # "NAC" es otra mención de la neumonía (#203); "(COVID)" no aclara a una mención.
    assert {e.texto: e.estado for e in resultado.entidades} == {
        "neumonía adquirida": Estado.POSIBLE,
        "COVID": Estado.CONFIRMADA,
    }


def test_lo_que_no_es_aclaracion_no_hereda():
    # El paréntesis no va pegado a otra mención: la sigla conserva su propio contexto.
    nota = "Sospecha de neumonía. Antecedente de tabaquismo (EPOC desde 2010)."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "neumonía", "EPOC"))

    assert {(e.texto, e.estado) for e in resultado.entidades} == {
        ("neumonía", Estado.POSIBLE),
        ("EPOC", Estado.CONFIRMADA),
    }


# --- Enumeraciones "A, B o C" (#196) -------------------------------------------------


@pytest.mark.parametrize(
    ("nota", "palabras", "estado"),
    [
        (
            "Se niega antecedente de asma bronquial, tosferina o tuberculosis pulmonar.",
            ("asma bronquial", "tosferina", "tuberculosis pulmonar"),
            Estado.NEGADA,
        ),
        (
            "El paciente niega antecedentes oncología, episodios convulsivos o hepatopatía "
            "crónica.",
            ("oncología", "episodios convulsivos", "hepatopatía crónica"),
            Estado.NEGADA,
        ),
        (
            "Niega asma bronquial alérgica, diabetes mellitus tipo 2 (DM2) o enfermedad "
            "renal crónica.",
            ("asma bronquial alérgica", "diabetes mellitus tipo 2", "DM2", "enfermedad renal"),
            Estado.NEGADA,
        ),
        (
            "Madre con diabetes mellitus tipo 2, cardiopatía isquémica crónica y cáncer de mama.",
            ("diabetes mellitus tipo 2", "cardiopatía isquémica crónica", "cáncer de mama"),
            Estado.FAMILIAR,
        ),
        (
            "Niega asma bronquial alérgica y/o enfermedad renal crónica avanzada.",
            ("asma bronquial alérgica", "enfermedad renal crónica avanzada"),
            Estado.NEGADA,
        ),
        # #218: "de forma categórica y explícita" no corta, y la lista sigue.
        (
            "Se niega de forma categórica y explícita haber presentado episodios previos de "
            "dolor torácico opresivo, infarto agudo de miocardio, angina de pecho o "
            "equivalentes anginosos.",
            ("infarto agudo de miocardio", "angina de pecho", "equivalentes anginosos"),
            Estado.NEGADA,
        ),
        # #213: el modelo no marcó el primer elemento ni el último.
        (
            "La paciente niega rotundamente haber presentado episodios de dolor precordial "
            "opresivo, disnea paroxística nocturna o palpitaciones, y tras un ecocardiograma.",
            ("disnea paroxística nocturna",),
            Estado.NEGADA,
        ),
        (
            "Sin fiebre, tos productiva ni insuficiencia cardiaca.",
            ("tos productiva", "insuficiencia cardiaca"),
            Estado.NEGADA,
        ),
        # #207: un elemento que el modelo no marcó y palabras con "de" delante del último.
        (
            "El paciente niega haber presentado síncope, palpitaciones sostenidas o historia "
            "previa de infarto agudo de miocardio.",
            ("síncope", "infarto agudo de miocardio"),
            Estado.NEGADA,
        ),
        (
            "Niega diabetes mellitus, dislipidemia mixta o antecedente de enfermedad "
            "cerebrovascular.",
            ("diabetes mellitus", "enfermedad cerebrovascular"),
            Estado.NEGADA,
        ),
        (
            "Se sospecha infección respiratoria baja, derrame pleural y neumonía basal.",
            ("infección respiratoria baja", "derrame pleural", "neumonía basal"),
            Estado.POSIBLE,
        ),
    ],
)
def test_la_enumeracion_comparte_el_estado_del_primero(nota, palabras, estado):
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, *palabras))

    # La sigla entre paréntesis ("DM2") va como mención de su entidad (#203).
    esperado = {p: estado for p in palabras if p != "DM2"}
    assert {e.texto: e.estado for e in resultado.entidades} == esperado


@pytest.mark.parametrize(
    ("nota", "palabras", "esperado"),
    [
        # Sin conjunción final la coma separa cláusulas (#84).
        (
            "Sin secuelas motoras, demencia vascular leve.",
            ("secuelas motoras", "demencia vascular leve"),
            {"secuelas motoras": Estado.NEGADA, "demencia vascular leve": Estado.CONFIRMADA},
        ),
        # Lo que hay entre las dos no es solo coma o conjunción.
        (
            "Niega fiebre y presenta diabetes mellitus tipo 2.",
            ("fiebre", "diabetes mellitus tipo 2"),
            {"fiebre": Estado.NEGADA, "diabetes mellitus tipo 2": Estado.CONFIRMADA},
        ),
        # Un salto de línea entre elementos corta la enumeración (queda fijado).
        (
            "Niega asma bronquial alérgica,\ndiabetes mellitus tipo 2 o cirrosis hepática.",
            ("asma bronquial alérgica", "diabetes mellitus tipo 2", "cirrosis hepática"),
            {
                "asma bronquial alérgica": Estado.NEGADA,
                "diabetes mellitus tipo 2": Estado.CONFIRMADA,
                "cirrosis hepática": Estado.CONFIRMADA,
            },
        ),
        # #207: las palabras delante del último elemento tienen que terminar en "de".
        (
            "Niega fiebre y tiene diabetes mellitus tipo 2.",
            ("fiebre", "diabetes mellitus tipo 2"),
            {"fiebre": Estado.NEGADA, "diabetes mellitus tipo 2": Estado.CONFIRMADA},
        ),
        # Revisión de #213: el tramo de un disparador que va detrás no abre enumeración,
        # y "antecedente de" tras una coma sola abre otra cláusula.
        (
            "TEP descartado, neumonía y derrame pleural.",
            ("TEP", "neumonía", "derrame pleural"),
            {
                "TEP": Estado.NEGADA,
                "neumonía": Estado.CONFIRMADA,
                "derrame pleural": Estado.CONFIRMADA,
            },
        ),
        (
            "Sin signos de infección, antecedente de EPOC y diabetes mellitus.",
            ("EPOC", "diabetes mellitus"),
            {"EPOC": Estado.CONFIRMADA, "diabetes mellitus": Estado.CONFIRMADA},
        ),
        # Un cierre que empieza con su propio disparador es otra cláusula (NUBes, #213).
        (
            "No presenta otras comorbilidades cardiovasculares, tabaquismo leve y sin tratamiento.",
            ("comorbilidades cardiovasculares", "tabaquismo leve"),
            {
                "comorbilidades cardiovasculares": Estado.NEGADA,
                "tabaquismo leve": Estado.CONFIRMADA,
            },
        ),
        # Revisión de #207: una cláusula con verbo o artículo no es un elemento.
        (
            "Madre con cáncer de mama, el paciente tiene historia de diabetes y EPOC.",
            ("cáncer de mama", "diabetes", "EPOC"),
            {
                "cáncer de mama": Estado.FAMILIAR,
                "diabetes": Estado.CONFIRMADA,
                "EPOC": Estado.CONFIRMADA,
            },
        ),
        (
            "Niega fiebre y tiene antecedente de diabetes mellitus.",
            ("fiebre", "diabetes mellitus"),
            {"fiebre": Estado.NEGADA, "diabetes mellitus": Estado.CONFIRMADA},
        ),
        (
            "Niega fiebre, tiene historia de diabetes y EPOC.",
            ("fiebre", "diabetes", "EPOC"),
            {"fiebre": Estado.NEGADA, "diabetes": Estado.CONFIRMADA, "EPOC": Estado.CONFIRMADA},
        ),
        (
            "Sin fiebre, el paciente es portador de VIH o EPOC.",
            ("fiebre", "VIH", "EPOC"),
            {"fiebre": Estado.NEGADA, "VIH": Estado.CONFIRMADA, "EPOC": Estado.CONFIRMADA},
        ),
        (
            "Niega fiebre, tos, hoy ingresa por cuadro de neumonía y EPOC.",
            ("fiebre", "neumonía", "EPOC"),
            {"fiebre": Estado.NEGADA, "neumonía": Estado.CONFIRMADA, "EPOC": Estado.CONFIRMADA},
        ),
        (
            "Niega fiebre, excepto en contexto de asma y diabetes.",
            ("fiebre", "asma", "diabetes"),
            {"fiebre": Estado.NEGADA, "asma": Estado.CONFIRMADA, "diabetes": Estado.CONFIRMADA},
        ),
        (
            "Niega fiebre y ahora tiene cuadro de neumonía.",
            ("fiebre", "neumonía"),
            {"fiebre": Estado.NEGADA, "neumonía": Estado.CONFIRMADA},
        ),
        # Un elemento intermedio con barrera ("refiere") corta la enumeración.
        (
            "Niega fiebre, refiere tos o insuficiencia cardiaca.",
            ("fiebre", "insuficiencia cardiaca"),
            {"fiebre": Estado.NEGADA, "insuficiencia cardiaca": Estado.CONFIRMADA},
        ),
        # Sin coma ni conjunción entre las dos no hay enumeración.
        (
            "Niega dolor torácico de esfuerzo.",
            ("dolor torácico", "esfuerzo"),
            {"dolor torácico": Estado.NEGADA, "esfuerzo": Estado.NEGADA},
        ),
        # Una enumeración que empieza confirmada no cambia nada.
        (
            "Hipertensión arterial, dislipidemia y sin diabetes.",
            ("Hipertensión arterial", "dislipidemia", "diabetes"),
            {
                "Hipertensión arterial": Estado.CONFIRMADA,
                "dislipidemia": Estado.CONFIRMADA,
                "diabetes": Estado.NEGADA,
            },
        ),
    ],
)
def test_lo_que_no_es_enumeracion_no_propaga(nota, palabras, esperado):
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, *palabras))

    assert {e.texto: e.estado for e in resultado.entidades} == esperado


# --- La sigla o el código entre paréntesis es otra mención de su entidad (#203) ------


@pytest.mark.parametrize(
    ("nota", "larga", "sigla"),
    [
        ("Se confirma artritis reumatoide (M05) seropositiva.", "artritis reumatoide", "M05"),
        (
            "Sospecha de enfermedad pulmonar obstructiva crónica (EPOC) por tabaco.",
            "enfermedad pulmonar obstructiva crónica",
            "EPOC",
        ),
    ],
)
def test_la_sigla_entre_parentesis_es_otra_mencion_de_la_entidad(nota, larga, sigla):
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, larga, sigla))

    assert [e.texto for e in resultado.entidades] == [larga]
    assert [m.texto for m in resultado.entidades[0].menciones] == [larga, sigla]


def test_la_sigla_suelta_en_otro_lugar_sigue_siendo_su_entidad():
    nota = "Antecedente de enfermedad renal crónica (ERC). Control de HTA. Niega EPOC."
    extractor = extractor_para(nota, "enfermedad renal crónica", "ERC", "HTA", "EPOC")
    resultado = pipeline.analizar(nota, extractor=extractor)

    textos = {e.texto: [m.texto for m in e.menciones] for e in resultado.entidades}
    assert textos["enfermedad renal crónica"] == ["enfermedad renal crónica", "ERC"]
    assert "EPOC" in textos and "HTA" in textos


def test_un_parentesis_con_explicacion_no_se_une():
    nota = "Niega metrorragias (menopausia hace 4 años)."
    extractor = extractor_para(nota, "metrorragias", "menopausia")
    resultado = pipeline.analizar(nota, extractor=extractor)

    assert sorted(e.texto for e in resultado.entidades) == ["menopausia", "metrorragias"]


@pytest.mark.parametrize(
    ("nota", "palabras", "categorias"),
    [
        # El paréntesis dice la causa: son dos entidades y cada una conserva su categoría.
        ("Paciente con nefropatía (DM2).", ("nefropatía", "DM2"), {"DM2": "diabetes"}),
        (
            "Demencia (ACV) en seguimiento.",
            ("Demencia", "ACV"),
            {"Demencia": "demencia", "ACV": "cerebrovascular"},
        ),
        ("Anemia (ERC) en estudio.", ("Anemia", "ERC"), {"ERC": "renal_cronica"}),
    ],
)
def test_un_parentesis_con_la_causa_es_otra_entidad(nota, palabras, categorias):
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, *palabras))

    assert sorted(e.texto for e in resultado.entidades) == sorted(palabras)
    assert {
        e.texto: e.categoria_charlson.value for e in resultado.entidades if e.categoria_charlson
    } == categorias


def test_la_sigla_unida_cruza_con_lo_registrado():
    nota = "Antecedente de enfermedad pulmonar obstructiva crónica (EPOC)."
    extractor = extractor_para(nota, "enfermedad pulmonar obstructiva crónica", "EPOC")
    resultado = pipeline.analizar(nota, extractor=extractor, diagnosticos_registrados=["EPOC"])

    [entidad] = resultado.entidades
    assert entidad.registrado is True


# --- Introductor de tema (#212) ----------------------------------------------------


def test_respecto_a_no_decide_el_estado_si_hay_otras_menciones():
    nota = (
        "Respecto a la insuficiencia cardíaca, la historia anterior mencionaba una sospecha "
        "de insuficiencia cardíaca con fracción de eyección preservada."
    )
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "insuficiencia cardíaca"))

    [ic] = resultado.entidades
    assert (ic.estado, ic.otros_estados) == (Estado.POSIBLE, (Estado.CONFIRMADA,))
    assert resultado.indice_charlson == 0


def test_respecto_a_sola_conserva_su_estado():
    nota = "En cuanto a la diabetes mellitus, buen control con metformina."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "diabetes mellitus"))

    [dm] = resultado.entidades
    assert dm.estado is Estado.CONFIRMADA


@pytest.mark.parametrize(
    "nota",
    [
        "Respecto a la HTA, en manejo con losartán. Se descarta HTA secundaria.",
        "Con relación a la EPOC, estable con broncodilatadores. Sin exacerbación de EPOC.",
    ],
)
def test_respecto_a_en_otra_oracion_si_cuenta(nota):
    palabra = "HTA" if "HTA" in nota else "EPOC"
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, palabra))

    [entidad] = resultado.entidades
    assert entidad.estado is Estado.CONFIRMADA
