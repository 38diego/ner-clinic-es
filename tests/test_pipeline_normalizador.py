"""`analizar()` con la segunda etapa del normalizador (#165), sin red ni modelos."""

import pytest

from ner_clinic import pipeline
from ner_clinic.normalizacion.decision import Decision
from ner_clinic.normalizacion.indice import ORIGEN_CATALOGO, Candidato
from ner_clinic.tipos import AlternativaCie10, CandidatoIndice, Estado
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_pipeline import extractor_para, por_texto


class SegundaFalsa:
    """Decide por tabla (texto → `Decision`) y guarda los lotes que recibe."""

    def __init__(self, tabla):
        self.tabla = tabla
        self.lotes = []

    def decidir(self, textos):
        self.lotes.append(list(textos))
        return {t: self.tabla.get(t) for t in dict.fromkeys(textos)}

    def alternativas(self, textos):
        self.lotes_alternativas = getattr(self, "lotes_alternativas", []) + [list(textos)]
        return {t: [AlternativaCie10("X00", f"opción de {t}", 0.9)] for t in textos}


def _decision(categoria, codigo, texto="descripción", origen=ORIGEN_CATALOGO, candidato=True):
    c = Candidato(codigo, texto, 0.91, origen) if candidato else None
    return Decision(categoria, codigo, 0.88, c)


def test_sin_normalizador_la_categoria_es_de_las_reglas():
    nota = "Paciente con EPOC. Refiere tos."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "EPOC", "tos"))
    entidades = por_texto(resultado)
    assert entidades["EPOC"].mapeo == "reglas" and entidades["EPOC"].candidato is None
    assert entidades["tos"].mapeo is None and entidades["tos"].categoria_charlson is None
    assert entidades["EPOC"].to_dict()["candidato"] is None


def test_con_normalizador_decide_la_segunda_etapa():
    nota = "Paciente con IRC estadio 4. Refiere tos."
    segunda = SegundaFalsa(
        {"IRC estadio 4": _decision(C.RENAL_CRONICA, "N18.4", "ERC estadio 4", "sinónimo")}
    )
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "IRC estadio 4", "tos"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    entidades = por_texto(resultado)
    irc = entidades["IRC estadio 4"]
    assert (irc.categoria_charlson, irc.cie10, irc.mapeo) == (
        C.RENAL_CRONICA,
        "N18.4",
        "normalizador",
    )
    assert irc.candidato == CandidatoIndice("ERC estadio 4", "N18.4", "sinónimo", 0.91)
    assert irc.to_dict()["candidato"] == {
        "texto": "ERC estadio 4",
        "codigo": "N18.4",
        "origen": "sinónimo",
        "similitud": 0.91,
    }
    assert entidades["tos"].mapeo is None
    assert resultado.indice_charlson == 2  # renal crónica
    # Un solo lote con los nombres y las menciones
    assert len(segunda.lotes) == 1 and set(segunda.lotes[0]) == {"IRC estadio 4", "tos"}


def test_el_normalizador_reemplaza_a_las_reglas():
    """Si la segunda etapa no da categoría, no se cae a las reglas: es el método medido."""
    nota = "Paciente con EPOC."
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "EPOC"),
        normalizador=SegundaFalsa({}),
        modo_mapeo="normalizador",
    )
    (epoc,) = resultado.entidades
    assert epoc.categoria_charlson is None and epoc.mapeo is None
    assert resultado.indice_charlson == 0


def test_si_el_nombre_no_da_categoria_prueba_las_menciones():
    nota = "Paciente con EPOC, epoc y eppoc."
    segunda = SegundaFalsa({"EPOC": _decision(C.PULMONAR_CRONICA, "J44.9")})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "EPOC", "epoc", "eppoc"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    (entidad,) = resultado.entidades
    assert entidad.texto != "EPOC"  # la fusión la nombra con otra variante
    assert entidad.categoria_charlson is C.PULMONAR_CRONICA and entidad.mapeo == "normalizador"


def test_la_mencion_no_vale_si_la_exclusion_quita_el_nombre():
    """ "diabetes gestacional" fusionada con "diabetes": la mención no le da diabetes."""
    nota = "Antecedente de diabetes gestacional. Control de diabetes."
    segunda = SegundaFalsa({"diabetes": _decision(C.DIABETES, "E11.9")})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "diabetes gestacional", "diabetes"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    gestacional = next(e for e in resultado.entidades if e.texto == "diabetes gestacional")
    assert gestacional.categoria_charlson is None


def test_la_exclusion_se_busca_tambien_en_el_nombre_corregido():
    """Como `test_la_exclusion_se_busca_en_el_nombre_corregido` de las reglas (#124)."""
    nota = "Paciente con pre diabetis. Control de diabetis."
    segunda = SegundaFalsa({"diabetis": _decision(C.DIABETES, "E11.9")})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "pre diabetis", "diabetis"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    assert all(e.categoria_charlson is None for e in resultado.entidades)


def test_una_mencion_contenida_en_el_nombre_no_le_da_categoria():
    """ "insuficiencia renal" fusionada dentro de "insuficiencia renal aguda": una parte del
    nombre no es una variante de él."""
    nota = "Insuficiencia renal aguda. Control de insuficiencia renal."
    segunda = SegundaFalsa({"insuficiencia renal": _decision(C.RENAL_CRONICA, "N19")})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "Insuficiencia renal aguda", "insuficiencia renal"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    aguda = next(e for e in resultado.entidades if e.texto == "Insuficiencia renal aguda")
    assert len(aguda.menciones) == 2 and aguda.categoria_charlson is None


def test_negadas_y_familiares_no_pasan_por_el_normalizador():
    nota = "Niega EPOC. Madre con diabetes. Paciente con IRC."
    segunda = SegundaFalsa({"IRC": _decision(C.RENAL_CRONICA, "N18.9")})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "EPOC", "diabetes", "IRC"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    estados = {e.texto: e.estado for e in resultado.entidades}
    assert estados == {"EPOC": Estado.NEGADA, "diabetes": Estado.FAMILIAR, "IRC": Estado.CONFIRMADA}
    assert set(segunda.lotes[0]) == {"IRC"}


def test_sin_entidades_a_mapear_no_codifica():
    nota = "Niega EPOC."
    segunda = SegundaFalsa({})
    pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "EPOC"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    assert segunda.lotes == [[]]


def test_correcciones_solo_si_las_reglas_dan_la_misma_categoria():
    nota = "Paciente con diabetis mellitus y enfisemma."
    segunda = SegundaFalsa(
        {
            "diabetis mellitus": _decision(C.DIABETES, "E11.9"),
            "enfisemma": _decision(C.INSUFICIENCIA_CARDIACA, "I50.9"),  # otra categoría
        }
    )
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "diabetis mellitus", "enfisemma"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    entidades = por_texto(resultado)
    assert entidades["diabetis mellitus"].correcciones == (("diabetis", "diabetes"),)
    assert entidades["enfisemma"].correcciones == ()


def test_categoria_solo_de_las_reglas_dentro_del_clasificador():
    nota = "Paciente con EPOC."
    segunda = SegundaFalsa({"EPOC": _decision(C.PULMONAR_CRONICA, "J44", candidato=False)})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "EPOC"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    (epoc,) = resultado.entidades
    assert epoc.mapeo == "normalizador" and epoc.candidato is None


@pytest.fixture
def rejilla(monkeypatch):
    from tests.test_clasificador import _rejilla_chica

    _rejilla_chica(monkeypatch)


def test_con_la_segunda_etapa_real(rejilla):
    """De punta a punta con `SegundaEtapa` (índice de trigramas y clasificador entrenado)."""
    import json

    from evaluacion import medicion_final as mf
    from ner_clinic.normalizacion import Normalizador, SegundaEtapa, modelo_a_json, modelo_de_json
    from ner_clinic.normalizacion.decision import K
    from tests.test_dos_etapas import TRAIN_6, _indice

    indice, codificador = _indice()
    modelo, eleccion = mf.entrenar_final(indice, codificador, TRAIN_6, [], {}, 3, K)
    guardado = modelo_de_json(json.loads(json.dumps(modelo_a_json(modelo, eleccion["umbral"]))))
    final = indice.con_sinonimos(mf.sinonimos_de(TRAIN_6), codificador)
    segunda = SegundaEtapa(Normalizador(final, codificador), *guardado)
    nota = "Tuvo un infarto cerebral. Refiere tos."
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "infarto cerebral", "tos"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    infarto = por_texto(resultado)["infarto cerebral"]
    assert infarto.categoria_charlson is C.CEREBROVASCULAR and infarto.mapeo == "normalizador"
    assert infarto.candidato is not None and infarto.candidato.codigo.startswith("I6")
    assert segunda.normalizador._memoria == {}  # la nota no queda guardada


def test_con_normalizador_los_registrados_se_mapean_con_el_normalizador():
    """#176: la categoría de los diagnósticos registrados sale del mismo mapeo."""
    registrado = "Diabetes mellitus no insulinodependiente con complicaciones oftálmicas"
    nota = "Paciente con retinopatía diabética."
    segunda = SegundaFalsa(
        {
            "retinopatía diabética": _decision(C.DIABETES_COMPLICADA, "E11.3"),
            registrado: _decision(C.DIABETES_COMPLICADA, "E11.39"),
        }
    )
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=[registrado],
        extractor=extractor_para(nota, "retinopatía diabética"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    (entidad,) = resultado.entidades
    assert (entidad.registrado, entidad.coincide_por) == (True, "categoria")
    assert segunda.lotes[-1] == [registrado]


# --- Modo híbrido (#180): reglas primero, normalizador de respaldo ---------------


def test_hibrido_es_el_modo_por_defecto_con_normalizador():
    """Lo que las reglas mapean sale de ellas; lo demás, del normalizador, en un lote."""
    nota = "Paciente con ICC y falla renal crónica. Refiere tos."
    segunda = SegundaFalsa(
        {
            # El normalizador confundiría la sigla: las reglas mandan
            "ICC": _decision(C.TUMOR_MALIGNO, "C22.0"),
            "falla renal crónica": _decision(C.RENAL_CRONICA, "N18.9"),
        }
    )
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "ICC", "falla renal crónica", "tos"),
        normalizador=segunda,
    )
    entidades = por_texto(resultado)
    icc, renal = entidades["ICC"], entidades["falla renal crónica"]
    assert (icc.categoria_charlson, icc.mapeo, icc.candidato) == (
        C.INSUFICIENCIA_CARDIACA,
        "reglas",
        None,
    )
    assert (renal.categoria_charlson, renal.cie10, renal.mapeo) == (
        C.RENAL_CRONICA,
        "N18.9",
        "normalizador",
    )
    assert renal.candidato is not None
    assert entidades["tos"].mapeo is None
    assert resultado.indice_charlson == 3  # ICC 1 + renal 2
    # El normalizador solo ve lo que las reglas no mapearon
    assert len(segunda.lotes) == 1 and set(segunda.lotes[0]) == {"falla renal crónica", "tos"}


def test_hibrido_sin_nada_que_normalizar_no_llama_al_normalizador():
    nota = "Paciente con EPOC."
    segunda = SegundaFalsa({})
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "EPOC"), normalizador=segunda
    )
    (epoc,) = resultado.entidades
    assert epoc.mapeo == "reglas" and segunda.lotes == []


def test_hibrido_las_exclusiones_de_las_reglas_siguen_en_el_normalizador():
    """Lo que una regla excluye ("diabetes gestacional") no se lo da el normalizador."""
    nota = "Paciente con diabetes gestacional."
    segunda = SegundaFalsa({"diabetes gestacional": _decision(C.DIABETES, "E11.9")})
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "diabetes gestacional"), normalizador=segunda
    )
    (entidad,) = resultado.entidades
    assert entidad.categoria_charlson is None


def test_hibrido_los_registrados_usan_reglas_y_luego_normalizador():
    """La entidad la mapean las reglas; el registrado, que no comparte palabras ni lo
    conocen las reglas, el normalizador. Coinciden por categoría (#176)."""
    registrados = ["Complicación ocular crónica"]
    nota = "Paciente con retinopatía diabética."
    segunda = SegundaFalsa(
        {"Complicación ocular crónica": _decision(C.DIABETES_COMPLICADA, "E11.3")}
    )
    resultado = pipeline.analizar(
        nota,
        diagnosticos_registrados=registrados,
        extractor=extractor_para(nota, "retinopatía diabética"),
        normalizador=segunda,
    )
    (entidad,) = resultado.entidades
    assert entidad.mapeo == "reglas"
    assert (entidad.registrado, entidad.coincide_por) == (True, "categoria")
    assert segunda.lotes == [["Complicación ocular crónica"]]


def test_modo_de_mapeo_desconocido():
    with pytest.raises(ValueError, match="modo_mapeo"):
        nota = "Paciente con EPOC."
        pipeline.analizar(nota, extractor=extractor_para(nota, "EPOC"), modo_mapeo="otro")


@pytest.mark.parametrize("modo", ["hibrido", "normalizador"])
def test_categorias_de_los_registrados_mantienen_el_orden(modo):
    textos = ["X sin regla", "EPOC", "Y sin regla"]
    segunda = SegundaFalsa(
        {
            "X sin regla": _decision(C.DEMENCIA, "F03"),
            "EPOC": _decision(C.PULMONAR_CRONICA, "J44.9"),
        }
    )
    categorias = pipeline._categorias(textos, segunda, modo)
    assert categorias == [C.DEMENCIA, C.PULMONAR_CRONICA, None]
    esperado = ["X sin regla", "Y sin regla"] if modo == "hibrido" else textos
    assert segunda.lotes == [esperado]


def test_normalizador_excluye_el_nombre_corregido():
    """La segunda etapa excluye "pre diabetis" tal como viene; el nombre corregido
    ("pre diabetes") también se mira en el pipeline."""
    nota = "Paciente con pre diabetis."
    segunda = SegundaFalsa({"pre diabetis": _decision(C.DIABETES, "E11.9")})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "pre diabetis"),
        normalizador=segunda,
        modo_mapeo="normalizador",
    )
    (entidad,) = resultado.entidades
    assert entidad.categoria_charlson is None


def test_hibrido_las_reglas_por_mencion_o_correccion_no_pasan_por_el_normalizador():
    nota = "Paciente con diabetis tipo 2."
    segunda = SegundaFalsa({})
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "diabetis tipo 2"), normalizador=segunda
    )
    (entidad,) = resultado.entidades
    assert (entidad.categoria_charlson, entidad.mapeo) == (C.DIABETES, "reglas")
    assert entidad.correcciones and segunda.lotes == []


def test_modo_normalizador_sin_normalizador_usa_las_reglas():
    nota = "Paciente con EPOC."
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, "EPOC"), modo_mapeo="normalizador"
    )
    assert resultado.entidades[0].mapeo == "reglas"


# --- Filtros del respaldo del híbrido (#185) --------------------------------------


def _con_candidato(categoria, codigo, descripcion):
    return Decision(categoria, codigo, 0.9, Candidato(codigo, descripcion, 0.7, ORIGEN_CATALOGO))


def _hibrido(nota, texto, decision, **extra):
    segunda = SegundaFalsa({texto: decision})
    resultado = pipeline.analizar(
        nota, extractor=extractor_para(nota, texto), normalizador=segunda, **extra
    )
    return resultado.entidades[0]


def test_respaldo_una_sigla_solo_vale_con_el_candidato_igual():
    distinta = _con_candidato(C.TUMOR_MALIGNO, "C83.70", "LB")
    assert _hibrido("Paciente con SB.", "SB", distinta).categoria_charlson is None
    igual = _con_candidato(C.TUMOR_MALIGNO, "C22.0", "HCC")
    assert _hibrido("Paciente con HCC.", "HCC", igual).categoria_charlson is C.TUMOR_MALIGNO


def test_respaldo_metastasico_necesita_senal_en_el_texto():
    generica = _con_candidato(C.TUMOR_METASTASICO, "C80.1", "tumor maligno")
    nota = "Se observa proliferación neoplásica."
    assert _hibrido(nota, "proliferación neoplásica", generica).categoria_charlson is None
    nota = "Se observa infiltración tumoral."
    entidad = _hibrido(nota, "infiltración tumoral", generica)
    assert entidad.categoria_charlson is C.TUMOR_METASTASICO


def test_los_filtros_no_aplican_en_modo_normalizador():
    distinta = _con_candidato(C.TUMOR_MALIGNO, "C83.70", "LB")
    entidad = _hibrido("Paciente con SB.", "SB", distinta, modo_mapeo="normalizador")
    assert entidad.categoria_charlson is C.TUMOR_MALIGNO


def test_el_interruptor_interno_quita_los_filtros():
    nota = "Paciente con SB."
    segunda = SegundaFalsa({"SB": _con_candidato(C.TUMOR_MALIGNO, "C83.70", "LB")})
    resultado = pipeline._analizar(
        nota, extractor=extractor_para(nota, "SB"), normalizador=segunda, filtros_respaldo=False
    )
    assert resultado.entidades[0].categoria_charlson is C.TUMOR_MALIGNO


def test_los_registrados_pasan_por_los_mismos_filtros():
    segunda = SegundaFalsa(
        {
            "SB": _con_candidato(C.DIABETES, "E11.9", "DM"),
            "Complicación X": _con_candidato(C.DIABETES, "E11.9", "descripción"),
        }
    )
    assert pipeline._categorias(["SB", "Complicación X"], segunda, "hibrido") == [None, C.DIABETES]


@pytest.mark.parametrize(
    "texto", ["lesiones hepáticas", "lesión ocupante de espacio en hígado", "LOE"]
)
def test_lesion_hepatica_no_es_hepatopatia_leve(texto):
    decision = _con_candidato(C.HEPATICA_LEVE, "K76.9", texto)
    nota = f"Se observa {texto}."
    assert _hibrido(nota, texto, decision).categoria_charlson is None


# Revisión del PR #188
@pytest.mark.parametrize(
    "texto",
    [
        "cirrosis con LOE hepática",
        "hepatopatía crónica con lesiones hepáticas",
        "hepatitis C crónica con LOE",
    ],
)
def test_lesion_hepatica_no_quita_la_hepatopatia_de_las_reglas(texto):
    from ner_clinic.charlson.correccion import mapear

    assert mapear(texto).categoria is C.HEPATICA_LEVE


@pytest.mark.parametrize(
    ("texto", "es_sigla"),
    [
        ("SB", True),
        ("sb", True),
        ("S.B.", True),
        ("SDMO", True),
        ("GLIOMA", False),
        ("TUMOR", False),
    ],
)
def test_que_cuenta_como_sigla(texto, es_sigla):
    assert pipeline._es_sigla(texto) is es_sigla


def test_sigla_sin_candidato_no_vale():
    assert not pipeline._respaldo_valido("SB", Decision(C.TUMOR_MALIGNO, "C83", 0.9, None))


@pytest.mark.parametrize(
    "texto",
    ["T2N1M1", "pT3N2M1", "cM1", "estadio IVb", "Estadío 4", "mts hepáticas", "metas óseas"],
)
def test_senales_de_metastasis(texto):
    d = Decision(C.TUMOR_METASTASICO, "C80", 0.9, Candidato("C80", "tumor", 0.7, ORIGEN_CATALOGO))
    assert pipeline._respaldo_valido(texto, d)


def test_el_respaldo_rechaza_el_nombre_y_acepta_una_mencion_con_senal():
    d = Decision(C.TUMOR_METASTASICO, "C80", 0.9, Candidato("C80", "tumor", 0.7, ORIGEN_CATALOGO))
    nota = "Lesión tumoral y lesión tumoral metastásica."
    segunda = SegundaFalsa({"lesión tumoral": d, "lesión tumoral metastásica": d})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "lesión tumoral metastásica"),
        normalizador=segunda,
    )
    assert resultado.entidades[0].categoria_charlson is C.TUMOR_METASTASICO


def test_los_filtros_de_los_registrados_respetan_el_modo_y_el_interruptor():
    d = Decision(C.TUMOR_METASTASICO, "C80", 0.9, Candidato("C80", "tumor", 0.7, ORIGEN_CATALOGO))
    segunda = SegundaFalsa({"tumor X": d})
    assert pipeline._categorias(["tumor X"], segunda, "hibrido") == [None]
    assert pipeline._categorias(["tumor X"], segunda, "normalizador") == [C.TUMOR_METASTASICO]
    assert pipeline._categorias(["tumor X"], segunda, "hibrido", False) == [C.TUMOR_METASTASICO]


def test_alternativas_cie10_para_confirmadas_y_posibles():
    """#230: las confirmadas y posibles traen las alternativas, también si mapearon las
    reglas; las negadas, no."""
    nota = "Paciente con EPOC. Probable diabetes. Niega asma."
    segunda = SegundaFalsa({})
    resultado = pipeline.analizar(
        nota,
        extractor=extractor_para(nota, "EPOC", "diabetes", "asma"),
        normalizador=segunda,
    )
    entidades = por_texto(resultado)
    assert entidades["EPOC"].mapeo == "reglas"
    assert entidades["EPOC"].alternativas_cie10 == (AlternativaCie10("X00", "opción de EPOC", 0.9),)
    assert entidades["diabetes"].to_dict()["alternativas_cie10"] == [
        {"codigo": "X00", "descripcion": "opción de diabetes", "similitud": 0.9}
    ]
    assert entidades["asma"].alternativas_cie10 == ()
    assert segunda.lotes_alternativas == [["EPOC", "diabetes"]]


def test_sin_normalizador_no_hay_alternativas():
    nota = "Paciente con EPOC."
    resultado = pipeline.analizar(nota, extractor=extractor_para(nota, "EPOC"))
    assert resultado.entidades[0].to_dict()["alternativas_cie10"] == []
