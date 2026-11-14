"""Tests de la segunda etapa por votos (#150) y de la jerarquía CIE-10, sin red."""

import json
import zipfile
from dataclasses import replace

import pytest

from evaluacion import dos_etapas as de
from evaluacion import jerarquia_cie10 as jc
from evaluacion import votos as vt
from evaluacion.__main__ import main
from evaluacion.normalizador import Candidato
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_dos_etapas import TRAIN_6, _indice, _mini
from tests.test_normalizador import DEV, Trigramas, _documento


def _p(codigo, similitud, categoria):
    return de.Propuesta(Candidato(codigo, "-", similitud), codigo, categoria)


# --- Jerarquía CIE-10 ----------------------------------------------------------------

XML = b"""<ICD10CM.tabular>
<chapter><name>1</name><desc>Infecciosas</desc>
  <section id="A00-A09"><diag><name>A00</name></diag><diag><name>A01</name></diag></section>
</chapter>
<chapter><name>2</name><desc>Neoplasias</desc>
  <section id="C00-C96"><desc>agrupa</desc></section>
  <section id="C30-C39"><diag><name>C34</name><diag><name>C34.1</name></diag></diag></section>
</chapter>
</ICD10CM.tabular>"""


def test_tabla_de_xml():
    tabla = jc.tabla_de_xml(XML)
    assert tabla == {
        "A00": ("A00-A09", "I"),
        "A01": ("A00-A09", "I"),
        "C34": ("C30-C39", "II"),
    }


def test_escribir_y_leer(tmp_path):
    ruta = tmp_path / "jerarquia.tsv"
    jc.escribir(jc.tabla_de_xml(XML), ruta)
    assert jc.leer(ruta)["C34"] == jc.Jerarquia("C30-C39", "II")


def test_tabla_versionada():
    tabla = jc.leer()
    assert len(tabla) == 1910
    assert jc.jerarquia_de("i63.9") == ("I60-I69", "IX")
    assert jc.jerarquia_de("C34.90") == ("C30-C39", "II")
    assert jc.jerarquia_de("O9A.1") == ("O94-O9A", "XV")
    assert jc.jerarquia_de("XYZ") is None
    assert jc.jerarquia_de("CAP.02") == ("CAP.02", "II")
    assert jc.jerarquia_de("CAP.99") is None


def test_descargar_verifica_el_md5(tmp_path, monkeypatch):
    zip_ = tmp_path / "origen.zip"
    with zipfile.ZipFile(zip_, "w") as z:
        z.writestr(jc.XML_TABULAR, XML)
    monkeypatch.setattr(jc.urllib.request, "urlopen", lambda url: zip_.open("rb"))
    monkeypatch.setattr(jc, "MD5_TABULAR", "0" * 32)
    with pytest.raises(ValueError, match="MD5"):
        jc.descargar(tmp_path / "destino")
    assert not list((tmp_path / "destino").iterdir())
    monkeypatch.setattr(jc, "MD5_TABULAR", jc.md5_de(zip_))
    ruta = jc.descargar(tmp_path / "destino")
    assert ruta.read_bytes() == zip_.read_bytes()


def test_propuesta_trae_bloque_y_capitulo():
    p = _p("I63.9", 0.9, C.CEREBROVASCULAR)
    assert (p.codigo_3, p.bloque, p.capitulo) == ("I63", "I60-I69", "IX")


# --- Votos ---------------------------------------------------------------------------

# Tres candidatos de tumor maligno detrás de un vecino sin categoría con más similitud
TOP = [
    _p("D49.2", 0.80, None),
    _p("C34.9", 0.78, C.TUMOR_MALIGNO),
    _p("C34.1", 0.77, C.TUMOR_MALIGNO),
    _p("C78.0", 0.75, C.TUMOR_METASTASICO),
    _p("R20.0", 0.74, None),
]


def test_con_k_1_es_el_primer_candidato():
    decisor = vt.votar(vt.Configuracion(k=1, umbral=0.5))
    referencia = de.primer_candidato(0.5)
    for top in (TOP, TOP[1:], TOP[3:]):
        assert decisor("x", top) == referencia("x", top)


def test_voto_por_categoria():
    c = vt.Configuracion("categoria", k=5, umbral=0.7)
    # "ninguna" pesa 0,80 + 0,74; tumor maligno, 0,78 + 0,77: gana tumor maligno
    assert vt.votar(c)("x", TOP) == (C.TUMOR_MALIGNO, "C34.9")
    assert [p.candidato.codigo for p in vt.votantes(c, TOP)] == ["C34.9", "C34.1"]
    assert vt.votar(replace(c, umbral=0.79))("x", TOP) is None


def test_temperatura_baja_manda_el_primero():
    c = vt.Configuracion("categoria", k=5, temperatura=0.001, umbral=0.5)
    assert vt.votar(c)("x", TOP) is None  # gana "ninguna" por el D49.2


def test_voto_jerarquico():
    # Por bloque: C30-C39 (0,78 + 0,77) le gana a D49 (0,80); dentro, tumor maligno
    bloque = vt.Configuracion("bloque", k=5, umbral=0.5)
    assert vt.votar(bloque)("x", TOP) == (C.TUMOR_MALIGNO, "C34.9")
    # Por capítulo: el II junta D49, C34, C34 y C78; dentro, gana tumor maligno
    capitulo = vt.Configuracion("capitulo", k=5, umbral=0.5)
    assert [p.candidato.codigo for p in vt.votantes(capitulo, TOP)] == ["C34.9", "C34.1"]
    # Por código de 3: C34 (1,55) le gana a D49 (0,80)
    assert vt.votar(vt.Configuracion("codigo_3", k=5, umbral=0.5))("x", TOP)[0] is (C.TUMOR_MALIGNO)


def test_umbral_por_categoria():
    c = vt.Configuracion("categoria", k=5, umbral=0.8, umbrales=((C.TUMOR_MALIGNO, 0.7),))
    assert vt.votar(c)("x", TOP) == (C.TUMOR_MALIGNO, "C34.9")
    assert c.a_json()["umbrales"] == {"tumor_maligno": 0.7}


def test_empate_gana_el_primero_del_top():
    top = [_p("I63.9", 0.8, C.CEREBROVASCULAR), _p("F03", 0.8, C.DEMENCIA)]
    assert vt.votar(vt.Configuracion(k=2, umbral=0.5))("x", top)[0] is C.CEREBROVASCULAR


# --- Búsqueda ------------------------------------------------------------------------


def _pliegues():
    indice, codificador = _indice()
    return de.pliegues_topk(indice, codificador, TRAIN_6, n=3, k=5)


@pytest.mark.parametrize(
    "configuracion",
    [
        vt.Configuracion("categoria", 1, None, 0.5),
        vt.Configuracion("categoria", 3, 0.05, 0.4),
        vt.Configuracion("bloque", 5, None, 0.3),
        vt.Configuracion("capitulo", 5, 0.02, 0.6),
        vt.Configuracion("codigo_3", 3, None, 0.5, ((C.DIABETES, 0.2),)),
    ],
)
def test_el_buscador_da_el_mismo_f1_que_el_arnes(configuracion):
    pliegues = _pliegues()
    buscador = vt._Buscador(vt._menciones(pliegues))
    votos = buscador.votos(configuracion)
    rapidas = [
        v[0] if v is not None and v[1] >= configuracion.umbral_de(v[0]) else None for v in votos
    ]
    lista = de.casos(pliegues, vt.votar(configuracion))
    assert rapidas == [c.prediccion for c in lista]  # mención por mención
    rapido = buscador.f1(votos, configuracion)
    assert rapido == pytest.approx(de._prf(lista, de._ids(pliegues))["f1"], abs=1e-4)


def test_mejor_prefiere_lo_simple_a_igual_f1():
    menciones = [vt._Mencion("ictus", C.CEREBROVASCULAR, (_p("I63.9", 0.9, C.CEREBROVASCULAR),))]
    configuracion, f1 = vt._Buscador(menciones).mejor()
    assert f1 == 1.0
    assert configuracion == vt.Configuracion("categoria", 1, None, max(vt.UMBRALES))


def test_por_categoria_sube_el_umbral_solo_si_mejora(monkeypatch):
    monkeypatch.setattr(vt, "SOPORTE_MINIMO", 1)
    menciones = [
        vt._Mencion("a", C.DEMENCIA, (_p("F03", 0.9, C.DEMENCIA),)),
        vt._Mencion("b", None, (_p("F03", 0.6, C.DEMENCIA),)),  # falso positivo con 0,5
    ]
    buscador = vt._Buscador(menciones)
    base = vt.Configuracion(umbral=0.5)
    configuracion, f1 = buscador.por_categoria(base)
    assert f1 == 1.0
    assert configuracion.umbral == 0.5
    assert configuracion.umbral_de(C.DEMENCIA) == 0.8  # de 0,62 a 0,80 empatan: el más alto


def test_el_buscador_aplica_las_exclusiones():
    menciones = [vt._Mencion("mioma uterino", None, (_p("C90.0", 0.9, C.TUMOR_MALIGNO),))]
    buscador = vt._Buscador(menciones)
    assert buscador.votos(vt.Configuracion()) == [None]


def test_recuperadas():
    documento = _documento("a", "Tos e ictus.", ("R05", "Tos"), ("I63.9", "ictus"))
    top = {
        "Tos": [_p("F03", 0.9, C.DEMENCIA)],
        "ictus": [
            _p("R20", 0.9, None),
            _p("I63.9", 0.8, C.CEREBROVASCULAR),
            _p("I63.0", 0.79, C.CEREBROVASCULAR),
        ],
    }
    pliegues = [de.Pliegue([documento], top)]
    base = de.casos(pliegues, de.primer_candidato(0.69))
    nuevos = de.casos(pliegues, vt.votar(vt.Configuracion(k=3, umbral=0.5)))
    resultado = vt.recuperadas(pliegues, base, nuevos)
    assert resultado["perdidas_base"]["top5"] == 1
    assert resultado["recuperadas"]["top5"] == 1
    assert resultado["falsos_positivos_nuevos"] == 0
    assert resultado["falsos_positivos_quitados"] == 0  # "Tos" sigue como demencia


# --- Evaluación y línea de comandos --------------------------------------------------

VARIANTES_CHICAS = {
    "votos": {"niveles": ("categoria", "bloque"), "por_categoria": False},
    "votos_umbral_por_categoria": {"niveles": ("categoria",), "por_categoria": True},
}


def test_evaluar(monkeypatch):
    monkeypatch.setattr(vt, "KS", (1, 3))
    monkeypatch.setattr(vt, "TEMPERATURAS", (None, 0.05))
    indice, codificador = _indice()
    resultados = vt.evaluar(
        "sapbert",
        indice,
        codificador,
        TRAIN_6,
        {},
        pliegues=3,
        k=5,
        replicas=5,
        variantes=VARIANTES_CHICAS,
    )
    votos = resultados["variantes"]["votos"]
    assert len(votos["por_pliegue"]) == 3
    assert set(votos["contra_hibrido"]) == {"precision", "recall", "f1"}
    assert set(resultados["configuracion_final"]) == {"votos", "votos_umbral_por_categoria"}
    assert "desarrollo" not in json.dumps(resultados).replace("desarrollo y prueba no", "")
    texto = json.dumps(resultados, ensure_ascii=False).casefold()
    assert "infarto cerebral" not in texto and "ictus" not in texto


def test_casos_anidados_usan_la_configuracion_de_cada_pliegue():
    pliegues = _pliegues()
    sin = vt.Configuracion(umbral=1.01)  # nada pasa el umbral
    todo = vt.Configuracion(umbral=0.0)
    lista = vt.casos_anidados(pliegues, [sin, todo, sin])
    por_documento = {c.documento: c.prediccion for c in lista}
    assert all(por_documento[d.id] is None for d in pliegues[0].documentos)
    assert any(c.prediccion is not None for c in lista if c.documento in {"t-2", "t-5"})


def test_subcomando_votos(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV})
    monkeypatch.setattr(cli.normalizador, "CodificadorHF", lambda modelo: Trigramas())
    monkeypatch.setattr(vt, "KS", (1, 3))
    monkeypatch.setattr(vt, "TEMPERATURAS", (None,))
    monkeypatch.setattr(vt, "UMBRALES", (0.3, 0.6))
    main(
        ["votos", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
        + ["--pliegues", "3", "--bootstrap", "5", "--salida", "votos.json"]
    )
    salida = capsys.readouterr().out
    assert "| votos |" in salida and "| solo bloque |" in salida
    guardado = json.loads((tmp_path / "votos.json").read_text(encoding="utf-8"))
    assert guardado["pliegues"] == 3


def test_sin_aciertos_el_f1_rapido_es_cero():
    menciones = [vt._Mencion("x", None, (_p("F03", 0.9, C.DEMENCIA),))]
    buscador = vt._Buscador(menciones)
    assert buscador.f1(buscador.votos(vt.Configuracion()), vt.Configuracion()) == 0.0


def test_buscador_y_votar_coinciden_con_una_exclusion():
    top = (_p("C90.0", 0.9, C.TUMOR_MALIGNO), _p("D25.9", 0.8, None))
    menciones = [
        vt._Mencion("mioma uterino", None, top),
        vt._Mencion("mieloma", C.TUMOR_MALIGNO, top),
    ]
    configuracion = vt.Configuracion(k=2, umbral=0.5)
    votos = vt._Buscador(menciones).votos(configuracion)
    assert [v and v[0] for v in votos] == [None, C.TUMOR_MALIGNO]
    topk = {m.texto: list(m.top) for m in menciones}
    mapeador = de.mapeador_de(vt.votar(configuracion), topk)
    assert [mapeador(m.texto) for m in menciones] == [None, (C.TUMOR_MALIGNO, "C90.0")]
