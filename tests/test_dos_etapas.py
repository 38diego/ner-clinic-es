"""Tests del arnés de la decisión en dos etapas (#149), sin red: índice de trigramas."""

import json

import pytest

from evaluacion import charlson_codiesp, codiesp
from evaluacion import dos_etapas as de
from evaluacion import normalizacion_codiesp as nc
from evaluacion import normalizador as nz
from evaluacion import sinonimos_codiesp as sc
from evaluacion.__main__ import main
from evaluacion.charlson_codiesp import Caso
from evaluacion.normalizador import Candidato
from ner_clinic.tipos import CategoriaCharlson as C
from tests.test_normalizador import CATALOGO, DEV, TRAIN, Trigramas, _documento

TRAIN_6 = TRAIN + [
    _documento("t-5", "Tuvo un infarto cerebral.", ("I63.9", "infarto cerebral")),
    _documento("t-6", "Con diabetes y EPOC.", ("E11.9", "diabetes"), ("J44.9", "EPOC")),
]


def _indice():
    codificador = Trigramas()
    return nz.construir_indice(CATALOGO, codificador), codificador


def _propuesta(codigo, similitud, categoria, origen=nz.ORIGEN_CATALOGO):
    return de.Propuesta(Candidato(codigo, "-", similitud, origen), codigo, categoria)


# --- Propuestas y decisor ------------------------------------------------------------


@pytest.mark.parametrize(
    ("codigo", "capitulo"),
    [
        ("I63.9", "IX"),
        ("D49.0", "II"),
        ("D50", "III"),
        ("H59.8", "VII"),
        ("H60.1", "VIII"),
        ("O9A.1", "XV"),
        ("T88.7", "XIX"),
        ("Y99.9", "XX"),
        ("z95.1", "XXI"),
        ("1AB", None),
    ],
)
def test_capitulo(codigo, capitulo):
    assert de.capitulo(codigo) == capitulo


def test_propuestas_leen_el_codigo_padre_como_su_hijo():
    indice, codificador = _indice()
    normalizador = nz.Normalizador(indice, codificador, 3)
    primera = de.propuestas(normalizador, "diabetes mellitus tipo 2")[0]
    assert primera.candidato.codigo == "E11"
    assert primera.codigo_categoria == "E11.9"
    assert primera.categoria is C.DIABETES
    assert (primera.codigo_3, primera.capitulo) == ("E11", "IV")


def test_primer_candidato_igual_que_el_mapeador_del_normalizador():
    indice, codificador = _indice()
    normalizador = nz.Normalizador(indice, codificador, 5)
    textos = ["diabetes mellitus tipo 2", "infarto cerebral", "tos", "hodgkin", "xyz"]
    topk = {t: de.propuestas(normalizador, t) for t in textos}
    for umbral in (0.0, 0.5, 0.9):
        mapeador = de.mapeador_de(de.primer_candidato(umbral), topk, exclusiones=False)
        assert [mapeador(t) for t in textos] == [normalizador.mapeador(umbral)(t) for t in textos]


def test_mapeador_de_aplica_las_exclusiones():
    topk = {"mioma uterino": [_propuesta("C90.0", 0.9, C.TUMOR_MALIGNO)]}
    decisor = de.primer_candidato(0.5)
    assert de.mapeador_de(decisor, topk, exclusiones=False)("mioma uterino") == (
        C.TUMOR_MALIGNO,
        "C90.0",
    )
    assert de.mapeador_de(decisor, topk)("mioma uterino") is None


# --- Vectores en disco ---------------------------------------------------------------


def test_codificador_en_disco(tmp_path):
    creados = []

    def crear():
        creados.append(Trigramas())
        return creados[-1]

    ruta = tmp_path / "vectores.pt"
    primero = de.CodificadorEnDisco(crear, ruta)
    vectores = primero(["tos", "ictus", "tos"])
    assert creados[0].textos == 2 and primero.nuevos == 2
    primero.guardar()

    segundo = de.CodificadorEnDisco(crear, ruta)
    assert segundo(["ictus", "tos"]).tolist() == vectores[[1, 0]].tolist()
    assert len(creados) == 1  # todo estaba en disco: no se creó el modelo
    segundo(["epoc"])
    assert len(creados) == 2
    segundo.guardar()
    assert de.CodificadorEnDisco(crear, ruta)(["epoc"]).shape == (1, vectores.shape[1])


def test_ruta_vectores_cambia_con_el_modelo(tmp_path):
    otro = nz.ModeloEmbeddings("prueba/otro", "abc", "cls", "MIT")
    assert de.ruta_vectores(nz.CANDIDATOS["sapbert"], tmp_path) != de.ruta_vectores(otro, tmp_path)


# --- Pliegues ------------------------------------------------------------------------


def _sinonimos_en(pliegue):
    """Los sinónimos que había en el índice del pliegue (con k mayor que el índice, el
    top-k de cualquier texto trae todas las filas)."""
    return {
        p.candidato.descripcion.casefold()
        for top in pliegue.topk.values()
        for p in top
        if p.candidato.origen == sc.ORIGEN_CODIESP
    }


def test_pliegues_topk_sin_los_sinonimos_del_pliegue():
    indice, codificador = _indice()
    pliegues = de.pliegues_topk(indice, codificador, TRAIN, n=2, k=50)
    assert [[d.id for d in p.documentos] for p in pliegues] == [["t-1", "t-3"], ["t-2", "t-4"]]
    assert set(pliegues[0].topk) == {"ictus", "DM", "EPOC", "Tos"}
    # Como en test_cada_pliegue_solo_tiene_los_sinonimos_de_los_otros
    assert _sinonimos_en(pliegues[0]) == {"ictus", "tos"}
    assert _sinonimos_en(pliegues[1]) == {"ictus", "dm", "epoc", "tos"}


def test_pliegues_internos_sin_el_externo_ni_el_propio():
    indice, codificador = _indice()
    # 3 pliegues: [t-1, t-4], [t-2, t-5], [t-3, t-6]
    internos = de.pliegues_internos(indice, codificador, TRAIN_6, externo=0, n=3, k=50)
    assert [[d.id for d in p.documentos] for p in internos] == [["t-2", "t-5"], ["t-3", "t-6"]]
    # El índice del pliegue [t-2, t-5] solo tiene sinónimos de [t-3, t-6], y viceversa
    assert _sinonimos_en(internos[0]) == {"tos", "epoc", "diabetes"}
    assert _sinonimos_en(internos[1]) == {"ictus", "tos", "infarto cerebral"}
    with pytest.raises(ValueError, match="al menos 3"):
        de.pliegues_internos(indice, codificador, TRAIN, externo=0, n=2)
    with pytest.raises(ValueError, match="entre 0 y 2"):
        de.pliegues_internos(indice, codificador, TRAIN_6, externo=3, n=3)


@pytest.mark.parametrize("umbral", [0.3, 0.6, 0.9])
def test_el_arnes_reproduce_la_validacion_cruzada_de_132(umbral):
    indice, codificador = _indice()
    pliegues = de.pliegues_topk(indice, codificador, TRAIN_6, n=3, k=5)
    lista = de.casos(pliegues, de.primer_candidato(umbral), exclusiones=False)
    prf = de._prf(lista, de._ids(pliegues))
    curva = nc.curva_umbral(sc._pliegues_con_sinonimos(indice, codificador, TRAIN_6, 3))
    punto = next(p for p in curva if p["umbral"] == umbral)
    assert prf == {k: punto[k] for k in ("precision", "recall", "f1")}


# --- Medición ------------------------------------------------------------------------


def _caso(gold, prediccion=None, texto="x", documento="d"):
    return Caso(documento, texto, "-", gold, None, prediccion)


@pytest.mark.parametrize(
    ("top", "reglas", "causa"),
    [
        ([_propuesta("I63.9", 0.9, C.CEREBROVASCULAR)], None, "excluida"),
        ([_propuesta("I63.9", 0.5, C.CEREBROVASCULAR)], None, "bajo_umbral"),
        (
            [_propuesta("R20", 0.9, None)] * 4 + [_propuesta("I63.9", 0.8, C.CEREBROVASCULAR)],
            None,
            "top5",
        ),
        (
            [_propuesta("R20", 0.9, None)] * 5 + [_propuesta("I63.9", 0.8, C.CEREBROVASCULAR)],
            None,
            "topk",
        ),
        ([_propuesta("R20", 0.9, None)], (C.CEREBROVASCULAR, "I63.9"), "solo_reglas"),
        ([_propuesta("R20", 0.9, None)], (C.DEMENCIA, "F03"), "fuera"),
    ],
)
def test_causa_perdida(top, reglas, causa):
    assert de.causa_perdida(_caso(C.CEREBROVASCULAR), top, 0.69, reglas) == causa


def test_causa_perdida_sin_perdida():
    top = [_propuesta("I63.9", 0.9, C.CEREBROVASCULAR)]
    assert de.causa_perdida(_caso(C.CEREBROVASCULAR, C.CEREBROVASCULAR), top, 0.69, None) is None
    assert de.causa_perdida(_caso(None, C.CEREBROVASCULAR), top, 0.69, None) is None


def _pliegue_a_mano(id_, top_ictus):
    documento = _documento(id_, "Tuvo ictus.", ("I63.9", "ictus"))
    return de.Pliegue([documento], {"ictus": top_ictus})


def test_errores_usan_el_top_k_de_cada_pliegue():
    # El mismo texto con otro top-k en cada pliegue: la causa sale del de su pliegue
    pliegues = [
        _pliegue_a_mano("a", [_propuesta("I63.9", 0.5, C.CEREBROVASCULAR)]),
        _pliegue_a_mano("b", [_propuesta("R20", 0.9, None)]),
    ]
    lista = de.casos(pliegues, de.primer_candidato(0.69))
    resultado = de.errores(pliegues, lista, 0.69, reglas=lambda texto: None)
    assert resultado["causas"]["bajo_umbral"] == 1 and resultado["causas"]["fuera"] == 1
    assert resultado["perdidas"] == 2
    assert resultado["por_categoria"] == {"cerebrovascular": 2}
    assert resultado["detalle"][0] == {
        "documento": "a",
        "inicio": 5,
        "fin": 10,
        "codigo_gold": "I63.9",
        "gold": "cerebrovascular",
        "prediccion": None,
        "causa": "bajo_umbral",
    }


def test_errores_cuentan_los_falsos_positivos():
    documento = _documento("a", "Tos y DM.", ("R05", "Tos"), ("E11.9", "DM"))
    top = {
        "Tos": [_propuesta("I63.9", 0.9, C.CEREBROVASCULAR)],
        "DM": [_propuesta("I63.9", 0.9, C.CEREBROVASCULAR)],
    }
    pliegues = [de.Pliegue([documento], top)]
    lista = de.casos(pliegues, de.primer_candidato(0.69))
    resultado = de.errores(pliegues, lista, 0.69, reglas=lambda texto: None)
    assert resultado["falsos_positivos"] == {"gold_sin_categoria": 1, "gold_otra_categoria": 1}
    assert resultado["causas"]["fuera"] == 1


def test_techo():
    documento = _documento(
        "a", "Tos, ictus y DM.", ("R05", "Tos"), ("I63.9", "ictus"), ("E11.9", "DM")
    )
    top = {
        "Tos": [_propuesta("I63.9", 0.9, C.CEREBROVASCULAR)],
        "ictus": [_propuesta("R20", 0.9, None), _propuesta("I63.9", 0.8, C.CEREBROVASCULAR)],
        "DM": [_propuesta("R20", 0.9, None)] * 2,
    }
    resultado = de.techo([de.Pliegue([documento], top)], ks=(1, 2))
    with pytest.raises(ValueError, match="tiene 2 candidatos"):
        de.techo([de.Pliegue([documento], top)], ks=(1, 3))
    # Nunca da falsos positivos: la "tos" (sin categoría gold) queda sin categoría
    assert resultado["1"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert resultado["2"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667}


def test_comparar():
    a = [_caso(C.DEMENCIA, None, documento="a"), _caso(C.DEMENCIA, C.DEMENCIA, documento="b")]
    b = [_caso(C.DEMENCIA, C.DEMENCIA, documento="a"), _caso(C.DEMENCIA, C.DEMENCIA, documento="b")]
    resultado = de.comparar(a, b, ["a", "b"], replicas=50)
    assert resultado["recall"]["diferencia"] == 0.5
    assert resultado["recall"]["ic95"][0] >= 0
    # Una réplica con solo el documento "a" da precisión 0/0 = 0 para `a` y 1 para `b`
    assert resultado["precision"]["diferencia"] == 0.0
    assert de.comparar(a, a, ["a", "b"], replicas=50)["f1"] == {
        "diferencia": 0.0,
        "ic95": [0.0, 0.0],
    }


# --- Evaluación y línea de comandos --------------------------------------------------


def _evaluar(**kwargs):
    indice, codificador = _indice()
    return de.evaluar(
        "sapbert",
        indice,
        codificador,
        TRAIN_6,
        DEV,
        {"codigos": len(CATALOGO)},
        ["x-1"],
        k=5,
        pliegues=3,
        replicas=5,
        **kwargs,
    )


def test_evaluar_reproduce_el_hibrido_en_desarrollo_y_no_guarda_texto():
    resultados = _evaluar(umbral=0.5)
    indice, codificador = _indice()
    normalizador = nz.Normalizador(
        indice.con_sinonimos(sc.sinonimos_de(TRAIN_6), codificador), codificador
    )
    esperado = charlson_codiesp.metricas(
        charlson_codiesp.casos(DEV, nz.con_exclusiones(normalizador.mapeador(0.5))),
        [d.id for d in DEV],
        5,
        por_categoria=False,
    )["micro"]
    assert resultados["desarrollo"]["hibrido_exclusiones"] == esperado
    assert resultados["verificacion_origen"] == {
        "documentos_origen": 6,
        "ids_dev": 2,
        "ids_test": 1,
    }
    assert set(resultados["validacion_cruzada"]) == {
        "reglas_corrector",
        "modelo_umbral",
        "hibrido_exclusiones",
    }
    assert len(resultados["validacion_cruzada"]["hibrido_exclusiones"]["por_pliegue"]) == 3
    assert "detalle" not in resultados["errores_validacion_cruzada"]
    assert list(resultados["techo_validacion_cruzada"]) == ["1", "5"]  # k = 5
    texto = json.dumps(resultados, ensure_ascii=False).casefold()
    for documento in TRAIN_6 + DEV:
        for m in documento.menciones_diagnostico:
            # "tos" y "DM" aparecen dentro de otras palabras; "diabetes" es una categoría
            if len(m.texto) > 3 and m.texto.casefold() not in {c.value for c in C}:
                assert m.texto.casefold() not in texto


def test_evaluar_rechaza_sinonimos_de_desarrollo():
    indice, codificador = _indice()
    with pytest.raises(ValueError, match="'dev'"):
        de.evaluar("sapbert", indice, codificador, TRAIN, TRAIN[:1], {}, [], pliegues=2)


def _mini(raiz, por_particion):
    for particion, documentos in por_particion.items():
        carpeta = raiz / codiesp.CARPETA_CODIESP / particion
        (carpeta / "text_files").mkdir(parents=True)
        filas_x, filas_d = [], []
        for doc in documentos:
            (carpeta / "text_files" / f"{doc.id}.txt").write_text(doc.texto, encoding="utf-8")
            for m in doc.menciones:
                filas_x.append(
                    f"{doc.id}\tDIAGNOSTICO\t{m.codigo.lower()}\t{m.texto}\t{m.inicio} {m.fin}\n"
                )
                filas_d.append(f"{doc.id}\t{m.codigo.lower()}\n")
        (carpeta / f"{particion}X.tsv").write_text("".join(filas_x), encoding="utf-8")
        (carpeta / f"{particion}D.tsv").write_text("".join(filas_d), encoding="utf-8")
    carpeta = raiz / codiesp.CARPETA_CATALOGO
    carpeta.mkdir()
    (carpeta / codiesp.ARCHIVO_CATALOGO).write_text(
        "".join(f"{c}\t{d}\t-\n" for c, d in CATALOGO.items()), encoding="utf-8"
    )


def test_subcomando_dos_etapas(tmp_path, monkeypatch, capsys):
    from evaluacion import __main__ as cli

    monkeypatch.chdir(tmp_path)
    _mini(tmp_path, {"train": TRAIN_6, "dev": DEV})
    prueba = tmp_path / codiesp.CARPETA_CODIESP / "test" / "text_files"
    prueba.mkdir(parents=True)
    (prueba / "x-1.txt").write_text("no se lee", encoding="utf-8")
    creados = []
    monkeypatch.setattr(
        cli.normalizador, "CodificadorHF", lambda modelo: creados.append(1) or Trigramas()
    )
    argumentos = ["dos-etapas", "--raiz", ".", "--catalogo", ".", "--indices", "indices"]
    argumentos += ["--k", "5", "--pliegues", "3", "--bootstrap", "5"]
    main([*argumentos, "--salida", "uno.json"])
    salida = capsys.readouterr().out
    assert "| hibrido exclusiones |" in salida and "| bajo_umbral |" in salida
    assert json.loads((tmp_path / "uno.json").read_text(encoding="utf-8"))["k"] == 5
    assert len(creados) == 1

    # La segunda corrida lee índice y vectores del disco: no carga el modelo
    main([*argumentos, "--salida", "dos.json"])
    assert len(creados) == 1


def test_errores_fallan_si_casos_y_menciones_no_cuadran():
    pliegues = [_pliegue_a_mano("a", [_propuesta("R20", 0.9, None)])]
    with pytest.raises(ValueError, match="desalineados"):
        de.errores(pliegues, [_caso(C.CEREBROVASCULAR, documento="b")], reglas=lambda t: None)
