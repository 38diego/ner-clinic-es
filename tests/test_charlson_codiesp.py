"""Tests de la línea base de las reglas de Charlson en CodiEsp (#130), sin red.

Los documentos son sintéticos y se arman a mano con la estructura que da el lector.
"""

import json
from pathlib import Path

import pytest

from evaluacion import charlson_codiesp as cc
from evaluacion.__main__ import main
from evaluacion.codiesp import CARPETA_CODIESP, DocumentoCodificado, MencionCodificada
from ner_clinic.tipos import CategoriaCharlson as C


def _documento(id_: str, texto: str, *menciones: tuple[str, str, str]) -> DocumentoCodificado:
    """`menciones`: (tipo, código, fragmento); la posición es la del fragmento en `texto`."""
    lista = []
    for tipo, codigo, fragmento in menciones:
        inicio = texto.index(fragmento)
        lista.append(
            MencionCodificada(tipo, codigo, fragmento, ((inicio, inicio + len(fragmento)),))
        )
    return DocumentoCodificado(id_, texto, [m.codigo for m in lista], lista)


D, P = "DIAGNOSTICO", "PROCEDIMIENTO"

DOC_1 = _documento(
    "doc-1",
    "Varón con diabetes mellitus tipo 2, EPOC e hipertensión arterial. Se realiza colecistectomía.",
    (D, "E11.9", "diabetes mellitus tipo 2"),  # acierto: diabetes
    (D, "J44.9", "EPOC"),  # acierto: pulmonar crónica
    (D, "I10", "hipertensión arterial"),  # sin categoría en el gold ni en las reglas
    (P, "0FT44ZZ", "colecistectomía"),  # procedimiento: no entra
)
DOC_2 = _documento(
    "doc-2",
    "Mujer con enfermedad de Binswanger y disfunción ventricular. Sin insuficiencia renal crónica.",
    (D, "I67.3", "enfermedad de Binswanger"),  # falso negativo: las reglas no la reconocen
    (D, "I51.9", "disfunción ventricular"),  # falso positivo: las reglas dan ICC
    (D, "N18.9", "insuficiencia renal crónica"),  # acierto, y negada según el contexto
)
DOCUMENTOS = [DOC_1, DOC_2]


def test_casos_toma_solo_diagnosticos_con_la_categoria_de_quan():
    lista = cc.casos(DOCUMENTOS, cc.METODOS["reglas"])
    assert [(c.texto, c.gold, c.prediccion) for c in lista] == [
        ("diabetes mellitus tipo 2", C.DIABETES, C.DIABETES),
        ("EPOC", C.PULMONAR_CRONICA, C.PULMONAR_CRONICA),
        ("hipertensión arterial", None, None),
        ("enfermedad de Binswanger", C.CEREBROVASCULAR, None),
        ("disfunción ventricular", None, C.INSUFICIENCIA_CARDIACA),
        ("insuficiencia renal crónica", C.RENAL_CRONICA, C.RENAL_CRONICA),
    ]


def test_conteos_por_categoria_y_micro():
    caso = cc.Caso("d", "x", "C80", C.TUMOR_METASTASICO, "C80", C.TUMOR_MALIGNO)
    # Una confusión entre categorías es un falso positivo y un falso negativo
    assert cc._conteos(caso) == (0, 1, 1)
    assert cc._conteos(caso, C.TUMOR_METASTASICO) == (0, 0, 1)
    assert cc._conteos(caso, C.TUMOR_MALIGNO) == (0, 1, 0)
    assert cc._conteos(caso, C.DIABETES) == (0, 0, 0)
    sin_nada = cc.Caso("d", "x", "I10", None, None, None)
    assert cc._conteos(sin_nada) == (0, 0, 0)


def test_metricas():
    lista = cc.casos(DOCUMENTOS, cc.METODOS["reglas"])
    r = cc.metricas(lista, ["doc-1", "doc-2", "doc-3"], replicas=20)
    assert (r["micro"]["vp"], r["micro"]["fp"], r["micro"]["fn"]) == (3, 1, 1)
    assert r["micro"]["precision"] == r["micro"]["recall"] == 0.75
    assert r["menciones"] == 6
    assert r["menciones_gold_charlson"] == 4
    assert r["menciones_con_categoria"] == 4
    assert r["con_categoria_sin_gold"] == 1
    # E11.9/E11.9, J44.9/J44.9 y N18.9/N18.x coinciden a 3 caracteres; Binswanger no tiene código
    assert r["cie10_3"] == 0.75
    assert r["por_categoria"]["cerebrovascular"]["fn"] == 1
    assert r["por_categoria"]["insuficiencia_cardiaca"]["fp"] == 1
    assert set(r["por_categoria"]) == {c.value for c in C}
    inferior, superior = r["micro"]["ic95"]["f1"]
    assert 0 <= inferior <= superior <= 1


def test_metricas_sin_categorias():
    r = cc.metricas([], ["doc-1"], replicas=5, por_categoria=False)
    assert r["cie10_3"] == 0.0
    assert "por_categoria" not in r


def test_texto_en_lee_los_tramos_de_la_variante():
    mencion = MencionCodificada(D, "L98.9", "lesión piel", ((6, 12), (0, 4)))
    assert cc.texto_en("piel, lesión", mencion) == "lesión piel"
    assert cc.texto_en("PIEL, LESIÓN", mencion) == "LESIÓN PIEL"


def test_las_variantes_releen_la_mencion_del_texto_variado():
    original = cc.casos(DOCUMENTOS, cc.METODOS["reglas"])
    minusculas = cc.casos(DOCUMENTOS, cc.METODOS["reglas"], "minusculas")
    assert minusculas[1].texto == "epoc"
    assert [c.prediccion for c in minusculas] == [c.prediccion for c in original]
    con_errores = cc.casos(DOCUMENTOS, cc.METODOS["reglas"], "errores_10", semilla=3)
    assert len(con_errores) == len(original)
    assert all(len(a.texto) == len(b.texto) for a, b in zip(con_errores, original, strict=True))
    assert con_errores == cc.casos(DOCUMENTOS, cc.METODOS["reglas"], "errores_10", semilla=3)


def test_variante_que_cambia_el_largo_falla(monkeypatch):
    monkeypatch.setattr(cc.robustez, "variar", lambda texto, *_: texto + "!")
    with pytest.raises(ValueError, match="cambió el largo"):
        cc.casos(DOCUMENTOS, cc.METODOS["reglas"], "minusculas")


def test_el_corrector_recupera_un_error_de_ortografia():
    doc = _documento("doc-3", "Paciente con diabetis mellitus.", (D, "E11.9", "diabetis mellitus"))
    assert cc.casos([doc], cc.METODOS["reglas"])[0].prediccion is None
    assert cc.casos([doc], cc.METODOS["reglas_corrector"])[0].prediccion is C.DIABETES


def test_resumen_estados_usa_el_contexto():
    lista = cc.casos(DOCUMENTOS, cc.METODOS["reglas"])
    resumen = cc.resumen_estados(DOCUMENTOS, lista)
    assert resumen["negada"] == {"menciones_gold_charlson": 1, "recall": 1.0}
    assert resumen["confirmada"] == {"menciones_gold_charlson": 3, "recall": 0.6667}
    assert resumen["familiar"] == {"menciones_gold_charlson": 0, "recall": None}
    with pytest.raises(ValueError, match="no corresponde"):
        cc.resumen_estados(DOCUMENTOS, lista[:2])


def test_resumen_estados_rechaza_casos_de_otros_documentos():
    lista = cc.casos([DOC_2, DOC_1], cc.METODOS["reglas"])
    with pytest.raises(ValueError, match="no corresponde"):
        cc.resumen_estados([DOC_1, DOC_2], lista[:3] + lista[3:])


def test_el_estado_de_una_mencion_desordenada_sale_del_tramo_que_va_primero():
    # "renal" va antes en la nota, dentro de la negación; "insuficiencia" va después
    texto = "Sin enfermedad renal. Presenta insuficiencia."
    a, b = texto.index("insuficiencia"), texto.index("renal")
    mencion = MencionCodificada(
        D, "N19", "insuficiencia renal", ((a, a + len("insuficiencia")), (b, b + 5))
    )
    doc = DocumentoCodificado("doc-4", texto, ["N19"], [mencion])
    assert list(cc.estados([doc]).values()) == [cc.Estado.NEGADA]


def test_evaluar_rechaza_la_particion_de_prueba():
    with pytest.raises(ValueError, match="queda para #134"):
        cc.evaluar({"train": DOCUMENTOS, "test": DOCUMENTOS}, replicas=5)


def test_evaluar():
    resultados, detalle = cc.evaluar(
        {"train": [DOC_1], "dev": [DOC_2]}, replicas=5, variantes=("original", "errores_5")
    )
    assert resultados["conjunto_principal"] == "train+dev"
    assert resultados["documentos"] == {"train": 1, "dev": 1, "train+dev": 2}
    assert set(resultados["metodos"]) == {"reglas", "reglas_corrector"}
    original = resultados["metodos"]["reglas"]["original"]
    assert set(original) == {"train", "dev", "train+dev"}
    assert "por_categoria" in original["train+dev"]
    assert "por_categoria" not in original["train"]
    assert original["train+dev"]["micro"]["f1"] == 0.75
    assert set(resultados["estado_segun_contexto"]) == {"reglas", "reglas_corrector"}
    # Sin texto del corpus en lo que se versiona
    serializado = json.dumps(resultados, ensure_ascii=False)
    assert "Binswanger" not in serializado
    assert "disfunción" not in serializado
    # El detalle trae solo los errores, con texto
    assert [e["texto"] for e in detalle["metodos"]["reglas"]] == [
        "enfermedad de Binswanger",
        "disfunción ventricular",
    ]


def test_una_sola_particion_no_arma_conjunto():
    resultados, _ = cc.evaluar({"dev": [DOC_2]}, replicas=5, variantes=("original",))
    assert resultados["conjunto_principal"] == "dev"
    assert "por_categoria" in resultados["metodos"]["reglas"]["original"]["dev"]


def test_tablas():
    resultados, _ = cc.evaluar({"dev": DOCUMENTOS}, replicas=5, variantes=("original",))
    tabla = cc.tabla(resultados)
    assert "| reglas | original | 0.750 [" in tabla
    assert "Conjunto: dev (2 documentos)" in tabla
    categorias = cc.tabla_categorias(resultados)
    assert "| diabetes | 1 | 1.000 | 1.000 | 1.000 |" in categorias


# --- Línea de comandos ---------------------------------------------------------------


def _mini_corpus(raiz: Path) -> None:
    for particion, docs in {"train": [DOC_1], "dev": [DOC_2]}.items():
        carpeta = raiz / CARPETA_CODIESP / particion
        (carpeta / "text_files").mkdir(parents=True)
        filas_x, filas_d = [], []
        for doc in docs:
            (carpeta / "text_files" / f"{doc.id}.txt").write_text(doc.texto, encoding="utf-8")
            for m in doc.menciones:
                (a, b), *_ = m.tramos
                filas_x.append(f"{doc.id}\t{m.tipo}\t{m.codigo.lower()}\t{m.texto}\t{a} {b}\n")
                filas_d.append(f"{doc.id}\t{m.codigo.lower()}\n")
        (carpeta / f"{particion}X.tsv").write_text("".join(filas_x), encoding="utf-8")
        (carpeta / f"{particion}D.tsv").write_text("".join(filas_d), encoding="utf-8")


def test_subcomando_codiesp(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _mini_corpus(tmp_path / "corpus")
    main(
        [
            "codiesp",
            "--raiz",
            "corpus",
            "--salida",
            "resultados.json",
            "--detalle",
            "data/errores.json",
            "--bootstrap",
            "5",
        ]
    )
    salida = capsys.readouterr().out
    assert "Conjunto: train+dev (2 documentos)" in salida
    assert "Por categoría (reglas_corrector, original)" in salida
    resultados = json.loads((tmp_path / "resultados.json").read_text(encoding="utf-8"))
    assert resultados["particiones"] == ["train", "dev"]
    assert (tmp_path / "data" / "errores.json").is_file()


def test_subcomando_codiesp_exige_el_detalle_en_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="va en data/"):
        main(["codiesp", "--raiz", "corpus", "--detalle", "errores.json"])


def test_subcomando_codiesp_no_acepta_la_particion_de_prueba(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["codiesp", "--particiones", "test"])
    assert "invalid choice: 'test'" in capsys.readouterr().err
