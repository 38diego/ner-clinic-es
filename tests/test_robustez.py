"""Tests de la prueba de estrés (`evaluacion/robustez.py`, issue #112).

Todas las notas son sintéticas y se usa un extractor falso: no hace falta red ni modelo.
"""

import json
import random
from pathlib import Path

import pytest

from evaluacion import robustez
from evaluacion.__main__ import main
from evaluacion.corpus import Documento, MencionGold
from ner_clinic.extraccion import ExtractorFalso

TEXTO = (
    "Paciente masculino de 68 años con antecedente de HTA, diabetes mellitus tipo 2 y "
    "EPOC. Niega cáncer. Insuficiencia cardíaca con fracción de eyección reducida; "
    "pingüino, Ñandú, İstanbul."
)


# --- Variantes del texto -------------------------------------------------------------


@pytest.mark.parametrize("variante", robustez.VARIANTES_TEXTO)
@pytest.mark.parametrize("semilla", [0, 1, 7])
def test_las_variantes_conservan_el_largo(variante, semilla):
    assert len(robustez.variar(TEXTO, variante, semilla, "doc")) == len(TEXTO)


def test_original_no_cambia():
    assert robustez.variar(TEXTO, "original") == TEXTO


def test_minusculas_deja_igual_lo_que_cambiaria_de_largo():
    variado = robustez.variar(TEXTO, "minusculas")
    assert "hta" in variado and "epoc" in variado
    assert "İ" in variado  # "İ".lower() tiene dos caracteres
    assert variado.replace("İ", "i") == TEXTO.lower().replace("i̇", "i")


def test_sin_tildes_conserva_la_enie():
    variado = robustez.variar("Cardíaca, pingüino, Ñandú, niño", "minusculas_sin_tildes")
    assert variado == "cardiaca, pinguino, ñandu, niño"


def test_errores_son_deterministas_y_dependen_de_semilla_y_clave():
    a = robustez.variar(TEXTO, "errores_10", 3, "doc1")
    assert a == robustez.variar(TEXTO, "errores_10", 3, "doc1")
    otras = {robustez.variar(TEXTO, "errores_10", s, c) for s, c in [(4, "doc1"), (3, "doc2")]}
    assert len({a, *otras}) > 1


def test_errores_con_valor_fijo():
    """Fija la salida: un cambio de algoritmo o de semilla (p. ej. a `hash()`) lo rompe."""
    texto = "Paciente con diabetes mellitus e insuficiencia cardiaca congestiva descompensada"
    assert robustez.variar(texto, "errores_10", 2, "fijo") == (
        "Paciente con diabetes mellitus e invuficiencia cardpaca congestiva descompfnsada"
    )


def test_texto_en_nfd_conserva_el_largo():
    import unicodedata

    texto = unicodedata.normalize("NFD", "Cardíaca crónica")
    assert len(robustez.variar(texto, "minusculas_sin_tildes")) == len(texto)


def test_errores_solo_en_palabras_de_4_letras_y_nunca_la_primera():
    texto = "HTA y DM. " + " ".join(["diabetes"] * 200)
    variado = robustez.variar(texto, "errores_10", 0, "x")
    assert variado.startswith("HTA y DM. ")
    palabras = variado.split()[3:]
    cambiadas = [p for p in palabras if p != "diabetes"]
    assert all(p[0] == "d" for p in palabras)
    assert all(sorted(p) == sorted("diabetes") or _una_sustitucion(p) for p in cambiadas)
    # El 10 % de 200 palabras, con margen.
    assert 5 <= len(cambiadas) <= 40


def _una_sustitucion(palabra: str) -> bool:
    return sum(a != b for a, b in zip(palabra, "diabetes", strict=True)) == 1


def test_con_errores_fraccion_cero_no_cambia_y_uno_cambia_todo():
    texto = "hipertension diabetes cirrosis demencia"
    assert robustez.con_errores(texto, 0.0, random.Random(0)) == texto
    variado = robustez.con_errores(texto, 1.0, random.Random(0))
    assert all(a != b for a, b in zip(variado.split(), texto.split(), strict=True))


def test_sustitucion_conserva_mayusculas():
    variado = robustez.con_errores("DIABETES", 1.0, random.Random(5))
    assert variado.isupper() and variado != "DIABETES"


def test_variante_desconocida():
    with pytest.raises(ValueError, match="Variante desconocida"):
        robustez.variar(TEXTO, "errores_7")


# --- DisTEMIST -----------------------------------------------------------------------


def _documento(id_: str, texto: str, *palabras: str) -> Documento:
    menciones = []
    for palabra in palabras:
        inicio = texto.index(palabra)
        menciones.append(MencionGold(inicio=inicio, fin=inicio + len(palabra), texto=palabra))
    return Documento(id=id_, texto=texto, menciones=menciones)


class ExtractorSensible(ExtractorFalso):
    """Marca `TERMINOS` solo en mayúscula, como el modelo real con las siglas.

    No usa siglas de `SIGLAS_PARA_EL_MODELO`: esas el pipeline ya las sube a mayúscula
    antes del modelo (#114). "SIADH" tiene 5 letras para pasar el filtro de longitud.
    """

    TERMINOS = ("SIADH", "ENFISEMA")

    def __init__(self) -> None:
        super().__init__({})

    def extraer(self, fragmentos: list[str]) -> list[list[dict]]:
        salidas = []
        for texto in fragmentos:
            entidades = []
            for termino in self.TERMINOS:
                desde = 0
                while (inicio := texto.find(termino, desde)) != -1:
                    fin = inicio + len(termino)
                    entidades.append(
                        {
                            "entity_group": "ENFERMEDAD",
                            "score": 0.99,
                            "word": termino,
                            "start": inicio,
                            "end": fin,
                        }
                    )
                    desde = fin
            salidas.append(sorted(entidades, key=lambda e: e["start"]))
        return salidas


def test_variar_documentos_conserva_el_gold_y_excluye_los_que_cambian_de_largo(monkeypatch):
    documentos = [_documento("a", "Paciente con SIADH.", "SIADH"), _documento("b", "Otro texto.")]
    variados, excluidos = robustez.variar_documentos(documentos, "minusculas")
    assert [d.texto for d in variados] == ["paciente con siadh.", "otro texto."]
    assert variados[0].menciones == documentos[0].menciones
    assert excluidos == []

    monkeypatch.setattr(robustez, "variar", lambda texto, *_: texto + "!")
    variados, excluidos = robustez.variar_documentos(documentos, "minusculas")
    assert (variados, excluidos) == ([], ["a", "b"])


def test_medir_distemist_por_variante():
    documentos = [_documento("a", "Paciente con SIADH en manejo.", "SIADH")]
    resultados = robustez.medir_distemist(
        documentos, ExtractorSensible(), ["original", "minusculas"], replicas=10
    )
    assert resultados["original"]["pipeline"]["laxo"]["f1"] == 1.0
    assert resultados["original"]["crudo"]["estricto"]["vp"] == 1
    assert resultados["minusculas"]["pipeline"]["laxo"]["f1"] == 0.0
    assert resultados["minusculas"]["documentos"] == 1
    assert resultados["original"]["pipeline"]["siglas"]["laxo"]["recall"] == 1.0
    assert resultados["minusculas"]["pipeline"]["siglas"]["laxo"]["encontradas"] == 0
    assert resultados["minusculas"]["pipeline"]["siglas"]["laxo"]["total"] == 1
    assert resultados["minusculas"]["excluidos"] == []


# --- Set sintético -------------------------------------------------------------------

NOTAS = [
    {
        "id": "n1",
        "texto": "Paciente con ENFISEMA en manejo.",
        "entidades": [
            {"texto": "ENFISEMA", "estado": "confirmada", "categoria": "pulmonar_cronica"}
        ],
        "indice": 1,
    },
    {
        "id": "n2",
        "texto": "Niega ENFISEMA.",
        "entidades": [{"texto": "ENFISEMA", "estado": "negada", "categoria": "pulmonar_cronica"}],
        "indice": 0,
    },
]


def test_medir_notas_en_original_y_en_minusculas():
    original = robustez.medir_notas(NOTAS, ExtractorSensible(), "original", 0)
    assert original["indice_exacto"] == 2
    assert original["categorias"] == {
        "encontradas": 1,
        "esperadas": 1,
        "recall": 1.0,
        "sobrantes": 0,
    }
    assert original["errores"] == []

    minusculas = robustez.medir_notas(NOTAS, ExtractorSensible(), "minusculas", 0)
    assert minusculas["indice_exacto"] == 1
    assert minusculas["indice_a_1"] == 2
    assert minusculas["categorias"]["recall"] == 0.0
    assert minusculas["errores"] == [
        {"nota": "n1", "indice": [1, 0], "faltan": ["pulmonar_cronica"], "sobran": []}
    ]


def test_medir_sintetico_separa_las_semillas_de_los_errores():
    resultados = robustez.medir_sintetico(
        NOTAS, ExtractorSensible(), ["original", "errores_5"], semillas=[0, 1]
    )
    assert "indice_exacto" in resultados["original"]
    assert set(resultados["errores_5"]) == {"semilla_0", "semilla_1"}


def test_tabla_y_evaluar_completo():
    documentos = [_documento("a", "Paciente con SIADH en manejo.", "SIADH")]
    resultados = robustez.evaluar(
        ExtractorSensible(), documentos=documentos, notas=NOTAS, semillas=[0, 1], replicas=10
    )
    assert set(resultados["distemist"]["variantes"]) == set(robustez.VARIANTES_TEXTO)
    assert set(resultados["sintetico"]["variantes"]) == set(robustez.VARIANTES_TEXTO)
    json.dumps(resultados)  # serializable, sin objetos del pipeline
    texto = robustez.tabla(resultados)
    assert "| `minusculas` | 0.000" in texto
    assert "`errores_10` semilla 0" in texto
    assert "`errores_10` semillas 1–1" in texto


def test_evaluar_sin_documentos_ni_notas():
    resultados = robustez.evaluar(ExtractorSensible())
    assert resultados["distemist"] is None and resultados["sintetico"] is None
    assert robustez.tabla(resultados) == ""


def test_cli_no_sobrescribe(tmp_path):
    salida = tmp_path / "robustez.json"
    salida.write_text("{}")
    with pytest.raises(SystemExit, match="ya existe"):
        main(["robustez", "--salida", str(salida)])


def test_cli_solo_sintetico(tmp_path, monkeypatch):
    notas = tmp_path / "notas.jsonl"
    notas.write_text("\n".join(json.dumps(n) for n in NOTAS) + "\n", encoding="utf-8")
    salida = tmp_path / "r.json"
    monkeypatch.setattr("evaluacion.__main__.ExtractorHF", ExtractorSensible)
    main(
        [
            "robustez",
            "--solo",
            "sintetico",
            "--notas",
            str(notas),
            "--semillas",
            "0",
            "--salida",
            str(salida),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )
    resultados = json.loads(salida.read_text(encoding="utf-8"))
    assert resultados["distemist"] is None
    assert resultados["sintetico"]["notas"] == 2
    assert resultados["sintetico"]["sha256"]


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("EPOC", True),
        ("DM2", True),
        (" HTA ", True),
        ("TB", True),
        ("M0", True),
        ("12", False),
        ("Epoc", False),
        ("diabetes", False),
        ("A", False),
        ("EPOCHTA", False),
    ],
)
def test_es_sigla(texto, esperado):
    assert robustez.es_sigla(texto) is esperado


def test_recall_siglas_solo_cuenta_las_siglas_del_gold():
    documento = _documento("a", "Paciente con SIADH y diabetes.", "SIADH", "diabetes")
    corrida = robustez.medir_distemist([documento], ExtractorSensible(), ["original"], replicas=10)
    siglas = corrida["original"]["pipeline"]["siglas"]["estricto"]
    assert (siglas["encontradas"], siglas["total"], siglas["recall"]) == (1, 1, 1.0)


def test_recall_siglas_sin_siglas_en_el_gold():
    documento = _documento("a", "Paciente con diabetes.", "diabetes")
    corrida = robustez.medir_distemist([documento], ExtractorSensible(), ["original"], replicas=10)
    assert corrida["original"]["pipeline"]["siglas"]["laxo"]["recall"] is None


def test_recall_siglas_dentro_de_un_span_largo():
    """La sigla anotada dentro de otra mención cuenta si alguna predicción la cubre."""
    texto = "Paciente con secreción inadecuada (SIADH) en manejo."
    documento = _documento("a", texto, "secreción inadecuada (SIADH)", "SIADH")
    corrida = robustez.medir_distemist([documento], ExtractorSensible(), ["original"], replicas=10)
    assert corrida["original"]["pipeline"]["siglas"]["laxo"]["recall"] == 1.0


def test_recall_siglas_cuenta_predicciones_que_fueron_falsos_positivos():
    """La predicción de la sigla quedó como falso positivo del emparejamiento global y la
    sigla del gold como falso negativo; aparte, la predicción sí la cubre."""
    from evaluacion.metricas import emparejar

    texto = "Paciente con EPOC grave."
    gold = _documento("a", texto, "EPOC grave", "EPOC").menciones
    predichas = [MencionGold(13, 23, "EPOC grave"), MencionGold(13, 17, "EPOC")]
    global_ = emparejar(gold[:1], predichas, "estricto")
    assert [p.texto for p in global_.falsos_positivos] == ["EPOC"]
    global_.falsos_negativos.append(gold[1])  # la sigla del gold, sin pareja

    siglas = robustez.recall_siglas([global_], "estricto", replicas=10, semilla=0)
    assert (siglas["encontradas"], siglas["total"]) == (1, 1)


def test_medir_notas_sin_indice_en_el_gold():
    nota = {**NOTAS[0], "id": "n3"}
    del nota["indice"]
    r = robustez.medir_notas([nota], ExtractorSensible(), "original", 0)
    assert (r["notas_con_indice"], r["indice_exacto"]) == (0, 0)


def test_correcciones_en_lista_solo_las_entidades_corregidas():
    texto = "Paciente con ENFISEMMA y SIADH."
    extractor = ExtractorFalso(
        {
            texto: [
                {"entity_group": "ENFERMEDAD", "score": 0.9, "word": "x", "start": 13, "end": 22},
                {"entity_group": "ENFERMEDAD", "score": 0.9, "word": "x", "start": 25, "end": 30},
            ]
        }
    )
    (fila,) = robustez.correcciones_en([Documento("d", texto)], extractor)
    assert fila["texto"] == "ENFISEMMA"
    assert fila["categoria"] == "pulmonar_cronica"
    assert fila["correcciones"] == [["enfisemma", "enfisema"]]
    assert "ENFISEMMA" in fila["contexto"]


def test_cli_correcciones(tmp_path, monkeypatch):
    texto = "Paciente con ENFISEMMA."
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("evaluacion.__main__.leer_particion", lambda *_: [Documento("d", texto)])
    monkeypatch.setattr(
        "evaluacion.__main__.ExtractorHF",
        lambda: ExtractorFalso(
            {texto: [{"entity_group": "E", "score": 0.9, "word": "x", "start": 13, "end": 22}]}
        ),
    )
    salida = Path("data/evaluacion/c.json")
    main(["robustez", "--correcciones", str(salida), "--cache", str(tmp_path / "cache")])
    datos = json.loads(salida.read_text(encoding="utf-8"))
    assert datos["documentos"] == 1
    assert [e["texto"] for e in datos["entidades"]] == ["ENFISEMMA"]


def test_cli_correcciones_fuera_de_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="texto del corpus"):
        main(["robustez", "--correcciones", "evaluacion/resultados/c.json"])
