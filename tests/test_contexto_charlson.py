"""Tests de las métricas del set sintético (`evaluacion.contexto_charlson`, subcomando
`sintetico`).

Sin red ni modelo: cuatro notas sintéticas escritas aquí y un `ExtractorFalso` que marca
palabras fijas. El gold de las notas tiene errores puestos a propósito (una categoría y un
índice que no son, un `registrado` que no corresponde) para que cada métrica tenga
aciertos y errores conocidos.
"""

import json

import pytest

from evaluacion import contexto_charlson
from evaluacion.__main__ import main
from evaluacion.cache import ExtractorConCache
from evaluacion.contexto_charlson import (
    Corrida,
    buscar,
    coincide,
    correr,
    evaluar,
    huella,
    leer_notas,
    metricas_categoria,
    metricas_estado,
    metricas_registrado,
    tabla,
)
from ner_clinic.extraccion import ExtractorFalso
from ner_clinic.limpieza import limpiar
from ner_clinic.segmentacion import segmentar
from ner_clinic.tipos import CategoriaCharlson, Entidad, Estado, Mencion

NOTAS = [
    {
        "id": "t1",
        "texto": "PACIENTE CON ANTECEDENTE DE HTA, DM2 Y EPOC. CONSULTA POR NEUMONIA. NIEGA ASMA.",
        "entidades": [
            {"texto": "HTA", "estado": "confirmada"},
            {"texto": "DM2", "estado": "confirmada", "categoria": "diabetes"},
            {"texto": "EPOC", "estado": "confirmada", "categoria": "pulmonar_cronica"},
            {"texto": "ASMA", "estado": "negada", "categoria": "pulmonar_cronica"},
        ],
        "indice": 2,
    },
    {
        "id": "t2",
        "texto": "Madre con cáncer de mama. Paciente con insuficiencia cardiaca e hipotiroidismo.",
        "entidades": [
            {"texto": "cáncer de mama", "estado": "familiar", "categoria": "tumor_maligno"},
            {
                "texto": "insuficiencia cardiaca",
                "estado": "confirmada",
                "categoria": "insuficiencia_cardiaca",
            },
            {"texto": "hipotiroidismo", "estado": "confirmada"},  # el extractor no la marca
        ],
        "indice": 1,
    },
    {
        "id": "t3",
        "texto": "Paciente con enfermedad renal crónica estadio 3 e hipertensión arterial.",
        "registrados": ["Enfermedad renal crónica"],
        "entidades": [
            {
                "texto": "enfermedad renal crónica",
                "estado": "confirmada",
                "categoria": "renal_cronica",
                "registrado": True,
            },
            # Error puesto a propósito: la HTA no está entre los registrados.
            {"texto": "hipertensión arterial", "estado": "confirmada", "registrado": True},
        ],
        "indice": 3,  # sale 2: a un punto
    },
    {
        "id": "t4",
        "texto": "Se descarta cirrosis. Paciente con demencia.",
        "entidades": [
            # El pipeline la niega: error de estado, fuera de la métrica de categoría.
            {"texto": "cirrosis", "estado": "confirmada", "categoria": "hepatica_leve"},
            {"texto": "demencia", "estado": "confirmada", "categoria": "hemiplejia"},
        ],
    },
]
PALABRAS = (
    "HTA",
    "DM2",
    "EPOC",
    "NEUMONIA",
    "ASMA",
    "cáncer de mama",
    "insuficiencia cardiaca",
    "enfermedad renal crónica",
    "hipertensión arterial",
    "cirrosis",
    "demencia",
)


def extractor_para(textos, palabras) -> ExtractorFalso:
    """`ExtractorFalso` que marca cada aparición de `palabras` en los fragmentos."""
    salidas = {}
    for texto in textos:
        for fragmento in segmentar(limpiar(texto).texto):
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


@pytest.fixture
def falso():
    return extractor_para([n["texto"] for n in NOTAS], PALABRAS)


@pytest.fixture
def resultados(falso):
    return evaluar(NOTAS, falso)


def _entidad(texto, estado=Estado.CONFIRMADA, categoria=None):
    return Entidad(texto, [Mencion(texto, 0, len(texto))], estado, categoria_charlson=categoria)


# --- Emparejamiento ------------------------------------------------------------------


def test_coincide_por_contencion_sin_mayusculas():
    assert coincide("DM2", "dm2 insulinorrequiriente")
    assert coincide("diabetes mellitus tipo 2", "Diabetes Mellitus")
    assert not coincide("EPOC", "asma")


def test_buscar_prefiere_la_igual_y_si_no_la_primera():
    larga, igual = _entidad("DM2 con nefropatía"), _entidad("dm2")
    assert buscar("DM2", [larga, igual]) is igual
    assert buscar("DM2", [larga]) is larga
    assert buscar("EPOC", [larga, igual]) is None


def test_correr_empareja_y_separa_las_de_fuera_del_gold(falso):
    corrida = correr(NOTAS, falso)
    assert len(corrida.resultados) == 4
    assert len(corrida.pares) == 11
    assert [(n, e.texto) for n, e in corrida.sobrantes] == [("t1", "NEUMONIA")]


# --- Métricas ------------------------------------------------------------------------


def test_entidades(resultados):
    ent = resultados["entidades"]
    assert (ent["esperadas"], ent["encontradas"], ent["recall"]) == (11, 10, 0.9091)
    assert ent["perdidas"] == [
        {
            "id": "t2",
            "entidad": "hipotiroidismo",
            "salida": None,
            "esperado": "confirmada",
            "obtenido": None,
        }
    ]
    assert (ent["salida_total"], ent["fuera_del_gold"]) == (11, 1)
    assert ent["lista_fuera_del_gold"] == [
        {
            "id": "t1",
            "entidad": "NEUMONIA",
            "esperado": None,
            "obtenido": "confirmada",
            "categoria": None,
        }
    ]


def test_estado(resultados):
    est = resultados["estado"]
    # Filas: lo que anota el gold; columnas: también `posible` (#88).
    assert est["matriz"] == {
        "confirmada": {"confirmada": 7, "negada": 1, "posible": 0, "familiar": 0},
        "negada": {"confirmada": 0, "negada": 1, "posible": 0, "familiar": 0},
        "familiar": {"confirmada": 0, "negada": 0, "posible": 0, "familiar": 1},
    }
    assert "posible" not in est["por_clase"]
    assert (est["aciertos"], est["total"], est["exactitud"]) == (9, 10, 0.9)
    assert est["por_clase"]["confirmada"] == {"aciertos": 7, "total": 8, "exactitud": 0.875}
    assert est["por_clase"]["familiar"]["exactitud"] == 1.0
    assert est["errores"] == [
        {
            "id": "t4",
            "entidad": "cirrosis",
            "salida": "cirrosis",
            "esperado": "confirmada",
            "obtenido": "negada",
        }
    ]


def test_estado_posible_es_un_error_en_su_columna():
    # El gold no anota `posible`: una confirmada que sale posible es un error de estado.
    par = contexto_charlson.Par(
        "x", {"texto": "cáncer", "estado": "confirmada"}, _entidad("cáncer", Estado.POSIBLE)
    )
    est = metricas_estado(Corrida(notas=[], resultados=[], pares=[par]))
    assert est["matriz"]["confirmada"]["posible"] == 1
    assert (est["aciertos"], est["total"]) == (0, 1)
    assert est["errores"][0]["obtenido"] == "posible"


def test_categoria(resultados):
    cat = resultados["categoria"]
    assert (cat["entidades"], cat["excluidas_por_estado"]) == (7, 1)
    assert cat["global"] == {
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "vp": 4,
        "fp": 1,
        "fn": 1,
    }
    assert list(cat["por_categoria"]) == [
        "insuficiencia_cardiaca",
        "demencia",
        "pulmonar_cronica",
        "diabetes",
        "hemiplejia",
        "renal_cronica",
    ]
    # demencia solo tiene un falso positivo y hemiplejía solo un falso negativo
    assert cat["por_categoria"]["demencia"]["precision"] == 0.0
    assert cat["por_categoria"]["demencia"]["recall"] is None
    assert cat["por_categoria"]["hemiplejia"]["precision"] is None
    assert cat["por_categoria"]["hemiplejia"]["recall"] == 0.0
    assert cat["por_categoria"]["hemiplejia"]["f1"] is None
    assert cat["errores"] == [
        {
            "id": "t4",
            "entidad": "demencia",
            "salida": "demencia",
            "esperado": "hemiplejia",
            "obtenido": "demencia",
        }
    ]


def test_categoria_esperada_sin_categoria_obtenida():
    nota = {"id": "x", "texto": "", "entidades": []}
    pares = [
        contexto_charlson.Par("x", {"texto": "ERC", "estado": "confirmada"}, _entidad("ERC")),
        contexto_charlson.Par(
            "x",
            {"texto": "HTA", "estado": "confirmada", "categoria": "renal_cronica"},
            _entidad("HTA"),
        ),
        contexto_charlson.Par(
            "x",
            {"texto": "EPOC", "estado": "confirmada", "categoria": "pulmonar_cronica"},
            _entidad("EPOC", categoria=CategoriaCharlson.PULMONAR_CRONICA),
        ),
    ]
    cat = metricas_categoria(Corrida(notas=[nota], resultados=[], pares=pares))
    assert cat["por_categoria"]["renal_cronica"] == {
        "precision": None,
        "recall": 0.0,
        "f1": None,
        "vp": 0,
        "fp": 0,
        "fn": 1,
    }
    assert cat["global"]["precision"] == 1.0
    assert cat["global"]["recall"] == 0.5


def test_indice(resultados):
    ind = resultados["indice"]
    assert ind["notas"] == 3  # t4 no trae índice
    assert ind["exacto"] == {"aciertos": 2, "total": 3, "exactitud": 0.6667}
    assert ind["a_1_punto"] == {"aciertos": 3, "total": 3, "exactitud": 1.0}
    assert ind["errores"] == [{"id": "t3", "esperado": 3, "obtenido": 2}]


def test_registrado(resultados):
    reg = resultados["registrado"]
    assert (reg["con_valor_esperado"], reg["no_encontradas"]) == (2, 0)
    assert (reg["aciertos"], reg["total"], reg["exactitud"]) == (1, 2, 0.5)
    assert reg["errores"] == [
        {
            "id": "t3",
            "entidad": "hipertensión arterial",
            "salida": "hipertensión arterial",
            "esperado": True,
            "obtenido": False,
        }
    ]


def test_registrado_no_encontrada_no_cuenta():
    par = contexto_charlson.Par(
        "x", {"texto": "ERC", "estado": "confirmada", "registrado": True}, None
    )
    reg = metricas_registrado(Corrida(notas=[], resultados=[], pares=[par]))
    assert (reg["con_valor_esperado"], reg["no_encontradas"], reg["total"]) == (1, 1, 0)
    assert reg["exactitud"] == 0.0


def test_sin_notas_no_divide_por_cero():
    resultados = evaluar([], extractor_para([], ()))
    assert resultados["entidades"]["recall"] == 0.0
    assert resultados["estado"]["exactitud"] == 0.0
    assert resultados["categoria"]["global"]["precision"] is None
    assert resultados["indice"]["exacto"]["exactitud"] == 0.0


# --- Evaluación, lectura y reporte ---------------------------------------------------


def test_evaluar_trae_entorno_y_cache(tmp_path, falso):
    ruta = tmp_path / "notas.jsonl"
    ruta.write_text("\n".join(json.dumps(n, ensure_ascii=False) for n in NOTAS) + "\n\n")
    notas = leer_notas(ruta)
    assert notas == NOTAS

    con_cache = ExtractorConCache(falso, carpeta=tmp_path / "cache")
    resultados = evaluar(notas, con_cache, ruta_notas=ruta, cache=con_cache)
    assert resultados["set"] == {"notas": 4, "ruta": str(ruta), "sha256": huella(ruta)}
    assert len(resultados["set"]["sha256"]) == 64
    assert {"modelo", "transformers", "commit"} <= set(resultados["entorno"])
    assert resultados["cache"]["fallos"] > 0
    assert evaluar(NOTAS, falso)["cache"] is None


def test_tabla(resultados):
    texto = tabla(resultados)
    assert "| Recall de entidades esperadas | 0.909 (10/11) |" in texto
    assert "| Índice a 1 punto o menos | 1.000 (3/3) |" in texto
    assert "| confirmada | 7 | 1 | 0 | 0 | 0.875 |" in texto
    assert "| negada | posible | familiar | Exactitud |" in texto
    assert "\n| posible |" not in texto  # columna, pero no fila
    assert "| `hemiplejia` | — | 0.000 | 0 | 0 | 1 |" in texto


def test_ruta_por_defecto_del_set():
    notas = leer_notas()
    assert len(notas) == 100
    assert notas[0]["id"] == "sin001"


# --- Subcomando ----------------------------------------------------------------------


def test_subcomando_sintetico(tmp_path, falso, monkeypatch, capsys):
    ruta = tmp_path / "notas.jsonl"
    ruta.write_text("\n".join(json.dumps(n, ensure_ascii=False) for n in NOTAS))
    salida = tmp_path / "res" / "sintetico.json"
    monkeypatch.setattr("evaluacion.__main__.ExtractorHF", lambda: falso)
    argumentos = ["sintetico", "--notas", str(ruta), "--salida", str(salida)]
    main([*argumentos, "--cache", str(tmp_path / "cache")])

    guardado = json.loads(salida.read_text(encoding="utf-8"))
    assert guardado["entidades"]["recall"] == 0.9091
    assert guardado["cache"]["fallos"] > 0
    assert "Set sintético: 4 notas" in capsys.readouterr().out

    # Medición única: no sobrescribe el resultado guardado.
    with pytest.raises(SystemExit, match="ya existe"):
        main(argumentos)
