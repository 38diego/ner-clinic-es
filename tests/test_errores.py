"""Tests de la muestra de errores (`evaluacion.errores`), con un detalle escrito a mano."""

import json

import pytest

from evaluacion import errores
from evaluacion.errores import (
    CATEGORIAS,
    MAX_CONTEXTO,
    conteo,
    contexto,
    muestrear,
    tabla_categorias,
    validar,
)

TEXTO = "Paciente con diabetes mellitus tipo 2 y asma. Niega hipertensión arterial. Fiebre."


def _m(texto: str, fragmento: str) -> dict:
    inicio = texto.index(fragmento)
    return {"inicio": inicio, "fin": inicio + len(fragmento), "texto": fragmento}


def detalle_mini() -> dict:
    """Un documento: un TP, dos FP (uno solapado con un gold) y dos FN."""
    doc = {
        "documento": "caso-1",
        "verdaderos": [{"gold": _m(TEXTO, "asma"), "prediccion": _m(TEXTO, "asma")}],
        "falsos_positivos": [_m(TEXTO, "diabetes mellitus"), _m(TEXTO, "Fiebre")],
        "falsos_negativos": [
            _m(TEXTO, "diabetes mellitus tipo 2"),
            _m(TEXTO, "hipertensión arterial"),
        ],
    }
    return {"particion": "test", "variantes": {"pipeline": {"estricto": [doc]}}}


def test_contexto_centra_el_span_y_respeta_el_maximo():
    texto = "a" * 200 + "SPAN" + "b" * 200
    fragmento = contexto(texto, 200, 204)
    assert len(fragmento) == MAX_CONTEXTO
    assert "SPAN" in fragmento
    assert fragmento.index("SPAN") == 58


def test_contexto_en_los_bordes_y_con_span_largo():
    assert contexto("corto\ncon salto", 0, 5) == "corto con salto"
    assert contexto("x" * 300, 10, 290) == "x" * MAX_CONTEXTO
    texto = "SPAN" + "b" * 300
    assert contexto(texto, 0, 4).startswith("SPAN")
    texto = "a" * 300 + "SPAN"
    assert contexto(texto, 300, 304).endswith("SPAN")


def test_muestrear_marca_acierto_laxo_y_solapes():
    muestra = muestrear(detalle_mini(), {"caso-1": TEXTO}, n=50, revisiones=3)
    assert muestra["poblacion"] == {"FP": 2, "FN": 2}
    por_texto = {e["texto"]: e for e in muestra["errores"]}
    assert por_texto["diabetes mellitus"]["acierto_laxo"] is True
    assert por_texto["diabetes mellitus"]["gold_solapado"][0]["texto"] == (
        "diabetes mellitus tipo 2"
    )
    assert por_texto["Fiebre"]["acierto_laxo"] is False
    assert por_texto["diabetes mellitus tipo 2"]["prediccion_solapada"][0]["texto"] == (
        "diabetes mellitus"
    )
    assert por_texto["hipertensión arterial"]["prediccion_solapada"] == []
    assert "acierto_laxo" not in por_texto["hipertensión arterial"]
    assert len(muestra["revisar"]) == 3
    assert set(muestra["revisar"]) <= {e["id"] for e in muestra["errores"]}
    assert all(e["categoria"] is None for e in muestra["errores"])
    assert "CC BY 4.0" in muestra["atribucion"]


def test_muestrear_es_reproducible_y_depende_de_la_semilla():
    detalle = detalle_mini()
    doc = detalle["variantes"]["pipeline"]["estricto"][0]
    doc["falsos_positivos"] = [{"inicio": i, "fin": i + 1, "texto": TEXTO[i]} for i in range(0, 60)]
    a = muestrear(detalle, {"caso-1": TEXTO}, n=5)
    b = muestrear(detalle, {"caso-1": TEXTO}, n=5)
    c = muestrear(detalle, {"caso-1": TEXTO}, n=5, semilla=7)
    assert a == b
    assert [e["inicio"] for e in a["errores"]] != [e["inicio"] for e in c["errores"]]
    assert [e["id"] for e in a["errores"]][:5] == [f"FP-0{i}" for i in range(1, 6)]


def _clasificada() -> dict:
    muestra = muestrear(detalle_mini(), {"caso-1": TEXTO})
    categorias = ["limite_span", "sintoma_no_anotado", "limite_span", "no_detectada"]
    for error, categoria in zip(muestra["errores"], categorias, strict=True):
        error["categoria"] = categoria
    return muestra


def test_validar_conteo_y_tabla():
    muestra = _clasificada()
    assert validar(muestra) == []
    tipos = {e["categoria"]: e["tipo"] for e in muestra["errores"]}
    assert set(tipos) <= set(CATEGORIAS)
    total = conteo(muestra)
    assert sum(n["FP"] + n["FN"] for n in total.values()) == 4
    assert list(total) == [c for c in CATEGORIAS if c in total]
    texto = tabla_categorias(muestra)
    assert "| **Total** | 2 | 2 | 4 |" in texto
    assert "acierto laxo" in texto and "1 de 2" in texto


def test_validar_detecta_problemas():
    muestra = _clasificada()
    muestra["errores"][0]["categoria"] = "inventada"
    muestra["errores"][1]["contexto"] = "x" * (MAX_CONTEXTO + 1)
    problemas = validar(muestra)
    assert len(problemas) == 2


def test_cli_muestra_y_tabla(tmp_path, monkeypatch, capsys):
    detalle = tmp_path / "detalle.json"
    detalle.write_text(json.dumps(detalle_mini()), encoding="utf-8")

    class Doc:
        id = "caso-1"
        texto = TEXTO

    monkeypatch.setattr(errores, "leer_particion", lambda raiz, particion: [Doc()])
    salida = tmp_path / "muestra.json"
    errores.main(["muestra", "--detalle", str(detalle), "--salida", str(salida)])
    assert "4 errores" in capsys.readouterr().out

    muestra = json.loads(salida.read_text(encoding="utf-8"))
    with pytest.raises(SystemExit):
        errores.main(["tabla", str(salida)])

    for error in muestra["errores"]:
        error["categoria"] = "otra"
    salida.write_text(json.dumps(muestra), encoding="utf-8")
    errores.main(["tabla", str(salida)])
    assert "| Otra | 2 | 2 | 4 |" in capsys.readouterr().out
