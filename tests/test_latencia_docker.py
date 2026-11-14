import pytest

from evaluacion.latencia_docker import a_bytes, percentil, recortar, resumen


def test_percentil_por_rango_mas_cercano():
    valores = list(range(1, 101))

    assert percentil(valores, 95) == 95
    assert percentil(valores, 50) == 50
    assert percentil([3.0], 95) == 3.0
    assert percentil([5, 1, 3], 0) == 1


def test_percentil_sin_valores_falla():
    with pytest.raises(ValueError):
        percentil([], 95)


def test_resumen_calcula_mediana_p95_y_ms_por_mil_caracteres():
    datos = resumen([1.0, 2.0, 3.0], [1000, 2000, 3000])

    assert datos == {
        "notas": 3,
        "caracteres_mediana": 2000,
        "caracteres_max": 3000,
        "mediana_s": 2.0,
        "p95_s": 3.0,
        "max_s": 3.0,
        "ms_por_1000_caracteres": 1000.0,
    }


def test_resumen_exige_un_largo_por_tiempo():
    with pytest.raises(ValueError):
        resumen([1.0], [10, 20])


def test_recortar_no_parte_la_ultima_palabra():
    texto = "palabra " * 300

    recortado = recortar(texto, 2000)

    assert len(recortado) <= 2000
    assert len(recortado) > 1800
    assert recortado.endswith("palabra")
    assert recortar("corto", 2000) == "corto"


def test_recortar_sin_espacios_corta_en_seco():
    assert recortar("x" * 50, 10) == "x" * 10


@pytest.mark.parametrize(
    ("cantidad", "esperado"),
    [("812MiB", 812 * 2**20), ("1.5GiB", int(1.5 * 2**30)), ("0B", 0), ("10kB", 10_000)],
)
def test_a_bytes(cantidad, esperado):
    assert a_bytes(cantidad) == esperado


def test_a_bytes_unidad_desconocida():
    with pytest.raises(ValueError):
        a_bytes("12 parsecs")
