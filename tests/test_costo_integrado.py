"""Lo que se puede probar sin Docker del costo integrado (#167)."""

from evaluacion import costo_integrado as ci
from ner_clinic.normalizacion.indice import SAPBERT


def _medicion(arranque, p95, memoria, tamano):
    servidor = {"p95_s": p95, "mediana_s": p95 / 2}
    return {
        "arranque_en_frio_s": arranque,
        "notas_2000_caracteres": {"servidor": servidor},
        "distemist_test": {"servidor": servidor},
        "memoria_pico_mb": {"cgroup_memory_peak": memoria},
        "imagen": {"tamano_mb": tamano},
    }


def test_dockerfile_sapbert_trae_el_modelo_fijo_y_el_indice():
    texto = ci.dockerfile_sapbert()
    assert texto.startswith(f"FROM {ci.IMAGEN_BASE}")
    assert SAPBERT.revision in texto and SAPBERT.tokenizador[1] in texto
    assert f"COPY indice_final.pt {ci.RUTA_INDICE_EN_IMAGEN}" in texto
    assert f"ENV NER_INDICE_NORMALIZADOR={ci.RUTA_INDICE_EN_IMAGEN}" in texto
    assert texto.rstrip().endswith("USER ner")


def test_resumen_comparado():
    comparacion = ci.resumen_comparado(
        {
            "reglas": _medicion(10.0, 0.5, 400, 900),
            "normalizador": _medicion(18.0, 0.6, 1100, 2000),
            "control": _medicion(11.0, 0.55, 420, 2000),
        }
    )
    assert comparacion["normalizador_menos_reglas"]["arranque_s"] == 8.0
    assert comparacion["normalizador_menos_reglas"]["memoria_pico_mb"] == 700
    assert comparacion["normalizador_menos_control"]["imagen_mb"] == 0
    assert comparacion["normalizador_menos_control"]["p95_2000_s"] == 0.05


def test_corridas_en_orden():
    assert [n for n, _, _ in ci.CORRIDAS] == ["reglas", "normalizador", "control"]
    assert [v for _, _, v in ci.CORRIDAS] == [
        {"NER_NORMALIZADOR": "reglas"},
        {"NER_NORMALIZADOR": "sapbert"},
        {"NER_NORMALIZADOR": "reglas"},
    ]
