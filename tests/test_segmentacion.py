import random
import re

import pytest

from ner_clinic.segmentacion import Fragmento, segmentar


def _segmentar_tesis(texto_limpio, max_caracteres=2000):
    """Lógica de `segmentar_historia` de la tesis, como referencia."""
    parrafos = texto_limpio.split("\n")
    fragmentos = []
    actual = ""
    for parrafo in parrafos:
        if len(parrafo) > max_caracteres:
            for oracion in parrafo.split(". "):
                if len(actual) + len(oracion) < max_caracteres:
                    actual += oracion + ". "
                else:
                    if actual.strip():
                        fragmentos.append(actual.strip())
                    actual = oracion + ". "
            continue
        if len(actual) + len(parrafo) + 1 <= max_caracteres:
            actual += parrafo + "\n"
        else:
            if actual.strip():
                fragmentos.append(actual.strip())
            actual = parrafo + "\n"
    if actual.strip():
        fragmentos.append(actual.strip())
    return fragmentos


PALABRAS = ["paciente", "con", "hipertensión", "niega", "diabetes", "EPOC", "dolor", "HTA"]


def _oracion(rng, largo_max=12):
    return " ".join(rng.choice(PALABRAS) for _ in range(rng.randint(1, largo_max)))


def _parrafo(rng, oraciones_max=6, largo_max=12):
    return ". ".join(_oracion(rng, largo_max) for _ in range(rng.randint(1, oraciones_max)))


def _nota(rng, largo_max=12):
    return "\n".join(_parrafo(rng, largo_max=largo_max) for _ in range(rng.randint(1, 10)))


def _normalizar(texto):
    # La tesis agrega ". " tras cada oración y pierde el "\n" que sigue a un párrafo largo.
    return re.sub(r"[\s.]", "", texto)


@pytest.mark.parametrize("vacio", [None, ""])
def test_texto_vacio_o_none_devuelve_lista_vacia(vacio):
    assert segmentar(vacio) == []


def test_texto_corto_es_un_solo_fragmento():
    nota = "Paciente con HTA.\nNiega DM."
    assert segmentar(nota) == [Fragmento(texto=nota, desplazamiento=0)]


def test_corta_por_parrafo():
    nota = "aaaa\nbbbb\ncccc"
    assert [f.texto for f in segmentar(nota, max_caracteres=10)] == ["aaaa\nbbbb", "cccc"]


def test_parrafo_largo_se_corta_por_oracion():
    nota = "uno uno. dos dos. tres tres"
    fragmentos = segmentar(nota, max_caracteres=12)
    assert [f.texto for f in fragmentos] == ["uno uno.", "dos dos.", "tres tres"]
    assert [f.desplazamiento for f in fragmentos] == [0, 9, 18]


def test_oracion_mas_larga_que_el_maximo_queda_sola():
    larga = "x" * 30
    nota = f"corta. {larga}. otra"
    fragmentos = segmentar(nota, max_caracteres=20)
    assert [f.texto for f in fragmentos] == ["corta.", larga + ".", "otra"]


@pytest.mark.parametrize("semilla", range(300))
@pytest.mark.parametrize("maximo", [40, 120, 2000])
def test_fragmentos_son_rebanadas_del_texto(semilla, maximo):
    rng = random.Random(semilla)
    nota = _nota(rng)
    for fragmento in segmentar(nota, max_caracteres=maximo):
        inicio = fragmento.desplazamiento
        assert nota[inicio : inicio + len(fragmento.texto)] == fragmento.texto
        assert fragmento.texto == fragmento.texto.strip()
        assert fragmento.texto


@pytest.mark.parametrize("semilla", range(300))
@pytest.mark.parametrize("maximo", [40, 120, 2000])
def test_ningun_fragmento_supera_el_maximo_salvo_una_oracion_larga(semilla, maximo):
    rng = random.Random(semilla)
    nota = _nota(rng, largo_max=rng.choice([5, 12, 40]))
    for fragmento in segmentar(nota, max_caracteres=maximo):
        if len(fragmento.texto) > maximo:
            # Solo puede ser una oración: sin saltos de línea ni separadores ". " dentro.
            assert "\n" not in fragmento.texto
            assert ". " not in fragmento.texto


@pytest.mark.parametrize("semilla", range(300))
@pytest.mark.parametrize("maximo", [40, 120, 2000])
def test_mismo_corte_que_la_tesis(semilla, maximo):
    rng = random.Random(semilla)
    nota = _nota(rng)
    propios = [_normalizar(f.texto) for f in segmentar(nota, max_caracteres=maximo)]
    tesis = [_normalizar(f) for f in _segmentar_tesis(nota, max_caracteres=maximo)]
    assert propios == tesis


@pytest.mark.parametrize("semilla", range(100))
def test_sin_parrafos_largos_el_texto_es_identico_a_la_tesis(semilla):
    rng = random.Random(semilla)
    nota = _nota(rng)
    propios = [f.texto for f in segmentar(nota, max_caracteres=2000)]
    assert propios == _segmentar_tesis(nota, max_caracteres=2000)
