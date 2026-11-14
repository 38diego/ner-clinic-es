import random
import re

import pytest

from ner_clinic.limpieza import TextoLimpio, limpiar
from ner_clinic.tipos import Mencion

ENCABEZADO = "Folio: 12 | Fecha: 2023-04-05 08:30:15.0"


def _limpiar_tesis(texto):
    """Lógica de `limpiar_encabezados_historia` de la tesis, como referencia."""
    if not isinstance(texto, str):
        return ""
    patron = r"Folio:\s*\d+\s*\|\s*Fecha:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s*"
    texto = re.sub(patron, " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"-{3,}", " ", texto)
    texto = re.sub(r"\n+", "\n", texto)
    return texto.strip()


FRASES = [
    "Paciente con hipertensión arterial",
    "niega diabetes mellitus",
    "madre con cáncer de mama",
    "EPOC en manejo con salbutamol",
    "sin antecedentes de infarto agudo de miocardio",
    "insuficiencia renal crónica estadio 3",
    "a-b c",  # guion suelto: no se elimina
]

ESPECIALES = [
    ENCABEZADO + "\n",
    "folio:3|fecha:2021-12-31 23:59:59.123  \n\n",
    "---",
    "------------",
    "\n\n\n",
    "\n",
]


def _nota_aleatoria(rng, especiales=ESPECIALES):
    """Arma una nota sintética marcando qué tramos del original sobreviven a la limpieza.

    Devuelve el texto y la lista de tramos conservados `(inicio, fin)` en el original.
    Cada especial va entre dos frases para que los tramos conservados sean conocidos.
    """
    partes = [rng.choice(FRASES)]
    for _ in range(rng.randint(0, 8)):
        partes.append(rng.choice(especiales))
        partes.append(rng.choice(FRASES))
    texto, conservados, pos = "", [], 0
    for i, parte in enumerate(partes):
        if i % 2 == 0:
            conservados.append((pos, pos + len(parte)))
        texto += parte
        pos += len(parte)
    return texto, conservados


def test_quita_encabezados_guiones_y_saltos_repetidos():
    nota = f"{ENCABEZADO}\nPaciente con HTA.\n\n\n-----\nNiega DM.\n"
    assert limpiar(nota).texto == "Paciente con HTA.\n \nNiega DM."


def test_conserva_el_texto_original():
    nota = f"{ENCABEZADO}\nPaciente con HTA."
    assert limpiar(nota).original == nota


@pytest.mark.parametrize("vacio", [None, "", "   \n\n "])
def test_texto_vacio_o_none_devuelve_resultado_vacio(vacio):
    limpio = limpiar(vacio)
    assert isinstance(limpio, TextoLimpio)
    assert limpio.texto == ""
    assert limpio.a_original(0, 0) == (0, 0)


def test_no_string_devuelve_resultado_vacio():
    assert limpiar(123).texto == ""


@pytest.mark.parametrize("semilla", range(200))
def test_mismo_texto_que_la_tesis(semilla):
    rng = random.Random(semilla)
    nota, _ = _nota_aleatoria(rng)
    if rng.random() < 0.5:
        nota = "\n\n" + ENCABEZADO + "\n" + nota + "\n---\n"
    assert limpiar(nota).texto == _limpiar_tesis(nota)


def test_a_original_traduce_posiciones():
    nota = f"{ENCABEZADO}\nPaciente con HTA."
    limpio = limpiar(nota)
    i = limpio.texto.index("HTA")
    inicio, fin = limpio.a_original(i, i + 3)
    assert nota[inicio:fin] == "HTA"
    assert inicio == nota.index("HTA")


@pytest.mark.parametrize("semilla", range(200))
def test_propiedad_subcadenas_sin_cruzar_tramos_eliminados(semilla):
    rng = random.Random(semilla)
    nota, conservados = _nota_aleatoria(rng)
    limpio = limpiar(nota)
    # Las frases no tienen espacios en los bordes, así que el strip final no las toca
    # y aparecen en el texto limpio en el mismo orden.
    cursor = 0
    for ini_o, fin_o in conservados:
        frase = nota[ini_o:fin_o]
        ini_l = limpio.texto.index(frase, cursor)
        cursor = ini_l + len(frase)
        for _ in range(10):
            i = rng.randint(ini_l, cursor)
            f = rng.randint(i, cursor)
            a, b = limpio.a_original(i, f)
            assert nota[a:b] == limpio.texto[i:f]
        assert limpio.a_original(ini_l, cursor) == (ini_o, fin_o)


@pytest.mark.parametrize("semilla", range(50))
def test_propiedad_saltos_de_linea_conservados(semilla):
    """Cada carácter del texto limpio que no es un reemplazo apunta al mismo carácter."""
    rng = random.Random(semilla)
    nota, _ = _nota_aleatoria(rng)
    limpio = limpiar(nota)
    for i, caracter in enumerate(limpio.texto):
        a, b = limpio.a_original(i, i + 1)
        assert b == a + 1
        if caracter != " ":
            assert nota[a] == caracter


@pytest.mark.parametrize("rango", [(-1, 2), (3, 2), (0, 10_000)])
def test_a_original_rechaza_rangos_invalidos(rango):
    limpio = limpiar("Paciente con HTA.")
    with pytest.raises(ValueError):
        limpio.a_original(*rango)


def test_a_original_rango_vacio_al_final():
    nota = "Paciente con HTA.\n---\n"
    limpio = limpiar(nota)
    n = len(limpio.texto)
    assert limpio.a_original(n, n) == (len("Paciente con HTA."), len("Paciente con HTA."))


# --- Traducción de menciones al original -----------------------------------------------


def test_mencion_a_original_sin_tramo_eliminado_conserva_el_texto():
    original = ENCABEZADO + "\nPaciente con EPOC."
    limpio = limpiar(original)
    inicio = limpio.texto.index("EPOC")
    mencion = Mencion("EPOC", inicio, inicio + 4)

    traducida = limpio.mencion_a_original(mencion)

    assert traducida.texto == "EPOC"
    assert original[traducida.inicio : traducida.fin] == "EPOC"


def test_mencion_a_original_que_cruza_un_tramo_eliminado_toma_el_texto_del_original():
    original = "Paciente con diabetes ------ mellitus."
    limpio = limpiar(original)
    inicio = limpio.texto.index("diabetes")
    fin = limpio.texto.index("mellitus") + len("mellitus")
    mencion = Mencion(limpio.texto[inicio:fin], inicio, fin)

    traducida = limpio.mencion_a_original(mencion)

    assert mencion.texto == "diabetes   mellitus"
    assert traducida.texto == "diabetes ------ mellitus"
    assert original[traducida.inicio : traducida.fin] == traducida.texto


# --- Saltos de línea de Windows y Mac antiguo (#24) ------------------------------------

ESPECIALES_CR = ESPECIALES + [
    "\r\n",
    "\r\n\r\n\r\n",
    "\r",
    "\n\r\n\r",
    ENCABEZADO + "\r\n",
    "---\r\n",
]


@pytest.mark.parametrize(
    ("nota", "esperado"),
    [
        ("Paciente con HTA.\r\n\r\n\r\nNiega DM.", "Paciente con HTA.\nNiega DM."),
        ("Paciente con HTA.\rNiega DM.", "Paciente con HTA.\nNiega DM."),
        (f"{ENCABEZADO}\r\nPaciente con HTA.\r\n", "Paciente con HTA."),
    ],
)
def test_normaliza_saltos_de_linea_de_windows_y_mac(nota, esperado):
    # Antes de #24 "\r\n\r\n" quedaba como "\r\n\r\n": los saltos no se colapsaban.
    assert limpiar(nota).texto == esperado


@pytest.mark.parametrize("semilla", range(100))
def test_con_retornos_de_carro_da_el_mismo_texto_que_con_saltos_unix(semilla):
    rng = random.Random(semilla)
    nota, _ = _nota_aleatoria(rng, ESPECIALES_CR)
    unix = nota.replace("\r\n", "\n").replace("\r", "\n")
    assert limpiar(nota).texto == limpiar(unix).texto


@pytest.mark.parametrize("semilla", range(200))
def test_propiedad_subcadenas_con_retornos_de_carro(semilla):
    rng = random.Random(semilla)
    nota, conservados = _nota_aleatoria(rng, ESPECIALES_CR)
    limpio = limpiar(nota)
    assert "\r" not in limpio.texto
    cursor = 0
    for ini_o, fin_o in conservados:
        frase = nota[ini_o:fin_o]
        ini_l = limpio.texto.index(frase, cursor)
        cursor = ini_l + len(frase)
        for _ in range(10):
            i = rng.randint(ini_l, cursor)
            f = rng.randint(i, cursor)
            a, b = limpio.a_original(i, f)
            assert nota[a:b] == limpio.texto[i:f]
        assert limpio.a_original(ini_l, cursor) == (ini_o, fin_o)


@pytest.mark.parametrize("semilla", range(50))
def test_propiedad_cada_caracter_apunta_al_original_con_retornos_de_carro(semilla):
    rng = random.Random(semilla)
    nota, _ = _nota_aleatoria(rng, ESPECIALES_CR)
    limpio = limpiar(nota)
    for i, caracter in enumerate(limpio.texto):
        a, b = limpio.a_original(i, i + 1)
        assert b == a + 1
        if caracter == "\n":
            # Un "\r" suelto se reemplaza por "\n" y apunta a ese "\r".
            assert nota[a] in "\n\r"
        elif caracter != " ":
            assert nota[a] == caracter


def test_mencion_que_cruza_un_salto_de_windows_toma_el_texto_del_original():
    original = "Paciente con diabetes\r\nmellitus."
    limpio = limpiar(original)
    inicio = limpio.texto.index("diabetes")
    fin = limpio.texto.index("mellitus") + len("mellitus")

    traducida = limpio.mencion_a_original(Mencion(limpio.texto[inicio:fin], inicio, fin))

    assert traducida.texto == "diabetes\r\nmellitus"
    assert original[traducida.inicio : traducida.fin] == traducida.texto
