"""Tests de la clasificación por contexto. Todas las notas son sintéticas.

Desde #73 se clasifica cada mención por su posición. Los tests que venían de la
clasificación por texto usan `estados`, que clasifica la primera aparición de cada
entidad en la nota: con una sola aparición, el resultado es el mismo.
"""

import time

import pytest

from ner_clinic import contexto
from ner_clinic.contexto import PRIORIDAD, Intervalo, clasificar_mencion, intervalos
from ner_clinic.tipos import Estado, Mencion


def mencion(nota: str, entidad: str, aparicion: int = 0) -> Mencion:
    """La `aparicion`-ésima aparición (desde 0) de `entidad` en `nota`."""
    inicio = -1
    for _ in range(aparicion + 1):
        inicio = nota.index(entidad, inicio + 1)
    return Mencion(entidad, inicio, inicio + len(entidad))


def estado(nota: str, entidad: str, aparicion: int = 0) -> Estado:
    return clasificar_mencion(mencion(nota, entidad, aparicion), intervalos(nota))


def estados(nota: str, *entidades: str) -> dict[str, Estado]:
    """Estado de la primera aparición de cada entidad."""
    return {entidad: estado(nota, entidad) for entidad in entidades}


@pytest.mark.parametrize(
    ("nota", "entidad", "esperado"),
    [
        ("Paciente de 64 años. Niega EPOC.", "EPOC", Estado.NEGADA),
        ("Madre con cáncer de mama.", "cáncer de mama", Estado.FAMILIAR),
        ("HTA en tratamiento con losartán.", "HTA", Estado.CONFIRMADA),
        ("Rx de tórax sin evidencia de neumonía.", "neumonía", Estado.NEGADA),
        ("Angiotac de tórax: TEP descartado.", "TEP", Estado.NEGADA),
        ("Prueba de VIH negativa.", "VIH", Estado.NEGADA),
        ("Sospecha de apendicitis aguda.", "apendicitis aguda", Estado.POSIBLE),
        ("No refiere asma.", "asma", Estado.NEGADA),
        ("Ni tabaquismo ni alcoholismo.", "alcoholismo", Estado.NEGADA),
        ("Antecedentes familiares: diabetes en la abuela.", "diabetes", Estado.FAMILIAR),
        ("Abuelo materno con infarto agudo de miocardio.", "infarto", Estado.FAMILIAR),
    ],
)
def test_clasifica_segun_contexto(nota, entidad, esperado):
    assert estado(nota, entidad) is esperado


def test_barrera_corta_la_negacion():
    nota = "Niega fiebre pero presenta diabetes mellitus tipo 2."
    assert estados(nota, "fiebre", "diabetes mellitus tipo 2") == {
        "fiebre": Estado.NEGADA,
        "diabetes mellitus tipo 2": Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    "barrera", ["aunque", "actualmente", "refiere", "presenta", "con", "se evidencia"]
)
def test_cada_barrera_corta_la_negacion(barrera):
    nota = f"Niega disnea {barrera} cefalea."
    assert estado(nota, "cefalea") is Estado.CONFIRMADA


def test_sin_embargo_corta_la_negacion_y_no_abre_otra():
    """Corregido (#24): en la tesis el "sin" de "sin embargo" abría otra negación."""
    nota = "Niega disnea sin embargo cefalea."
    assert estados(nota, "disnea", "cefalea") == {
        "disnea": Estado.NEGADA,
        "cefalea": Estado.CONFIRMADA,
    }


def test_no_obstante_corta_la_negacion_y_no_abre_otra():
    nota = "Niega disnea, no obstante cefalea intensa."
    assert estado(nota, "cefalea intensa") is Estado.CONFIRMADA


def test_sin_sigue_negando_cuando_no_es_sin_embargo():
    nota = "Sin embolia. Sin embarazo."
    assert estados(nota, "embolia", "embarazo") == {
        "embolia": Estado.NEGADA,
        "embarazo": Estado.NEGADA,
    }


def test_la_negacion_termina_en_coma_o_punto():
    # Tras la coma viene una cláusula larga, no un elemento de lista: la coma corta (#62).
    nota = "Niega tos, hipotiroidismo diagnosticado hace dos años. Sin edemas. Gastritis crónica."
    assert estados(nota, "tos", "hipotiroidismo", "edemas", "Gastritis crónica") == {
        "tos": Estado.NEGADA,
        "hipotiroidismo": Estado.CONFIRMADA,
        "edemas": Estado.NEGADA,
        "Gastritis crónica": Estado.CONFIRMADA,
    }


def test_el_bloque_familiar_llega_hasta_el_fin_de_linea():
    nota = "Antecedentes familiares: madre hipertensa, padre con cáncer de próstata\nHTA."
    resultado = estados(nota, "cáncer de próstata", "HTA")
    assert resultado == {"cáncer de próstata": Estado.FAMILIAR, "HTA": Estado.CONFIRMADA}


def test_familiar_tiene_prioridad_sobre_negada():
    # "Hermano sin asma" es bloque familiar y "sin asma" alcance negado: ganan los familiares.
    nota = "Hermano sin asma."
    assert intervalos(nota) == [
        Intervalo(0, 16, Estado.FAMILIAR),
        Intervalo(8, 16, Estado.NEGADA),
    ]
    assert estado(nota, "asma") is Estado.FAMILIAR


# --- Estado posible (#88) ---------------------------------------------------------------


def test_prioridad_familiar_posible_negada():
    assert PRIORIDAD == (Estado.FAMILIAR, Estado.POSIBLE, Estado.NEGADA)
    m = Mencion("asma", 5, 9)
    familiar, posible, negada = (Intervalo(0, 20, e) for e in PRIORIDAD)
    assert clasificar_mencion(m, [negada, posible]) is Estado.POSIBLE
    assert clasificar_mencion(m, [posible, familiar, negada]) is Estado.FAMILIAR
    assert clasificar_mencion(m, [negada]) is Estado.NEGADA
    assert clasificar_mencion(m, [Intervalo(9, 20, Estado.POSIBLE)]) is Estado.CONFIRMADA


def test_intervalos_posibles_entre_familiares_y_negados():
    nota = "Sospecha de TEP. Madre con asma."
    assert intervalos(nota) == [
        Intervalo(17, 31, Estado.FAMILIAR),
        Intervalo(0, 15, Estado.POSIBLE),  # "sospecha de" ya no niega (#83)
    ]
    assert estado(nota, "TEP") is Estado.POSIBLE
    assert estado("Hermano con sospecha de TEP.", "TEP") is Estado.FAMILIAR


# --- Reglas de incertidumbre (#83) ------------------------------------------------------


@pytest.mark.parametrize(
    ("nota", "entidad", "esperado"),
    [
        # US1 del PRD `incertidumbre`.
        ("Sospecha de TEP.", "TEP", Estado.POSIBLE),
        ("Ingresa por probable neumonía.", "neumonía", Estado.POSIBLE),
        ("No se descarta IAM.", "IAM", Estado.POSIBLE),
        ("SCA vs. pericarditis.", "SCA", Estado.POSIBLE),
        ("SCA vs. pericarditis.", "pericarditis", Estado.POSIBLE),
        ("Lesión compatible con metástasis.", "metástasis", Estado.POSIBLE),
        ("Se descartó TEP.", "TEP", Estado.NEGADA),
        ("Paciente que niega TEP.", "TEP", Estado.NEGADA),
        ("TEP.", "TEP", Estado.CONFIRMADA),
    ],
)
def test_us1_sospechas_negadas_y_confirmadas(nota, entidad, esperado):
    assert estado(nota, entidad) is esperado


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("Sospecha clínica de apendicitis.", "apendicitis"),
        ("Hallazgos sospechosos de malignidad.", "malignidad"),
        ("Posible ITU.", "ITU"),
        ("Probablemente secundario a hipotiroidismo.", "hipotiroidismo"),
        ("Probables metástasis hepáticas.", "metástasis hepáticas"),
        ("Imagen sugestiva de absceso.", "absceso"),
        ("Clínica que sugiere colecistitis.", "colecistitis"),
        ("Nódulos compatibles con granulomas.", "granulomas"),
        ("Orientación diagnóstica: pancreatitis aguda.", "pancreatitis aguda"),
        ("Cuadro que pudiera corresponder a sepsis.", "sepsis"),
        ("Podría tratarse de una celulitis.", "celulitis"),
        ("Puede corresponder a un lipoma.", "lipoma"),
        ("Parece una dermatitis de contacto.", "dermatitis"),
        ("No parece haber neumonía.", "neumonía"),
        ("Al parecer convulsiones en casa.", "convulsiones"),
        ("Impresiona de cuadro viral.", "cuadro viral"),
        ("Se plantea vasculitis.", "vasculitis"),
        ("Se interpretó como migraña.", "migraña"),
        ("Ante la posibilidad de endocarditis.", "endocarditis"),
        ("Dudosa trombosis venosa.", "trombosis venosa"),
        ("Sin clara neumonía.", "neumonía"),
        ("No se puede descartar sangrado activo.", "sangrado activo"),
        ("Sin descartar TVP.", "TVP"),
        ("Sin que se pueda descartar TVP.", "TVP"),
        ("No descartamos endocarditis.", "endocarditis"),
        ("NO SE DESCARTA IAM.", "IAM"),
        ("AIT versus migraña.", "migraña"),
        ("Formaciones tubulares, a valorar flebitis.", "flebitis"),
    ],
)
def test_claves_de_incertidumbre(nota, entidad):
    assert estado(nota, entidad) is Estado.POSIBLE


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        # Hacia atrás: las 4 palabras de antes.
        ("Fiebre sin foco infeccioso aparente.", "foco infeccioso"),
        ("Hepatopatía difusa vs esteatosis.", "Hepatopatía difusa"),
        ("Caída en domicilio ( síncope ? ).", "síncope"),
        ("Caída en domicilio (¿síncope?).", "síncope"),
        ("Trombosis dudosa en la vena poplítea.", "Trombosis"),
    ],
)
def test_claves_de_incertidumbre_hacia_atras(nota, entidad):
    assert estado(nota, entidad) is Estado.POSIBLE


def test_hacia_atras_solo_las_palabras_cercanas():
    nota = "Diabetes mellitus tipo 2 en manejo con metformina vs insulina."
    assert estado(nota, "Diabetes mellitus") is Estado.CONFIRMADA


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        # Las claves son palabras enteras.
        ("Improbable neumonía.", "neumonía"),
        ("Es imposible valorar la neumonía.", "neumonía"),
        ("Aparece neumonía.", "neumonía"),
        # Lo que ya está antes de la clave queda como estaba.
        ("Lesión compatible con metástasis.", "Lesión"),
        # Se rechazaron con la medición: dicen qué se busca o qué se planea.
        ("Remitido para descartar TVP.", "TVP"),
        ("Valorar amigdalectomía por amigdalitis.", "amigdalitis"),
    ],
)
def test_lo_que_no_es_clave_de_incertidumbre(nota, entidad):
    assert estado(nota, entidad) is Estado.CONFIRMADA


@pytest.mark.parametrize(
    "nota",
    [
        "Sin datos sugestivos de neumonía.",
        "No se evidencian imágenes compatibles con neumonía.",
        "Tampoco refiere clínica sugestiva de neumonía.",
        "Ninguna lesión sugiere neumonía.",
    ],
)
def test_una_clave_dentro_de_un_alcance_negado_no_abre_posible(nota):
    """ "Sin datos sugestivos de X": lo que se niega es la sospecha, así que X queda negada."""
    assert estado(nota, "neumonía") is Estado.NEGADA


def test_el_alcance_posible_se_corta_donde_empieza_una_negacion():
    nota = "Probable neumonía sin derrame pleural."
    assert estados(nota, "neumonía", "derrame pleural") == {
        "neumonía": Estado.POSIBLE,
        "derrame pleural": Estado.NEGADA,
    }
    # "descartar" niega, aunque vaya tras "sugiere".
    assert estado("Se sugiere descartar neoplasia.", "neoplasia") is Estado.NEGADA


def test_la_negacion_dentro_de_la_clave_no_cuenta():
    # El "no" y el "descarta" de "no se descarta" son parte de la clave: no hay negados.
    assert [i.estado for i in intervalos("No se descarta IAM.")] == [Estado.POSIBLE]
    # Si el "sin" de "sin clara" negara, "sugestiva de" quedaría negada dentro de él.
    nota = "Sin clara imagen sugestiva de litiasis."
    assert {i.estado for i in intervalos(nota)} == {Estado.POSIBLE}
    assert estado(nota, "litiasis") is Estado.POSIBLE


def test_la_incertidumbre_sigue_por_la_enumeracion_y_respeta_las_barreras():
    nota = "Probable neumonía, ITU o sepsis."
    assert set(estados(nota, "neumonía", "ITU", "sepsis").values()) == {Estado.POSIBLE}
    nota = "Probable neumonía pero presenta diabetes."
    assert estados(nota, "neumonía", "diabetes") == {
        "neumonía": Estado.POSIBLE,
        "diabetes": Estado.CONFIRMADA,
    }
    assert estado("Probable neumonía. Diabetes.", "Diabetes") is Estado.CONFIRMADA


def test_dos_claves_seguidas_abren_cada_una_su_alcance():
    # El tramo de "pudiera" se corta en la barrera "con"; "compatible con" abre otro.
    nota = "Debilidad que pudiera ser compatible con un infarto lacunar."
    assert estado(nota, "infarto lacunar") is Estado.POSIBLE


def test_seccion_de_alergicos_que_no_refiere():
    nota = "Alérgicos: no refiere"
    assert estado(nota, "Alérgicos") is Estado.NEGADA


def test_los_disparadores_ignoran_mayusculas():
    assert estado("NIEGA EPOC.", "EPOC") is Estado.NEGADA


def test_entidad_con_caracteres_especiales_de_regex():
    nota = "Niega infección por VIH (+)."
    assert estado(nota, "VIH (+)") is Estado.NEGADA


def test_intervalos_familiares_primero_y_luego_negados():
    nota = "Niega EPOC. Madre con asma."
    assert intervalos(nota) == [
        Intervalo(12, 26, Estado.FAMILIAR),
        Intervalo(0, 10, Estado.NEGADA),
    ]
    assert intervalos("Paciente con HTA.") == []


def test_cada_mencion_se_clasifica_por_su_posicion():
    """Corregido (#73): la tesis buscaba el texto de la entidad en los bloques negados, así
    que la diabetes en manejo también salía negada."""
    nota = "Niega diabetes. Hace 3 años: diabetes en manejo con metformina."
    assert estado(nota, "diabetes", aparicion=0) is Estado.NEGADA
    assert estado(nota, "diabetes", aparicion=1) is Estado.CONFIRMADA


def test_la_misma_palabra_negada_y_de_un_familiar():
    nota = "Niega asma. Hermano con asma."
    assert estado(nota, "asma", aparicion=0) is Estado.NEGADA
    assert estado(nota, "asma", aparicion=1) is Estado.FAMILIAR


def test_mencion_dentro_si_su_inicio_cae_en_el_intervalo():
    # El alcance de "Niega" se corta antes de la barrera "con": "Niega dolor ".
    nota = "Niega dolor con fiebre."
    assert intervalos(nota) == [Intervalo(0, 12, Estado.NEGADA)]
    # Una mención que empieza dentro del alcance y sale de él cuenta como negada...
    assert estado(nota, "dolor con fiebre") is Estado.NEGADA
    # ... y una que empieza justo donde termina, no.
    pegada = Mencion("con fiebre", 12, 22)
    assert clasificar_mencion(pegada, intervalos(nota)) is Estado.CONFIRMADA


def test_mencion_que_empieza_antes_del_alcance_no_se_niega():
    """Con solapamiento, "IAM no Q" quedaba negada porque el disparador está dentro de la
    mención; con el inicio, no: la mención empieza antes del alcance "no Q"."""
    nota = "Antecedente de IAM no Q."
    assert intervalos(nota) == [Intervalo(19, 23, Estado.NEGADA)]
    assert estado(nota, "IAM no Q") is Estado.CONFIRMADA
    assert estado(nota, "Q") is Estado.NEGADA


def test_mencion_que_cruza_una_barrera_sigue_negada():
    """Con contención, "diabetes con nefropatía" no quedaba negada porque la barrera "con"
    corta el alcance a mitad de la mención; con el inicio, sí."""
    nota = "Niega diabetes con nefropatía."
    assert intervalos(nota) == [Intervalo(0, 15, Estado.NEGADA)]
    assert estado(nota, "diabetes con nefropatía") is Estado.NEGADA
    assert estado(nota, "nefropatía") is Estado.CONFIRMADA


@pytest.mark.parametrize(
    "nota",
    [
        "Se considera descartado TEP.",
        "Angiotac: descartada TEP masiva.",
        "Se descarta TEP por angiotac.",
        "Dímero D negativo para TEP.",
    ],
)
def test_descartado_antes_de_la_entidad_niega(nota):
    """Corregido (#24): en la tesis "descartado" solo negaba lo que iba antes."""
    assert estado(nota, "TEP") is Estado.NEGADA


def test_descartado_antes_respeta_las_barreras():
    nota = "Se descarta TEP pero presenta neumonía."
    assert estados(nota, "TEP", "neumonía") == {
        "TEP": Estado.NEGADA,
        "neumonía": Estado.CONFIRMADA,
    }


@pytest.mark.parametrize("posicion", [0, 6, 28])
def test_mencion_vacia_no_se_marca_como_familiar(posicion):
    nota = "Madre con diabetes mellitus."
    vacia = Mencion("", posicion, posicion)
    assert clasificar_mencion(vacia, intervalos(nota)) is Estado.CONFIRMADA


# --- Enumeraciones (#62) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("nota", "entidades"),
    [
        ("Niega HTA, DM, cardiopatias.", ["HTA", "DM", "cardiopatias"]),
        ("niega DM y HTA.", ["DM", "HTA"]),
        ("NIEGA ANTECEDENTES DE IAM, ACV O ENFERMEDAD RENAL.", ["IAM", "ACV", "ENFERMEDAD RENAL"]),
        ("Negativo para VIH y hepatitis B.", ["VIH", "hepatitis B"]),
        ("niega HTA, DM, EPOC, ERC.", ["HTA", "DM", "EPOC", "ERC"]),
        ("Niega úlcera péptica y enfermedad hepática.", ["úlcera péptica", "enfermedad hepática"]),
        ("Sin HTA ni DM.", ["HTA", "DM"]),
        ("Sin disnea, tos e hemoptisis.", ["disnea", "tos", "hemoptisis"]),
        ("No refiere asma, EPOC, y tabaquismo.", ["asma", "EPOC", "tabaquismo"]),
    ],
)
def test_la_negacion_cubre_toda_la_enumeracion(nota, entidades):
    assert set(estados(nota, *entidades).values()) == {Estado.NEGADA}


@pytest.mark.parametrize(
    ("nota", "negada", "confirmada"),
    [
        # "y" seguida de una barrera no sigue la lista.
        ("niega fiebre y presenta diabetes", "fiebre", "diabetes"),
        ("Niega disnea y refiere cefalea.", "disnea", "cefalea"),
        ("niega fiebre, refiere diabetes", "fiebre", "diabetes"),
        ("Niega HTA, DM pero presenta EPOC.", "DM", "EPOC"),
        ("Sin HTA. DM2 en manejo.", "HTA", "DM2"),
        ("Niega HTA\nDM2 en manejo.", "HTA", "DM2"),
        # Palabras que cambian de tema no abren un elemento nuevo.
        ("Sin antecedentes, salvo gastritis.", "antecedentes", "gastritis"),
        ("No DM, sí HTA.", "DM", "HTA"),
        # Un elemento de más de tres palabras no es parte de la lista.
        ("Niega tos y dolor torácico opresivo irradiado.", "tos", "dolor torácico"),
    ],
)
def test_la_enumeracion_se_corta_en_las_barreras(nota, negada, confirmada):
    assert estados(nota, negada, confirmada) == {
        negada: Estado.NEGADA,
        confirmada: Estado.CONFIRMADA,
    }


def test_la_lista_acaba_en_el_ultimo_elemento_corto():
    # "EPOC diagnosticada hace dos años" no es un elemento: la lista acaba en "DM".
    nota = "Niega HTA, DM, EPOC diagnosticada hace dos años."
    # El tramo de siempre ("Niega HTA") y la lista ("Niega HTA, DM").
    assert intervalos(nota) == [
        Intervalo(0, 9, Estado.NEGADA),
        Intervalo(0, 13, Estado.NEGADA),
    ]
    assert estados(nota, "DM", "EPOC") == {"DM": Estado.NEGADA, "EPOC": Estado.CONFIRMADA}


def test_la_lista_no_recorta_el_tramo_de_siempre():
    """El tramo de siempre sigue negando: la lista es otro patrón, no una alternativa que
    se quede con un alcance más corto."""
    nota = "Sin tos ni fiebre o signos de dificultad respiratoria."
    assert estados(nota, "tos", "fiebre", "dificultad respiratoria") == {
        "tos": Estado.NEGADA,
        "fiebre": Estado.NEGADA,
        "dificultad respiratoria": Estado.NEGADA,
    }


def test_fuera_de_una_lista_la_y_sigue_cortando():
    """La "y" corta cuando el primer elemento no es corto: "Niega dolor torácico opresivo
    irradiado y disnea" no se lee como una lista."""
    nota = "Niega dolor torácico opresivo irradiado y disnea."
    assert estados(nota, "dolor torácico", "disnea") == {
        "dolor torácico": Estado.NEGADA,
        "disnea": Estado.CONFIRMADA,
    }


# --- La lista no salta a la cláusula siguiente (#84) ------------------------------------


@pytest.mark.parametrize(
    ("nota", "negada", "confirmada"),
    [
        ("Sin secuelas motoras, demencia vascular leve.", "secuelas motoras", "demencia vascular"),
        ("sin gran esfuerzo, tres síncopes.", "gran esfuerzo", "síncopes"),
        ("Sin evidencia de sangrado activo, leve hipertiroidismo.", "sangrado", "hipertiroidismo"),
    ],
)
def test_tras_un_elemento_largo_la_coma_no_sigue_con_otro_largo(nota, negada, confirmada):
    assert estados(nota, negada, confirmada) == {
        negada: Estado.NEGADA,
        confirmada: Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    ("nota", "entidades"),
    [
        ("Niega HTA, DM, EPOC.", ["HTA", "DM", "EPOC"]),
        ("sin cefalea, diplopía, fotofobia.", ["cefalea", "diplopía", "fotofobia"]),
        ("NIEGA ANTECEDENTES DE IAM, ACV O ENFERMEDAD RENAL.", ["IAM", "ACV", "ENFERMEDAD RENAL"]),
        ("Negativo para VIH y hepatitis B.", ["VIH", "hepatitis B"]),
        # Un largo tras un corto, y después del largo otro corto.
        ("Niega hematuria, sangrado de encías, melenas.", ["sangrado de encías", "melenas"]),
        ("No cefalea, asimetrías faciales.", ["cefalea", "asimetrías faciales"]),
        # Dos largos seguidos, si los une una conjunción.
        ("Niega úlcera péptica, y enfermedad hepática.", ["úlcera péptica", "enfermedad hepática"]),
        ("Sin dolor torácico ni disnea súbita, HTA.", ["dolor torácico", "disnea súbita", "HTA"]),
    ],
)
def test_la_lista_sigue_negando_con_elementos_cortos_y_conjunciones(nota, entidades):
    assert set(estados(nota, *entidades).values()) == {Estado.NEGADA}


# --- "No presenta" (#61) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("No presenta EPOC en espirometría.", "EPOC"),
        ("No presenta hipertensión pulmonar en ecocardiograma.", "hipertensión pulmonar"),
        ("no presenta encefalopatía hepática", "encefalopatía hepática"),
        ("No presenta antecedente de IAM.", "IAM"),
        ("NO PRESENTA  signos de TVP.", "TVP"),
    ],
)
def test_no_presenta_niega(nota, entidad):
    """Corregido (#61): la barrera "presenta" dejaba vacío el bloque abierto por "no"."""
    assert estado(nota, entidad) is Estado.NEGADA


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("No presentó fiebre.", "fiebre"),
        ("No ha presentado IAM.", "IAM"),
        ("No presentaba disnea.", "disnea"),
        ("No presentan lesiones.", "lesiones"),
    ],
)
def test_las_otras_formas_de_no_presentar_ya_negaban(nota, entidad):
    """La barrera es solo la palabra "presenta": las otras formas nunca cortaron."""
    assert estado(nota, entidad) is Estado.NEGADA


def test_no_presenta_niega_toda_la_enumeracion():
    nota = "No presenta HTA, DM ni EPOC."
    assert set(estados(nota, "HTA", "DM", "EPOC").values()) == {Estado.NEGADA}


def test_no_presenta_respeta_las_barreras():
    nota = "No presenta fiebre pero refiere cefalea."
    assert estados(nota, "fiebre", "cefalea") == {
        "fiebre": Estado.NEGADA,
        "cefalea": Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    "nota",
    [
        "Niega fiebre pero presenta diabetes.",
        "niega fiebre y presenta diabetes",
        "Niega fiebre, presenta diabetes.",
        "Niega fiebre presenta diabetes.",
        "Sin fiebre. Presenta diabetes.",
    ],
)
def test_presenta_sigue_cortando_si_no_va_tras_no(nota):
    """La barrera "presenta" sigue cortando cuando no va justo después del "no"."""
    assert estados(nota, "fiebre", "diabetes") == {
        "fiebre": Estado.NEGADA,
        "diabetes": Estado.CONFIRMADA,
    }


# --- Claves que faltaban (#65) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("Se descartó TEP por angiotac.", "TEP"),
        ("Angiotac que descartó la presencia de TEP.", "TEP"),
        ("TAC que descarta complicaciones.", "complicaciones"),
        ("Ecografía que descartan colecistitis.", "colecistitis"),
        ("Se realiza doppler, descartándose TVP.", "TVP"),
        ("Se realiza doppler, descartandose TVP.", "TVP"),
        ("Se hizo TAC descartando apendicitis.", "apendicitis"),
        ("Se han descartado metástasis.", "metástasis"),
        ("Inicia anticoagulación tras descartar hemorragia.", "hemorragia"),
        ("Descartar neumonía.", "neumonía"),
        ("SE DESCARTARON FRACTURAS.", "FRACTURAS"),
    ],
)
def test_las_formas_de_descartar_niegan_lo_que_sigue(nota, entidad):
    """Corregido (#65): antes solo negaban "descartado/a(s)" y "se descarta"."""
    assert estado(nota, entidad) is Estado.NEGADA


@pytest.mark.parametrize(
    "nota",
    [
        "Remitido para descartar TVP.",
        "Ingresa a descartar TVP.",
        "PARA DESCARTAR TVP.",
    ],
)
def test_para_descartar_no_niega(nota):
    """Las frases con "para descartar" o "a descartar" dicen qué se busca, no qué se encontró."""
    assert estado(nota, "TVP") is Estado.CONFIRMADA


def test_descartado_hacia_atras_sigue_igual():
    assert estado("Angiotac: TEP descartado.", "TEP") is Estado.NEGADA


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("Tampoco asma.", "asma"),
        ("Tampoco refiere asma.", "asma"),
        ("Tampoco presenta edemas.", "edemas"),
        ("Nunca ha tenido convulsiones.", "convulsiones"),
        ("NUNCA HA TENIDO CONVULSIONES.", "CONVULSIONES"),
        ("Ningún signo de focalidad neurológica.", "focalidad"),
        ("Ninguna lesión sugiere malignidad.", "lesión"),
        ("Ninguno de los criterios de sepsis.", "sepsis"),
        ("En ningún momento pérdida de conciencia.", "pérdida de conciencia"),
        ("En ningun momento pérdida de conciencia.", "pérdida de conciencia"),
        ("No se evidencia neumonía.", "neumonía"),
        ("No se evidencian lesiones.", "lesiones"),
    ],
)
def test_otras_claves_niegan(nota, entidad):
    """Corregido (#65): "tampoco", "nunca", "ningún" y "no se evidencia" no negaban."""
    assert estado(nota, entidad) is Estado.NEGADA


def test_tampoco_niega_toda_la_enumeracion():
    nota = "Tampoco refiere HTA, DM ni EPOC."
    assert set(estados(nota, "HTA", "DM", "EPOC").values()) == {Estado.NEGADA}


@pytest.mark.parametrize(
    ("nota", "negada", "confirmada"),
    [
        ("Se descartó TEP pero presenta neumonía.", "TEP", "neumonía"),
        ("Nunca tuvo convulsiones y refiere cefalea.", "convulsiones", "cefalea"),
        ("Tampoco asma, presenta EPOC.", "asma", "EPOC"),
    ],
)
def test_las_claves_nuevas_respetan_las_barreras(nota, negada, confirmada):
    assert estados(nota, negada, confirmada) == {
        negada: Estado.NEGADA,
        confirmada: Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        # Palabras que contienen una clave no disparan.
        ("Lesión descartable como origen: neumonía.", "neumonía"),
        ("Ninguneo social por depresión.", "depresión"),
        ("Se evidencia neumonía.", "neumonía"),
        # "sin embargo" y "no obstante" siguen sin negar.
        ("Sin embargo presenta HTA.", "HTA"),
        ("No obstante, DM en manejo.", "DM"),
        # Una enfermedad resuelta no es una negada.
        ("Cede la cefalea con analgesia.", "cefalea"),
        ("Se resuelve la neumonía.", "neumonía"),
    ],
)
def test_lo_que_no_es_clave_sigue_sin_negar(nota, entidad):
    assert estado(nota, entidad) is Estado.CONFIRMADA


# Alcance (#66): la negación no pasa de la cláusula.


@pytest.mark.parametrize(
    ("nota", "negada", "confirmada"),
    [
        ("Sin dolor; fractura costal derecha.", "dolor", "fractura"),
        ("Niega fiebre: neumonía basal derecha.", "fiebre", "neumonía"),
        ("Sin hallazgos agudos ) TAC con fractura costal.", "hallazgos", "fractura"),
        ("Sin edemas (ingresa con celulitis).", "edemas", "celulitis"),
        ("No disnea - EPOC en tratamiento.", "disnea", "EPOC"),
    ],
)
def test_la_negacion_se_corta_en_otros_signos(nota, negada, confirmada):
    """Corregido (#66): el alcance solo se cortaba en coma, punto y salto de línea."""
    assert estados(nota, negada, confirmada) == {
        negada: Estado.NEGADA,
        confirmada: Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        # Un guion dentro de una palabra no corta.
        ("Sin infección post-quirúrgica.", "infección"),
        ("Niega dolor torácico tipo-anginoso.", "tipo-anginoso"),
        # Los cortes no rompen la lista: el elemento termina en el signo.
        ("Niega HTA, DM y EPOC; fumador activo.", "EPOC"),
        ("Niega HTA, DM y EPOC (revisado).", "EPOC"),
    ],
)
def test_lo_que_no_es_corte_sigue_negando(nota, entidad):
    assert estado(nota, entidad) is Estado.NEGADA


def test_la_lista_termina_en_el_signo_de_corte():
    nota = "Niega HTA, DM y EPOC; refiere asma."
    assert estados(nota, "HTA", "DM", "EPOC", "asma") == {
        "HTA": Estado.NEGADA,
        "DM": Estado.NEGADA,
        "EPOC": Estado.NEGADA,
        "asma": Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("Poliartritis seronegativa en seguimiento.", "Poliartritis"),
        ("Bacteriemia por estafilococo coagulasa negativo.", "Bacteriemia"),
        ("Hemocultivo con estafilococo coagulasa negativo.", "estafilococo"),
        ("Artritis seronegativa.", "Artritis"),
        ("Neumonía por bacilos gram negativos.", "bacilos"),
        ("Infección por Gram-negativos.", "Infección"),
    ],
)
def test_negativo_dentro_de_otra_palabra_no_niega(nota, entidad):
    """Corregido (#66): "seronegativa" y "coagulasa negativo" negaban la enfermedad.

    "coagulasa negativo" y "gram negativos" nombran el germen, no un resultado negativo.
    """
    assert estado(nota, entidad) is Estado.CONFIRMADA


def test_negativo_hacia_atras_solo_niega_las_palabras_cercanas():
    """Corregido (#66): antes negaba toda la cláusula hasta la coma anterior."""
    nota = "Paciente con diabetes tipo 2 en tratamiento con metformina y urocultivo negativo."
    assert estados(nota, "diabetes", "urocultivo") == {
        "diabetes": Estado.CONFIRMADA,
        "urocultivo": Estado.NEGADA,
    }


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        ("Prueba de VIH negativa.", "VIH"),
        ("Antígeno de superficie de hepatitis B negativo.", "hepatitis B"),
        ("Angiotac: TEP descartado.", "TEP"),
        ("TEP masivo y TVP descartados.", "TVP"),
        # Los dos puntos cortan hacia adelante, pero no entre la prueba y su resultado.
        ("Serología VIH: negativa.", "VIH"),
        ("Serología VIH:negativa.", "VIH"),
        ("Cultivo para tuberculosis negativo.", "tuberculosis"),
    ],
)
def test_negativo_hacia_atras_sigue_negando_lo_cercano(nota, entidad):
    assert estado(nota, entidad) is Estado.NEGADA


@pytest.mark.parametrize(
    ("nota", "negada", "confirmada"),
    [
        ("Sin anticoagular ya que presentó hemorragia digestiva.", "anticoagular", "hemorragia"),
        ("No anticoagulado dada la hemorragia cerebral.", "anticoagulado", "hemorragia"),
    ],
)
def test_las_barreras_de_causa_cortan_la_negacion(nota, negada, confirmada):
    """Corregido (#66): "ya que" y "dado/a" abren otra cláusula."""
    assert estados(nota, negada, confirmada) == {
        negada: Estado.NEGADA,
        confirmada: Estado.CONFIRMADA,
    }


@pytest.mark.parametrize(
    ("nota", "entidad"),
    [
        # "que" y "debido a" no son barreras: medidas en NUBes, bajaban el F1.
        # "por" tampoco: es parte del nombre de muchas enfermedades.
        ("Niega infección por VIH.", "VIH"),
        ("Sin neumonía por COVID-19.", "COVID-19"),
        ("Niega cirrosis por virus C.", "virus C"),
        ("Niega anemia por déficit de hierro.", "déficit de hierro"),
        ("Niega dolor que irradie a brazo izquierdo.", "brazo"),
        ("Sin disnea debido a insuficiencia cardiaca.", "insuficiencia cardiaca"),
        # "por" dentro de otra palabra ("portador") no corta.
        ("Niega portador de hepatitis B.", "hepatitis B"),
    ],
)
def test_lo_que_no_es_barrera_sigue_negando(nota, entidad):
    assert estado(nota, entidad) is Estado.NEGADA


def test_patrones_posible_por_grupos():
    """`patrones_posible` arma los patrones con cualquier subconjunto de claves (para medir)."""
    assert contexto.patrones_posible([], []) == []
    (atras,) = contexto.patrones_posible([], [r"\?"])
    assert atras.search("síncope ?")
    adelante = contexto.patrones_posible([contexto.CLAVES_POSIBLE["probable"]], [])
    assert len(adelante) == 2
    assert not any(p.search("Sospecha de TEP.") for p in adelante)


def test_un_texto_largo_con_muchas_claves_se_clasifica_rapido():
    """Las reglas entre posibles y negados (#83) no comparan cada clave con cada negación."""
    frase = "no fiebre ? probable neumonía vs sin ITU ? "
    for nota in (frase * 2000, "?" * 20000, " ?" * 20000, "posible " * 5000):
        inicio = time.perf_counter()
        intervalos(nota)
        assert time.perf_counter() - inicio < 1


def test_un_texto_largo_sin_comas_se_clasifica_rapido():
    """Corregido (#66): el patrón hacia atrás tardaba en proporción al cuadrado del largo.

    Con 5.000 palabras sin comas ni puntos tardaba unos 28 s. El último caso es una
    racha larga de espacios, que hacía lo mismo con el bloque "Antecedentes familiares".
    """
    palabras = ("dolor", "toracico", "intermitente", "desde", "hace", "dias", "refiere")
    texto = " ".join(palabras[i % len(palabras)] for i in range(5000))
    for nota in (texto, texto + " negativo", " " * 30000 + texto):
        inicio = time.perf_counter()
        intervalos(nota)
        assert time.perf_counter() - inicio < 1


# --- "no claro" (#136) y resultados en la lista negada (#137) ---------------------


def _estado_de(nota: str, palabra: str) -> Estado:
    inicio = nota.index(palabra)
    return clasificar_mencion(Mencion(palabra, inicio, inicio + len(palabra)), intervalos(nota))


@pytest.mark.parametrize(
    ("nota", "palabra"),
    [
        ("Antecedente no claro de EPOC.", "EPOC"),
        ("Antecedente no claro de urolitiasis en la infancia.", "urolitiasis"),
    ],
)
def test_no_claro_es_posible(nota, palabra):
    assert _estado_de(nota, palabra) is Estado.POSIBLE


NOTA_REFLEJOS = (
    "EXAMEN NEUROLÓGICO: NO DÉFICIT MOTOR O SENSITIVO, REFLEJOS OSTEOTENDINOSOS +++/+++, "
    "MARCHA NORMAL."
)


def test_un_resultado_corta_la_lista_negada():
    assert _estado_de(NOTA_REFLEJOS, "REFLEJOS OSTEOTENDINOSOS") is Estado.CONFIRMADA
    assert _estado_de(NOTA_REFLEJOS, "DÉFICIT MOTOR") is Estado.NEGADA
    assert _estado_de(NOTA_REFLEJOS, "SENSITIVO") is Estado.NEGADA


@pytest.mark.parametrize(
    ("nota", "palabra"),
    [
        ("Niega HTA, diabetes tipo 2 y EPOC.", "EPOC"),
        ("Niega HTA, diabetes tipo 2 y EPOC.", "diabetes tipo 2"),
        ("Sin edema, disnea ni cianosis.", "cianosis"),
    ],
)
def test_las_listas_con_numeros_siguen_negando(nota, palabra):
    assert _estado_de(nota, palabra) is Estado.NEGADA


def test_tras_x_o_y_largo_la_coma_no_sigue_con_otro_largo():
    nota = "No déficit motor o sensitivo, reflejos osteotendinosos normales, marcha normal."
    assert _estado_de(nota, "reflejos osteotendinosos") is Estado.CONFIRMADA
    assert _estado_de(nota, "sensitivo") is Estado.NEGADA
    # Con un elemento corto antes, la lista sí sigue
    assert _estado_de("Niega HTA, DM, úlcera péptica.", "úlcera péptica") is Estado.NEGADA


@pytest.mark.parametrize(
    ("nota", "palabra"),
    [
        ("Niega VIH+, HTA y EPOC.", "HTA"),
        ("Niega VIH+, HTA y EPOC.", "EPOC"),
        ("Niega HTA, VIH-1/2, EPOC.", "EPOC"),
        ("Niega hernia L4/5, HTA.", "HTA"),
        ("Niega HTA, Rh+, EPOC.", "EPOC"),
        ("Niega HTA+DM, EPOC.", "EPOC"),
        ("Niega HTA, NIC 2/3, EPOC.", "NIC"),
        ("Niega úlcera péptica, HTA, enfermedad hepática.", "enfermedad hepática"),
        ("Niega antecedentes de IAM, ACV, enfermedad renal.", "enfermedad renal"),
    ],
)
def test_lo_que_parece_resultado_o_lista_con_coma_sigue_negado(nota, palabra):
    assert _estado_de(nota, palabra) is Estado.NEGADA


def test_causa_poco_clara_no_vuelve_posible_la_enfermedad():
    nota = "Causa poco clara de la insuficiencia renal crónica."
    assert _estado_de(nota, "insuficiencia renal crónica") is Estado.CONFIRMADA


# --- Claves mal escritas (#126) ---------------------------------------------------


@pytest.mark.parametrize(
    ("nota", "palabra", "esperado"),
    [
        ("niegh HTA, DM, EPOC.", "EPOC", Estado.NEGADA),
        ("nieag HTA.", "HTA", Estado.NEGADA),
        ("NIEGH HTA.", "HTA", Estado.NEGADA),
        ("Se descarra enfermedad metastásica.", "enfermedad metastásica", Estado.NEGADA),
        ("parde con ICC.", "ICC", Estado.FAMILIAR),
        ("majre con DM2.", "DM2", Estado.FAMILIAR),
        ("Sospeca de TEP.", "TEP", Estado.POSIBLE),
    ],
)
def test_claves_mal_escritas(nota, palabra, esperado):
    assert _estado_de(nota, palabra) is esperado


@pytest.mark.parametrize(
    "nota", ["nieta con HTA.", "parte de HTA.", "Se observa HTA.", "podre con HTA."]
)
def test_palabras_reales_no_se_corrigen(nota):
    assert _estado_de(nota, "HTA") is Estado.CONFIRMADA


def test_las_posiciones_son_las_del_texto_original():
    nota = "Sospeca de TEP. HTA."  # "Sospeca" tiene una letra menos que "sospecha"
    tramos = [i for i in intervalos(nota) if i.estado is Estado.POSIBLE]
    assert tramos and tramos[0].inicio == 0
    assert tramos[0].fin <= nota.index("HTA")
    assert _estado_de(nota, "HTA") is Estado.CONFIRMADA


@pytest.mark.parametrize("nota", ["Descarga de HTA.", "Se descartaba HTA ayer.", "Madure HTA."])
def test_palabras_reales_cercanas_no_cambian_el_estado(nota):
    assert _estado_de(nota, "HTA") is Estado.CONFIRMADA


@pytest.mark.parametrize(
    "palabra", ["descarga", "descartaba", "madure", "pudre", "aquela", "pasible"]
)
def test_no_se_corrigen(palabra):
    from ner_clinic.contexto import _clave_corregida

    assert _clave_corregida(palabra) is None


def test_variantes_de_la_misma_clave_no_son_ambiguas():
    from ner_clinic.contexto import _clave_corregida

    assert _clave_corregida("hermanp") == "hermano"
    assert _clave_corregida("niegn") == "niega"


def test_posiciones_con_una_palabra_mas_larga_y_varias_correcciones():
    nota = "niegaa HTA. majre con DM2. Refiere tos."
    assert _estado_de(nota, "HTA") is Estado.NEGADA
    assert _estado_de(nota, "DM2") is Estado.FAMILIAR
    assert _estado_de(nota, "tos") is Estado.CONFIRMADA


def test_tramo_que_termina_en_la_palabra_corregida():
    nota = "TEP descartadoq. HTA."
    assert _estado_de(nota, "TEP") is Estado.NEGADA
    assert _estado_de(nota, "HTA") is Estado.CONFIRMADA


# --- Formas verbales y "confirmar o descartar" (#202) --------------------------------


@pytest.mark.parametrize(
    ("nota", "entidad", "esperado"),
    [
        (
            "Realizar electromiografía para confirmar o descartar neuropatía periférica diabética.",
            "neuropatía periférica diabética",
            Estado.POSIBLE,
        ),
        (
            "Solicitar ecocardiograma para descartar o confirmar endocarditis.",
            "endocarditis",
            Estado.POSIBLE,
        ),
        (
            "Neoplasia en remisión completa, negando enfermedad oncológica activa.",
            "enfermedad oncológica",
            Estado.NEGADA,
        ),
        ("La paciente negó dolor torácico.", "dolor torácico", Estado.NEGADA),
        (
            "Inicialmente se sospechó enfermedad de Parkinson.",
            "enfermedad de Parkinson",
            Estado.POSIBLE,
        ),
        ("Se sospechó en su momento una artropatía gotosa.", "artropatía gotosa", Estado.POSIBLE),
        ("Cuadro sospechado de tuberculosis pulmonar.", "tuberculosis pulmonar", Estado.POSIBLE),
        # "se negó a" es rechazar, no negar la enfermedad.
        (
            "Se negó a recibir tratamiento para su diabetes mellitus.",
            "diabetes mellitus",
            Estado.CONFIRMADA,
        ),
        # "descartar" solo, en una frase que no dice qué se busca, sigue negando.
        ("Se descartó endocarditis.", "endocarditis", Estado.NEGADA),
    ],
)
def test_formas_verbales_y_confirmar_o_descartar(nota, entidad, esperado):
    assert estado(nota, entidad) is esperado


def test_negandose_niega():
    # #211: como "negando" (#202).
    nota = "Función renal normal, negándose antecedentes de litiasis o infecciones urinarias."
    assert estado(nota, "litiasis") is Estado.NEGADA
    assert estado(nota, "infecciones urinarias") is Estado.NEGADA


def test_negandose_a_es_rechazar_y_no_niega():
    nota = "Paciente negándose a recibir tratamiento para su diabetes mellitus."
    assert estado(nota, "diabetes mellitus") is Estado.CONFIRMADA


# --- Modificador de modo tras el disparador (#218) -----------------------------------


@pytest.mark.parametrize(
    ("nota", "entidad", "esperado"),
    [
        (
            "Se niega de forma categórica y explícita haber presentado infarto agudo de miocardio.",
            "infarto agudo de miocardio",
            Estado.NEGADA,
        ),
        ("Niega de manera rotunda haber tenido angina de pecho.", "angina de pecho", Estado.NEGADA),
        (
            "Niega HTA y posteriormente se diagnostica diabetes mellitus.",
            "diabetes mellitus",
            Estado.CONFIRMADA,
        ),
        # La "y" sigue cortando fuera del modificador.
        ("Niega fiebre y presenta diabetes mellitus.", "diabetes mellitus", Estado.CONFIRMADA),
    ],
)
def test_modificador_de_modo_no_corta_la_negacion(nota, entidad, esperado):
    assert estado(nota, entidad) is esperado
