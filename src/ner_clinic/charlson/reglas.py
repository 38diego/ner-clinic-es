"""Reglas de mapeo de comorbilidades a categorías de Charlson y códigos CIE-10.

Portadas de `04_Perfil_Morbimortalidad/MapeoCIE10 OK.ipynb` de la tesis. Las regex de
inclusión, exclusión y subclasificación son las de la tesis; los textos largos se parten
en varias líneas (cadenas adyacentes) para respetar el ancho.

Errores de la tesis corregidos en el issue #24 (un test por cada uno en
`tests/test_charlson_correcciones.py`), en esta parte: la `ñ` de `cuña` y `acuñamiento`
(el texto normalizado ya no la tiene), `\bles\b` en la ICC, las siglas de arterias del
ACV con `\b` y la estenosis u oclusión sin infarto en I65/I66 antes que I63,
"bronquiectasias" e "hipertension pulmonar severa" en la inclusión pulmonar, "l.e.s."
seguido de espacio, "aneurisma de aorta" sin "la" en la EVP, la errata `gestacvional`
(sin efecto: `gestacional` ya estaba en la misma exclusión), la estenosis basilar en I65
(arteria precerebral) y `\b` antes de las negaciones de todas las exclusiones y de la
etiología de la ICC: el "no" de "año" o "daño" (normalizados como "ano", "dano") se leía
como negación.

Cada categoría es una `Regla` (en `base.py`): una regex de inclusión, una de exclusión y
una función que elige el código CIE-10 específico. Este módulo tiene la parte 1 (IAM a
enfermedad hepática leve); la parte 2 (diabetes a VIH/SIDA) vive en `reglas_parte2.py`.
`REGLAS` une las 17 en el orden de la tesis.

`mapear` normaliza el texto como `limpiar_texto_clinico` y recorre todas las reglas en
orden: la primera que aplica gana, salvo las tres reglas que sobrescriben (diabetes
complicada, hepática severa y tumor metastásico), igual que la tesis. Allí las reglas se
aplicaban sobre un DataFrame con la máscara `Mapeado_Charlson`, que esas tres reglas no
respetaban; aquí se aplican sobre un texto, con el mismo resultado por fila.
"""

import re

from ner_clinic.charlson.base import METASTASIS_AFIRMADA, Regla, normalizar
from ner_clinic.charlson.reglas_parte2 import REGLAS_PARTE2
from ner_clinic.tipos import CategoriaCharlson

__all__ = [
    "REGLAS",
    "REGLAS_PARTE1",
    "REGLAS_PARTE2",
    "Regla",
    "mapear",
    "normalizar",
    "reconoce",
]


# --- 1. Infarto de miocardio (mapear_iam_corregido) ---------------------------
# En la tesis no tiene subclasificación: siempre I21.

IAM = Regla(
    categoria=CategoriaCharlson.INFARTO_MIOCARDIO,
    inclusion=re.compile(
        r"\biam\b|\bima\b|iam[cs]est|\bminoca\b|sindrom.*coronari.*agudo|\bsca\b|"
        r"(infarto|postinfarto).*(miocardio|coronari|ventricul|cardiac|cardiogenic|\bagudo\b|\bantiguo\b)"
    ),
    exclusion=re.compile(
        r"cerebral|cerebelos|lacunar|pulmonar|oseo|testicular|esplenic|intestinal|mesenteric|"
        r"renal|talamic|acv|encefalo|bulbar|multiinfarto|subdural|\bcuna\b|macul|basal|lenticul|"
        r"fazekas|isquemic|\bacm[id]?\b|coroide|retin|oftalmic|angina|cronic|parietal|"
        r"occipital|temporal|frontal|fronto|parieto|temporo|pontic|puente|vertebrobasilar|"
        r"perforante|malacic"
    ),
    subclasificar=lambda _texto: "I21",
)


# --- 2. Insuficiencia cardiaca (mapear_icc_pipeline) ---------------------------
# En la tesis se buscaba con re.IGNORECASE; se conserva aunque el texto ya está en minúsculas.

_ICC_NEGACIONES = re.compile(
    r"\b(sin|descartar|ausencia\s+de|no\s+hay|secundaria\s+a\s+otra|no)\s+(\w+\s+){0,3}"
    r"(isquemi|hipertens|hta|dilatad|restrictiv|coronari|toxic|chagas|reumatic)"
)


def subclasificar_icc(texto: str) -> str:
    """`subclasificar_cie10_icc`: borra las negaciones antes de buscar la etiología."""
    texto_limpio = _ICC_NEGACIONES.sub("", texto.lower())

    if re.search(r"neonatal|recien\s+nacido", texto_limpio):
        return "P29.0"
    elif re.search(
        r"(hipertens|hta).*(renal|nefropatia)|(renal|nefropatia).*(hipertens|hta)", texto_limpio
    ):
        return (
            "I13.2"
            if re.search(r"avanzada|severa|estadio\s+[45]|terminal", texto_limpio)
            else "I13.0"
        )
    elif re.search(
        r"farmaco|toxic|quimioterapia|antraciclina|inmunoterapia|trastuzumab|doxorrubicina",
        texto_limpio,
    ):
        return "I42.7"
    elif re.search(r"isquemic|isquemia|coronariopatia|enfermedad\s+coronaria", texto_limpio):
        return "I25.5"
    elif re.search(r"hipertensiv|hipertension|\bhta\b", texto_limpio):
        return "I11.0"
    elif re.search(r"chagas|\bles\b|lupus|amiloide|infiltrativ|sarcoidosis", texto_limpio):
        return "I43"
    elif re.search(r"reumatic|fiebre\s+reumatica", texto_limpio):
        return "I09.9"
    elif re.search(r"alcoholic|etilic|enolism", texto_limpio):
        return "I42.6"
    elif re.search(r"dilatada", texto_limpio):
        return "I42.0"
    elif re.search(r"restrictiva", texto_limpio):
        return "I42.5"
    elif re.search(r"miocardiopatia|cardiomiopatia", texto_limpio):
        return "I42.9"
    else:
        return "I50"


ICC = Regla(
    categoria=CategoriaCharlson.INSUFICIENCIA_CARDIACA,
    inclusion=re.compile(
        r"\bicc\b|\b(miocardiopatia|cardiomiopatia)\b|"
        # "cardiopatía isquémica" (I25.9) e "hipertensiva" (I11.9) sin falla no son ICC en
        # Quan; la miocardiopatía isquémica (I25.5) entra por `miocardiopatia` (#201).
        r"cardiopatia\s+(?:\w+\s+)?(dilatada|restrictiva|chagasica)|"
        r"(insuficiencia|falla|disfuncion)\s+"
        r"(?:(agud[oa]|cronic[oa]|descompensad[oa]|global|congestiv[oa]|sever[oa]|moderad[oa]|"
        r"leve|tipo\s+\w+|del?|la|y|o)\s+){0,3}"
        r"(cardiac[oa]s?|ventricul\w*|biventricular|sistolica|diastolica|miocardic[oa]s?)",
        re.IGNORECASE,
    ),
    exclusion=re.compile(
        r"\b(sin|descartar|ausencia\s+de|no\s+hay|no\s+(se\s+)?evidencia|niega|riesgo\s+de|"
        r"baja\s+probabilidad\s+de)\s+(?:\w+\s+){0,3}(falla|insuficiencia|disfuncion|icc\b)",
        re.IGNORECASE,
    ),
    subclasificar=subclasificar_icc,
)


# --- 3. Enfermedad vascular periférica (mapear_evp_final) ----------------------


def subclasificar_evp(texto: str) -> str:
    """`subclasificar_cie10_evp`."""
    if re.search(r"mesenteric|intestinal|intestino", texto):
        return "K55.9"
    elif re.search(
        r"aneurisma.*(aorta|aortic|abdominal|infrarrenal|toracic|cayado|raiz|ascendente|descendente)",
        texto,
    ):
        return "I71.9"
    elif re.search(r"buerger|tromboangeitis", texto):
        return "I73.1"
    elif re.search(
        r"injerto|bypass|implante\s+vascular|stent\s+vascular|injerto\s+cardiovascular", texto
    ):
        return "Z95.8"
    elif re.search(r"aterosclero|arteriosclero|ateromato|ateroma|placa", texto):
        return "I70.9"
    elif re.search(r"claudicacion", texto):
        return "I73.9"
    elif re.search(r"estenosis|oclusion|obstruccion|isquemia|trombosis|angiopatia", texto):
        return "I77.1"
    else:
        return "I73.9"


EVP = Regla(
    categoria=CategoriaCharlson.VASCULAR_PERIFERICA,
    inclusion=re.compile(
        r"\b(aterosclero[a-z]*|arteriosclero[a-z]*|ateromato[a-z]*|ateroma|claudicacion|"
        r"tromboangeitis|buerger|angiopatia|isquemia\s+mesenteric[oa]|trombosis\s+mesenteric[oa]|"
        r"evp|eap|dolicoectasia)\b|placas?\s+de\s+ateroma|"
        r"arteriopatia\s+(periferic[oa]|obliterante|oclusiva|de\s+miembros)|"  # #123
        r"(aneurisma|diseccion|pseudoaneurisma)\s+(de\s+(la\s+)?)?(aorta|aortic[oa]|iliac[oa]|femoral|"
        r"abdominal|infrarrenal|suprarrenal|toracic[oa]|fusiform\w*|cayado|raiz|ascendente|"
        r"descendente)|"
        r"(estenosis|oclusion|obstruccion|isquemia|trombosis|insuficiencia|enfermedad|trastorno)\s+"
        # #63: "insuficiencia arterial cronica de miembros"
        r"(arterial|vascular)\s+((cronic[oa]|obliterante|oclusiva)\s+)?(del?\s+|las?\s+)?(periferic[oa]|iliac[oa]|femoral|poplite[oa]|"
        r"tibial|perone[oa]|axilar|braquial|miembro|pedia|mesenteric[oa]|intestinal|intestino|"
        r"subclavia)|(implante|injerto|stent|bypass)\s+(cardio)?vascular"
    ),
    exclusion=re.compile(
        r"coronari|miocardio|\biam\b|\bima\b|valvular|mitral|tricuspide|e?obstrucci[oó]n\s+intestinal|"
        r"ileo\s+paralitico|amiloide|"
        r"aneurisma\s+(cerebral|pulmonar|septal|intracraneal|sacular|willis|acm)|"
        r"\b(sin|descartar|ausencia|no\s+se\s+evidencia|no\s+hay)\s+(?:\w+\s+){0,3}"
        r"(aneurisma|ateromato|aterosclero|ateroma|placa|isquemia|evp|eap|estenosis|oclusion|"
        r"trombosis|trastorno)|carotid\w*|vertebral|vertebrobasilar|intracraneal|supraclinoide|"
        r"cavernos[oa]|cerebral|encefal|bulbo\s+carotideo|poligono|clinoide|cerebel"
    ),
    subclasificar=subclasificar_evp,
)


# --- 4. Enfermedad cerebrovascular (mapear_acv_final) --------------------------


def subclasificar_acv(texto: str) -> str:
    """`subclasificar_cie10_acv`. Las secuelas (I69) tienen prioridad."""
    texto = texto.lower()

    if re.search(
        r"secuela|hemiparesia|hemiplejia|afasia|disartria|antiguo|cronico|malaci|gliosis|"
        r"involutiv|secuelar|paralisis|deficit\s+neurologico",
        texto,
    ):
        return "I69"

    if re.search(r"subaracnoidea|hsa\b", texto):
        return "I60"
    elif re.search(
        r"intraparenquimatosa|intracerebral|gangliobasal|talamica|"
        r"hemorragia\s+(pontin|bulbar|cerebelosa)",
        texto,
    ) or (
        # ACV "hemorrágico" (#123), salvo el isquémico con transformación hemorrágica o
        # "no hemorrágico"
        re.search(r"hemorragic", texto)
        and not re.search(r"isquemi|transformacion|no\s+hemorragic", texto)
    ):
        return "I61"
    elif re.search(r"subdural|epidural", texto):
        return "I62"
    elif re.search(r"retinian[oa]", texto):
        return "H34.0"
    elif re.search(r"\bait\b|\btia\b|transitorio|amaurosis\s+fugax", texto):
        return "G45"

    if re.search(
        r"microangiopat|leucoencefalopatia|fazekas|amiloide|aneurisma|dolicoectasia|"
        r"vascular\s+cronica",
        texto,
    ):
        return "I67"

    # I63 va antes que I65/I66 solo si hay infarto o isquemia; una estenosis u oclusión
    # sin infarto es I65 (precerebral) o I66 (cerebral). Corrección de #24: en la tesis
    # las siglas de arterias iban sin `\b` ("placa" contiene "aca") y antes de I65/I66.
    if re.search(r"infarto|isquemi|stroke", texto):
        return "I63"

    if re.search(r"carotid|vertebral|basilar|precerebral", texto) and re.search(
        r"estenosis|oclusion|obstruccion|placa|ateroma", texto
    ):
        return "I65"
    elif re.search(r"\b(acm|aca|acp)[id]?\b|cerebral", texto) and re.search(
        r"estenosis|oclusion|obstruccion", texto
    ):
        return "I66"

    if re.search(
        r"\b(acm|aca|acp|pica|aica)[id]?\b|basilar|arteria\s+media|"
        r"vascular\s+anterior|territorio|insul|pontin",
        texto,
    ):
        return "I63"
    return "I64"


ACV = Regla(
    categoria=CategoriaCharlson.CEREBROVASCULAR,
    inclusion=re.compile(
        r"\b(acv\w*|ecv\w*|avc|ait|tia|hsa|stroke|ehi|ictus|apoplejia)\b|"
        r"(derrame|embolia|embolismo)\s+cerebral|"
        # #123: la forma escrita completa
        r"(accidente|enfermedad|evento|ataque|insulto)\s+(cerebro\s*-?\s*vascular|vascular\s+cerebral)|"
        r"(infarto|isquemi\w*|hemorragia|hematoma|trombosis|oclusion|estenosis|malaci\w*|gliosis|"
        r"leucoencefalopatia|microangiopat\w*|fazekas|angiopatia|dolicoectasia)\s+(?:\w+\s+){0,3}"
        r"(cerebr\w*|encefal\w*|cerebel\w*|talam\w*|basal\w*|bulbar|pontin\w*|puente|lacunar|"
        r"hemisfer\w*|pariet\w*|front\w*|occipit\w*|tempor\w*|insul\w*|medular|fosa\s+posterior|"
        r"tronco|capsula|ganglio|lenticular|caudado|periventricular|sifon|clinoide|cavernoso|"
        r"supraclinoide|retinian\w*|acm|aca|acp|pica|aica|basilar)|"
        r"\b(aci|pica|aica|acm[id]|aca[id]|acp[id]|basilar|vertebrobasilar|carotid\w*|vertebral|"
        r"percheron|willis)\b|"
        r"(amiloide|mav|aneurisma\s+(cerebral|intracraneal|sacular|fusiforme))"
    ),
    exclusion=re.compile(
        r"coronari|miocardio|\biam\b|\bima\b|pulmonar|esplenico|renal|mesenterico|intestinal|"
        r"testicular|oseo|"
        r"fractur\w*|acunamiento|acunado|osteodegenerat\w*|osteocondros\w*|espondilo\w*|disco\b|"
        r"litica|blastica|osteolisis|osteoporos\w*|osteopen\w*|platillo|cuerpo\s+vertebral|"
        r"columna|"
        r"subgaleal|subcutaneo|cefalohematoma|extra-axial|epidural|blandos|equimosis|herida|"
        r"abrasiva|"
        r"glomus|tumor|masa|neoplasia|schwannoma|adenopatia|meningioma|papilomavirus|"
        r"fetal|placenta|gestacional|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|mimic|niega)\s+(?:\w+\s+){0,3}"
        r"(acv|stroke|infarto|hemorragia|ecv|ait|isquemia|ictus|accidente|evento|enfermedad\s+cerebro)|"
        # #123: más negaciones y contextos que no afirman el ACV
        r"\b(no|descart\w*|negativ[oa]\s+para|ausencia\s+de|riesgo\s+de|prevencion\s+de|"
        r"profilaxis\s+de)\s+(?:\w+\s+){0,2}(acv|ictus|ecv|stroke|evento\s+cerebro|"
        r"accidente\s+cerebro|enfermedad\s+cerebro|evento\s+vascular|embolia)|"
        r"(ictus|acv|cerebrovascular|cerebral)\s+(descartad|negativ)\w*|codigo\s+ictus|"
        r"hipofis\w*|pituitari\w*"
    ),
    subclasificar=subclasificar_acv,
)


# --- 5. Demencia (mapear_demencia_final) ---------------------------------------


def subclasificar_demencia(texto: str) -> str:
    """`subclasificar_cie10_demencia`."""
    texto = texto.lower()

    if re.search(r"deliri[ou]m?.*(sobrepuesto|superpuesto|sobre|\+)\s+demen[cs]ia", texto):
        return "F05.1"
    elif re.search(r"alz[ah]e?i?mer|alzh[eé]imer|alzai\w+|alzaimer", texto):
        return "F00"
    elif re.search(r"vascular|multiinfarto|neurovascular", texto):
        return "F01"
    elif re.search(
        r"parkinson|lewy|lewis|pick|huntington|frontotemporal|autoinmune|jakob|cuerpos\s+de",
        texto,
    ):
        return "F02"
    elif re.search(r"degeneracion\s+senil", texto):
        return "G31.1"
    else:
        return "F03"


DEMENCIA = Regla(
    categoria=CategoriaCharlson.DEMENCIA,
    inclusion=re.compile(
        r"\b(demen[cs]ia|alz[ah]e?i?mer|alzh[eé]imer|alzai\w+|alzaimer|antidemencial|tncm?|dcl)\b|"
        r"(trastorno|deterioro|deficit)\s+(neuro)?cognitivo|"
        r"degeneracion\s+senil"
    ),
    exclusion=re.compile(
        r"alzas?\s+(termic|febril)|sensacion\s+de\s+alza|"
        r"\b(re)?alza[ntn]|hipo-realzante|fase\s+(arterial|venosa)|nodular|"
        r"\bcalza\b|alzad[oa]|"
        r"perforacion|edeme|hipotonia"
    ),
    subclasificar=subclasificar_demencia,
)


# --- 6. Enfermedad pulmonar crónica (mapear_pulmonar_final) --------------------


def subclasificar_pulmonar(texto: str) -> str:
    """`subclasificar_cie10_pulmonar`. Por defecto, EPOC (J44)."""
    texto = texto.lower()

    if re.search(r"cor\s+pulmonale|hipertension\s+pulmonar\s+severa|cardiopulmonar", texto):
        return "I27.9"
    elif re.search(r"bronquiectasia", texto):
        return "J47"
    elif re.search(r"fibrosis\s+pulmonar|intersticial|\bepid\b", texto):
        return "J84"
    elif re.search(r"\bepoc\b|obstructiva\s+cronica|oxigeno\s*requir|oxigeno\s*dependiente", texto):
        return "J44"
    elif re.search(r"asma|asmatic[oa]|debut\s+asmatico", texto):
        return "J45"
    elif re.search(r"enfisema", texto):
        return "J43"
    elif re.search(r"neumoconiosis|silicosis|antracosis|asbestosis|exposicional", texto):
        return "J64"
    elif re.search(r"bronquitis\s+cronic[oa]", texto):
        return "J42"
    else:
        return "J44"


PULMONAR = Regla(
    categoria=CategoriaCharlson.PULMONAR_CRONICA,
    inclusion=re.compile(
        r"\b(epoc|asma|asmatic[oa]|enfisema|bronquiectasias?|neumoconiosis|silicosis|antracosis|"
        r"asbestosis|cor\s+pulmonale|hipertension\s+pulmonar\s+severa|epid)\b|"
        # #63: fibrosis y enfermedad intersticial (J84 en Quan)
        r"fibrosis\s+pulmonar|(enfermedad|patologia)\s+pulmonar\s+intersticial|"
        r"neumopatia\s+intersticial|"
        r"bronquitis\s+cronic[oa]|enfermedad\s+pulmonar\s+obstructiva|obstruccion\s+cronica|"
        r"(enfermedad|compromiso|patologia)\s+cardiopulmonar"
    ),
    exclusion=re.compile(
        r"subcutane[oa]|tejidos\s+blandos|pared\s+(toracica|abdominal)|cuello|orbita|nasal|"
        r"extracraneal|"
        r"psoas|poplitea|perirenal|hipocondrio|hipogastrio|facial|neurologico|subgaleal|flanco|"
        r"auscultacion|asintom[aá]tic[oa]|sistema|ruidos|herida|esternotomia|signos|hematoma|"
        r"^cardiopulmonar$|"
        r"bronquitis\s+aguda|bronquiolitis|neumonia|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|no\s+obesidad|no\s+diabetes|"
        r"epoc:\s+no)\s+(?:\w+\s+){0,3}(epoc|asma|enfisema|bronquitis)|"
        r"\b(no|sin|descart\w*)\s+(?:\w+\s+){0,2}(fibrosis|neumopatia|enfermedad\s+pulmonar)|"
        r"(fibrosis\s+pulmonar|intersticial)\s+descartad"
    ),
    subclasificar=subclasificar_pulmonar,
)


# --- 7. Enfermedad reumatológica (mapear_reumatologica_final) ------------------


def subclasificar_reumatologica(texto: str) -> str:
    """`subclasificar_cie10_reumatologica`. Por defecto, artritis reumatoide (M06)."""
    texto = texto.lower()

    if re.search(r"lupus|l\.e\.s\.|nefritis\s+l[uú]pica", texto):
        return "M32"
    elif re.search(r"artritis\s+reumatoi.*seropositiva", texto):
        return "M05"
    elif re.search(r"artritis\s+reumatoi", texto):
        return "M06"
    elif re.search(r"esclerodermia|esclerosis\s+sist[eé]mica|\bcrest\b|esclerodactilia", texto):
        return "M34"
    elif re.search(r"dermato.*miositis|polimiositis|antisintetasa", texto):
        return "M33"
    elif re.search(r"c[eé]lulas\s+gigantes|arteritis\s+de\s+la\s+temporal", texto):
        return "M31.5"
    elif re.search(r"polimialgia\s+reum[aá]tica", texto):
        return "M35.3"
    elif re.search(
        r"superpuesta|mixta\s+del\s+tejido\s+conectivo|enfermedad\s+del\s+tejido\s+conectivo",
        texto,
    ):
        return "M35.1"
    else:
        # Inalcanzable desde `mapear`: cada término de la inclusión cae en una rama
        # anterior. Se deja como respaldo; ningún texto clínico razonable llega aquí
        # sin ampliar antes la inclusión.
        return "M06"


REUMATOLOGICA = Regla(
    categoria=CategoriaCharlson.REUMATOLOGICA,
    inclusion=re.compile(
        r"\bl\.e\.s\.|"
        r"\b(artritis\s+reumatoi[de][ea]?|lupus|nefritis\s+l[uú]pica|esclerodermia|"
        r"esclerosis\s+sist[eé]mica|\bcrest\b|esclerodactilia|"
        r"polimiositis|dermatomiositis|dermatopolimiositis|antisintetasa|"
        r"polimialgia\s+reum[aá]tica|"
        r"arteritis\s+de\s+(c[eé]lulas\s+gigantes|la\s+temporal)|"
        r"enfermedad\s+(mixta\s+)?del\s+tejido\s+conectivo)\b"
    ),
    exclusion=re.compile(
        r"artritis\s+(gotosa|s[eé]ptica|reactiva|psori[aá]sica|degenerativa|viral)|artrosis|"
        r"osteoartritis|"
        r"esclerosis\s+(m[uú]ltiple|lateral|amiotr[oó]fica|a[oó]rtica|valvular|mesial)|"
        r"aterosclerosis|arteriosclerosis|"
        r"fiebre\s+reum[aá]tica|"
        r"madre|padre|hno|familiar|materna\b|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|negativo\s+para)\s+(?:\w+\s+){0,3}"
        r"(artritis|lupus|esclerodermia|polimiositis|polimialgia|enfermedad)"
    ),
    subclasificar=subclasificar_reumatologica,
)


# --- 8. Úlcera péptica (mapear_ulcera_final) -----------------------------------


def subclasificar_ulcera(texto: str) -> str:
    """`subclasificar_cie10_ulcera`. Por defecto, úlcera péptica no especificada (K27)."""
    texto = texto.lower()

    if re.search(r"yeyun[oa]|anastomosis|baston", texto):
        return "K28"
    elif re.search(r"duodenal|bulbo|bulboduodenal", texto):
        return "K26"
    elif re.search(
        r"g[aá]stric[oa]|antral|prepiloric[oa]|subcardial|cameron|est[oó]mago|curvatura|"
        r"antropil[oó]rica|cardial",
        texto,
    ):
        return "K25"
    else:
        return "K27"


ULCERA = Regla(
    categoria=CategoriaCharlson.ULCERA_PEPTICA,
    inclusion=re.compile(
        r"\bulcer[asóo]\w*\b.*\b(g[aá]stric[oa]|duodenal|p[eé]ptica|yeyunal|antral|"
        r"prepiloric[oa]|subcardial|bulbo|bulboduodenal|cameron|forrest|baston)\b|"
        r"\b(g[aá]stric[oa]|duodenal|p[eé]ptica|antral|prepiloric[oa]|subcardial|yeyunal|"
        r"bulboduodenal)\b.*\bulcer[asóo]\w*\b|"
        r"\bhvd[as]\b.*\bulceras?\b|"
        r"ulceraci[oó]n\s+g[aá]strica|perforaci[oó]n\s+de\s+ulcera"
    ),
    exclusion=re.compile(
        r"adenocarcinoma|carcinoma|neoplasia|malign|tumor|gist\b|"
        r"presi[oó]n|sacra|gl[uú]te[oa]|isqui[aá]tica|trocant[eé]rica|calc[aá]nea|talon|"
        r"maleol[oa]|"
        r"varicos[oa]|venos[oa]|diab[eé]tico|pierna|miembro|pie\b|"
        r"corneal|cornean[oa]|ojo|queratitis|geogr[aá]fica|"
        r"oral|boca|labio|lengua|paladar|am[ií]gdala|"
        r"colitis\s+ulcerativa|proctocolitis|rectocolitis|"
        r"placa|ateroma|arteria|coronaria|car[oó]tida|ulcerado\s+del\s+90%|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia)\s+(?:\w+\s+){0,3}ulcera"
    ),
    subclasificar=subclasificar_ulcera,
)


# --- 9. Enfermedad hepática leve (mapear_hepatica_final) -----------------------


def subclasificar_hepatica(texto: str) -> str:
    """`subclasificar_cie10_hepatica`. Por defecto, enfermedad hepática no especificada."""
    texto = texto.lower()

    if re.search(r"trasplante\s+hep[aá]tico", texto):
        return "Z94.4"

    if re.search(
        r"alcoh[oó]lic[oa]|etilismo|et[ií]lic[oa]|consumo\s+de\s+alcohol", texto
    ) and not re.search(r"no\s+alcoh[oó]lic[oa]", texto):
        if re.search(r"cirrosis", texto):
            return "K70.3"
        if re.search(r"hepatitis|esteatohepatitis", texto):
            return "K70.1"
        if re.search(r"graso|esteatosis", texto):
            return "K70.0"
        return "K70.9"

    if re.search(r"cirrosis|fibrosis|child|pugh|meld|hipertension\s+portal", texto):
        return "K74"

    if re.search(r"hepatitis\s+(b|c\b|viral\s+cr[oó]nica)", texto):
        return "B18"

    if re.search(r"autoinmune|hepatitis\s+cr[oó]nica", texto):
        return "K73"

    if re.search(r"graso|esteatosis|nash|nafld|mafld|esteatohepatitis", texto):
        return "K76.0"

    # Las dos ramas dan K76.9; se conservan como en la tesis
    if re.search(r"hepatopat[ií]a|da[ñn]o\s+hep[aá]tico", texto):
        return "K76.9"
    else:
        return "K76.9"


HEPATICA_LEVE = Regla(
    categoria=CategoriaCharlson.HEPATICA_LEVE,
    inclusion=re.compile(
        r"\b(cirrosis|hepatopat[ií]a\s+cr[oó]nica|dhc|esteatosis\s+hep[aá]tica|h[ií]gado\s+graso|"
        r"esteatohepatitis)\b|"
        r"\b(nash|nafld|mafld)\b|"
        r"hepatitis\s+(b|c\b|cr[oó]nica|autoinmune|alcoh[oó]lica)|"
        r"fibrosis\s+hep[aá]tica|trasplante\s+hep[aá]tico|"
        r"da[ñn]o\s+hep[aá]tico\s+cr[oó]nico"
    ),
    exclusion=re.compile(
        r"negativ[oa]\s+(para\s+)?hepatitis|"
        r"hepatitis(?:\s+[a-z])?\s+(es\s+)?(negativ[oa]|no\s+reactiv[oa])|"
        r"no\s+reactiv[oa]\s+(para\s+)?hepatitis|"
        r"anticuerpos?\s+(s\b|superficie|central)|"
        r"hepatitis\s+a\b|"
        r"hepatocarcinoma|c[aá]ncer|neoplasia|tumor|malign|lesi[oó]n\s+focal|"
        + METASTASIS_AFIRMADA
        + r"|"
        r"n[oó]dulo\s+hep[aá]tico|"
        r"esteatosis\s+pancre[aá]tica|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|negativo\s+para)\s+(?:\w+\s+){0,3}"
        r"(cirrosis|hepatitis|hepatopat[ií]a|esteatosis|h[ií]gado|nash|nafld)"
    ),
    subclasificar=subclasificar_hepatica,
)


# --- Orden de la tesis --------------------------------------------------------

REGLAS_PARTE1: tuple[Regla, ...] = (
    IAM,
    ICC,
    EVP,
    ACV,
    DEMENCIA,
    PULMONAR,
    REUMATOLOGICA,
    ULCERA,
    HEPATICA_LEVE,
)


REGLAS: tuple[Regla, ...] = REGLAS_PARTE1 + REGLAS_PARTE2


def mapear(texto: str) -> tuple[CategoriaCharlson, str] | None:
    """Categoría de Charlson y código CIE-10 del texto, o `None` si ninguna regla aplica.

    Recorre las 17 reglas sin cortar: la primera que aplica gana, salvo las tres reglas
    que sobrescriben (`Regla.sobrescribe`), igual que la tesis. El código CIE-10 lo da
    la subclasificación de la regla que queda asignada.
    """
    texto_normalizado = normalizar(texto)
    asignada: Regla | None = None
    for regla in REGLAS:
        if (asignada is None or regla.sobrescribe) and regla.aplica(texto_normalizado):
            asignada = regla
    if asignada is None:
        return None
    return asignada.categoria, asignada.subclasificar(texto_normalizado)


def reconoce(texto: str, categoria: CategoriaCharlson | None = None) -> bool:
    """Si la regla de `categoria` (o, sin `categoria`, alguna regla) conoce el texto: lo
    incluye o lo excluye a propósito.

    Distingue un texto que la regla no conoce ("eppoc") de uno que excluye ("diabetes
    gestacional", "carcinoma basocelular", "prediabetes"): `mapear` da `None` en los dos
    casos, pero solo el primero puede buscarse en sus menciones (#113). Se pregunta por la
    regla de la categoría que daría la mención: la exclusión de otra regla ("sin
    insuficiencia" en "eppoc sin insuficiencia respiratoria") no cuenta.
    """
    texto_normalizado = normalizar(texto)
    return any(
        regla.inclusion.search(texto_normalizado) or regla.exclusion.search(texto_normalizado)
        for regla in REGLAS
        if categoria is None or regla.categoria is categoria
    )
