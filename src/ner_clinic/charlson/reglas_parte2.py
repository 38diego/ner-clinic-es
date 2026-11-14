"""Reglas de Charlson y CIE-10, parte 2: las 8 últimas categorías del índice.

Portadas de las funciones `mapear_*_final` y `subclasificar_cie10_*` de la tesis
(`04_Perfil_Morbimortalidad/MapeoCIE10 OK.ipynb`). Las regex de inclusión, exclusión y
subclasificación son las de la tesis; algunas conservan clases como `[eé]` que, sobre el
texto normalizado (sin tildes), solo usan la letra sin tilde.

Errores de la tesis corregidos en el issue #24 (un test por cada uno en
`tests/test_charlson_correcciones.py`): `daño` en la exclusión de diabetes complicada
(`dano` tras normalizar), la errata `diab[eé]tical?` (ahora `diab[eé]tic[oa]s?`, con
plurales), coma y cetoacidosis de la diabetes sin tipo en E14.0/E14.1 (no E11), `\b` en
las siglas `ms`/`mi` de la monoplejía, `\b` antes de las negaciones (`no`, `ni`, `sin`)
de todas las exclusiones, y "osea(s)" en la subclasificación de metástasis y en la
localización ósea del cáncer.

Tres reglas tienen `sobrescribe=True` (diabetes complicada, hepática severa y tumor
metastásico): en la tesis no respetan la máscara `Mapeado_Charlson` y reemplazan lo que
haya asignado cualquier regla anterior, también las de la parte 1.

`reglas.py` une estas reglas con las de la parte 1 en `REGLAS` y las aplica en `mapear`.
"""

import re

from ner_clinic.charlson.base import METASTASIS_AFIRMADA, Regla
from ner_clinic.tipos import CategoriaCharlson

# ==========================================
# Diabetes mellitus sin complicación crónica
# ==========================================


def subclasificar_cie10_diabetes(texto: str) -> str:
    es_cetoacidosis = re.search(r"cetoacidosis|cad\b|cetosis", texto)
    es_coma = re.search(r"\bcoma\b", texto)
    es_descompensada = re.search(
        r"descompensad[oa]|hipergl[iu]cemia|hiperosmolar|no\s+compensada|inestable|"
        r"mal\s+control|crisis",
        texto,
    )

    # E10: tipo 1
    if re.search(r"tipo\s*1\b|tipo\s*i\b|dm1\b|dmt1\b|juvenil|insulinodependiente", texto):
        if es_coma:
            return "E10.0"
        if es_cetoacidosis:
            return "E10.1"
        if es_descompensada:
            return "E10.6"
        return "E10.9"

    # E11: tipo 2 (absorbe NIR, IR e ISR por probabilidad clínica)
    if re.search(
        r"tipo\s*2\b|tipo\s*ii\b|dm2\b|dmt2\b|adulto|nir\b|ir\b|isr\b|"
        r"insulino\s*r?r?equiriente|no\s+insulino",
        texto,
    ):
        if es_coma:
            return "E11.0"
        if es_cetoacidosis:
            return "E11.1"
        if es_descompensada:
            return "E11.6"
        return "E11.9"

    # E14: no especificada. Corrección de #24: la tesis mandaba coma y cetoacidosis a
    # E11.0/E11.1 sin que el texto dijera el tipo; la CIE-10 tiene E14.0 y E14.1.
    if es_coma:
        return "E14.0"
    if es_cetoacidosis:
        return "E14.1"
    if es_descompensada:
        return "E14.6"
    return "E14.9"


DIABETES = Regla(
    categoria=CategoriaCharlson.DIABETES,
    inclusion=re.compile(
        r"\b(dm\b|dm[12]\b|dmt[12]\b|dm\s+tipo\s+[12i]+\b|diabetes|diab[eé]tic[oa]|cetoacidosis|cetosis)\b|"
        r"\b(?:insulino|insul|inslino|isulino|insulinop)[-\s]?r?[eé]q[ui]+r[ie]+nte\b|"
        r"\b(ir|nir|isr)\b(?=\s+hace|\s+compensada|\s+mal|\s*$)|\bdmt?2\b|"
        r"\binsulinodependiente\b"
    ),
    exclusion=re.compile(
        r"gestacional|pregestacional|pre[- ]?diabetes|pre[- ]?dm\b|ins[ií]pida|"
        r"nefropat[ií]a|retinopat[ií]a|neuropat[ií]a|polineuropat[ií]a|gastropat[ií]a|oftalmopat[ií]a|"
        r"pie\s+diab[eé]tico|microvascular|macrovascular|enfermedad\s+renal\s+diab[eé]tica|erc\s+diab[eé]tica|enf\s+renal\s+diab[eé]tica|"
        r"madre|padre|hno|familiar|materna\b|"
        r"\b(diabetes|dm\b|dm[12])\s*:\s*no\b|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|niega|negativo\s+para)\s+(?:\w+\s+){0,3}(diabetes|dm\b|dm[12]|diab[eé]tic[oa])"
    ),
    subclasificar=subclasificar_cie10_diabetes,
)


# ==========================================
# Diabetes con daño de órgano blanco
# ==========================================


def subclasificar_cie10_diabetes_cronica(texto: str) -> str:
    prefijo = "E11"
    if re.search(r"tipo\s*1\b|tipo\s*i\b|dm1\b|dmt1\b|juvenil|insulinodependiente", texto):
        prefijo = "E10"

    # Varios sistemas afectados: .7 (complicaciones múltiples)
    sistemas = 0
    if re.search(r"nefropat|renal|erc|nefrolitiasis", texto):
        sistemas += 1
    if re.search(r"retinopat|oftalmo|ojo|ceguera|vitrea", texto):
        sistemas += 1
    if re.search(r"neuropat|neuro|gastroparesia|amiotrofia", texto):
        sistemas += 1
    if re.search(r"pie\s+diab[eé]|vasculopat|angiopat|circulatoria|mionecrosis|isquemia", texto):
        sistemas += 1

    if sistemas > 1 or re.search(r"complicaciones\s+(m[uú]ltiples|micro|macro)", texto):
        return f"{prefijo}.7"

    # Sistema dominante
    if re.search(r"nefropat|enfermedad\s+renal|erc", texto):
        return f"{prefijo}.2"
    if re.search(r"retinopat|oftalmo|ceguera", texto):
        return f"{prefijo}.3"
    if re.search(r"neuropat|neuro|gastroparesia", texto):
        return f"{prefijo}.4"
    if re.search(r"pie\s+diab[eé]|vasculopat|circulatoria|mionecrosis", texto):
        return f"{prefijo}.5"
    return f"{prefijo}.8"


DIABETES_COMPLICADA = Regla(
    categoria=CategoriaCharlson.DIABETES_COMPLICADA,
    inclusion=re.compile(
        r"(nefropat[ií]as?|retinopat[ií]as?|neuropat[ií]as?|gastroparesias?|mionecrosis|nefroesclerosis|angiopat[ií]as?)\s+diab[eé]tic[oa]s?|"
        r"\bpie\s+diab[eé]tico\b|"
        r"(dm\b|dm[12]\b|dmt[12]\b|diabetes|diab[eé]tic[oa]).*(microvascular|macrovascular|renal|oftalmo|neuro|compromiso|complicaci[oó]n|organo\s+blanco)|"
        r"(erc|enfermedad\s+renal|insuficiencia\s+renal).*(diab[eé]tic[oa]|dm\b|dm[12])"
    ),
    exclusion=re.compile(
        r"pre[- ]?diabetes|ins[ií]pida|gestacional|"
        r"madre|padre|hno|familiar|materna\b|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|niega)\s+(?:\w+\s+){0,3}(nefropat|retinopat|neuropat|complicacion|pie|dano)"
    ),
    subclasificar=subclasificar_cie10_diabetes_cronica,
    sobrescribe=True,
)


# ==========================================
# Hemiplejía / paraplejía
# ==========================================


def subclasificar_cie10_hemiplejia(texto: str) -> str:
    if re.search(
        r"paraplej[ií]a|paraparesia|tetraplej[ií]a|tetraparesia|cuadriplej[ií]a|cuadriparesia|biparesia",
        texto,
    ):
        return "G82"
    if re.search(r"hemiplej[ií]a|hemiparesia", texto):
        return "G81"
    if re.search(r"mielitis\s+transversa", texto):
        return "G04.1"
    if re.search(r"par[aá]lisis\s+cerebral", texto):
        if re.search(r"hemi", texto):
            return "G80.2"
        return "G80.1"
    if re.search(r"cola\s+de\s+caballo|cauda\s+equina", texto):
        return "G83.4"
    if re.search(r"monoplej[ií]a|monoparesia", texto):
        # `\b` en las siglas (corrección de #24): "mi" coincidía dentro de "mielopatia"
        if re.search(r"superior|\bms[di]?\b|brazo", texto):
            return "G83.2"
        if re.search(r"inferior|\bmi[di]?\b|pierna", texto):
            return "G83.1"
        return "G83.3"
    if re.search(r"diplej[ií]a|diparesia", texto):
        return "G83.0"
    # Inalcanzable desde `mapear`: cada término de la inclusión tiene su rama. Se deja
    # como respaldo; G83.9 (síndrome paralítico no especificado) no tiene un término
    # propio en la inclusión.
    return "G83.9"


HEMIPLEJIA = Regla(
    categoria=CategoriaCharlson.HEMIPLEJIA,
    inclusion=re.compile(
        r"\b(hemiplej[ií]a|hemiparesia|paraplej[ií]a|paraparesia|tetraplej[ií]a|tetraparesia|"
        r"cuadriplej[ií]a|cuadriparesia|monoplej[ií]a|monoparesia|diplej[ií]a|diparesia|biparesia|"
        r"mielitis\s+transversa|cola\s+de\s+caballo|cauda\s+equina|par[aá]lisis\s+cerebral)\b"
    ),
    exclusion=re.compile(
        r"(hemiplej[ií]a|hemiparesia|diplej[ií]a|diparesia)\s+(derecha\s+|izquierda\s+|de\s+|en\s+(la\s+)?)?(cara|hemicara|labios?|facial)\b|"
        r"\b(labios?|cara|facial)\s+(con\s+)?(hemiplej[ií]a|hemiparesia|diplej[ií]a|diparesia)\b|"
        r"\b(nervio\s+craneal|(iii|iv|vi|vii)\s+nervio|(iii|iv|vi|vii)\s+pc|cuerdas\s+vocales|esf[ií]nter|hipocalemia)\b|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|niega|tampoco)\s+(?:\w+\s+){0,8}(hemiplej[ií]a|hemiparesia|paraplej[ií]a|paraparesia|paresia|par[aá]lisis|cauda\s+equina|cola\s+de\s+caballo)"
    ),
    subclasificar=subclasificar_cie10_hemiplejia,
)


# ==========================================
# Enfermedad renal crónica
# ==========================================


def subclasificar_cie10_renal(texto: str) -> str:
    # Z94.0: trasplante renal (prioridad máxima por la inmunosupresión)
    if re.search(r"trasplante\s+renal", texto):
        return "Z94.0"
    # Z99.2: dependencia de diálisis / TRR / catéter de diálisis
    if re.search(
        r"hemodi[aá]lisis|di[aá]lisis\s+peritoneal|\btrr\b|\btsr\b|cat[eé]ter\s+de\s+(di[aá]lisis|hemodi[aá]lisis)",
        texto,
    ):
        return "Z99.2"
    # N18: el estadio va antes que la causa subyacente (HTA, IgA)
    if re.search(r"\berc\b|\birc\b|cr[oó]nica|estadio\s+[2345v]|terminal", texto):
        return "N18"
    if re.search(r"hipertensiva|nefroesclerosis", texto):
        return "I12.0"
    if re.search(
        r"nefr[ií]tico\s+cr[oó]nico|glomerulonefritis\s+cr[oó]nica|glomerulopat[ií]a|iga", texto
    ):
        return "N03.9"
    if re.search(r"osteodistrofia", texto):
        return "N25.0"
    # Inalcanzable desde `mapear`: cada término de la inclusión tiene su rama. Se deja
    # así a propósito: N19 es insuficiencia renal no especificada (sin cronicidad), que
    # Charlson no cuenta, así que no se amplía la inclusión para llegar aquí.
    return "N19"


RENAL_CRONICA = Regla(
    categoria=CategoriaCharlson.RENAL_CRONICA,
    inclusion=re.compile(
        r"\b(erc|irc)\b(?!\s+aguda)|"
        r"enfermedad\s+renal\s+cr[oó]nica|insuficiencia\s+renal\s+cr[oó]nica|falla\s+renal\s+terminal|"
        r"hemodi[aá]lisis|di[aá]lisis\s+peritoneal|\btrr\b|\btsr\b|"
        r"trasplante\s+renal|"
        r"nefropat[ií]a\s+hipertensiva|nefroesclerosis|"
        r"s[ií]ndrome\s+nefr[ií]tico\s+cr[oó]nico|glomerulonefritis\s+cr[oó]nica|nefropat[ií]a\s+por\s+iga|"
        r"osteodistrofia\s+renal|"
        r"enfermedad\s+renal\s+estadio\s+[2345v]|\berc\s+(estadio\s+|etapa\s+|g\s*)?[2345v]\b"
    ),
    exclusion=re.compile(
        r"\bira\b|\blra\b|\baki\b(?!\s+en\s+paciente\s+con\s+erc)(?!\s+vs\s+erc)|fracaso\s+renal\s+agudo|"
        r"(?<!erc\s)(?<!irc\s)(?<!enfermedad\srenal\scronica\s)lesi[oó]n\s+renal\s+aguda|"
        r"nefrolitiasis|urolitiasis|c[aá]lculos?|piedras?|c[oó]lico|"
        r"pielonefritis|infecci[oó]n\s+urinaria|"
        r"c[aá]ncer\s+renal|neoplasia\s+renal|tumor\s+renal|carcinoma\s+renal|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|niega)\s+(?:\w+\s+){0,4}(erc|irc|enfermedad\s+renal|insuficiencia\s+renal|di[aá]lisis)"
    ),
    subclasificar=subclasificar_cie10_renal,
)


# ==========================================
# Tumor maligno (sin metástasis)
# ==========================================


def subclasificar_cie10_cancer(texto: str) -> str:
    # Hematológicos y linfoides
    if re.search(r"leucemia", texto):
        if re.search(r"linfoide|linfoblas|linfoc[ií]tica", texto):
            return "C91"
        if re.search(r"mieloide|mieloblas", texto):
            return "C92"
        return "C95"
    if re.search(r"linfoma", texto):
        if re.search(r"hodgkin|hodgking", texto) and not re.search(r"no\s+hodgkin", texto):
            return "C81"
        return "C85"
    if re.search(r"mieloma", texto):
        return "C90"
    if re.search(r"kaposi|capozzi", texto):
        return "C46"
    if re.search(r"sarcoma", texto):
        return "C49"

    # Tumores sólidos, en el orden de la tesis (la primera localización que coincide gana)
    localizaciones = [
        (r"mama|seno|mamario|ductal|lobulillar", "C50"),
        (r"c[eé]rvix|cuello\s+uterino|cervicouterino|endocervix", "C53"),
        (r"endometrio|[uú]tero|endometrial", "C54"),
        (r"ovario", "C56"),
        (r"pr[oó]stata|prost[aá]tic[oa]", "C61"),
        (r"test[ií]culo", "C62"),
        (
            r"colon|recto|colorrectal|ciego|sigmoide|v[aá]lvula\s+ileocecal|[ií]leon|anorectal|canal\s+anal",
            "C18",
        ),
        (r"est[oó]mago|g[aá]strico|antropil[oó]ric[oa]", "C16"),
        (r"es[oó]fago", "C15"),
        (r"pulm[oó]n|pulmonar|bronquio|broncog[eé]nico", "C34"),
        (r"laringe|garganta|gl[oó]tic[oa]|cuerdas\s+vocales", "C32"),
        (r"faringe|am[ií]gdala|cavum|sinonasal|maxilar|tr[ií]gono|boca|oral|lengua|labio", "C14"),
        (r"vejiga|urotelial|uroteria|uraco|vesical|ureteral", "C67"),
        (r"ri[ñn][oó]n|renal|c[eé]lulas\s+claras", "C64"),
        (r"tiroides|tiroideo|troides|tirodides", "C73"),
        (r"p[aá]ncreas|pancre[aá]tico", "C25"),
        (r"h[ií]gado|hep[aá]tico|hepatocarcinoma|colangiocarcinoma|biliar", "C22"),
        (r"cerebro|enc[eé]falo|glioblastoma|astrocitoma|glial", "C71"),
        (r"hueso|[oó]se[oa]s?|il[ií]aco", "C41"),
        (r"melanoma", "C43"),
    ]
    for patron, codigo in localizaciones:
        if re.search(patron, texto):
            return codigo
    return "C80"


TUMOR_MALIGNO = Regla(
    categoria=CategoriaCharlson.TUMOR_MALIGNO,
    inclusion=re.compile(
        r"\b(c[aá]ncer|carcinoma|adenocarcinoma|sarcoma|melanoma|linfoma|leucemia|mieloma|hepatocarcinoma|colangiocarcinoma|glioblastoma|neoplasia\s+maligna|tumor\s+maligno)\b|"
        r"\bca\s+de\b|\bca\s+(g[aá]strico|pulmonar|mamario|prost[aá]tico|renal|col[oó]nico|c[eé]rvico|endometrio|ovario|tiroides)\b"
    ),
    exclusion=re.compile(
        METASTASIS_AFIRMADA
        + r"|secundarismo|estadio\s+iv\b|\bm1[a-c]?\b|afectando\s+tambi[eé]n|krukenberg|"
        r"compromiso\s+(pulmonar|[oó]seo|hep[aá]tico|cerebral|secundario)|"
        r"basocelular|escamocelular\s+(cut[aá]neo|en\s+piel|en\s+nariz|en\s+oreja|en\s+rostro|en\s+cara)|"
        r"\b(c[aá]ncer|ca|carcinoma|tumor)\s+de\s+piel\b|seb[aá]ceo|merkel|"
        r"\b(hermana?|madre|padre|hijo|abuelo)\b|"
        r"\?|sospecha\s+de|a\s+descartar|probable\s+(ca\b|c[aá]ncer|linfoma|neoplasia)|"
        r"\bvs\s+(ca\b|c[aá]ncer|carcinoma|linfoma|leucemia|mieloma|neoplasia)|"
        r"(ca\b|c[aá]ncer|carcinoma|linfoma|leucemia|mieloma|neoplasia)\s+vs\b|"
        r"benign[oa]|in\s+situ|insitus|intraepitelial|\bnic\s+[123]\b|displasia|"
        r"p[oó]lipo|adenoma(?!\s+carcinoma)|quiste|fibroma|mioma|lipoma|hemangioma|meningioma|"
        r"calcio|ca\s+antagonista|suplemento\s+de\s+ca|"
        r"\b(sin|descartar|no\s+se\s+evidencia|no\s+hay|ausencia|niega)\s+(?:\w+\s+){0,4}(c[aá]ncer|ca\b|carcinoma|tumor|linfoma|leucemia|neoplasia|adenocarcinoma)"
    ),
    subclasificar=subclasificar_cie10_cancer,
)


# ==========================================
# Enfermedad hepática severa
# ==========================================


def subclasificar_cie10_hepatica_severa(texto: str) -> str:
    # Várices
    if re.search(r"v[aá]rices\s+esof[aá]gicas", texto):
        if re.search(r"hemorragia|sangrad[oa]|sangrante|ruptura|rota|chorro|ligadura|evda", texto):
            return "I85.0"  # con hemorragia o tratada endoscópicamente
        return "I85.9"
    if re.search(r"v[aá]rices\s+g[aá]stricas|fundus", texto):
        return "I86.4"
    if re.search(r"v[aá]rices", texto) and re.search(r"h[ií]gado|cirrosis|hep[aá]tic[oa]", texto):
        return "I98.2"
    if re.search(r"hipertensi[oó]n\s+portal|portosist[eé]mic[oa]", texto):
        return "K76.6"
    if re.search(r"hepatorrenal", texto):
        return "K76.7"
    if re.search(r"(insuficiencia|falla)\s+hep[aá]tica", texto) and re.search(
        r"alcoh[oó]lic[oa]|etilismo|et[ií]lic[oa]", texto
    ):
        return "K70.4"
    if re.search(r"necrosis\s+hep[aá]tica|t[oó]xica", texto):
        return "K71.1"
    if re.search(r"veno-?oclusiva", texto):
        return "K76.5"
    if re.search(
        r"insuficiencia\s+hep[aá]tica\s+cr[oó]nica|en[fc]e[fc]alopat[ií]a|west\s*haven", texto
    ):
        return "K72.1"
    if re.search(r"insuficiencia\s+hep[aá]tica|falla\s+hep[aá]tica", texto):
        return "K72.9"
    # #63: cirrosis severa sin otro marcador
    if re.search(r"con\s+ascitis|ascitis\s+(por|secundaria)", texto):
        return "K76.6"  # aproximación: la ascitis de la cirrosis es por hipertensión portal
    if re.search(r"child|descompensa", texto):
        return "K72.1"  # insuficiencia hepática crónica
    return "K72.9"


HEPATICA_SEVERA = Regla(
    categoria=CategoriaCharlson.HEPATICA_SEVERA,
    inclusion=re.compile(
        r"\b(v[aá]rices\s+(esof[aá]gicas|g[aá]stricas|paraesof[aá]gicas)|hipertensi[oó]n\s+portal|portosist[eé]mic[oa])\b|"
        r"\b(s[ií]ndrome\s+hepatorrenal|en[fc]e[fc]alopat[ií]a\s+hep[aá]tica|west\s*haven|necrosis\s+hep[aá]tica)\b|"
        r"\b(insuficiencia\s+hep[aá]tica|falla\s+hep[aá]tica|falla\s+hep[aá]tica\s+(aguda|fulminante|hiperaguda))\b|"
        r"\benfermedad\s+veno-?oclusiva\s+hep[aá]tica\b|"
        # #63: Child-Pugh C, cirrosis descompensada o con ascitis
        r"\bchild(\s*-?\s*pugh)?\s*c\b|"
        # Cerca de "cirrosis" y sin "no"/"sin" en medio: "insuficiencia cardiaca
        # descompensada con antecedente de cirrosis" no es hepática severa.
        r"cirrosis\s+(?:(?!no\b|sin\b)\w+\s+){0,3}(descompensad[oa]|con\s+ascitis)|"
        r"descompensacion\s+(de\s+(la\s+)?)?cirrosis|ascitis\s+(por|secundaria\s+a)\s+cirrosis"
    ),
    exclusion=re.compile(
        r"v[aá]rices\s+(en\s+)?(miembros|piernas|mmii|inferiores|uterin[oa]s)|venas\s+varicosas|"
        r"(?<!portal\s)hipertensi[oó]n\s+(arterial|pulmonar|endocraneana|pulmonar|esencial|sist[eé]mica)|"
        r"riesgo\s+de\s+(falla\s+hep[aá]tica|insuficiencia|en[fc]e[fc]alopat[ií]a)|"
        r"(sospecha|probable|posible)\s+((de\s+)?v[aá]rices|(de\s+)?falla\s+hep[aá]tica|(de\s+)?en[fc]e[fc]alopat[ií]a|(de\s+)?s[ií]ndrome\s+hepatorrenal)|"
        r"\?|\bvs\b\s+(v[aá]rices|falla\s+hep[aá]tica|en[fc]e[fc]alopat[ií]a)|"
        r"(v[aá]rices|falla\s+hep[aá]tica|en[fc]e[fc]alopat[ií]a|hipertensi[oó]n\s+portal)\s+\bvs\b|"
        r"a\s+estudio|"
        r"\b(sin|descart[aóoe]r?|no\s+se\s+evidencia|no\s+es|no\s+hay|ausencia|niega|ni)\s+(?:\w+\s+){0,6}(v[aá]rices|hipertensi[oó]n\s+portal|insuficiencia|falla\s+hep[aá]tica|hepatorrenal|en[fc]e[fc]alopat[ií]a|west\s*haven)"
    ),
    subclasificar=subclasificar_cie10_hepatica_severa,
    sobrescribe=True,
)


# ==========================================
# Tumor metastásico
# ==========================================


def subclasificar_cie10_metastasis(texto: str) -> str:
    # C77: ganglios
    if re.search(r"ganglio|linf[aá]tico|adenopat[ií]a|adenomegalia|retroperitoneal", texto):
        return "C77"
    # C78: respiratorio / digestivo
    if re.search(
        r"hep[aá]tic[oa]|h[ií]gado|pulm[oó]n|pulmonar|pleura|respiratorio|digestivo|gastrointestinal|intestino",
        texto,
    ):
        return "C78"
    # C79: otros (hueso, cerebro, peritoneo)
    if re.search(
        r"hueso|[oó]se[oa]s?|esqueleto|columna|femur|iliaco|sacro|cr[aá]neo|vertebral|cerebro|cerebral|enc[eé]falo|snc|peritone[oa]l|carcinomatosis|krukenberg|piel|suprarrenal|uretra|ovario",
        texto,
    ):
        return "C79"
    return "C80"


TUMOR_METASTASICO = Regla(
    categoria=CategoriaCharlson.TUMOR_METASTASICO,
    inclusion=re.compile(
        r"\b(met[aá]stasis|metast[aá]sic[oa]s?|secundarismo|carcinomatosis|krukenberg|omental\s+cake)\b|"
        r"\bm1[a-c]?\b(?!\s*-\s*m[2-6])(?!\s*izquierd[oa])(?!\s*derech[oa])|"
        r"estadio\s+iv\b.*(c[aá]ncer|ca\b|carcinoma|tumor|adenocarcinoma|melanoma|sarcoma|oncolog)|"
        r"(c[aá]ncer|ca\b|carcinoma|tumor|adenocarcinoma|melanoma|sarcoma|oncolog).*estadio\s+iv\b|"
        r"siembra\s+(peritoneal|pleural|subcapsular)|"
        r"compromiso\s+(secundario|metastasico)|"
        r"enfermedad\s+diseminada"
    ),
    exclusion=re.compile(
        r"\bacmd?\b|\bacmi?\b|trombo|isqu[eé]mico|aneurisma\s+acm|arteria\s+cerebral|desgarro|retina|\ba[1-4]\b|\bp[1-4]\b|"
        r"eventraci[oó]n|\bw[1-4]\b|"
        r"\berc\b|enfermedad\s+renal|nefroangioesclerosis|"
        r"secundarismo\s+(infeccioso|carencial)|"
        r"compromiso\s+secundario\s+a\s+(autoinmunidad|infecci[oó]n|trauma|medicamento)|"
        r"\bm0\b|\bmx\b|"
        r"\?|sospecha\s+de|a\s+descartar|considerar|(probable|posiblemente|posible)\s+(met[aá]stasis|secundarismo|carcinomatosis)|"
        r"(sugestiv[oa]s?\s+(de|para)|sospechos[oa]s?\s+(de|para)|al\s+parecer)\s+(met[aá]stasis|secundarismo)|"
        r"\b(vs|versus|o)\b\s*(ca\b|c[aá]ncer|carcinoma|linfoma|leucemia|mieloma|neoplasia|met[aá]stasis|secundarismo|carcinomatosis|secundari[oa]|primari[oa])|"
        r"(ca\b|c[aá]ncer|carcinoma|linfoma|leucemia|mieloma|neoplasia|met[aá]stasis|secundarismo|carcinomatosis|secundari[oa]|primari[oa])\s*\b(vs|versus|o)\b|"
        r"\b(sin|descart[aóoe]r?|no\s+se\s+evidencia|no\s+es|no\s+hay|ausencia|niega|negativ[oa]\s+para|ni|no\b)\s+(?:\w+\s+){0,5}(met[aá]stasis|metast[aá]sic[oa]|secundarismo|siembra|diseminaci[oó]n|carcinomatosis)|"
        r"libre\s+de\s+(met[aá]stasis|enfermedad|carcinomatosis)|"
        r"(met[aá]stasis|secundarismo)\s+(descartad|negativ)\w*"
    ),
    subclasificar=subclasificar_cie10_metastasis,
    sobrescribe=True,
)


# ==========================================
# VIH / SIDA
# ==========================================


def subclasificar_cie10_vih(texto: str) -> str:
    # B20: VIH con enfermedades oportunistas
    if re.search(
        r"tuberculosis|tbc|miliar|kaposi|capozzi|oportunista|toxoplasma|pneumocystis|pcp|jirovecii|cmv|criptococo",
        texto,
    ):
        return "B20"
    # B22: VIH con síndrome de desgaste o encefalopatía
    if re.search(r"caquexia|desgaste|emaciaci[oó]n|encefalopat[ií]a|wasting", texto):
        return "B22"
    # B24: VIH no especificado
    return "B24"


VIH_SIDA = Regla(
    categoria=CategoriaCharlson.VIH_SIDA,
    inclusion=re.compile(
        r"\b(vih|hiv|sida)\b|"
        r"inmunodeficiencia\s+(humana|adquirida)|"
        r"\bb24x?\b|"
        r"retrovirus"
    ),
    exclusion=re.compile(
        r"\bhiv\b\s+grado\s+[iv]+\b|"  # hemorragia intraventricular
        r"grado\s+[iv]+\b\s+(derecha|izquierda|bilateral)|"
        r"\b(vih|hiv|sida)\b\s+(negativ[oa]|no\s+reactiv[oa]|neg\b)|"
        r"(negativ[oa]|no\s+reactiv[oa]|neg\b)\s+(para\s+)?(vih|hiv|sida)|"
        r"\?|sospecha\s+de|a\s+descartar|posible\s+(vih|hiv|sida)|"
        r"\bvs\b\s*(vih|hiv|sida)|(vih|hiv|sida)\s*\bvs\b|"
        r"\b(sin|descart[aóoe]r?|no\s+se\s+evidencia|no\s+es|no\s+hay|ausencia|niega|ni)\s+(?:\w+\s+){0,5}(vih|hiv|sida)"
    ),
    subclasificar=subclasificar_cie10_vih,
)


# --- Orden de la tesis --------------------------------------------------------

REGLAS_PARTE2: tuple[Regla, ...] = (
    DIABETES,
    DIABETES_COMPLICADA,
    HEMIPLEJIA,
    RENAL_CRONICA,
    TUMOR_MALIGNO,
    HEPATICA_SEVERA,
    TUMOR_METASTASICO,
    VIH_SIDA,
)
