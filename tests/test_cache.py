"""Tests de la caché en disco del extractor (`evaluacion.cache`).

Sin red ni modelo: envuelven un `ExtractorFalso` y una carpeta temporal.
"""

import json

from evaluacion.cache import ExtractorConCache, clave_de, version_de
from ner_clinic.extraccion import NOMBRE_MODELO, STRIDE_TOKENS, ExtractorFalso

FRAGMENTO = "Paciente con EPOC."
ENTIDAD = {"entity_group": "ENFERMEDAD", "score": 0.99, "word": "EPOC", "start": 13, "end": 17}


class Float32:
    """Imita un escalar de numpy: no se serializa a JSON pero tiene `item()`."""

    def __init__(self, valor: float):
        self.valor = valor

    def item(self) -> float:
        return self.valor


def test_segunda_llamada_no_usa_el_extractor(tmp_path):
    falso = ExtractorFalso({FRAGMENTO: [ENTIDAD]})
    con_cache = ExtractorConCache(falso, carpeta=tmp_path)
    assert con_cache.extraer([FRAGMENTO]) == [[ENTIDAD]]
    assert con_cache.extraer([FRAGMENTO]) == [[ENTIDAD]]
    assert falso.llamadas == [[FRAGMENTO]]
    assert (con_cache.aciertos, con_cache.fallos, con_cache.llamadas) == (1, 1, 1)


def test_la_cache_sobrevive_entre_instancias(tmp_path):
    ExtractorConCache(ExtractorFalso({FRAGMENTO: [ENTIDAD]}), carpeta=tmp_path).extraer([FRAGMENTO])
    falso = ExtractorFalso({})
    assert ExtractorConCache(falso, carpeta=tmp_path).extraer([FRAGMENTO]) == [[ENTIDAD]]
    assert falso.llamadas == []


def test_solo_infiere_los_que_faltan_sin_repetidos(tmp_path):
    otro = "Niega asma."
    falso = ExtractorFalso({FRAGMENTO: [ENTIDAD]})
    con_cache = ExtractorConCache(falso, carpeta=tmp_path)
    con_cache.extraer([FRAGMENTO])
    salidas = con_cache.extraer([otro, FRAGMENTO, otro])
    assert salidas == [[], [ENTIDAD], []]
    assert falso.llamadas == [[FRAGMENTO], [otro]]


def test_lista_vacia_no_llama_al_extractor(tmp_path):
    falso = ExtractorFalso({})
    assert ExtractorConCache(falso, carpeta=tmp_path).extraer([]) == []
    assert falso.llamadas == []


def test_un_archivo_por_fragmento_sin_el_texto(tmp_path):
    con_cache = ExtractorConCache(ExtractorFalso({FRAGMENTO: [ENTIDAD]}), carpeta=tmp_path)
    con_cache.extraer([FRAGMENTO, "Otro fragmento."])
    archivos = sorted(tmp_path.glob("*.json"))
    assert len(archivos) == 2
    assert con_cache.ruta(FRAGMENTO) in archivos
    for archivo in archivos:
        assert "Paciente" not in archivo.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_la_clave_depende_de_modelo_version_stride_y_texto():
    base = clave_de(FRAGMENTO, NOMBRE_MODELO, "5.0.0", STRIDE_TOKENS)
    assert base == clave_de(FRAGMENTO, NOMBRE_MODELO, "5.0.0", STRIDE_TOKENS)
    assert len(base) == 64
    distintas = {
        clave_de(FRAGMENTO + " ", NOMBRE_MODELO, "5.0.0", STRIDE_TOKENS),
        clave_de(FRAGMENTO, "otro/modelo", "5.0.0", STRIDE_TOKENS),
        clave_de(FRAGMENTO, NOMBRE_MODELO, "5.1.0", STRIDE_TOKENS),
        clave_de(FRAGMENTO, NOMBRE_MODELO, "5.0.0", 64),
    }
    assert base not in distintas
    assert len(distintas) == 4


def test_otra_version_de_transformers_no_usa_la_entrada_vieja(tmp_path):
    ExtractorConCache(
        ExtractorFalso({FRAGMENTO: [ENTIDAD]}), carpeta=tmp_path, version_transformers="5.0.0"
    ).extraer([FRAGMENTO])
    falso = ExtractorFalso({})
    con_cache = ExtractorConCache(falso, carpeta=tmp_path, version_transformers="9.9.9")
    assert con_cache.extraer([FRAGMENTO]) == [[]]
    assert falso.llamadas == [[FRAGMENTO]]


def test_version_por_defecto_es_la_instalada(tmp_path):
    con_cache = ExtractorConCache(ExtractorFalso({}), carpeta=tmp_path)
    assert con_cache.version_transformers == version_de("transformers")
    assert version_de("paquete-que-no-existe") is None


def test_convierte_escalares_de_numpy(tmp_path):
    entidad = dict(ENTIDAD, score=Float32(0.5))
    con_cache = ExtractorConCache(ExtractorFalso({FRAGMENTO: [entidad]}), carpeta=tmp_path)
    assert con_cache.extraer([FRAGMENTO]) == [[dict(ENTIDAD, score=0.5)]]
    guardado = json.loads(con_cache.ruta(FRAGMENTO).read_text(encoding="utf-8"))
    assert guardado == {"entidades": [dict(ENTIDAD, score=0.5)]}


def test_archivo_danado_se_vuelve_a_inferir(tmp_path):
    con_cache = ExtractorConCache(ExtractorFalso({FRAGMENTO: [ENTIDAD]}), carpeta=tmp_path)
    con_cache.ruta(FRAGMENTO).parent.mkdir(parents=True, exist_ok=True)
    con_cache.ruta(FRAGMENTO).write_text("{no es json", encoding="utf-8")
    assert con_cache.extraer([FRAGMENTO]) == [[ENTIDAD]]
    assert con_cache.fallos == 1
    assert json.loads(con_cache.ruta(FRAGMENTO).read_text(encoding="utf-8"))["entidades"]


def test_las_salidas_son_copias(tmp_path):
    con_cache = ExtractorConCache(ExtractorFalso({FRAGMENTO: [ENTIDAD]}), carpeta=tmp_path)
    con_cache.extraer([FRAGMENTO])[0][0]["start"] = 999
    assert con_cache.extraer([FRAGMENTO]) == [[ENTIDAD]]
