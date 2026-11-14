"""Tests del front estático (`src/ner_clinic_api/estatico/`), sin navegador ni modelo.

Comprueban que la API sirve el HTML, el CSS y el JS con la CSP, que el HTML no trae
scripts ni estilos inline (la CSP los bloquearía) y, con un grep simple, que el JS nunca
inserta HTML (riesgo de XSS con el texto de la nota o la respuesta de la API) ni guarda
el texto en el navegador.
"""

import re
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from ner_clinic.extraccion import ExtractorFalso
from ner_clinic_api import crear_app
from ner_clinic_api.app import ESTATICO
from ner_clinic_api.configuracion import Configuracion

JS = (ESTATICO / "app.js").read_text(encoding="utf-8")
HTML = (ESTATICO / "index.html").read_text(encoding="utf-8")
CSS = (ESTATICO / "estilos.css").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cliente():
    app = crear_app(
        extractor=ExtractorFalso({}),
        configuracion=Configuracion(sal_ip="sal", normalizador="reglas"),
    )
    with TestClient(app) as cliente:
        yield cliente


class Recolector(HTMLParser):
    """Junta los `<script>`, `<link>`, `<style>` y los atributos inline del HTML."""

    def __init__(self):
        super().__init__()
        self.scripts: list[dict] = []
        self.hojas: list[str] = []
        self.estilos_inline = 0
        self.atributos_inline: list[str] = []
        self.dentro_de_script = False
        self.codigo_inline = ""

    def handle_starttag(self, etiqueta, atributos):
        attrs = dict(atributos)
        for nombre in attrs:
            if nombre == "style" or nombre.startswith("on"):
                self.atributos_inline.append(f"{etiqueta}[{nombre}]")
        if etiqueta == "script":
            self.scripts.append(attrs)
            self.dentro_de_script = True
        elif etiqueta == "style":
            self.estilos_inline += 1
        elif etiqueta == "link" and attrs.get("rel") == "stylesheet":
            self.hojas.append(attrs["href"])

    def handle_endtag(self, etiqueta):
        if etiqueta == "script":
            self.dentro_de_script = False

    def handle_data(self, datos):
        if self.dentro_de_script:
            self.codigo_inline += datos.strip()


def test_inicio_sirve_el_front_con_la_csp(cliente):
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in respuesta.headers["content-security-policy"]
    texto = respuesta.text
    assert "No pegues datos reales de pacientes; el texto no se guarda" in texto
    assert "Es una ayuda, no reemplaza leer la historia clínica." in texto
    assert 'id="formulario"' in texto and 'id="nota"' in texto


@pytest.mark.parametrize(
    ("ruta", "tipo"),
    [("/estatico/app.js", "javascript"), ("/estatico/estilos.css", "text/css")],
)
def test_estaticos_se_sirven_con_la_csp(cliente, ruta, tipo):
    respuesta = cliente.get(ruta)
    assert respuesta.status_code == 200
    assert tipo in respuesta.headers["content-type"]
    assert "default-src 'self'" in respuesta.headers["content-security-policy"]
    assert respuesta.headers["x-content-type-options"] == "nosniff"


def test_html_sin_scripts_ni_estilos_inline_ni_recursos_externos():
    recolector = Recolector()
    recolector.feed(HTML)
    assert recolector.scripts == [{"src": "/estatico/app.js", "defer": None}]
    assert recolector.codigo_inline == ""
    assert recolector.hojas == ["/estatico/estilos.css"]
    assert recolector.estilos_inline == 0
    assert recolector.atributos_inline == []
    # Sin CDN ni fuentes externas: nada externo se carga (src) y los únicos http(s) son
    # los enlaces del pie (repo y atribuciones de licencia).
    assert re.findall(r'src="(https?://[^"]+)"', HTML) == []
    permitidos = (
        "https://github.com/38diego",
        "https://github.com/cambridgeltl/sapbert",
        "https://huggingface.co/BSC-NLP4BIA/",
        "https://doi.org/10.5281/zenodo.",
    )
    externos = re.findall(r'href="(https?://[^"]+)"', HTML)
    assert externos and all(u.startswith(permitidos) for u in externos)
    assert re.findall(r'<link[^>]+href="https?://', HTML) == []


def test_html_trae_cuatro_ejemplos_y_el_js_cuatro_notas():
    assert len(re.findall(r"data-ejemplo=\"\d\"", HTML)) == 4
    bloque = JS[JS.index("const EJEMPLOS = [") : JS.index("];", JS.index("const EJEMPLOS"))]
    # Cada nota empieza con un encabezado en mayúsculas en su propia línea.
    assert len(re.findall(r'^  "[A-ZÁÉÍÓÚ -]+\\n" \+$', bloque, flags=re.M)) == 4


@pytest.mark.parametrize(
    "prohibido",
    [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "history.pushState",
        "history.replaceState",
        "location.hash",
        "location.search",
    ],
)
def test_js_no_inserta_html_ni_guarda_el_texto(prohibido):
    # Quita los comentarios: la cabecera del JS nombra lo que no se usa.
    codigo = re.sub(r"//.*|/\*.*?\*/", "", JS, flags=re.S)
    assert prohibido not in codigo


def test_js_construye_la_nota_con_nodos_de_texto():
    assert "createTextNode" in JS
    assert "textContent" in JS
    # Las posiciones de la API son puntos de código: la nota se parte con Array.from.
    assert "Array.from(texto)" in JS


def test_js_marca_la_categoria_por_correccion():
    """#116: la entidad corregida se marca con texto, no con HTML."""
    assert "categoria_por_correccion" in JS
    assert "Corregido: " in JS
    assert ".correccion" in CSS


@pytest.mark.parametrize("fuente", ["hanken-grotesk", "source-serif-4"])
def test_fuentes_propias_con_su_licencia(cliente, fuente):
    """Las fuentes se sirven desde la app (la CSP no deja cargarlas de afuera) y van
    con su licencia SIL OFL."""
    respuesta = cliente.get(f"/estatico/fuentes/{fuente}.woff2")
    assert respuesta.status_code == 200
    assert respuesta.content[:4] == b"wOF2"
    assert f"/estatico/fuentes/{fuente}.woff2" in CSS
    licencia = (ESTATICO / "fuentes" / f"OFL-{fuente}.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License" in licencia


def test_una_sola_superficie_para_la_nota():
    """La nota no se muestra dos veces: el textarea y la nota resaltada se alternan."""
    assert 'id="nota-resaltada"' in HTML and 'id="editar"' in HTML
    assert "function modoLectura" in JS and "function modoEdicion" in JS


def test_favicon_sirve_el_icono(cliente):
    """#226: `/favicon.ico` lo piden los navegadores por su cuenta; antes daba 404."""
    respuesta = cliente.get("/favicon.ico")
    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("image/svg+xml")
    assert respuesta.content == (ESTATICO / "icono.svg").read_bytes()


def test_js_muestra_las_opciones_de_codigo():
    """#230: las alternativas CIE-10 se pintan con nodos de texto y se eligen sin guardar."""
    assert "alternativas_cie10" in JS
    assert "function elegirCodigo" in JS
    assert ".opciones" in CSS and ".opcion[aria-pressed=" in CSS
