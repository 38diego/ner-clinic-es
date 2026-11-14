"""Primera etapa del normalizador: el índice CIE-10 y la búsqueda por similitud.

Cada fila del índice es un texto (la descripción de un código del catálogo CIE-10-ES o un
sinónimo, p. ej. una mención de CodiEsp entrenamiento con su código) y su vector, que da un
modelo de embeddings. Una mención se convierte igual y se buscan las filas con mayor
similitud de coseno (los vectores tienen norma 1, así que es el producto punto).

El modelo es SapBERT multilingüe (`SAPBERT`, revisión fija). Su código es MIT, pero sus
pesos no declaran licencia y se entrenó con UMLS, que tiene licencia propia: está pendiente
de resolver antes de redistribuirlo (`docs/datos.md`).

Viene del experimento de `evaluacion.normalizador` (#131, #132), que ahora importa de aquí.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import torch

from ner_clinic.charlson import quan
from ner_clinic.tipos import CategoriaCharlson

TOP_K = 5
LARGO_MAXIMO = 64  # tokens: las descripciones y menciones son cortas
LOTE = 128
ORIGEN_CATALOGO = "catalogo"

Pooling = Literal["cls", "media"]
Mapeador = Callable[[str], tuple[CategoriaCharlson, str] | None]


@dataclass(frozen=True)
class ModeloEmbeddings:
    nombre: str
    revision: str
    pooling: Pooling
    licencia: str
    prefijo: str = ""
    minusculas: bool = False
    # Si el tokenizador se carga de otro repo (nombre, revisión)
    tokenizador: tuple[str, str] | None = None


SAPBERT = ModeloEmbeddings(
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
    "47b6bd041ba61311584bb2494edfda5c7d9b719f",
    "cls",
    "no declarada (entrenado con UMLS)",
    # Se entrenó con los nombres en minúscula (`do_lower_case` en su configuración, que
    # el tokenizador XLM-R no aplica solo).
    minusculas=True,
    # Su repo solo trae el sentencepiece, que transformers 5 no convierte sin el paquete
    # `sentencepiece`. Es idéntico (mismo SHA-256) al de xlm-roberta-base (MIT), que
    # trae el tokenizador ya convertido.
    tokenizador=("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
)


class Codificador(Protocol):
    """Convierte textos en vectores de norma 1, uno por fila."""

    def __call__(self, textos: Sequence[str]) -> torch.Tensor: ...


class CodificadorHF:
    """Un modelo de Hugging Face en CPU, con la revisión y el *pooling* de `modelo`."""

    def __init__(self, modelo: ModeloEmbeddings, lote: int = LOTE):
        from transformers import AutoModel, AutoTokenizer

        self.modelo = modelo
        self.lote = lote
        nombre, revision = modelo.tokenizador or (modelo.nombre, modelo.revision)
        self.tokenizador = AutoTokenizer.from_pretrained(nombre, revision=revision)
        self.red = AutoModel.from_pretrained(modelo.nombre, revision=modelo.revision).eval()

    @torch.inference_mode()
    def __call__(self, textos: Sequence[str]) -> torch.Tensor:
        vectores = []
        for i in range(0, len(textos), self.lote):
            lote = [
                self.modelo.prefijo + (t.lower() if self.modelo.minusculas else t)
                for t in textos[i : i + self.lote]
            ]
            entrada = self.tokenizador(
                lote, padding=True, truncation=True, max_length=LARGO_MAXIMO, return_tensors="pt"
            )
            salida = self.red(**entrada).last_hidden_state
            if self.modelo.pooling == "cls":
                vector = salida[:, 0]
            else:
                mascara = entrada["attention_mask"].unsqueeze(-1).to(salida.dtype)
                vector = (salida * mascara).sum(1) / mascara.sum(1).clamp(min=1)
            vectores.append(torch.nn.functional.normalize(vector, dim=-1))
        if not vectores:
            return torch.empty(0, 0)
        return torch.cat(vectores)


@dataclass(frozen=True)
class Candidato:
    codigo: str
    descripcion: str
    similitud: float
    origen: str = ORIGEN_CATALOGO

    def explicacion(self) -> str:
        """Por qué salió este código, para mostrarlo al lado de la mención."""
        if self.origen == ORIGEN_CATALOGO:
            return f"{self.codigo}: {self.descripcion} (catálogo CIE-10)"
        return f"{self.codigo}: parecido a '{self.descripcion}', {self.origen}"


@dataclass(frozen=True)
class Sinonimo:
    """Otra forma de nombrar `codigo`. `documentos` son los de donde salió, para verificar
    que no viene de una partición de evaluación; `menciones`, cuántas veces aparece."""

    texto: str
    codigo: str
    origen: str
    documentos: tuple[str, ...] = ()
    menciones: int = 1


@dataclass
class Indice:
    """Códigos y el vector de su texto (fila `i` ↔ `codigos[i]`).

    El texto es la descripción del catálogo o un sinónimo; `origenes[i]` dice cuál
    (`None`: todas son del catálogo). Un código puede repetirse con varios sinónimos.
    """

    codigos: list[str]
    descripciones: list[str]
    vectores: torch.Tensor
    origenes: list[str] | None = None

    def origen(self, fila: int) -> str:
        return ORIGEN_CATALOGO if self.origenes is None else self.origenes[fila]

    def descripcion_catalogo(self, codigo: str) -> str | None:
        """La descripción del catálogo CIE-10-ES para `codigo` (`None` si solo está como
        sinónimo)."""
        if not hasattr(self, "_catalogo"):
            # Se arma completo y se asigna al final: otro hilo de la API nunca ve un dict a
            # medio llenar.
            catalogo: dict[str, str] = {}
            for fila, codigo_ in enumerate(self.codigos):
                if self.origen(fila) == ORIGEN_CATALOGO:
                    catalogo.setdefault(codigo_, self.descripciones[fila])
            self._catalogo = catalogo
        return self._catalogo.get(codigo)

    def con_sinonimos(self, sinonimos: Sequence[Sinonimo], codificador: Codificador) -> "Indice":
        """Un índice nuevo con las filas de este más una por sinónimo."""
        if not sinonimos:
            return self
        return Indice(
            self.codigos + [s.codigo for s in sinonimos],
            self.descripciones + [s.texto for s in sinonimos],
            torch.cat([self.vectores, codificador([s.texto for s in sinonimos])]),
            [self.origen(i) for i in range(len(self.codigos))] + [s.origen for s in sinonimos],
        )

    def buscar(
        self, consultas: torch.Tensor, k: int = TOP_K, lote: int = 1024
    ) -> list[list[Candidato]]:
        """Los `k` códigos más parecidos a cada vector de `consultas`, de mayor a menor."""
        salida = []
        for i in range(0, len(consultas), lote):
            similitudes = consultas[i : i + lote] @ self.vectores.T
            valores, posiciones = similitudes.topk(min(k, len(self.codigos)), dim=1)
            for fila_v, fila_p in zip(valores.tolist(), posiciones.tolist(), strict=True):
                salida.append(
                    [
                        Candidato(
                            self.codigos[p], self.descripciones[p], round(v, 4), self.origen(p)
                        )
                        for v, p in zip(fila_v, fila_p, strict=True)
                    ]
                )
        return salida


def huella_configuracion(modelo: ModeloEmbeddings) -> str:
    """SHA-256 de todo lo que cambia los vectores: el modelo (revisión, *pooling*, prefijo,
    minúscula, tokenizador) y el largo máximo."""
    contenido = json.dumps([asdict(modelo), LARGO_MAXIMO], ensure_ascii=False)
    return hashlib.sha256(contenido.encode()).hexdigest()


class Normalizador:
    """Mención → candidatos, con memoria de los textos ya buscados.

    `precalcular` codifica muchos textos en lotes (es lo que hace rápida la evaluación);
    `candidatos` usa lo precalculado o codifica el texto en el momento.
    """

    def __init__(self, indice: Indice, codificador: Codificador, k: int = TOP_K):
        self.indice = indice
        self.codificador = codificador
        self.k = k
        self._memoria: dict[str, list[Candidato]] = {}
        self._codigos = set(indice.codigos)

    def precalcular(self, textos: Iterable[str]) -> None:
        nuevos = list(dict.fromkeys(t for t in textos if t not in self._memoria))
        if nuevos:
            encontrados = self.indice.buscar(self.codificador(nuevos), self.k)
            self._memoria.update(zip(nuevos, encontrados, strict=True))

    def candidatos(self, texto: str) -> list[Candidato]:
        if texto not in self._memoria:
            self.precalcular([texto])
        return self._memoria[texto]

    def mapeador(self, umbral: float) -> Callable[[str], tuple[CategoriaCharlson, str] | None]:
        """Función texto → (categoría, código) con el `umbral`, como `charlson.reglas.mapear`.

        Si el primer candidato no llega al umbral, o su código no tiene categoría de
        Charlson, devuelve `None`. El código pasa antes por `codigo_para_categoria`.
        """

        def mapear(texto: str) -> tuple[CategoriaCharlson, str] | None:
            primero = self.candidatos(texto)[0]
            if primero.similitud < umbral:
                return None
            codigo = self.codigo_para_categoria(primero.codigo)
            categoria = quan.categoria_de(codigo)
            return None if categoria is None else (categoria, codigo)

        return mapear

    def codigo_para_categoria(self, codigo: str) -> str:
        """Un código padre sin categoría se lee como su hijo no especificado (`.9`).

        El catálogo trae los códigos de 3 caracteres ("E11 Diabetes mellitus tipo 2") y su
        descripción corta suele ser la más parecida a una mención. Quan no le da categoría
        a E11 porque reparte sus hijos entre diabetes con y sin complicación; "E11.9" es la
        que corresponde cuando no se especifica. Si el hijo `.9` no existe, queda igual.
        Con el catálogo de CodiEsp cambia la categoría de 13 códigos padre: E10, E11, E13,
        G83, I09, I27, I42, I73, K55, K70, K72, K76 y Z95.
        """
        if quan.categoria_de(codigo) is not None:
            return codigo
        hijo = f"{codigo}.9"
        return hijo if hijo in self._codigos else codigo
