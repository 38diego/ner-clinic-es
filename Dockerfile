# Imagen de la API (FastAPI + modelo NER + normalizador) para Cloud Run, en CPU.
#
# Construir:  uv run python -m evaluacion artefactos   # una vez: crea el índice
#             docker build -t ner-clinic-es .
# Sin SapBERT ni índice (CI): docker build --target base -t ner-clinic-es:base .
# Correr:     docker run --rm -p 8080:8080 --cpus=2 --memory=4g ner-clinic-es
#
# Etapas: `dependencias` instala las dependencias con uv y descarga el modelo NER;
# `constructor` instala el código;
# `base` solo copia el entorno virtual y la caché del modelo, sin uv ni herramientas de
# build; `sapbert` descarga SapBERT; `completa` (la de por defecto, #189) es `base` con
# SapBERT y el índice. Si no está el índice, `docker build` falla: nunca una imagen grande
# que en silencio solo usa reglas. Funciona con BuildKit y con el builder clásico.

# --- Etapa 1: dependencias y modelo -------------------------------------------------
FROM python:3.12-slim AS dependencias

# uv con versión fija, desde su imagen oficial.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /bin/uv

# - Copia los archivos al entorno (no enlaces): la etapa final copia `.venv` tal cual.
# - Compila a bytecode: el arranque en frío no lo paga.
# - Usa el Python de la imagen base y nunca descarga otro.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Primero solo las dependencias (torch de CPU según `[tool.uv.sources]`): esta capa se
# reutiliza mientras `uv.lock` no cambie.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra api --no-install-project

# El modelo, con la revisión fijada (commit del repo en Hugging Face). Va antes del
# código para que un cambio en `src/` no lo vuelva a descargar. Solo los pesos en
# safetensors, la configuración y el tokenizador: ni `pytorch_model.bin` ni
# `training_args.bin`.
ARG NER_MODELO=BSC-NLP4BIA/bsc-bio-ehr-es-distemist
ARG NER_MODELO_REVISION=a66816753b41d5acf2df8070a781265672a356ec
ENV HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1
RUN /app/.venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${NER_MODELO}', revision='${NER_MODELO_REVISION}', \
allow_patterns=['config.json', 'model.safetensors', '*.txt', 'tokenizer*', \
'special_tokens_map.json', 'vocab.json', 'merges.txt', 'README.md'])"

# El código: se instala como paquete (no editable) dentro del entorno. Etapa aparte para
# que un cambio en `src/` no vuelva a descargar SapBERT. En /proyecto quedan solo los
# archivos del paquete (ner_clinic, ner_clinic_api y su dist-info) para la etapa final.
FROM dependencias AS constructor
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra api --no-editable \
    && mkdir /proyecto \
    && cp -a /app/.venv/lib/python3.12/site-packages/ner_clinic* /proyecto/


# --- Etapa 2: imagen sin normalizador ------------------------------------------------
FROM python:3.12-slim AS base

ARG NER_MODELO_REVISION=a66816753b41d5acf2df8070a781265672a356ec

# Usuario sin privilegios, sin shell de login.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin ner

# Primero lo que no cambia con el código (dependencias y modelo, más de 1 GB) y después,
# en una capa aparte y pequeña, el paquete del proyecto (#236). Así un cambio en `src/` no
# vuelve a construir, guardar en caché ni subir la capa de torch.
COPY --from=dependencias /app/.venv /app/.venv
COPY --from=dependencias /opt/hf /opt/hf
COPY --from=constructor /proyecto/ /app/.venv/lib/python3.12/site-packages/

# - El modelo ya está en la imagen: en runtime no se descarga nada (modo offline) y se
#   carga exactamente la revisión del build.
# - `HOST=0.0.0.0` para recibir tráfico de fuera del contenedor; Cloud Run inyecta
#   `PORT` (8080 por defecto). El access log va apagado en `python -m ner_clinic_api`.
# - `OMP_NUM_THREADS=2`: los hilos de torch iguales a los vCPU asignados. Sin esto torch
#   ve todos los núcleos del host y se estorba con la cuota de CPU. Si se cambian los
#   vCPU en Cloud Run, se cambia aquí también (o con `--set-env-vars`).
# - Sin `NER_NORMALIZADOR`: la API usa el mapeo híbrido (#182). Esta etapa no trae el
#   índice, así que arranca con las reglas y `/api/salud` dice `mapeo: "reglas"`.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NER_MODELO_REVISION=${NER_MODELO_REVISION} \
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    HOST=0.0.0.0 \
    PORT=8080

USER ner
WORKDIR /app
EXPOSE 8080

CMD ["python", "-m", "ner_clinic_api"]


# --- Etapa 3: SapBERT y su tokenizador, con revisión fija -----------------------------
# Revisiones iguales a `ner_clinic.normalizacion.indice.SAPBERT` (hay un test). La
# licencia de los pesos de SapBERT está pendiente (`docs/datos.md`): resolverla antes de
# publicar esta imagen, desplegar la demo pública con ella o usarla con un cliente.
FROM dependencias AS sapbert
ARG SAPBERT=cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR
ARG SAPBERT_REVISION=47b6bd041ba61311584bb2494edfda5c7d9b719f
ARG TOKENIZADOR=FacebookAI/xlm-roberta-base
ARG TOKENIZADOR_REVISION=e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
ENV HF_HOME=/opt/hf-sapbert
RUN /app/.venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${SAPBERT}', revision='${SAPBERT_REVISION}', \
allow_patterns=['config.json', 'model.safetensors']); \
snapshot_download('${TOKENIZADOR}', revision='${TOKENIZADOR_REVISION}', \
allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json', \
'sentencepiece.bpe.model'])"


# --- Etapa 4: imagen completa (la de por defecto) ------------------------------------
FROM base AS completa
COPY --from=sapbert /opt/hf-sapbert/hub/ /opt/hf/hub/
# El índice se construye con `uv run python -m evaluacion artefactos`; no se versiona.
COPY data/normalizacion/indice_final.pt /opt/normalizacion/indice_final.pt
ENV NER_INDICE_NORMALIZADOR=/opt/normalizacion/indice_final.pt
