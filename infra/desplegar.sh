#!/usr/bin/env bash
# Construye la imagen `completa`, la sube a Artifact Registry y la despliega en Cloud Run.
#
#   uv run python -m evaluacion artefactos   # una vez: el índice del normalizador
#   PROYECTO=ner-clinic-es-demo infra/desplegar.sh
#
# Antes: infra/gcp.sh. Es el plan B de #103: lo normal es que despliegue GitHub Actions
# (.github/workflows/deploy.yml) al fusionar a main.
#
# Topes de costo: ver infra/servicio.sh, que comparte con el despliegue desde GitHub (#103).
set -euo pipefail

PROYECTO="${PROYECTO:-ner-clinic-es-demo}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-ner-clinic}"
SERVICIO="${SERVICIO:-ner-clinic-es}"
SECRETO="${SECRETO:-sal-ip}"
CUENTA="${CUENTA:-ner-clinic-run}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"

IMAGEN="${REGION}-docker.pkg.dev/${PROYECTO}/${REPO}/ner-clinic-es:${VERSION}"

INDICE="data/normalizacion/indice_final.pt"
if [[ ! -f "$INDICE" ]]; then
  echo "Falta ${INDICE}: corre 'uv run python -m evaluacion artefactos'." >&2
  exit 1
fi

# Se construye desde una copia limpia del commit (más el índice, que no se versiona), para
# que la imagen sea exactamente ${VERSION} aunque haya cambios locales.
LIMPIO="$(mktemp -d)"
trap 'git worktree remove --force "$LIMPIO" >/dev/null 2>&1 || rm -rf "$LIMPIO"' EXIT
git worktree add --detach "$LIMPIO" "$VERSION" >/dev/null
mkdir -p "$LIMPIO/data/normalizacion"
cp "$INDICE" "$LIMPIO/$INDICE"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet >/dev/null
docker build --target completa -t "$IMAGEN" "$LIMPIO"
docker push "$IMAGEN"

# La primera vez el servicio se crea público; las siguientes, --allow-unauthenticated no
# cambia nada. Las flags del servicio están en infra/servicio.sh (#103).
PROYECTO="$PROYECTO" REGION="$REGION" SERVICIO="$SERVICIO" SECRETO="$SECRETO" CUENTA="$CUENTA" \
  "$(dirname "$0")/servicio.sh" "$IMAGEN" --allow-unauthenticated
