#!/usr/bin/env bash
# Permisos para que GitHub Actions despliegue la demo sin llaves JSON (#103).
# Idempotente: se puede correr varias veces.
#
#   PROYECTO=ner-clinic-es-demo infra/github_wif.sh
#
# Requiere `gcloud auth login` con la cuenta dueña del proyecto y que ya se haya corrido
# infra/gcp.sh y un primer infra/desplegar.sh (el servicio tiene que existir).
# Crea, si no existen:
#   - el bucket privado de artefactos y sube el índice del normalizador
#     (data/normalizacion/indice_final.pt, 323 MB, no se versiona en git);
#   - el pool `github` de Workload Identity Federation con el proveedor `github-actions`,
#     que solo acepta tokens del repo 38diego/ner-clinic-es en la rama main;
#   - la cuenta de servicio `desplegador`, con permisos mínimos:
#       * escribir imágenes en el repositorio de Artifact Registry `ner-clinic`;
#       * desplegar revisiones del servicio `ner-clinic-es` (run.developer, solo ese servicio;
#         no puede cambiar quién lo invoca);
#       * actuar como la cuenta de runtime `ner-clinic-run` (sin eso no puede desplegar con ella);
#       * leer el bucket de artefactos.
# Al final imprime las dos variables que van en GitHub (no son secretas).
set -euo pipefail

PROYECTO="${PROYECTO:-ner-clinic-es-demo}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-ner-clinic}"
SERVICIO="${SERVICIO:-ner-clinic-es}"
CUENTA_RUNTIME="${CUENTA:-ner-clinic-run}"
REPO_GITHUB="${REPO_GITHUB:-38diego/ner-clinic-es}"
# ID numérico del repo (`gh api repos/38diego/ner-clinic-es -q .id`): si el repo cambiara de
# nombre o se borrara, otro repo con el mismo nombre no heredaría el acceso.
REPO_GITHUB_ID="${REPO_GITHUB_ID:-1457863510}"
BUCKET="${BUCKET:-${PROYECTO}-artefactos}"
POOL="github"
PROVEEDOR="github-actions"
DESPLEGADOR="desplegador"
INDICE="data/normalizacion/indice_final.pt"

SA="${DESPLEGADOR}@${PROYECTO}.iam.gserviceaccount.com"
SA_RUNTIME="${CUENTA_RUNTIME}@${PROYECTO}.iam.gserviceaccount.com"
g() { gcloud --project "$PROYECTO" --quiet "$@"; }
paso() { printf '\n== %s\n' "$*"; }

NUMERO="$(g projects describe "$PROYECTO" --format='value(projectNumber)')"

paso "APIs"
g services enable iamcredentials.googleapis.com sts.googleapis.com storage.googleapis.com

paso "Bucket gs://${BUCKET} con el índice"
if ! g storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  g storage buckets create "gs://${BUCKET}" --location="$REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi
if [[ -f "$INDICE" ]]; then
  # Solo sube si cambió. Se compara el CRC32C: un archivo de este tamaño se sube en partes
  # (objeto compuesto) y los objetos compuestos no tienen MD5.
  LOCAL="$(g storage hash "$INDICE" --skip-md5 --format='value(crc32c_hash)')"
  REMOTO="$(g storage objects describe "gs://${BUCKET}/${INDICE#data/}" \
    --format='value(crc32c_hash)' 2>/dev/null || true)"
  if [[ "$LOCAL" == "$REMOTO" ]]; then
    echo "el índice ya está al día"
  else
    g storage cp "$INDICE" "gs://${BUCKET}/${INDICE#data/}"
  fi
else
  echo "Aviso: falta ${INDICE}; el bucket queda sin índice y el despliegue fallará." >&2
fi

paso "Pool ${POOL} y proveedor ${PROVEEDOR}"
if ! g iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1; then
  g iam workload-identity-pools create "$POOL" --location=global \
    --display-name="GitHub Actions"
fi
CONDICION="assertion.repository == '${REPO_GITHUB}' && assertion.repository_id == '${REPO_GITHUB_ID}' && assertion.ref == 'refs/heads/main'"
MAPEO="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"
if ! g iam workload-identity-pools providers describe "$PROVEEDOR" \
     --workload-identity-pool="$POOL" --location=global >/dev/null 2>&1; then
  g iam workload-identity-pools providers create-oidc "$PROVEEDOR" \
    --workload-identity-pool="$POOL" --location=global \
    --display-name="Repo ${REPO_GITHUB}" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="$MAPEO" \
    --attribute-condition="$CONDICION"
else
  # Ya existe: se reaplica la condición por si cambió.
  g iam workload-identity-pools providers update-oidc "$PROVEEDOR" \
    --workload-identity-pool="$POOL" --location=global \
    --attribute-mapping="$MAPEO" \
    --attribute-condition="$CONDICION" >/dev/null
fi

paso "Cuenta de servicio ${SA}"
if ! g iam service-accounts describe "$SA" >/dev/null 2>&1; then
  g iam service-accounts create "$DESPLEGADOR" --display-name="Despliegue desde GitHub Actions"
fi
# Solo el repo (y por la condición del proveedor, solo su rama main) puede usarla.
g iam service-accounts add-iam-policy-binding "$SA" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${NUMERO}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${REPO_GITHUB}" \
  >/dev/null
g artifacts repositories add-iam-policy-binding "$REPO" --location="$REGION" \
  --member="serviceAccount:${SA}" --role=roles/artifactregistry.writer >/dev/null
g run services add-iam-policy-binding "$SERVICIO" --region="$REGION" \
  --member="serviceAccount:${SA}" --role=roles/run.developer >/dev/null
g iam service-accounts add-iam-policy-binding "$SA_RUNTIME" \
  --member="serviceAccount:${SA}" --role=roles/iam.serviceAccountUser >/dev/null
g storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA}" --role=roles/storage.objectViewer >/dev/null

paso "Listo. Variables de GitHub (Settings → Secrets and variables → Actions → Variables)"
echo "GCP_PROVEEDOR_WIF=projects/${NUMERO}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVEEDOR}"
echo "GCP_CUENTA_DESPLIEGUE=${SA}"
