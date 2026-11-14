#!/usr/bin/env bash
# Infraestructura de la demo en GCP (#102). Idempotente: se puede correr varias veces.
#
#   PROYECTO=ner-clinic-es-demo infra/gcp.sh
#
# Requiere `gcloud auth login` con una cuenta dueña del proyecto y de su facturación.
# Crea, si no existen:
#   - las APIs: Cloud Run, Artifact Registry, Firestore, Secret Manager y presupuestos;
#   - el presupuesto de 5 USD con avisos al 50, 90 y 100 % (llegan a los administradores
#     de la cuenta de facturación, es decir, al correo de Diego);
#   - el repositorio de Artifact Registry, que guarda solo las 2 imágenes más recientes;
#   - Firestore en modo nativo, con la TTL sobre `expira` en `uso` y `uso_ip`;
#   - el secreto `sal-ip` (32 bytes aleatorios; nunca se imprime);
#   - la cuenta de servicio de Cloud Run, con `datastore.user` y acceso solo a ese secreto.
# No despliega nada: eso lo hace infra/desplegar.sh.
set -euo pipefail

PROYECTO="${PROYECTO:-ner-clinic-es-demo}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-ner-clinic}"
SECRETO="${SECRETO:-sal-ip}"
CUENTA="${CUENTA:-ner-clinic-run}"
PRESUPUESTO_USD="${PRESUPUESTO_USD:-5}"
NOMBRE_PRESUPUESTO="ner-clinic-es demo"

SA="${CUENTA}@${PROYECTO}.iam.gserviceaccount.com"
g() { gcloud --project "$PROYECTO" --quiet "$@"; }
paso() { printf '\n== %s\n' "$*"; }

paso "APIs"
g services enable run.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com secretmanager.googleapis.com \
  billingbudgets.googleapis.com cloudbilling.googleapis.com iam.googleapis.com

paso "Presupuesto de ${PRESUPUESTO_USD} USD"
FACTURACION="$(g billing projects describe "$PROYECTO" --format='value(billingAccountName)')"
FACTURACION="${FACTURACION#billingAccounts/}"
NUMERO="$(g projects describe "$PROYECTO" --format='value(projectNumber)')"
if g billing budgets list --billing-account="$FACTURACION" \
     --format='value(displayName)' | grep -qx "$NOMBRE_PRESUPUESTO"; then
  echo "ya existe"
else
  g billing budgets create --billing-account="$FACTURACION" \
    --display-name="$NOMBRE_PRESUPUESTO" \
    --budget-amount="${PRESUPUESTO_USD}USD" \
    --filter-projects="projects/${NUMERO}" \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
fi

paso "Artifact Registry"
if ! g artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  g artifacts repositories create "$REPO" --repository-format=docker \
    --location="$REGION" --description="Imágenes de la demo de ner-clinic-es"
fi
POLITICA="$(mktemp)"
trap 'rm -f "$POLITICA"' EXIT
cat >"$POLITICA" <<'JSON'
[
  {"name": "conservar-2", "action": {"type": "Keep"},
   "mostRecentVersions": {"keepCount": 2}},
  {"name": "borrar-resto", "action": {"type": "Delete"},
   "condition": {"tagState": "any"}}
]
JSON
g artifacts repositories set-cleanup-policies "$REPO" --location="$REGION" \
  --policy="$POLITICA" --no-dry-run >/dev/null

paso "Firestore"
if ! g firestore databases describe --database='(default)' >/dev/null 2>&1; then
  g firestore databases create --location="$REGION" --type=firestore-native
fi
for coleccion in uso uso_ip; do
  g firestore fields ttls update expira --collection-group="$coleccion" --enable-ttl \
    >/dev/null
done

paso "Secreto ${SECRETO}"
if ! g secrets describe "$SECRETO" >/dev/null 2>&1; then
  python3 -c 'import secrets; print(secrets.token_hex(32), end="")' |
    g secrets create "$SECRETO" --replication-policy=automatic --data-file=-
fi

paso "Cuenta de servicio ${SA}"
if ! g iam service-accounts describe "$SA" >/dev/null 2>&1; then
  g iam service-accounts create "$CUENTA" --display-name="Cloud Run de ner-clinic-es"
fi
g projects add-iam-policy-binding "$PROYECTO" --member="serviceAccount:${SA}" \
  --role=roles/datastore.user --condition=None >/dev/null
g secrets add-iam-policy-binding "$SECRETO" --member="serviceAccount:${SA}" \
  --role=roles/secretmanager.secretAccessor >/dev/null

paso "Listo. Siguiente: infra/desplegar.sh"
