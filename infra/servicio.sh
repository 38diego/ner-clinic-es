#!/usr/bin/env bash
# Despliega una imagen ya subida en el servicio de Cloud Run, con los topes de costo.
# Lo usan infra/desplegar.sh (desde la máquina de Diego) y .github/workflows/deploy.yml
# (#103), para que las dos vías desplieguen exactamente igual.
#
#   PROYECTO=ner-clinic-es-demo infra/servicio.sh IMAGEN [FLAGS EXTRA DE gcloud run deploy]
#
# Topes de costo (CLAUDE.md, "Despliegue"): 2 vCPU, 4 GiB, instancias mínimas 0 y máximas 1,
# CPU solo durante las peticiones, 20 análisis por IP al día y 200 en total.
#
# La app abre el puerto antes de cargar el modelo (#3), así que Cloud Run da la revisión por
# lista aunque el modelo no cargue. Por eso se despliega sin tráfico, se espera a que la
# revisión nueva cargue el modelo (por su URL con la etiqueta `candidata`) y solo entonces
# recibe el tráfico. Si no carga, el tráfico sigue en la revisión anterior.
set -euo pipefail

IMAGEN="${1:?Uso: infra/servicio.sh IMAGEN [flags extra]}"
shift
PROYECTO="${PROYECTO:-ner-clinic-es-demo}"
REGION="${REGION:-us-central1}"
SERVICIO="${SERVICIO:-ner-clinic-es}"
SECRETO="${SECRETO:-sal-ip}"
CUENTA="${CUENTA:-ner-clinic-run}"
VERSION="${IMAGEN##*:}"

SA="${CUENTA}@${PROYECTO}.iam.gserviceaccount.com"

gcloud run deploy "$SERVICIO" --project "$PROYECTO" --region "$REGION" --quiet \
  --image "$IMAGEN" \
  --service-account "$SA" \
  --cpu 2 --memory 4Gi --cpu-boost --cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 4 \
  --timeout 60 \
  --execution-environment gen2 \
  --set-env-vars "CONTADOR=firestore,FIRESTORE_PROYECTO=${PROYECTO},NER_NORMALIZADOR=hibrido,TOPE_IP_DIA=20,TOPE_GLOBAL_DIA=200,MAX_CARACTERES=5000,TIMEOUT_S=30" \
  --set-secrets "SAL_IP=${SECRETO}:latest" \
  --labels "app=ner-clinic-es,version=${VERSION}" \
  --no-traffic --tag candidata \
  "$@"

URL=$(gcloud run services describe "$SERVICIO" --project "$PROYECTO" --region "$REGION" \
  --format='value(status.url)')
CANDIDATA="https://candidata---${URL#https://}"

# `esperar=25` deja la petición abierta mientras carga: con la CPU solo durante las
# peticiones, sin una petición abierta la carga casi no avanza.
listo=""
for _ in $(seq 1 12); do
  if curl -sf --max-time 40 "${CANDIDATA}/api/salud?esperar=25" | grep -q '"estado":"ok"'; then
    listo=1
    break
  fi
  sleep 5
done
if [[ -z "$listo" ]]; then
  echo "La revisión nueva no cargó el modelo: el tráfico sigue en la anterior." >&2
  exit 1
fi

gcloud run services update-traffic "$SERVICIO" --project "$PROYECTO" --region "$REGION" \
  --to-latest --quiet >&2
echo "$URL"
