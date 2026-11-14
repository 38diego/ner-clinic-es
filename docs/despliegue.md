# Despliegue de la demo en Google Cloud Run

Epic #97. La demo es la imagen `completa` del Dockerfile (pipeline, API, front y SapBERT)
en Cloud Run, con topes de costo y sin guardar el texto que se pega.

## Arquitectura

```
navegador ──HTTPS──▶ Cloud Run (ner-clinic-es, us-central1)
                       │  imagen completa: NER + reglas + SapBERT + índice CIE-10
                       │  1 instancia como máximo, 0 sin uso
                       ├──▶ Firestore: contadores por día (hash de la IP, sin texto)
                       └──▶ Secret Manager: sal-ip (la sal del hash)
Artifact Registry (ner-clinic) ── guarda las 2 imágenes más recientes
Presupuesto de 5 USD ── avisos al 50, 90 y 100 % al correo de la cuenta de facturación
```

| Pieza | Valor |
|---|---|
| Proyecto | `ner-clinic-es-demo` |
| Región | `us-central1` (tier 1 de precios) |
| Servicio | `ner-clinic-es`: 2 vCPU, 4 GiB, CPU solo durante las peticiones, CPU boost, mínimas 0 y máximas 1 instancia, concurrencia 4, timeout 60 s |
| Cuenta de servicio | `ner-clinic-run`: `roles/datastore.user` y acceso solo al secreto `sal-ip` |
| Límites de la API | 20 análisis por IP al día, 200 en total, 5.000 caracteres, 30 s de espera (`configuracion.py`) |

## Cómo desplegar

Requiere `gcloud auth login` con la cuenta dueña del proyecto, y Docker.

```bash
uv run python -m evaluacion artefactos              # una vez: data/normalizacion/indice_final.pt
PROYECTO=ner-clinic-es-demo infra/gcp.sh            # infraestructura; idempotente
PROYECTO=ner-clinic-es-demo infra/desplegar.sh      # build, push y deploy del commit actual
```

`desplegar.sh` construye desde una copia limpia del commit (más el índice), así que la
etiqueta de la imagen es el commit exacto. Las flags del servicio (CPU, memoria,
instancias, topes) están en `infra/servicio.sh`, que comparten este script y GitHub Actions.

## Despliegue automático desde GitHub (#103)

Al fusionar a `main` un cambio en `src/`, el `Dockerfile`, las dependencias o el propio
workflow, `.github/workflows/deploy.yml` corre los tests, construye la imagen `completa`,
la sube y la despliega con `infra/servicio.sh`. Al final comprueba `/api/salud`. Los
cambios de documentación no despliegan. A mano: pestaña Actions o
`gh workflow run Despliegue`.

**Arranque en frío (#3).** El servicio se apaga cuando nadie lo usa, y al despertar el
modelo tarda unos 40 s en cargar. La app abre el puerto de una vez y carga el modelo en
segundo plano: la página sale en un par de segundos y avisa "Despertando el modelo"
mientras tanto; `/api/analizar` responde 503 sin contar el uso, y lo que se pida desde la
página se analiza apenas carga. Antes, la carga bloqueaba el puerto y Cloud Run respondía
500 a los ~20 s de espera.

Como Cloud Run da la revisión por lista apenas abre el puerto, `infra/servicio.sh`
despliega **sin tráfico**, espera a que la revisión nueva cargue el modelo (por su URL con
la etiqueta `candidata`) y solo entonces le pasa el tráfico. Si no carga, el tráfico sigue
en la revisión anterior.

**Tiempo.** Los tests y la construcción de la imagen corren en paralelo, y el despliegue
espera a los dos. En la imagen, las dependencias y los modelos van en capas que no
cambian con el código: un cambio en `src/` solo reconstruye y sube una capa de ~1 MB
(#236). Antes, cualquier cambio rehacía la capa de torch, de más de 1 GB, y un despliegue
tardaba unos 12 minutos.

**Sin llaves JSON.** GitHub se identifica ante Google con Workload Identity Federation: el
token OIDC del workflow se cambia por un permiso temporal de la cuenta `desplegador`. No
hay ninguna llave guardada en GitHub que se pueda filtrar.

Se configura una vez con `PROYECTO=ner-clinic-es-demo infra/github_wif.sh` (idempotente),
después de `gcp.sh` y de un primer `desplegar.sh`, y luego se ponen en GitHub las dos
**variables** que imprime (no son secretas): `GCP_PROVEEDOR_WIF` y
`GCP_CUENTA_DESPLIEGUE`.

| Pieza | Para qué |
|---|---|
| Pool `github`, proveedor `github-actions` | Acepta tokens de GitHub solo si `repository == 38diego/ner-clinic-es`, su ID numérico coincide (otro repo con el mismo nombre no entra) y `ref == refs/heads/main`: ni otro repo ni otra rama ni un PR pueden desplegar |
| `iam.workloadIdentityUser` sobre `desplegador` | Solo las identidades de ese repo pueden usar la cuenta |
| `artifactregistry.writer` en el repositorio `ner-clinic` | Subir imágenes (solo a ese repositorio) |
| `run.developer` en el servicio `ner-clinic-es` | Desplegar revisiones de ese servicio; no puede cambiar quién lo invoca ni tocar otros servicios |
| `iam.serviceAccountUser` sobre `ner-clinic-run` | Desplegar con la cuenta de runtime (que es la que lee Firestore y el secreto) |
| `storage.objectViewer` en `gs://ner-clinic-es-demo-artefactos` | Descargar el índice del normalizador |

**El índice del normalizador** (`indice_final.pt`, 323 MB) no está en git: `github_wif.sh`
lo sube a un bucket privado (con acceso uniforme y sin acceso público) y el workflow lo
descarga al construir. Si se regenera, se vuelve a correr `github_wif.sh` (solo sube si
cambió) y se despliega a mano.

**Plan B:** si WIF falla o GitHub Actions no está disponible, `infra/desplegar.sh` despliega
lo mismo desde la máquina de Diego.

## Costo esperado

Precios de Cloud Run con facturación por petición (tier 1, consultados el 2026-11-09):
capa gratuita mensual de 180.000 vCPU-s, 360.000 GiB-s y 2 millones de peticiones; después,
0,000024 USD por vCPU-s y 0,0000025 USD por GiB-s. La capa gratuita es por cuenta de
facturación.

- Un análisis de una nota de 2.000 caracteres tarda ~0,5 s (p95 0,76 s, #189), más un
  arranque en frío de ~14 s cuando la instancia está apagada.
- Peor caso con el tope global: 200 análisis al día × 30 días = 6.000 al mes. Aun si cada
  uno pagara un arranque en frío completo (~15 s × 2 vCPU = 30 vCPU-s), son 180.000 vCPU-s:
  el límite de la capa gratuita. En la práctica, muy por debajo.
- Artifact Registry: 0,5 GB gratis y luego ~0,10 USD por GB al mes. Con 2 imágenes de ~2 GB
  que comparten capas, unos centavos al mes.
- Firestore: 2 lecturas y 2 escrituras por análisis, dentro de su capa gratuita diaria.

Esperado: entre 0 y unos centavos de dólar al mes. El presupuesto de 5 USD avisa si algo no
cuadra.

## Apagar todo

```bash
# Detener el servicio (deja la infraestructura):
gcloud run services delete ner-clinic-es --region us-central1 --project ner-clinic-es-demo
# Borrar todo, incluida la facturación del proyecto:
gcloud projects delete ner-clinic-es-demo
```

El apagado automático al llegar al presupuesto (FR5, opcional) no se implementó: con una
instancia como máximo y el tope global diario, el costo ya está acotado por diseño.

## Licencias en la demo

- Modelo NER `BSC-NLP4BIA/bsc-bio-ehr-es-distemist`: CC BY 4.0 (se cita en la demo y el README).
- SapBERT: los autores declaran el proyecto MIT y su repositorio dice que contiene "code,
  data, and pretrained weights" (ver `docs/datos.md`). Se entrenó con UMLS: zona gris para
  uso comercial, no para esta demo.
- Índice: catálogo CIE-10-ES de CodiEsp y CodiEsp entrenamiento, CC BY 4.0 (se cita en la
  demo y en el README).

## Verificación en producción (#102)

Primer despliegue el 2026-11-10 (imagen `fd6441f`, revisión `ner-clinic-es-00001-nrk`):

| Qué | Resultado |
|---|---|
| `/api/salud` | `modelo_cargado: true`, `mapeo: hibrido` (SapBERT e índice cargados) |
| Cabeceras | CSP `default-src 'self'`, `x-frame-options: DENY`, `referrer-policy: no-referrer` |
| Latencia, instancia caliente, nota corta | 1,02 / 0,61 / 0,61 / 0,59 / 0,64 s |
| Tope por IP | El análisis 20 responde 200 con `restantes_hoy: 0`; el 21 responde 429 |
| Firestore | `uso` y `uso_ip` solo tienen `fecha`, `contador` y `expira`; el ID de `uso_ip` es un hash de 64 caracteres, sin la IP |
| Logs | Sin texto de las notas: solo arranque, carga de pesos y peticiones con su código |

Pendiente: medir el arranque en frío en producción (requiere que la instancia se apague
por inactividad) y revisar la primera semana de logs y costo (#104).
