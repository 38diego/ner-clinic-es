---
name: despliegue
description: API FastAPI, imagen Docker con el modelo, demo web en JS nativo y despliegue en Cloud Run con tope de costo y sin guardar el texto
status: backlog
created: 2026-11-03T01:07:10Z
---

# PRD: despliegue

## Executive Summary

El pipeline está terminado y medido (epics #4, #39, #71 y #86). Falta lo que un cliente o un
reclutador puede **probar**: una demo web donde se pega una nota y se ven las comorbilidades
por estado, con un click que lleva a cada mención en el texto.

Este subproyecto entrega:

- **API FastAPI:** `POST /api/analizar`.
- **Imagen Docker** con el modelo incluido, en CPU.
- **Front en JS nativo**, servido por la misma imagen.
- **Despliegue en Google Cloud Run** con tope de costo: 20 análisis por IP al día, un tope
  global diario y alerta de presupuesto.
- **Privacidad:** el texto que se recibe nunca se guarda ni se registra en logs.

Decisiones de Diego (2026-11-03):
- Front en JS nativo, sin build.
- La URL que asigna Cloud Run, sin dominio propio.
- Contadores de uso en Firestore.
- La cuenta de GCP se crea desde cero y la crea Diego, porque la facturación pide tarjeta.

El resto viene de las decisiones de despliegue ya tomadas en el proyecto.

## Problem Statement

- Hoy el proyecto solo se puede probar clonando el repo, instalando `uv` y bajando el modelo.
  Ningún cliente va a hacer eso.
- La idea de producto para la demo necesita una interfaz:
  la lista por estado, el click al contexto y el resaltado por color.
- Una demo pública tiene dos riesgos:
  - **Costo:** alguien podría abusar de la demo y generar una factura.
  - **Privacidad:** alguien podría pegar una historia clínica real. Hay que cerrar los dos.

## User Stories

**US1. Cliente o reclutador (visitante de la demo)**
Como visitante, quiero pegar una nota (o elegir un ejemplo) y ver en segundos las
comorbilidades agrupadas por estado, para entender qué hace el sistema sin instalar nada.
- Criterios de aceptación:
  - Hay botones con 3 o 4 notas sintéticas de ejemplo, y un área de texto para pegar.
  - El resultado se ve en grupos: confirmadas, posibles ("a confirmar"), negadas y
    familiares. Cada entidad trae su categoría de Charlson y su CIE-10 cuando los tiene, y
    cuántas veces se menciona.
  - También se muestran el índice de Charlson y el índice "si se confirman las posibles".

**US2. Médico o auditor**
Como lector, quiero hacer click en una comorbilidad y que la nota salte a la mención y la
resalte, para leer el contexto y no confiar a ciegas.
- Criterios de aceptación:
  - Click en una entidad: la nota hace scroll a la primera mención y la resalta. Otro click
    lleva a la siguiente.
  - En la nota, todas las menciones están resaltadas según su estado. El estado nunca se
    indica solo con color: también va con un ícono y una etiqueta.
  - Cada entidad tiene un botón "esto está mal". No guarda nada; solo muestra que el diseño
    contempla la corrección humana.

**US3. Diego (dueño de la cuenta)**
Como dueño de la cuenta de GCP, quiero que la demo no me pueda costar más de unos pocos
dólares aunque la abusen, y que me avise si se acerca al límite.
- Criterios de aceptación:
  - Al pasar de 20 análisis en un día, la misma IP recibe un 429 con un mensaje claro.
  - Al pasar el tope global del día (p. ej. 200), todos reciben un 429.
  - La alerta de presupuesto está configurada (5 USD) y llega al correo de Diego.
  - Cloud Run tiene instancias máximas 1 y mínimas 0, así que sin uso no cobra CPU ni memoria.

**US4. Cualquiera que pegue una nota**
Como usuario, quiero saber que lo que pego no se guarda, y que se me advierta que no pegue
datos reales de pacientes.
- Criterios de aceptación:
  - Antes del área de texto hay un aviso visible: "No pegues datos reales de pacientes; el
    texto no se guarda".
  - Ni la API ni Cloud Run registran el texto: ni en los logs de acceso, ni en los errores,
    ni en las trazas. Un test lo verifica.
  - Firestore solo guarda un hash con sal de la IP y un contador por día.

**US5. Integrador**
Como desarrollador, quiero una API documentada (OpenAPI de FastAPI) para integrar el análisis
en otro sistema.

## Functional Requirements

**FR1. API (FastAPI)**
- `POST /api/analizar`:
  - recibe `{texto, diagnosticos_registrados?}`;
  - devuelve `Resultado.to_dict()` más `meta`: versión, tiempo de análisis y análisis
    restantes del día para esa IP.
- `GET /api/salud`: estado del servicio y si el modelo está cargado.
- `GET /`: el front estático.
- El modelo se carga **una vez, al arrancar**, y se calienta con una nota sintética. Las
  peticiones nunca cargan el modelo.
- La inferencia corre con **una sola nota a la vez** (semáforo), porque es CPU pura. Si la
  espera pasa del timeout, responde 503 con "intenta en unos segundos".
- Límites:
  - **tamaño máximo de la nota:** de entrada 5.000 caracteres; se ajusta con la medición de
    FR3. Si se pasa, responde 413;
  - **timeout del análisis:** de entrada 30 s.
- Validación con Pydantic. Los errores se responden en español y nunca incluyen el texto
  recibido.
- **Logs:**
  - solo método, ruta, código, duración y longitud del texto;
  - ningún middleware, excepción ni traza imprime el cuerpo de la petición;
  - el access log de uvicorn queda desactivado o filtrado.

**FR2. Límites de uso (Firestore)**
- Interfaz `Contador` con dos implementaciones:
  - Firestore, para producción;
  - en memoria, para tests y desarrollo local.
- **Por IP:** la clave es `sha256(sal + ip + fecha)` y el tope es 20 al día. La sal es un
  secreto, en Secret Manager o en una variable de entorno de Cloud Run, y nunca va en el
  repo. La IP se toma del primer valor de `X-Forwarded-For`, que Cloud Run sí garantiza.
- **Global:** el tope es de 200 al día (configurable).
- Se cuenta solo si el análisis se va a ejecutar, con una transacción o un incremento
  atómico.
- **Si Firestore falla, el servicio falla cerrado:** responde 503 y no analiza. Así se
  protege el costo.
- Los documentos expiran con un TTL de Firestore a los 2 días. No se guarda nada más.

**FR3. Imagen Docker**
- Base `python:3.12-slim`, con `uv` y `torch` de CPU. Corre con un usuario sin privilegios.
- El modelo se descarga **durante el build**, con la revisión fijada del repo de Hugging
  Face, y queda dentro de la imagen. En runtime no se descarga nada.
- Multi-etapa, sin herramientas de build en la imagen final. Se reporta su tamaño.
- **Medición local** con `--cpus=2 --memory=4g`, que es lo mismo que Cloud Run:
  - mediana y p95 por nota sobre las 250 notas de prueba de DisTEMIST;
  - el peor caso con una nota de 5.000 caracteres;
  - el arranque en frío: proceso más carga del modelo.

  Con esas cifras se fijan el tamaño máximo y el timeout de FR1, y quedan documentadas.

**FR4. Front (JS nativo)**
- HTML, CSS y JS sin build ni dependencias, en `src/ner_clinic_api/estatico/` o similar.
- Pantalla, en orden:
  1. el aviso de privacidad;
  2. los ejemplos y el área de texto con su contador de caracteres;
  3. el botón Analizar;
  4. el resultado, en dos columnas: la lista por estado y la nota con resaltado. En celular
     va en una sola columna.
- Colores por estado con contraste AA, cada uno con ícono y etiqueta. Modo claro y oscuro.
- Muestra el tiempo de análisis, los análisis que quedan hoy y los mensajes de 429, 413 y
  503 en lenguaje claro.
- Nota al pie: "Es una ayuda, no reemplaza leer la historia clínica". Enlace al repo, a la
  evaluación y a la autoría.
- El texto del usuario **no se guarda** en `localStorage` ni en ningún otro lado.

**FR5. Infraestructura en GCP**
- Diego crea la cuenta de GCP, activa la facturación y crea un proyecto solo para la demo. El
  resto se hace con comandos documentados en `docs/despliegue.md` que Diego corre con
  `gcloud`, o con un script idempotente.
- Se habilitan Cloud Run, Artifact Registry, Firestore (modo nativo), Secret Manager y Cloud
  Build si se usa.
- **Cuenta de servicio de Cloud Run** con permisos mínimos: `roles/datastore.user` y acceso
  al secreto de la sal. Nada más.
- **Cloud Run:**
  - 2 vCPU y 4 GiB;
  - instancias mínimas 0 y máximas 1;
  - concurrencia baja (p. ej. 4: las demás peticiones esperan el semáforo);
  - timeout de la petición acorde a FR1;
  - CPU boost al arrancar;
  - acceso público (`allUsers` como invoker).
- **Alerta de presupuesto** de 5 USD, con avisos al 50 %, 90 % y 100 %.
- **Apagado automático** opcional: si el presupuesto llega al 100 %, una función pone
  instancias máximas en 0. Se documenta, y se implementa si es simple.

**FR6. CI/CD**
- **En cada PR:** `docker build` sin push, un smoke test del contenedor con
  `GET /api/salud` y una nota sintética, y los tests de la API con un extractor falso y el
  contador en memoria.
- **Despliegue manual** (`workflow_dispatch`) desde GitHub Actions con **Workload Identity
  Federation**, sin llaves JSON. Hace build, push a Artifact Registry y deploy a Cloud Run.
  Si WIF resulta muy costoso de configurar, el plan B es desplegar desde la máquina de Diego
  con un script, y queda documentado.

**FR7. Documentación**
- `docs/despliegue.md`: arquitectura, costos esperados, cómo desplegar, cómo ver la alerta,
  cómo apagar todo y la medición de FR3.
- README: enlace a la demo, captura de pantalla y cómo correrlo local con Docker.

## Non-Functional Requirements

- **Privacidad (no negociable):** ningún texto se guarda ni se registra. Hay un test que
  analiza una nota con una cadena centinela y verifica que no aparezca en los logs
  capturados.
- **Costo:** sin uso, 0 USD de cómputo. En el peor caso, el tope global garantiza quedarse en
  la capa gratuita (unos 6.000 análisis al mes). Hay que confirmar los
  precios vigentes al configurar.
- **Seguridad:**
  - cabeceras básicas (CSP sin `unsafe-inline` si es posible, `X-Content-Type-Options`);
  - sin CORS abierto, porque el front y la API comparten origen;
  - dependencias fijadas en `uv.lock`.
- **El código de la API va aparte de la librería:** un paquete `ner_clinic_api`. La librería
  `ner_clinic` no depende de FastAPI.
- Tests de la API sin red ni modelo, y la cobertura global se mantiene en 85 % o más.

## Success Criteria

- La demo está publicada en la URL de Cloud Run y enlazada desde el README.
- Una nota de unos 2.000 caracteres se analiza en Cloud Run con **p95 ≤ 5 s** con la
  instancia caliente. El arranque en frío se mide y se reporta, sin meta fija.
- La IP número 21 del día recibe 429, y el tope global también. Se prueba en local con el
  contador en memoria y una vez contra Firestore.
- El test centinela confirma que el texto no aparece en los logs. En los logs de Cloud Run
  de la primera semana tampoco aparece ningún texto.
- La alerta de presupuesto está creada y probada.
- El costo del primer mes es de 0 USD o casi.

## Constraints & Assumptions

- La facturación de GCP pide tarjeta. Diego crea la cuenta y el proyecto, y corre
  `gcloud auth login` (con `! gcloud auth login` desde la sesión). Claude no maneja
  credenciales de GCP ni las guarda.
- Hay que confirmar que la licencia del modelo (CC BY 4.0 según la ficha) permite incluirlo
  en una imagen pública, citando su origen. La verificación de licencias pendiente
  se cierra aquí.
- La latencia con 2 vCPU todavía no se ha medido. Si el p95 no cumple, las opciones son
  bajar el tamaño máximo de nota o subir a 4 vCPU (y recalcular el costo).

## Out of Scope

- Cuentas de usuario, historial o guardar análisis.
- Dominio propio: se puede agregar después sin cambiar nada más.
- GPU.
- Procesar notas por lotes o archivos.
- Mejoras del pipeline (#77, #95 y otras).

## Dependencies

- Pipeline `ner_clinic` (epics #4, #71 y #86) y su evaluación (#39).
- Cuenta de GCP con facturación, a cargo de Diego.
- Repo público o privado: Cloud Run no lo necesita público. El README enlazará la demo
  cuando el repo se haga público.
