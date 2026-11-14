// Front de la demo de ner-clinic-es: JS nativo, sin dependencias ni build.
//
// Seguridad (XSS): el texto del usuario y todo lo que devuelve la API se insertan solo
// con `textContent`, `createTextNode` y `setAttribute` sobre atributos inertes. Nunca se
// inserta HTML como cadena ni se evalúa código. La CSP de la API (`default-src 'self'`)
// además bloquea scripts y estilos inline.
//
// Privacidad: el texto vive solo en memoria (el textarea y la variable `textoAnalizado`).
// No va al almacenamiento del navegador, a cookies ni a la URL.
//
// Una sola superficie para la nota: se escribe en el textarea y, al analizar, el mismo
// lugar muestra la nota con las menciones resaltadas. "Editar nota" vuelve al textarea.

"use strict";

// Debe coincidir con `MAX_CARACTERES` de la API (5000 por defecto). Si la API usa otro
// valor, el 413 de la API lo explica igual.
const MAX_CARACTERES = 5000;
// La API espera hasta 30 s el turno y luego corre la inferencia: margen amplio.
const TIMEOUT_MS = 120000;

const ESTADOS = [
  {
    clave: "confirmada",
    grupo: "Confirmadas",
    etiqueta: "confirmada",
    descripcion: "El paciente la tiene. Suman al índice.",
  },
  {
    clave: "posible",
    grupo: "A confirmar",
    etiqueta: "a confirmar",
    descripcion: "Planteadas como sospecha. Solo suman si se confirman.",
  },
  {
    clave: "negada",
    grupo: "Negadas",
    etiqueta: "negada",
    descripcion: "La nota dice que el paciente no las tiene.",
  },
  {
    clave: "familiar",
    grupo: "De un familiar",
    etiqueta: "de un familiar",
    descripcion: "Antecedentes familiares, no del paciente.",
  },
];
const ESTADO_POR_CLAVE = Object.fromEntries(ESTADOS.map((e) => [e.clave, e]));

// Las 17 categorías de Charlson et al. (1987), con su peso y un nombre corto para la
// hoja. Los pesos son los de `ner_clinic.charlson.indice.PESOS`.
const CATEGORIAS = [
  { clave: "infarto_miocardio", nombre: "Infarto de miocardio", corto: "Infarto", peso: 1 },
  { clave: "insuficiencia_cardiaca", nombre: "Insuficiencia cardíaca", corto: "Insuf. cardíaca", peso: 1 },
  { clave: "vascular_periferica", nombre: "Enfermedad vascular periférica", corto: "Vascular perif.", peso: 1 },
  { clave: "cerebrovascular", nombre: "Enfermedad cerebrovascular", corto: "Cerebrovascular", peso: 1 },
  { clave: "demencia", nombre: "Demencia", corto: "Demencia", peso: 1 },
  { clave: "pulmonar_cronica", nombre: "Enfermedad pulmonar crónica", corto: "Pulmonar crónica", peso: 1 },
  { clave: "reumatologica", nombre: "Enfermedad reumatológica", corto: "Reumatológica", peso: 1 },
  { clave: "ulcera_peptica", nombre: "Úlcera péptica", corto: "Úlcera péptica", peso: 1 },
  { clave: "hepatica_leve", nombre: "Hepatopatía leve", corto: "Hepática leve", peso: 1 },
  { clave: "diabetes", nombre: "Diabetes sin daño de órgano", corto: "Diabetes", peso: 1 },
  { clave: "diabetes_complicada", nombre: "Diabetes con daño de órgano", corto: "Diabetes con daño", peso: 2 },
  { clave: "hemiplejia", nombre: "Hemiplejía", corto: "Hemiplejía", peso: 2 },
  { clave: "renal_cronica", nombre: "Enfermedad renal crónica", corto: "Renal crónica", peso: 2 },
  { clave: "tumor_maligno", nombre: "Tumor maligno", corto: "Tumor maligno", peso: 2 },
  { clave: "hepatica_severa", nombre: "Hepatopatía moderada o severa", corto: "Hepática grave", peso: 3 },
  { clave: "tumor_metastasico", nombre: "Tumor sólido metastásico", corto: "Metástasis", peso: 6 },
  { clave: "vih_sida", nombre: "VIH / sida", corto: "VIH / sida", peso: 6 },
];
const CATEGORIA_POR_CLAVE = Object.fromEntries(CATEGORIAS.map((c) => [c.clave, c]));
// Forma leve -> forma severa: si están ambas, la leve no suma (`JERARQUIAS` del índice).
const JERARQUIAS = {
  diabetes: "diabetes_complicada",
  hepatica_leve: "hepatica_severa",
  tumor_maligno: "tumor_metastasico",
};

// Notas 100 % sintéticas, escritas para la demo. No vienen de ninguna historia real.
const EJEMPLOS = [
  "EPICRISIS - MEDICINA INTERNA\n" +
    "Paciente masculino de 74 años con antecedentes de HTA, diabetes mellitus tipo 2 " +
    "insulinorrequiriente y enfermedad renal crónica estadio 3b. Infarto agudo de " +
    "miocardio en 2019 con colocación de stent. Ingresa por disnea de medianos esfuerzos " +
    "y edemas en miembros inferiores; ecocardiograma con FEVI del 35 %, se documenta " +
    "insuficiencia cardiaca con fracción de eyección reducida.\n" +
    "Niega EPOC, asma y tabaquismo.\n" +
    "Antecedentes familiares: madre con cáncer de mama, padre con diabetes.\n" +
    "Plan: ajuste de diurético por la insuficiencia cardiaca, control de la HTA y " +
    "seguimiento por nefrología.",
  "NOTA DE URGENCIAS\n" +
    "Mujer de 81 años traída por su hija por cuadro de 2 días de desorientación y " +
    "deterioro del estado general. Antecedente de demencia tipo Alzheimer y fibrilación " +
    "auricular anticoagulada. Al examen hemiparesia derecha de inicio incierto.\n" +
    "Impresión diagnóstica: sospecha de accidente cerebrovascular isquémico.\n" +
    "No se descarta neumonía aspirativa.\n" +
    "Niega diabetes, HTA ni cardiopatía isquémica conocida.\n" +
    "Se solicita TAC simple de cráneo y radiografía de tórax.",
  "CONSULTA EXTERNA - MEDICINA FAMILIAR\n" +
    "Paciente femenina de 58 años en control de EPOC GOLD 2 y artritis reumatoide en " +
    "manejo con metotrexato. Refiere epigastralgia; endoscopia previa con úlcera gástrica " +
    "cicatrizada.\n" +
    "Niega infarto, enfermedad renal y hepatopatía.\n" +
    "Hermano con VIH, madre con enfermedad de Alzheimer.\n" +
    "Se continúa manejo y se solicitan paraclínicos de control.",
  "INTERCONSULTA - ONCOLOGÍA CLÍNICA\n" +
    "Paciente masculino de 66 años con adenocarcinoma de colon estadio IV con metástasis " +
    "hepáticas, en quimioterapia. Antecedente de cirrosis hepática por alcohol y várices " +
    "esofágicas. Diabetes mellitus tipo 2 con nefropatía diabética.\n" +
    "Fumador pesado. Probable EPOC, pendiente espirometría.\n" +
    "Niega enfermedad coronaria e insuficiencia cardiaca.\n" +
    "Padre fallecido por cáncer de próstata.",
];

const $ = (id) => document.getElementById(id);

const ui = {
  formulario: $("formulario"),
  nota: $("nota"),
  notaResaltada: $("nota-resaltada"),
  contador: $("contador"),
  analizar: $("analizar"),
  editar: $("editar"),
  limpiar: $("limpiar"),
  estadoPeticion: $("estado-peticion"),
  despertando: $("despertando"),
  despertandoSegundos: $("despertando-segundos"),
  error: $("error"),
  errorTitulo: $("error-titulo"),
  errorDetalle: $("error-detalle"),
  panel: $("panel"),
  indice: $("indice-charlson"),
  charlsonSi: $("charlson-si"),
  hojaCharlson: $("hoja-charlson"),
  hallazgosTitulo: $("hallazgos-titulo"),
  meta: $("meta"),
  desactualizado: $("desactualizado"),
  vacio: $("vacio"),
  grupos: $("grupos"),
};

let analizando = false;
// El modelo carga en segundo plano al despertar el servidor: hasta que esté listo no se
// analiza, y lo que se pida mientras tanto se analiza apenas termine.
let modeloListo = false;
let pendiente = false;
// El texto que produjo el resultado en pantalla (null si no hay resultado).
let textoAnalizado = null;
// Navegación por menciones: la fila activa y en qué mención va.
let navegacion = { fila: null, indice: -1 };
let marcaActiva = null;
// Fila de cada entidad (por índice en la respuesta) y entidad de cada marca de la nota.
let filas = [];
let entidadDeMarca = new Map();

// --- Utilidades ---

/** Largo en puntos de código, como `len()` de Python (no en unidades UTF-16). */
function largo(texto) {
  let n = 0;
  for (const _ of texto) n++;
  return n;
}

function formatoNumero(n) {
  return n.toLocaleString("es-CO");
}

function plural(n, uno, varios) {
  return `${formatoNumero(n)} ${n === 1 ? uno : varios}`;
}

/** Crea un elemento con clase y texto (siempre `textContent`, nunca HTML). */
function crear(etiqueta, clase, texto) {
  const el = document.createElement(etiqueta);
  if (clase) el.className = clase;
  if (texto !== undefined && texto !== null) el.textContent = texto;
  return el;
}

function oculto(texto) {
  return crear("span", "visualmente-oculto", texto);
}

function vaciar(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function movimiento() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

// --- Modo de la hoja: escribir o leer resaltado ---

function modoEdicion() {
  ui.nota.hidden = false;
  ui.notaResaltada.hidden = true;
  ui.editar.hidden = true;
  ui.analizar.hidden = false;
  if (textoAnalizado !== null) marcarDesactualizado(ui.nota.value !== textoAnalizado);
}

function modoLectura() {
  ui.nota.hidden = true;
  ui.notaResaltada.hidden = false;
  ui.editar.hidden = false;
  // La nota en pantalla ya está analizada: la acción siguiente es editarla.
  ui.analizar.hidden = true;
  marcarDesactualizado(false);
}

function marcarDesactualizado(si) {
  ui.desactualizado.hidden = !si;
  ui.panel.classList.toggle("es-desactualizado", si);
  ui.panel.querySelectorAll(".entidad-boton, .celda, .opcion").forEach((b) => {
    b.disabled = si || b.dataset.sinDestino === "true";
  });
}

function editar() {
  modoEdicion();
  ui.nota.focus();
}

// --- Entrada ---

function actualizarContador() {
  const n = largo(ui.nota.value);
  const excede = n > MAX_CARACTERES;
  ui.contador.textContent = excede
    ? `${formatoNumero(n)} de ${formatoNumero(MAX_CARACTERES)} caracteres: sobran ${formatoNumero(n - MAX_CARACTERES)}`
    : `${formatoNumero(n)} de ${formatoNumero(MAX_CARACTERES)} caracteres`;
  ui.contador.classList.toggle("excede", excede);
  ui.analizar.disabled = analizando || excede || !modeloListo;
}

function alEscribir() {
  actualizarContador();
  if (textoAnalizado !== null) marcarDesactualizado(ui.nota.value !== textoAnalizado);
}

function cargarEjemplo(evento) {
  if (analizando) return;
  const boton = evento.currentTarget;
  const ejemplo = EJEMPLOS[Number(boton.dataset.ejemplo)];
  document.querySelectorAll("[data-ejemplo]").forEach((b) => b.removeAttribute("aria-current"));
  boton.setAttribute("aria-current", "true");
  // Ya está analizado: se muestra sin gastar otro de los análisis del día.
  if (textoAnalizado === ejemplo) {
    ui.nota.value = ejemplo;
    actualizarContador();
    modoLectura();
    return;
  }
  ui.nota.value = ejemplo;
  modoEdicion();
  actualizarContador();
  ocultarError();
  ui.formulario.requestSubmit();
}

function limpiar() {
  ui.nota.value = "";
  textoAnalizado = null;
  document.querySelectorAll("[data-ejemplo]").forEach((b) => b.removeAttribute("aria-current"));
  vaciar(ui.notaResaltada);
  ocultarError();
  reiniciarPanel();
  modoEdicion();
  actualizarContador();
  ui.nota.focus();
}

// --- Errores ---

function mostrarError(titulo, detalle) {
  ui.errorTitulo.textContent = titulo;
  ui.errorDetalle.textContent = detalle || "";
  ui.error.hidden = false;
}

function ocultarError() {
  ui.error.hidden = true;
}

/** Título y detalle en lenguaje claro para cada código de error de la API. */
function mensajeDeError(codigo, detalleApi) {
  switch (codigo) {
    case 413:
      return [
        "La nota es demasiado larga.",
        detalleApi || `El máximo es de ${formatoNumero(MAX_CARACTERES)} caracteres. Recórtala y analízala de nuevo.`,
      ];
    case 422:
      return ["La nota no se pudo leer.", "Revisa que el texto no esté vacío y analízala de nuevo."];
    case 429:
      return [
        "Se acabaron los análisis de hoy.",
        `${detalleApi || "Llegaste al máximo de análisis por día."} ` +
          "El contador se reinicia a medianoche, hora de Colombia.",
      ];
    case 503:
      return [
        "El servicio está ocupado o no está disponible.",
        detalleApi || "Hay mucha demanda en este momento. Intenta en unos segundos.",
      ];
    default:
      return ["El servidor falló al analizar la nota.", "Intenta de nuevo en un momento."];
  }
}

// --- Despertar del modelo ---

const ESPERA_SALUD_S = 25;
let esperandoModelo = false;

function mostrarDespertando(desde) {
  ui.despertando.hidden = false;
  const contar = () => {
    const s = Math.round((Date.now() - desde) / 1000);
    ui.despertandoSegundos.textContent = s >= 3 ? ` (${s} s)` : "";
  };
  contar();
  return setInterval(contar, 1000);
}

/**
 * Pregunta a `/api/salud` hasta que el modelo esté listo. Con `esperar` la petición queda
 * abierta mientras carga: así Cloud Run no le quita la CPU a la carga.
 */
async function esperarModelo() {
  if (esperandoModelo) return;
  esperandoModelo = true;
  modeloListo = false;
  actualizarContador();
  const desde = Date.now();
  let reloj = null;
  let fallos = 0;
  let espera = 0;
  try {
    for (;;) {
      let datos = null;
      let codigo = 0;
      try {
        const respuesta = await fetch(`/api/salud?esperar=${espera}`, { cache: "no-store" });
        codigo = respuesta.status;
        datos = await respuesta.json();
      } catch {
        datos = null;
      }
      if (datos && datos.estado === "ok") break;
      if (datos && datos.estado === "error") {
        mostrarError(...mensajeDeError(503, "El modelo no pudo cargar. Intenta más tarde."));
        return;
      }
      // Ni la salud responde (el servidor está arrancando o no hay red): reintenta un rato.
      if (!datos && codigo !== 200 && ++fallos > 20) {
        mostrarError("No hay conexión con el servidor.", "Recarga la página en un momento.");
        return;
      }
      if (!reloj) reloj = mostrarDespertando(desde);
      if (!datos) await new Promise((r) => setTimeout(r, 3000));
      espera = ESPERA_SALUD_S;
    }
    modeloListo = true;
    actualizarContador();
    if (pendiente) {
      pendiente = false;
      ui.formulario.requestSubmit();
    }
  } finally {
    clearInterval(reloj);
    ui.despertando.hidden = true;
    esperandoModelo = false;
  }
}

// --- Análisis ---

function ponerAnalizando(activo) {
  analizando = activo;
  ui.analizar.textContent = activo ? "Analizando…" : "Analizar nota";
  ui.analizar.setAttribute("aria-busy", String(activo));
  document.body.classList.toggle("analizando", activo);
  // Mientras se analiza, la nota no cambia: así el resultado siempre es de lo que se ve.
  ui.limpiar.disabled = activo;
  ui.nota.readOnly = activo;
  ui.estadoPeticion.textContent = activo ? "Analizando la nota. Puede tardar unos segundos." : "";
  actualizarContador();
}

async function enviar(evento) {
  evento.preventDefault();
  if (analizando) return;
  ocultarError();

  const texto = ui.nota.value;
  if (!texto.trim()) {
    mostrarError("La nota está vacía.", "Escribe una nota o elige uno de los ejemplos.");
    modoEdicion();
    ui.nota.focus();
    return;
  }
  if (largo(texto) > MAX_CARACTERES) {
    mostrarError(...mensajeDeError(413));
    return;
  }
  if (!modeloListo) {
    pendiente = true;
    esperarModelo();
    return;
  }

  ponerAnalizando(true);
  let exito = false;
  const control = new AbortController();
  const temporizador = setTimeout(() => control.abort(), TIMEOUT_MS);
  try {
    const respuesta = await fetch("/api/analizar", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ texto }),
      signal: control.signal,
      cache: "no-store",
      credentials: "same-origin",
    });
    let datos = null;
    try {
      datos = await respuesta.json();
    } catch {
      datos = null;
    }
    if (respuesta.status === 503 && datos && datos.estado_modelo === "cargando") {
      // El servidor se apagó y volvió a arrancar con la página abierta: se espera y se
      // analiza solo.
      pendiente = true;
      esperarModelo();
      return;
    }
    if (!respuesta.ok) {
      const detalle = datos && typeof datos.detalle === "string" ? datos.detalle : "";
      mostrarError(...mensajeDeError(respuesta.status, detalle));
      return;
    }
    if (!datos || !Array.isArray(datos.entidades)) {
      mostrarError(...mensajeDeError(500));
      return;
    }
    mostrarResultado(texto, datos);
    exito = true;
  } catch (error) {
    if (error && error.name === "AbortError") {
      mostrarError(
        "El análisis tardó demasiado.",
        "El servidor no respondió a tiempo. Intenta de nuevo en un momento o con una nota más corta."
      );
    } else {
      mostrarError("No hay conexión con el servidor.", "Revisa tu conexión a internet e intenta de nuevo.");
    }
  } finally {
    clearTimeout(temporizador);
    ponerAnalizando(false);
    // El botón se deshabilitó con el foco encima: tras un error, el foco vuelve a la nota.
    if (!exito) ui.nota.focus({ preventScroll: true });
  }
}

// --- Resultado ---

function reiniciarPanel() {
  navegacion = { fila: null, indice: -1 };
  marcaActiva = null;
  filas = [];
  entidadDeMarca = new Map();
  vaciar(ui.grupos);
  ui.meta.textContent = "";
  ui.vacio.hidden = false;
  ui.panel.classList.remove("con-resultado");
  marcarDesactualizado(false);
  pintarHojaCharlson(null);
}

function mostrarResultado(texto, datos) {
  reiniciarPanel();
  textoAnalizado = texto;
  const entidades = datos.entidades.filter((e) => ESTADO_POR_CLAVE[e.estado]);

  const marcas = pintarNota(texto, entidades);
  pintarGrupos(entidades, marcas);
  pintarHojaCharlson(entidades, datos);
  pintarDescartes(datos.descartes);
  ui.meta.textContent = textoMeta(datos.meta, entidades.length);
  ui.vacio.hidden = true;
  ui.panel.classList.add("con-resultado");

  modoLectura();
  // En pantallas angostas el panel queda debajo de la nota: se lleva al lector allá.
  if (window.matchMedia("(max-width: 60rem)").matches) {
    ui.panel.scrollIntoView({ behavior: movimiento(), block: "start" });
  }
  ui.hallazgosTitulo.focus({ preventScroll: true });
}

function textoMeta(meta, cantidad) {
  const partes = [plural(cantidad, "encontrada", "encontradas")];
  if (meta && typeof meta.segundos === "number") {
    partes.push(`en ${meta.segundos.toLocaleString("es-CO", { maximumFractionDigits: 1 })} s`);
  }
  let texto = partes.join(" ");
  if (meta && typeof meta.restantes_hoy === "number") {
    texto += `. Te ${meta.restantes_hoy === 1 ? "queda" : "quedan"} ${plural(meta.restantes_hoy, "análisis", "análisis")} hoy.`;
  }
  return texto;
}

/**
 * Pinta la nota con un `<mark>` por mención, construida solo con nodos de texto.
 *
 * Las posiciones de la API son índices de Python (puntos de código), así que la nota se
 * parte con `Array.from(texto)` y no con `slice` sobre la cadena (que cuenta unidades
 * UTF-16 y se desfasa con emojis u otros caracteres fuera del plano básico).
 *
 * Menciones superpuestas (regla determinista): se ordenan por inicio; a igual inicio,
 * primero la más larga; a igual tramo, primero la de estado más relevante (confirmada,
 * posible, negada, familiar) y luego por orden en la respuesta. Se pinta cada mención que
 * no se cruce con una ya pintada. Una mención que se cruza no se pinta aparte: su click
 * lleva a la marca pintada que contiene su inicio. Las posiciones fuera de la nota o
 * vacías se ignoran.
 *
 * Devuelve un `Map` de "entidad:mención" a la marca que la representa.
 */
function pintarNota(texto, entidades) {
  vaciar(ui.notaResaltada);
  const caracteres = Array.from(texto);
  const total = caracteres.length;
  const prioridad = Object.fromEntries(ESTADOS.map((e, i) => [e.clave, i]));

  const menciones = [];
  entidades.forEach((entidad, e) => {
    (entidad.menciones || []).forEach((m, k) => {
      const inicio = Number(m.inicio);
      const fin = Number(m.fin);
      if (!Number.isInteger(inicio) || !Number.isInteger(fin)) return;
      if (inicio < 0 || fin > total || fin <= inicio) return;
      // Cada mención se pinta con su propio estado (#208); si no lo trae, el de la entidad.
      const estado = ESTADO_POR_CLAVE[m.estado] ? m.estado : entidad.estado;
      menciones.push({ clave: `${e}:${k}`, e, estado, inicio, fin, orden: menciones.length });
    });
  });
  menciones.sort(
    (a, b) =>
      a.inicio - b.inicio ||
      b.fin - a.fin ||
      prioridad[a.estado] - prioridad[b.estado] ||
      a.orden - b.orden
  );

  const pintadas = [];
  const solapadas = [];
  let finUltima = 0;
  for (const m of menciones) {
    if (m.inicio >= finUltima) {
      pintadas.push(m);
      finUltima = m.fin;
    } else {
      solapadas.push(m);
    }
  }

  const marcas = new Map();
  const fragmento = document.createDocumentFragment();
  let cursor = 0;
  for (const m of pintadas) {
    if (m.inicio > cursor) {
      fragmento.appendChild(document.createTextNode(caracteres.slice(cursor, m.inicio).join("")));
    }
    const estado = ESTADO_POR_CLAVE[m.estado];
    const marca = crear("mark", `estado-${estado.clave}`);
    marca.dataset.estado = estado.clave;
    marca.appendChild(document.createTextNode(caracteres.slice(m.inicio, m.fin).join("")));
    marca.appendChild(oculto(` (${estado.etiqueta})`));
    marca.setAttribute("title", `${entidades[m.e].texto}: ${estado.etiqueta}`);
    fragmento.appendChild(marca);
    marcas.set(m.clave, marca);
    entidadDeMarca.set(marca, m.e);
    m.marca = marca;
    cursor = m.fin;
  }
  if (cursor < total) {
    fragmento.appendChild(document.createTextNode(caracteres.slice(cursor).join("")));
  }
  ui.notaResaltada.appendChild(fragmento);

  for (const m of solapadas) {
    const contenedora = pintadas.find((p) => p.inicio <= m.inicio && m.inicio < p.fin);
    if (contenedora) marcas.set(m.clave, contenedora.marca);
  }
  return marcas;
}

/**
 * La hoja de Charlson: las 17 categorías agrupadas por peso. Una celda se llena si la
 * categoría está confirmada, queda punteada si solo está a confirmar y se tacha si la
 * forma severa ya la cubre. El total viene de la API; la hoja solo lo explica.
 */
function pintarHojaCharlson(entidades, datos) {
  vaciar(ui.hojaCharlson);
  const porCategoria = new Map();
  (entidades || []).forEach((entidad, e) => {
    const c = entidad.categoria_charlson;
    if (!CATEGORIA_POR_CLAVE[c]) return;
    if (entidad.estado !== "confirmada" && entidad.estado !== "posible") return;
    if (!porCategoria.has(c)) porCategoria.set(c, []);
    porCategoria.get(c).push(e);
  });
  const confirmada = (c) =>
    (porCategoria.get(c) || []).some((e) => entidades[e].estado === "confirmada");

  if (datos) {
    const total = Number(datos.indice_charlson ?? 0);
    const siConfirman = Number(datos.indice_charlson_si_se_confirman ?? total);
    ui.indice.textContent = String(total);
    ui.charlsonSi.textContent =
      siConfirman > total
        ? `Llega a ${siConfirman} si se confirman las que están a confirmar.`
        : "Suma el peso de cada categoría confirmada.";
  } else {
    ui.indice.textContent = "–";
    ui.charlsonSi.textContent = "Suma el peso de cada categoría confirmada.";
  }

  for (const peso of [1, 2, 3, 6]) {
    const fila = crear("div", "peso");
    const titulo = crear("p", "peso-titulo", String(peso));
    titulo.setAttribute("title", peso === 1 ? "1 punto" : `${peso} puntos`);
    titulo.appendChild(oculto(peso === 1 ? " punto" : " puntos"));
    fila.appendChild(titulo);
    const celdas = crear("ul", "celdas");
    for (const cat of CATEGORIAS.filter((c) => c.peso === peso)) {
      const item = crear("li");
      const celda = crear("button", "celda", cat.corto);
      celda.type = "button";
      celda.setAttribute("title", cat.nombre);
      const indices = porCategoria.get(cat.clave) || [];
      const severa = JERARQUIAS[cat.clave];
      let estado = "ausente";
      if (indices.length > 0 && severa && confirmada(severa)) {
        // La forma severa confirmada la cubre, esté la leve confirmada o a confirmar.
        estado = "cubierta";
      } else if (confirmada(cat.clave)) {
        estado = "suma";
      } else if (indices.length > 0) {
        estado = "posible";
      }
      celda.dataset.estado = estado;
      const lectura = {
        ausente: "no aparece",
        suma: `suma ${peso}`,
        cubierta: "no suma: la cubre su forma severa",
        posible: "a confirmar, no suma todavía",
      }[estado];
      celda.appendChild(oculto(`: ${lectura}`));
      // Lleva a la primera entidad de la categoría con una mención en la nota,
      // confirmada antes que posible. Sin ninguna, la celda no es un botón activo.
      const conMencion = indices.filter((e) => filas[e] && filas[e].destinos.length > 0);
      const destino =
        conMencion.find((e) => entidades[e].estado === "confirmada") ?? conMencion[0];
      if (destino === undefined) {
        celda.disabled = true;
        celda.dataset.sinDestino = "true";
      } else {
        celda.addEventListener("click", () => activarEntidad(destino));
      }
      item.appendChild(celda);
      celdas.appendChild(item);
    }
    fila.appendChild(celdas);
    ui.hojaCharlson.appendChild(fila);
  }
}

function detalleEntidad(entidad) {
  const detalle = crear("span", "entidad-detalle");
  const categoria = CATEGORIA_POR_CLAVE[entidad.categoria_charlson];
  // "Fuera de Charlson" solo informa en las que podrían sumar al índice.
  const puedeSumar = entidad.estado === "confirmada" || entidad.estado === "posible";
  if (categoria || puedeSumar) {
    detalle.appendChild(
      crear("span", "entidad-categoria", categoria ? categoria.nombre : "Fuera de Charlson")
    );
  }
  if (entidad.cie10) {
    const codigo = crear("span", "cie10");
    codigo.appendChild(oculto("CIE-10 "));
    codigo.appendChild(document.createTextNode(entidad.cie10));
    detalle.appendChild(codigo);
  }
  return detalle;
}

function notasEntidad(entidad) {
  const notas = [];
  // Otras menciones de la misma enfermedad en otro estado (#208).
  const otros = (entidad.otros_estados || []).filter((clave) => ESTADO_POR_CLAVE[clave]);
  if (otros.length > 0) {
    const nombres = otros.map((clave) => ESTADO_POR_CLAVE[clave].etiqueta).join(" y ");
    notas.push(crear("span", "otros-estados", `En otra mención aparece ${nombres}.`));
  }
  // La categoría salió de corregir una palabra mal escrita (#115): se dice cuál, para que
  // el lector sepa que no fue una coincidencia exacta.
  const correcciones = entidad.correcciones || [];
  if (entidad.categoria_por_correccion && correcciones.length > 0) {
    const cambios = correcciones.map((c) => `${c.escrita} → ${c.corregida}`).join(", ");
    const marca = crear("span", "correccion", `Corregido: ${cambios}`);
    // El aviso entra en el nombre accesible del botón (un `title` solo lo vería el mouse).
    marca.appendChild(oculto(" (categoría por corrección de ortografía; revisa la nota)"));
    notas.push(marca);
  }
  return notas;
}

/**
 * Las opciones de código CIE-10 que dio el normalizador (#230): el experto elige la que
 * corresponde y la fila la muestra como elegida. Solo en la pantalla: no se guarda.
 */
function opcionesCodigo(entidad, boton) {
  // Sin el código que la fila ya muestra (ni su padre: "E11" cuando se sugiere "E11.9").
  const sugerido = entidad.cie10 || "";
  const alternativas = (entidad.alternativas_cie10 || []).filter(
    (a) =>
      a &&
      typeof a.codigo === "string" &&
      typeof a.descripcion === "string" &&
      Number.isFinite(a.similitud) &&
      a.codigo !== sugerido &&
      `${a.codigo}.9` !== sugerido
  );
  if (alternativas.length === 0) return null;

  const bloque = crear("details", "opciones");
  bloque.appendChild(
    crear(
      "summary",
      null,
      `${entidad.cie10 ? "Otras opciones de código" : "Opciones de código"} (${alternativas.length})`
    )
  );
  bloque.appendChild(
    crear(
      "p",
      "opciones-ayuda",
      "Los más parecidos del catálogo CIE-10-ES. Elige el que corresponda; en la demo no se guarda."
    )
  );
  const lista = crear("ul", "opciones-lista");
  for (const a of alternativas) {
    const item = crear("li");
    const opcion = crear("button", "opcion");
    opcion.type = "button";
    opcion.setAttribute("aria-pressed", "false");
    opcion.appendChild(crear("span", "cie10", a.codigo));
    opcion.appendChild(crear("span", "opcion-descripcion", a.descripcion));
    const parecido = crear("span", "opcion-similitud", `${Math.round(Number(a.similitud) * 100)} %`);
    parecido.appendChild(oculto(" de parecido"));
    opcion.appendChild(parecido);
    opcion.addEventListener("click", () =>
      elegirCodigo(entidad, boton, lista, opcion, a.codigo, bloque)
    );
    item.appendChild(opcion);
    lista.appendChild(item);
  }
  bloque.appendChild(lista);
  const aviso = crear("p", "visualmente-oculto opciones-aviso");
  aviso.setAttribute("role", "status");
  bloque.appendChild(aviso);
  return bloque;
}

/** Pone en la fila el código elegido; elegir otra vez el mismo vuelve al sugerido. */
function elegirCodigo(entidad, boton, lista, opcion, codigo, bloque) {
  const deshacer = opcion.getAttribute("aria-pressed") === "true";
  lista.querySelectorAll(".opcion").forEach((o) => o.setAttribute("aria-pressed", "false"));
  const detalle = boton.querySelector(".entidad-detalle");
  let chip = detalle.querySelector(".cie10");
  if (!chip) {
    chip = crear("span", "cie10");
    detalle.appendChild(chip);
  }
  const nota = detalle.querySelector(".elegido");
  if (nota) nota.remove();
  vaciar(chip);
  const final = deshacer ? entidad.cie10 : codigo;
  // El cambio ocurre dentro del botón de la fila: se anuncia aparte.
  const aviso = bloque.querySelector(".opciones-aviso");
  aviso.textContent = deshacer
    ? `Volviste al código sugerido${final ? ` ${final}` : ""}.`
    : `Elegiste ${codigo} para «${entidad.texto}».`;
  if (!final) {
    chip.remove();
    return;
  }
  chip.appendChild(oculto("CIE-10 "));
  chip.appendChild(document.createTextNode(final));
  chip.classList.toggle("es-elegido", !deshacer);
  if (!deshacer) {
    opcion.setAttribute("aria-pressed", "true");
    detalle.appendChild(crear("span", "elegido", "Elegido por ti"));
  }
}

function pintarGrupos(entidades, marcas) {
  if (entidades.length === 0) {
    ui.grupos.appendChild(
      crear("p", "sin-hallazgos", "No se encontraron enfermedades en esta nota.")
    );
    return;
  }
  for (const estado of ESTADOS) {
    const delEstado = [];
    entidades.forEach((entidad, e) => {
      if (entidad.estado === estado.clave) delEstado.push([entidad, e]);
    });
    if (delEstado.length === 0) continue;

    const grupo = crear("section", `grupo estado-${estado.clave}`);
    const cabeza = crear("h3", "grupo-titulo");
    cabeza.appendChild(crear("span", "muestra", estado.grupo));
    cabeza.appendChild(crear("span", "grupo-cantidad", String(delEstado.length)));
    grupo.appendChild(cabeza);
    grupo.appendChild(crear("p", "grupo-descripcion", estado.descripcion));

    const lista = crear("ul", "entidades");
    for (const [entidad, e] of delEstado) {
      lista.appendChild(filaEntidad(entidad, e, estado, marcas));
    }
    grupo.appendChild(lista);
    ui.grupos.appendChild(grupo);
  }
}

function filaEntidad(entidad, e, estado, marcas) {
  // Menciones en orden de aparición, cada una con la marca que la representa.
  const destinos = (entidad.menciones || [])
    .map((m, k) => ({ inicio: m.inicio, marca: marcas.get(`${e}:${k}`) }))
    .filter((d) => d.marca)
    .sort((a, b) => a.inicio - b.inicio)
    .map((d) => d.marca);

  const item = crear("li", "entidad");
  const boton = crear("button", "entidad-boton");
  boton.type = "button";
  boton.setAttribute("aria-controls", "nota-resaltada");
  const nombre = crear("span", "entidad-nombre", entidad.texto);
  nombre.appendChild(oculto(` (${estado.etiqueta})`));
  boton.appendChild(nombre);
  const n = (entidad.menciones || []).length;
  boton.appendChild(crear("span", "entidad-menciones", n > 1 ? `×${n}` : ""));
  if (n > 1) boton.lastChild.appendChild(oculto(" menciones"));
  boton.appendChild(detalleEntidad(entidad));
  for (const nota of notasEntidad(entidad)) boton.appendChild(nota);
  if (destinos.length === 0) {
    boton.disabled = true;
    boton.dataset.sinDestino = "true";
    // Sin mención en la nota la fila nunca se activa: sus opciones se ven siempre.
    item.classList.add("sin-destino");
  }

  const posicion = crear("p", "posicion");
  posicion.setAttribute("aria-live", "polite");

  // Texto visible fijo y el estado en `aria-pressed`: el nombre accesible empieza igual
  // que lo que se ve (WCAG 2.5.3) y el nombre de la entidad va oculto al final.
  const reportar = crear("button", "reportar", "Marcar como error");
  reportar.type = "button";
  reportar.setAttribute("aria-pressed", "false");
  reportar.appendChild(oculto(`: ${entidad.texto} (${estado.etiqueta})`));
  const gracias = crear(
    "p",
    "reporte-gracias",
    "Marcado. En la demo no se guarda; en un despliegue real iría a revisión humana."
  );
  gracias.hidden = true;
  gracias.setAttribute("role", "status");
  reportar.addEventListener("click", () => {
    const marcado = reportar.getAttribute("aria-pressed") !== "true";
    reportar.setAttribute("aria-pressed", String(marcado));
    gracias.hidden = !marcado;
    item.classList.toggle("reportada", marcado);
  });

  item.appendChild(boton);
  item.appendChild(posicion);
  const opciones = opcionesCodigo(entidad, boton);
  if (opciones) item.appendChild(opciones);
  item.appendChild(reportar);
  item.appendChild(gracias);

  const fila = { item, boton, posicion, entidad, estado, destinos };
  filas[e] = fila;
  boton.addEventListener("click", () => irAMencion(fila));
  return item;
}

function pintarDescartes(descartes) {
  if (!Array.isArray(descartes) || descartes.length === 0) return;
  const bloque = crear("details", "descartes");
  bloque.appendChild(
    crear("summary", null, `Términos descartados por la depuración (${descartes.length})`)
  );
  const lista = crear("ul");
  for (const d of descartes) {
    const item = crear("li");
    item.appendChild(crear("strong", null, d.texto));
    if (d.detalle) item.appendChild(document.createTextNode(`: ${d.detalle}`));
    lista.appendChild(item);
  }
  bloque.appendChild(lista);
  ui.grupos.appendChild(bloque);
}

/** Activa la fila de una entidad (desde la hoja de Charlson o desde la nota). */
function activarEntidad(e, marca) {
  const fila = filas[e];
  if (!fila || fila.boton.disabled) return;
  if (marca && fila.destinos.includes(marca)) {
    // Desde la nota: se queda en la mención tocada.
    seleccionar(fila, fila.destinos.indexOf(marca));
  } else {
    irAMencion(fila);
  }
  fila.item.scrollIntoView({ behavior: movimiento(), block: "nearest" });
}

/** Primer click: primera mención. Cada click siguiente sobre la misma entidad: la siguiente. */
function irAMencion(fila) {
  const indice = navegacion.fila === fila ? (navegacion.indice + 1) % fila.destinos.length : 0;
  seleccionar(fila, indice);
  fila.destinos[indice].scrollIntoView({ behavior: movimiento(), block: "center", inline: "nearest" });
}

function seleccionar(fila, indice) {
  if (navegacion.fila && navegacion.fila !== fila) {
    navegacion.fila.boton.removeAttribute("aria-current");
    navegacion.fila.item.classList.remove("activa");
    navegacion.fila.posicion.textContent = "";
  }
  navegacion = { fila, indice };
  fila.boton.setAttribute("aria-current", "true");
  fila.item.classList.add("activa");

  const marca = fila.destinos[indice];
  if (marcaActiva) marcaActiva.classList.remove("activa");
  // Reinicia la animación aunque se repita la misma marca.
  void marca.offsetWidth;
  marca.classList.add("activa");
  marcaActiva = marca;

  const n = fila.destinos.length;
  const deLaMencion = ESTADO_POR_CLAVE[marca.dataset.estado] || fila.estado;
  let texto = `Mención ${indice + 1} de ${n}`;
  if (deLaMencion !== fila.estado) texto += `, aquí ${deLaMencion.etiqueta}`;
  if (n > 1) texto += ". Toca otra vez para ir a la siguiente.";
  fila.posicion.textContent = texto;
}

/** Tocar una marca en la nota activa su entidad en el panel. */
function alTocarNota(evento) {
  const marca = evento.target.closest("mark");
  if (!marca || !entidadDeMarca.has(marca)) return;
  activarEntidad(entidadDeMarca.get(marca), marca);
}

// --- Arranque ---

document.querySelectorAll("[data-ejemplo]").forEach((b) => b.addEventListener("click", cargarEjemplo));
ui.nota.addEventListener("input", alEscribir);
ui.formulario.addEventListener("submit", enviar);
ui.limpiar.addEventListener("click", limpiar);
ui.editar.addEventListener("click", editar);
ui.notaResaltada.addEventListener("click", alTocarNota);
// Ctrl/Cmd + Enter analiza desde el área de texto.
ui.nota.addEventListener("keydown", (evento) => {
  if (evento.key === "Enter" && (evento.ctrlKey || evento.metaKey)) {
    ui.formulario.requestSubmit();
  }
});
pintarHojaCharlson(null);
actualizarContador();
esperarModelo();
