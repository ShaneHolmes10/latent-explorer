// Tell ONNX Runtime Web to load its WebAssembly files from CDN
// rather than looking for them next to this script.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// ── Constants ──────────────────────────────────────────────────────────────

const MANIFEST_URL = "models/models.json";
const NUM_STD = 3.0;        // slider range = mean ± NUM_STD * std
const IMAGE_SIZE = 128;     // model output resolution
const DISPLAY_SIZE = 384;   // canvas display resolution

// ── State ──────────────────────────────────────────────────────────────────

let session = null;         // ort.InferenceSession
let stats = null;           // { latent_dim, means[], stds[] }
let values = null;          // Float32Array — current slider positions
let locks = null;           // bool[] — whether each dim is locked
let defaultVector = null;   // number[] — per-model default from models.json
let sliderEls = [];   // HTMLInputElement[]
let lockBtns = [];    // HTMLButtonElement[]

// ── DOM refs ───────────────────────────────────────────────────────────────

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const modelSelect = document.getElementById("model-select");
const sliderPanel = document.getElementById("slider-panel");

// ── Inference ──────────────────────────────────────────────────────────────

async function runInference() {
  if (!session || !stats) return;

  const input = new ort.Tensor("float32", Float32Array.from(values), [
    1,
    stats.latent_dim,
  ]);

  const results = await session.run({ z: input });
  const data = results.image.data; // Float32Array, shape [1, 3, H, W]

  drawTensor(data, IMAGE_SIZE, IMAGE_SIZE);
}

// Convert a CHW Float32Array in [0,1] to an upscaled canvas image.
function drawTensor(data, w, h) {
  const rgba = new Uint8ClampedArray(w * h * 4);
  const plane = w * h;

  for (let i = 0; i < plane; i++) {
    rgba[i * 4 + 0] = Math.round(data[i] * 255);             // R
    rgba[i * 4 + 1] = Math.round(data[plane + i] * 255);     // G
    rgba[i * 4 + 2] = Math.round(data[2 * plane + i] * 255); // B
    rgba[i * 4 + 3] = 255;                                    // A
  }

  // Draw to an off-screen canvas at native resolution, then scale up.
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  tmp.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmp, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
}

// ── Slider UI ──────────────────────────────────────────────────────────────

function buildSliders(s) {
  sliderPanel.innerHTML = "";
  sliderEls = [];
  lockBtns = [];
  values = new Float32Array(s.means);
  locks = new Array(s.latent_dim).fill(false);

  for (let i = 0; i < s.latent_dim; i++) {
    const mean = s.means[i];
    const std = s.stds[i];
    const lo = mean - NUM_STD * std;
    const hi = mean + NUM_STD * std;

    const row = document.createElement("div");
    row.className = "slider-row";

    // Lock button
    const lockBtn = document.createElement("button");
    lockBtn.className = "lock-btn";
    lockBtn.textContent = "○";
    lockBtn.title = "lock this dimension";
    lockBtn.addEventListener("click", () => toggleLock(i));
    lockBtns.push(lockBtn);

    // Dimension label
    const label = document.createElement("span");
    label.className = "slider-label";
    label.textContent = `D${String(i).padStart(2, "0")}`;

    // Range slider
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = lo;
    slider.max = hi;
    slider.step = (hi - lo) / 1000;
    slider.value = mean;
    slider.addEventListener("input", () => {
      values[i] = parseFloat(slider.value);
      runInference();
    });
    sliderEls.push(slider);

    row.appendChild(lockBtn);
    row.appendChild(label);
    row.appendChild(slider);
    sliderPanel.appendChild(row);
  }
}

function toggleLock(i) {
  locks[i] = !locks[i];
  lockBtns[i].classList.toggle("locked", locks[i]);
  lockBtns[i].textContent = locks[i] ? "●" : "○";
}

// Apply a vector (or zeros if null/short) to sliders and values[].
function applyVector(vec) {
  for (let i = 0; i < stats.latent_dim; i++) {
    const val = (vec && i < vec.length) ? vec[i] : 0.0;
    values[i] = val;
    sliderEls[i].value = val;
  }
}

// ── Button handlers ────────────────────────────────────────────────────────

// Box-Muller transform — sample from N(mean, std).
function randomNormal(mean, std) {
  const u = Math.random(), v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

document.getElementById("btn-random").addEventListener("click", () => {
  if (!stats) return;
  for (let i = 0; i < stats.latent_dim; i++) {
    if (locks[i]) continue;
    const val = randomNormal(stats.means[i], stats.stds[i]);
    values[i] = val;
    sliderEls[i].value = val;
  }
  runInference();
});

document.getElementById("btn-reset").addEventListener("click", () => {
  if (!stats) return;
  for (let i = 0; i < stats.latent_dim; i++) {
    if (locks[i]) continue;
    const val = (defaultVector && i < defaultVector.length) ? defaultVector[i] : 0.0;
    values[i] = val;
    sliderEls[i].value = val;
  }
  runInference();
});

document.getElementById("btn-toggle").addEventListener("click", () => {
  if (!stats) return;
  for (let i = 0; i < stats.latent_dim; i++) {
    toggleLock(i);
  }
});

// ── Model loading ──────────────────────────────────────────────────────────

async function loadModel(entry) {
  setStatus("loading model...");
  session = null;
  stats = null;

  try {
    // Load per-dimension stats (small JSON)
    const statsRes = await fetch(entry.stats);
    stats = await statsRes.json();

    // Build slider UI immediately so the user sees it while ONNX loads
    buildSliders(stats);
    setStatus("loading weights...");

    // Fetch model buffer and optional external data file in parallel
    const dataUrl = entry.onnx + ".data";
    const [onnxBuf, dataRes] = await Promise.all([
      fetch(entry.onnx).then(r => r.arrayBuffer()),
      fetch(dataUrl).catch(() => null),
    ]);

    const opts = {};
    if (dataRes && dataRes.ok) {
      const dataBuf = await dataRes.arrayBuffer();
      const dataFileName = entry.onnx.split("/").pop() + ".data";
      opts.externalData = [{ path: dataFileName, data: dataBuf }];
    }

    // Load per-model default vector if specified
    defaultVector = null;
    if (entry.init) {
      const initRes = await fetch(entry.init);
      if (initRes.ok) defaultVector = await initRes.json();
    }

    session = await ort.InferenceSession.create(onnxBuf, opts);

    setStatus("");
    applyVector(defaultVector);
    runInference();
  } catch (err) {
    setStatus(`error: ${err.message}`);
    console.error(err);
  }
}

// ── Manifest + dropdown ────────────────────────────────────────────────────

async function init() {
  let manifest;

  try {
    const res = await fetch(MANIFEST_URL);
    manifest = await res.json();
  } catch (err) {
    setStatus("could not load models.json");
    console.error(err);
    return;
  }

  if (manifest.length === 0) {
    modelSelect.innerHTML =
      '<option disabled selected>no models exported yet</option>';
    setStatus("run  python web/export_onnx.py  to export a model");
    return;
  }

  // Populate dropdown
  modelSelect.innerHTML = "";
  manifest.forEach((entry, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = entry.name;
    modelSelect.appendChild(opt);
  });

  modelSelect.addEventListener("change", () => {
    loadModel(manifest[parseInt(modelSelect.value)]);
  });

  // Auto-load the first model
  modelSelect.value = "0";
  loadModel(manifest[0]);
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

// ── Start ──────────────────────────────────────────────────────────────────

init();
