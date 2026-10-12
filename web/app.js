import { FilesetResolver, HandLandmarker, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { T, FEAT_DIM, frameVector, resample } from "./features.js";

const $ = (id) => document.getElementById(id);
const vid = $("vid"), cv = $("cv"), ctx = cv.getContext("2d");
const MIN_CONF = 0.75;          // below this we never speak — ask for a human interpreter instead
const FPS_WINDOW = 60;          // ~2 s buffer

let hands, pose, session, labels = [], vocab = {}, running = false;
let buffer = [], active = [], quietFrames = 0, recording = null, dataset = { signer: "H001", samples: [] };

// ---------- vocabulary ----------
const vocabList = await fetch("vocab.json").then(r => r.json()).catch(() => []);
for (const v of vocabList) vocab[v.id] = v;
$("sign").innerHTML = vocabList.map(v => `<option value="${v.id}">${v.ar} — ${v.en}</option>`).join("");
renderCounts();

// ---------- model ----------
async function loadModel() {
  try {
    const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/esm/ort.min.js");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    session = await ort.InferenceSession.create("model.onnx");
    labels = await (await fetch("labels.json")).json();
    window.__ort = ort;
    $("badge").textContent = `النموذج جاهز · ${labels.length} إشارة`;
  } catch (e) { $("badge").textContent = "لا يوجد نموذج بعد — سجّل بيانات ثم درّب"; }
}

async function classify(frames) {
  if (!session || frames.length < 8) return null;
  const ort = window.__ort;
  const x = new ort.Tensor("float32", resample(frames), [1, T, FEAT_DIM]);
  const logits = (await session.run({ x })).logits.data;
  const m = Math.max(...logits), ex = Array.from(logits, v => Math.exp(v - m)), s = ex.reduce((a, b) => a + b);
  let best = 0; ex.forEach((v, i) => { if (v > ex[best]) best = i; });
  return { id: labels[best], p: ex[best] / s };
}

// ---------- output ----------
async function show(res) {
  if (!res) return;
  const v = vocab[res.id] || { ar: res.id, en: res.id };
  $("conf").style.width = Math.round(res.p * 100) + "%";
  $("conf").style.background = res.p >= MIN_CONF ? "var(--ok)" : "var(--bad)";
  if (res.p < MIN_CONF) {
    $("out").textContent = "غير متأكد…"; $("outEn").textContent = `low confidence (${Math.round(res.p * 100)}%) — please repeat or call an interpreter`;
    return;
  }
  $("out").textContent = v.ar; $("outEn").textContent = `${v.en} · ${Math.round(res.p * 100)}%`;
  const h = document.createElement("div");
  h.textContent = `${new Date().toLocaleTimeString("ar-SA")} — ${v.ar} (${Math.round(res.p * 100)}%)`;
  $("hist").prepend(h);
  if ($("speak").checked) speak(v.ar);
}

async function speak(text) {
  try {
    const r = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (!r.ok) throw 0;
    new Audio(URL.createObjectURL(await r.blob())).play();
  } catch {
    const u = new SpeechSynthesisUtterance(text); u.lang = "ar-SA"; speechSynthesis.speak(u);
  }
}

// ---------- camera loop ----------
async function start() {
  $("start").disabled = true; $("status").textContent = "تحميل نماذج MediaPipe…";
  const files = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  hands = await HandLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO", numHands: 2 });
  pose = await PoseLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", delegate: "GPU" },
    runningMode: "VIDEO" });
  vid.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  await vid.play();
  cv.width = vid.videoWidth; cv.height = vid.videoHeight;
  running = true; $("status").textContent = "الكاميرا تعمل"; loop();
}

function draw(list, color) {
  ctx.fillStyle = color;
  for (const lm of list) for (const p of lm) { ctx.beginPath(); ctx.arc(p.x * cv.width, p.y * cv.height, 3, 0, 7); ctx.fill(); }
}

async function loop() {
  if (!running) return;
  const t = performance.now();
  const hr = hands.detectForVideo(vid, t), pr = pose.detectForVideo(vid, t);
  let left = null, right = null;
  hr.landmarks.forEach((lm, i) => { (hr.handednesses[i][0].categoryName === "Left" ? (left = lm) : (right = lm)); });
  const pl = pr.landmarks[0] || null;
  ctx.clearRect(0, 0, cv.width, cv.height);
  draw(hr.landmarks, "#c9a24b"); if (pl) draw([pl.slice(11, 17)], "#3fbf7f");
  const f = frameVector(left, right, pl);
  buffer.push(f); if (buffer.length > FPS_WINDOW) buffer.shift();
  if (recording) recording.frames.push(f);

  // automatic segmentation: a sign = hands visible, then hands dropped for ~8 frames
  const handsUp = !!(left || right);
  if ($("auto").checked && !recording) {
    if (handsUp) { active.push(f); quietFrames = 0; }
    else if (active.length) {
      if (++quietFrames >= 8) { const seg = active; active = []; quietFrames = 0; if (seg.length >= 12) show(await classify(seg)); }
    }
    if (active.length > 120) active = active.slice(-120);
  }
  requestAnimationFrame(loop);
}

// ---------- dataset recording ----------
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
async function recordBatch() {
  if (!running) return alert("شغّل الكاميرا أولًا");
  const label = $("sign").value; dataset.signer = $("signer").value || "H001";
  $("recBtn").disabled = true;
  for (let k = 0; k < 10; k++) {
    for (let c = 3; c > 0; c--) { $("recStatus").textContent = `${vocab[label].ar}: العينة ${k + 1}/10 بعد ${c}…`; await sleep(330); }
    recording = { label, signer: dataset.signer, frames: [] };
    $("recStatus").textContent = `🔴 أدِّ الإشارة الآن (${k + 1}/10)`;
    await sleep(2000);
    const s = recording; recording = null;
    if (s.frames.some(fr => fr[FEAT_DIM - 1] || fr[FEAT_DIM - 2])) dataset.samples.push(s);
    $("recStatus").textContent = "استراحة — أنزل يديك"; await sleep(1000);
  }
  $("recBtn").disabled = false; $("recStatus").textContent = "تم. صدّر الملف بعد إكمال الإشارات."; renderCounts();
}
function renderCounts() {
  const c = {}; for (const s of dataset.samples) c[s.label] = (c[s.label] || 0) + 1;
  $("counts").innerHTML = vocabList.map(v => `<div>${v.ar}: <b>${c[v.id] || 0}</b></div>`).join("");
}
function exportData() {
  const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `ssl_${dataset.signer}_${Date.now()}.json`; a.click();
}

// ---------- UI ----------
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  $("tab-rec").classList.toggle("hidden", b.dataset.tab !== "rec");
  $("tab-data").classList.toggle("hidden", b.dataset.tab !== "data");
});
$("start").onclick = () => start().catch(e => { $("status").textContent = "خطأ: " + e.message; $("start").disabled = false; });
$("now").onclick = async () => show(await classify(buffer));
$("ok").onclick = () => { $("hist").prepend(Object.assign(document.createElement("div"), { textContent: "✔ أكّد الطاقم الترجمة" })); };
$("bad").onclick = () => { $("out").textContent = "يُرجى استدعاء مترجم لغة إشارة"; $("outEn").textContent = "Please call a sign-language interpreter"; speak("يُرجى استدعاء مترجم لغة إشارة"); };
$("recBtn").onclick = recordBatch;
$("export").onclick = exportData;
loadModel();
