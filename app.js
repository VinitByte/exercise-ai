import { PoseLandmarker, FilesetResolver, DrawingUtils }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");

let landmarker = null, loadedKind = null, running = false, lastVideoTime = -1, stream = null;
let current = "squat", S, sm = {};

const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/";
const MODELS = {
  lite:  MODEL_BASE + "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full:  MODEL_BASE + "pose_landmarker_full/float16/1/pose_landmarker_full.task",
  heavy: MODEL_BASE + "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
};

/* ---------- helpers ---------- */
const W = () => canvas.width, H = () => canvas.height;
const pt = l => ({ x: l.x * W(), y: l.y * H(), v: l.visibility ?? 1 });
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

function ang(a, b, c) {
  const r = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let d = Math.abs(r * 180 / Math.PI);
  return d > 180 ? 360 - d : d;
}
function vertDeg(hip, sh) {
  return Math.abs(Math.atan2(sh.x - hip.x, hip.y - sh.y)) * 180 / Math.PI;
}
function smooth(key, v) {
  sm[key] = sm[key] == null ? v : sm[key] * 0.6 + v * 0.4;
  return sm[key];
}

function getPoints(lm) {
  const mk = ids => ({ sh: pt(lm[ids[0]]), el: pt(lm[ids[1]]), wr: pt(lm[ids[2]]),
                       hi: pt(lm[ids[3]]), kn: pt(lm[ids[4]]), an: pt(lm[ids[5]]) });
  const L = mk([11, 13, 15, 23, 25, 27]), R = mk([12, 14, 16, 24, 26, 28]);
  const vis = o => Object.values(o).reduce((s, p) => s + p.v, 0) / 6;
  const vl = vis(L), vr = vis(R);
  return { s: vl >= vr ? L : R, o: vl >= vr ? R : L,
           vis: Math.max(vl, vr), visBoth: Math.min(vl, vr) };
}

/* ---------- exercise definitions (rule-based form checks) ---------- */
const EX = {
  squat: {
    name: "Squat", type: "rep", start: 160, end: 100, mid: 130, minDur: 1.0, met: 5,
    depthMsg: "Go lower – thighs should be parallel to the floor",
    muscles: "Quadriceps, glutes, hamstrings, core, lower back",
    cam: "Place the camera at the SIDE (or 45°) of your body, full body visible, 2–3 m away.",
    analyze({ s }) {
      const a = smooth("a", ang(s.hi, s.kn, s.an));
      const lean = vertDeg(s.hi, s.sh), issues = [];
      if (lean > 60) issues.push("Keep your chest up (leaning too far forward)");
      return { ready: true, a, issues, info: `Knee ${a.toFixed(0)}° | Back lean ${lean.toFixed(0)}°` };
    }
  },
  pushup: {
    name: "Push-up", type: "rep", start: 155, end: 90, mid: 120, minDur: 0.8, met: 8,
    depthMsg: "Go lower – chest closer to the floor",
    muscles: "Chest, triceps, front shoulders, core",
    cam: "Place the camera at the SIDE of your body on the floor level, full body visible.",
    analyze({ s }) {
      const horizontal = Math.abs(s.sh.y - s.an.y) < 0.6 * Math.abs(s.sh.x - s.an.x);
      if (!horizontal) return { ready: false, msg: "Get into push-up position (side view)" };
      const a = smooth("a", ang(s.sh, s.el, s.wr));
      const body = smooth("b", ang(s.sh, s.hi, s.an)), issues = [];
      if (body < 160) issues.push("Keep your body straight (hips sagging / piking)");
      return { ready: true, a, issues, info: `Elbow ${a.toFixed(0)}° | Body line ${body.toFixed(0)}°` };
    }
  },
  lunge: {
    name: "Lunge", type: "rep", start: 150, end: 100, mid: 125, minDur: 1.0, met: 4, both: true,
    depthMsg: "Go lower – front knee about 90°",
    muscles: "Quadriceps, glutes, hamstrings, calves, core",
    cam: "Place the camera at the SIDE of your body, full body visible.",
    analyze({ s, o }) {
      const k1 = ang(s.hi, s.kn, s.an), k2 = ang(o.hi, o.kn, o.an);
      const a = smooth("a", Math.min(k1, k2)), issues = [];
      const lean = vertDeg(s.hi, s.sh);
      if (lean > 25) issues.push("Keep your torso upright");
      return { ready: true, a, issues, info: `Front knee ${a.toFixed(0)}° | Lean ${lean.toFixed(0)}°` };
    }
  },
  curl: {
    name: "Bicep Curl", type: "rep", start: 150, end: 50, mid: 100, minDur: 1.0, met: 3.5,
    depthMsg: "Curl all the way up",
    muscles: "Biceps, brachialis, forearms",
    cam: "Face the camera or stand slightly sideways, upper body visible.",
    analyze({ s }) {
      const a = smooth("a", ang(s.sh, s.el, s.wr)), issues = [];
      const swing = ang(s.el, s.sh, s.hi);
      if (swing > 35) issues.push("Keep your elbow close to your body");
      if (vertDeg(s.hi, s.sh) > 15) issues.push("Don't swing your back");
      return { ready: true, a, issues, info: `Elbow ${a.toFixed(0)}° | Arm swing ${swing.toFixed(0)}°` };
    }
  },
  plank: {
    name: "Plank", type: "plank", met: 3,
    muscles: "Core (abs), shoulders, glutes, lower back",
    cam: "Place the camera at the SIDE of your body, full body visible.",
    analyze({ s }) {
      const horizontal = Math.abs(s.sh.y - s.an.y) < 0.6 * Math.abs(s.sh.x - s.an.x);
      if (!horizontal) return { ready: false, msg: "Get into plank position (side view)" };
      const body = smooth("b", ang(s.sh, s.hi, s.an)), issues = [];
      if (body < 160) issues.push(s.hi.y > (s.sh.y + s.an.y) / 2
        ? "Hips are too low – lift them" : "Hips are too high – lower them");
      return { ready: true, issues, info: `Body line ${body.toFixed(0)}°` };
    }
  }
};

/* =====================================================================
   ON-DEVICE LEARNING AI  (k-nearest-neighbours on pose features)
   - You record "correct" and "wrong" examples.
   - The AI compares every new frame with your examples.
   - It is also used to guess which exercise you are doing.
   ===================================================================== */
const STORE = "exai_samples_v1", MAX_PER_LABEL = 400;
let samples = [], aiReady = false, rec = null, detCand = null, detN = 0;
try { samples = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch { samples = []; }

const SWAP = { 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27 };
const KEY_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

// pose -> numbers, centred on the hip and scaled by torso length.
// mirror = true gives the left/right flipped pose (so facing left or right both work)
function feat(lm, mirror) {
  const g = i => {
    const j = mirror ? (SWAP[i] ?? i) : i;
    return { x: (mirror ? -1 : 1) * lm[j].x * W(), y: lm[j].y * H() };
  };
  const hip = mid(g(23), g(24)), sh = mid(g(11), g(12));
  const sc = Math.hypot(sh.x - hip.x, sh.y - hip.y) || 1, f = [];
  for (const i of KEY_IDS) {
    const p = g(i);
    f.push(+((p.x - hip.x) / sc).toFixed(3), +((p.y - hip.y) / sc).toFixed(3));
  }
  return f;
}

function knn(f, filter, k = 7, group = l => l) {
  const near = [];
  for (const s of samples) {
    if (!filter(s.label)) continue;
    let d = 0;
    for (let i = 0; i < f.length; i++) { const t = f[i] - s.f[i]; d += t * t; }
    near.push([d, s.label]);
  }
  if (near.length < k) return null;
  near.sort((a, b) => a[0] - b[0]);
  const votes = {};
  near.slice(0, k).forEach(n => { const key = group(n[1]); votes[key] = (votes[key] || 0) + 1; });
  const [label, v] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  return { label, conf: v / k };
}

const countOf = l => samples.filter(s => s.label === l).length;

function saveSamples() {
  const seen = {};
  samples = samples.reverse().filter(s => (seen[s.label] = (seen[s.label] || 0) + 1) <= MAX_PER_LABEL).reverse();
  try { localStorage.setItem(STORE, JSON.stringify(samples)); } catch {}
  renderTeach();
}

function renderTeach() {
  const g = countOf(current + ":good"), b = countOf(current + ":bad");
  aiReady = g >= 40 && b >= 40;
  $("teachInfo").textContent = `${EX[current].name}: ${g} correct samples, ${b} wrong samples – ` +
    (aiReady ? "AI is ready ✅" : "record both (about 8 s each) to activate the AI");
  if (!aiReady) $("aiBox").textContent = "Not trained for " + EX[current].name + " yet";
  const names = new Set(samples.filter(s => countOf(s.label) >= 20).map(s => s.label.split(":")[0]));
  if (names.size < 2) $("detBox").textContent = "Train 2+ exercises to use this";
}

function startRecording(kind) {
  if (!running) { $("teachStatus").textContent = "Start the camera first."; return; }
  const t = performance.now();
  rec = { label: current + ":" + kind, startAt: t + 3000, endAt: t + 11000, last: 0, n: 0 };
}

function handleRecording(lm) {
  if (!rec) return;
  const t = performance.now();
  if (t < rec.startAt) {
    $("teachStatus").textContent = `Get ready… ${Math.ceil((rec.startAt - t) / 1000)}`;
  } else if (t < rec.endAt) {
    $("teachStatus").textContent = `🔴 Recording "${rec.label}" – ${Math.ceil((rec.endAt - t) / 1000)} s left, do the exercise`;
    if (t - rec.last > 100) {
      rec.last = t; rec.n++;
      samples.push({ label: rec.label, f: feat(lm, false) }, { label: rec.label, f: feat(lm, true) });
    }
  } else {
    $("teachStatus").textContent = `✅ Saved ${rec.n * 2} samples for "${rec.label}"`;
    rec = null; saveSamples();
  }
}

function switchExercise(name) {
  current = name; $("exercise").value = name;
  resetState(); renderTeach();
}

// guesses the exercise from your trained examples; switches after ~1 s of agreement
function autoDetectStep(lm) {
  const names = new Set(samples.map(s => s.label.split(":")[0]));
  if (names.size < 2) return false;
  const r = knn(feat(lm, false), () => true, 9, l => l.split(":")[0]);
  if (!r) return false;
  $("detBox").textContent = `${EX[r.label]?.name || r.label} (${Math.round(r.conf * 100)}% sure)`;
  if (!$("autoDet").checked || r.conf < 0.8 || r.label === current || !EX[r.label]) { detN = 0; return false; }
  if (r.label === detCand) detN++; else { detCand = r.label; detN = 1; }
  if (detN >= 25) { detN = 0; switchExercise(r.label); return true; }
  return false;
}

/* ---------- state ---------- */
function resetState() {
  S = { count: 0, bad: 0, phase: "start", partial: false, t0: null, errors: new Set(),
        repTimes: [], plankMs: 0, lastTs: performance.now(), startTs: performance.now(), lastSpoken: 0,
        errCount: {}, depthWarn: 0, aiN: 0, aiBad: 0 };
  sm = {};
  say(running ? "Counter reset" : "Choose an exercise and press Start.", "");
  render();
}

function say(msg, type) {
  const el = $("feedback");
  el.textContent = msg; el.className = "fb " + (type || "");
  if ($("voice").checked && type && performance.now() - S.lastSpoken > 1500) {
    S.lastSpoken = performance.now();
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(msg.replace(/[✅❌⚠]/g, "")));
  }
}

/* ---------- rep logic ---------- */
function repLogic(a, issues, cfg) {
  const now = performance.now();
  if (S.phase === "start") {
    if (a < cfg.start - 10) {
      if (!S.t0) { S.t0 = now; S.errors = new Set(); S.aiN = 0; S.aiBad = 0; }
      issues.forEach(i => S.errors.add(i));
      if (a < cfg.mid) S.partial = true;
    } else S.t0 = null;

    if (a < cfg.end) { S.phase = "end"; S.partial = false; }
    else if (a > cfg.start && S.partial) { S.partial = false; S.depthWarn++; say("⚠ " + cfg.depthMsg, "warn"); }
  } else {
    issues.forEach(i => S.errors.add(i));
    if (a > cfg.start) {
      S.phase = "start";
      const dur = (now - (S.t0 || now)) / 1000;
      if (dur < cfg.minDur) S.errors.add("Too fast – slow down and control the movement");
      // trained AI vote: reject the rep if most frames looked like your "wrong form" examples
      if ($("useAI").checked && aiReady && S.aiN >= 5 && S.aiBad / S.aiN > 0.5)
        S.errors.add("AI: your form looks like your 'wrong form' examples");
      if (S.errors.size === 0) { S.count++; S.repTimes.push(dur); say("✅ Good rep! (" + S.count + ")", "ok"); }
      else {
        S.bad++;
        S.errors.forEach(e => S.errCount[e] = (S.errCount[e] || 0) + 1);
        say("❌ Not counted: " + [...S.errors].join(" • "), "bad");
      }
      S.t0 = null;
    }
  }
}

/* ---------- main frame processing ---------- */
function processFrame(res) {
  ctx.clearRect(0, 0, W(), H());
  const cfg = EX[current], now = performance.now(), dt = now - S.lastTs; S.lastTs = now;

  if (!res.landmarks.length) { say("No person detected – step into the frame", "warn"); render(); return; }
  const lm = res.landmarks[0], P = getPoints(lm);

  handleRecording(lm);
  if (autoDetectStep(lm)) return;

  // trained AI verdict for the current frame
  let ai = null;
  if (aiReady && $("useAI").checked) {
    ai = knn(feat(lm, false), l => l.startsWith(current + ":"), 7);
    if (ai) $("aiBox").textContent =
      (ai.label.endsWith(":good") ? "✅ Looks correct" : "❌ Looks wrong") + ` (${Math.round(ai.conf * 100)}%)`;
  }
  const aiBad = ai && ai.label.endsWith(":bad");

  let color = "#22c55e", out;
  if (P.vis < 0.5 || (cfg.both && P.visBoth < 0.4)) {
    out = { ready: false, msg: "Full body not visible – move back / adjust the camera" };
  } else out = cfg.analyze(P);

  if (!out.ready) {
    color = "#f59e0b"; say(out.msg, "warn"); $("live").textContent = "";
  } else {
    if (cfg.type === "plank" && aiBad && ai.conf >= 0.7)
      out.issues.push("AI: this position matches your wrong-form examples");
    color = out.issues.length ? "#ef4444" : "#22c55e";
    $("live").textContent = out.issues.length ? "⚠ " + out.issues.join(" • ") : "👍 Form looks good";
    $("angle").textContent = out.info;
    if (cfg.type === "plank") {
      if (out.issues.length === 0) { S.plankMs += dt; say("✅ Perfect plank – hold it!", "ok"); }
      else {
        say("❌ " + out.issues[0] + " (timer paused)", "bad");
        out.issues.forEach(i => S.errCount[i] = (S.errCount[i] || 0) + dt / 1000);
      }
    } else {
      if (S.t0 && ai) { S.aiN++; if (aiBad) S.aiBad++; }
      repLogic(out.a, out.issues, cfg);
    }
  }

  const du = new DrawingUtils(ctx);
  du.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color, lineWidth: 4 });
  du.drawLandmarks(lm, { color: "#ffffff", radius: 3 });
  render();
}

function fmt(sec) {
  sec = Math.floor(sec);
  return String(Math.floor(sec / 60)).padStart(2, "0") + ":" + String(sec % 60).padStart(2, "0");
}

function render() {
  const cfg = EX[current], w = +$("weight").value || 70;
  const elapsed = (performance.now() - S.startTs) / 1000;
  const activeSec = cfg.type === "plank" ? S.plankMs / 1000 : elapsed;
  $("mainLabel").textContent = cfg.type === "plank" ? "SECONDS HELD (correct form)" : "CORRECT REPS";
  $("mainValue").textContent = cfg.type === "plank" ? (S.plankMs / 1000).toFixed(0) : S.count;
  $("bad").textContent = S.bad;
  const total = S.count + S.bad;
  $("acc").textContent = total ? Math.round(S.count / total * 100) + "%" : "--";
  $("avg").textContent = S.repTimes.length
    ? (S.repTimes.reduce((a, b) => a + b, 0) / S.repTimes.length).toFixed(1) + " s" : "--";
  $("time").textContent = running ? fmt(elapsed) : "00:00";
  $("kcal").textContent = running ? (cfg.met * 3.5 * w / 200 * (activeSec / 60)).toFixed(1) : "0";
  $("phase").textContent = cfg.type === "plank" ? "Hold" : (S.phase === "start" ? "Up / Start" : "Down / End");
  $("muscles").textContent = cfg.muscles;
  $("camTip").textContent = "📷 " + cfg.cam;
}

/* ---------- camera + pose AI model ---------- */
async function loadModel(kind) {
  $("loading").style.display = "flex";
  $("loading").textContent = `Loading AI model (${kind})…`;
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const make = d => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODELS[kind], delegate: d },
    runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5
  });
  let next;
  try { next = await make("GPU"); } catch { next = await make("CPU"); }
  landmarker?.close();
  landmarker = next; loadedKind = kind;
}

async function start() {
  try {
    if (!landmarker || loadedKind !== $("model").value) await loadModel($("model").value);
    $("loading").textContent = "Starting camera…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    $("loading").style.display = "none";
    running = true; resetState(); renderTeach();
    $("startBtn").textContent = "■ Stop";
    loop();
  } catch (e) {
    $("loading").style.display = "flex";
    $("loading").textContent = "Error: " + e.message + " (allow camera permission & use HTTPS)";
  }
}

function stop() {
  running = false; rec = null;
  stream?.getTracks().forEach(t => t.stop());
  $("startBtn").textContent = "▶ Start Camera";
  $("loading").style.display = "flex";
  $("loading").textContent = "Stopped. Press Start Camera.";
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    processFrame(landmarker.detectForVideo(video, performance.now()));
  }
  requestAnimationFrame(loop);
}

/* ---------- Free coach report (runs fully in the browser, no API, no server) ---------- */
const FIXES = [
  ["chest up", "Look slightly forward, brace your core and push your chest out."],
  ["body straight", "Squeeze your glutes and abs so shoulders, hips and ankles form one line."],
  ["torso upright", "Take a shorter step and keep your chest tall."],
  ["elbow close", "Pin your elbow to your side; use a lighter weight if it drifts."],
  ["swing your back", "Slow down and keep your back still; a lighter weight helps."],
  ["too low", "Squeeze your glutes and abs to lift your hips into line."],
  ["too high", "Lower your hips until your body is one straight line."],
  ["Too fast", "Take about 2 seconds going down and 1 second coming up."],
  ["AI:", "Watch your 'correct form' position and copy it; re-record examples if needed."]
];

function askCoach() {
  const cfg = EX[current], total = S.count + S.bad, plank = cfg.type === "plank";
  if (!total && !S.plankMs) { $("coachOut").textContent = "Do some reps first, then press the button."; return; }
  const mins = Math.max(1, Math.round((performance.now() - S.startTs) / 60000));
  const acc = total ? Math.round(S.count / total * 100) : null;
  const lines = [];

  lines.push(plank
    ? `Summary: you held a correct ${cfg.name} for ${Math.round(S.plankMs / 1000)} s in about ${mins} min.`
    : `Summary: ${S.count} correct and ${S.bad} wrong ${cfg.name} reps in about ${mins} min (${acc}% form accuracy).`);

  if (acc !== null) lines.push(acc >= 85 ? "Great form. You can add reps or slow the tempo."
    : acc >= 60 ? "Decent, but fix the mistakes below before adding more reps."
    : "Focus on quality first: do fewer, slower reps with the right form.");

  const top = Object.entries(S.errCount).sort((a, b) => b[1] - a[1]).slice(0, 2);
  if (top.length) {
    lines.push("", "Top problems and fixes:");
    top.forEach(([issue, n], i) => {
      const fix = (FIXES.find(f => issue.includes(f[0])) || [0, "Slow down and focus on control."])[1];
      lines.push(`${i + 1}) ${issue} (${Math.round(n)} ${plank ? "s" : "reps"}). Fix: ${fix}`);
    });
  }
  if (S.depthWarn) lines.push(`Depth: ${S.depthWarn} rep(s) were too shallow. ${cfg.depthMsg}.`);
  if (S.repTimes.length) {
    const avg = S.repTimes.reduce((a, b) => a + b, 0) / S.repTimes.length;
    lines.push(`Tempo: average ${avg.toFixed(1)} s per rep` + (avg < 1.5 ? " (a bit fast, slow down)." : "."));
  }
  lines.push("", `Muscles worked: ${cfg.muscles}.`,
    "Next session: " + (acc !== null && acc >= 85 ? "add 2–3 reps or a small weight." : "repeat the same reps and aim for fewer red messages."),
    "Note: general guidance, not medical advice. Stop if you feel pain.");
  $("coachOut").textContent = lines.join("\n");
}

/* ---------- events ---------- */
$("startBtn").onclick = () => running ? stop() : start();
$("resetBtn").onclick = resetState;
$("exercise").onchange = e => { current = e.target.value; resetState(); renderTeach(); };
$("model").onchange = async () => {
  if (!running) return;
  try { await loadModel($("model").value); } catch (e) { alert("Model error: " + e.message); }
  $("loading").style.display = "none";
};
$("recGood").onclick = () => startRecording("good");
$("recBad").onclick = () => startRecording("bad");
$("clearAI").onclick = () => {
  if (confirm("Delete the trained examples for " + EX[current].name + "?")) {
    samples = samples.filter(s => !s.label.startsWith(current + ":")); saveSamples();
  }
};
$("exportAI").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(samples)], { type: "application/json" }));
  a.download = "exercise-ai-training.json"; a.click();
};
$("importAI").onchange = async e => {
  try {
    const data = JSON.parse(await e.target.files[0].text());
    if (Array.isArray(data) && data.every(s => s.label && Array.isArray(s.f) && s.f.length === 26)) {
      samples = samples.concat(data); saveSamples();
      $("teachStatus").textContent = "✅ Imported " + data.length + " samples";
    } else throw new Error("bad file");
  } catch { $("teachStatus").textContent = "That file is not a valid training file."; }
};
$("coachBtn").onclick = askCoach;

resetState();
renderTeach();
