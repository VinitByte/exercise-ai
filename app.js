import { PoseLandmarker, FilesetResolver, DrawingUtils }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");

let landmarker = null, running = false, lastVideoTime = -1, stream = null;
let current = "squat", S, sm = {};

/* ---------- helpers ---------- */
const W = () => canvas.width, H = () => canvas.height;
const pt = l => ({ x: l.x * W(), y: l.y * H(), v: l.visibility ?? 1 });

function ang(a, b, c) {
  const r = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let d = Math.abs(r * 180 / Math.PI);
  return d > 180 ? 360 - d : d;
}
// angle of the line hip->shoulder from vertical (0 = upright)
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

/* ---------- exercise definitions ---------- */
// angles go from "start" (big angle) down to "end" (small angle) and back
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

/* ---------- state ---------- */
function resetState() {
  S = { count: 0, bad: 0, phase: "start", partial: false, t0: null, errors: new Set(),
        repTimes: [], plankMs: 0, lastTs: performance.now(), startTs: performance.now(), lastSpoken: 0 };
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
    speechSynthesis.speak(new SpeechSynthesisUtterance(msg.replace(/[✅❌]/g, "")));
  }
}

/* ---------- rep logic ---------- */
function repLogic(a, issues, cfg) {
  const now = performance.now();
  if (S.phase === "start") {
    if (a < cfg.start - 10) {
      if (!S.t0) { S.t0 = now; S.errors = new Set(); }
      issues.forEach(i => S.errors.add(i));
      if (a < cfg.mid) S.partial = true;
    } else S.t0 = null;

    if (a < cfg.end) { S.phase = "end"; S.partial = false; }
    else if (a > cfg.start && S.partial) { S.partial = false; say("⚠ " + cfg.depthMsg, "warn"); }
  } else {
    issues.forEach(i => S.errors.add(i));
    if (a > cfg.start) {
      S.phase = "start";
      const dur = (now - (S.t0 || now)) / 1000;
      if (dur < cfg.minDur) S.errors.add("Too fast – slow down and control the movement");
      if (S.errors.size === 0) { S.count++; S.repTimes.push(dur); say("✅ Good rep! (" + S.count + ")", "ok"); }
      else { S.bad++; say("❌ Not counted: " + [...S.errors].join(" • "), "bad"); }
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
  let color = "#22c55e", out;

  if (P.vis < 0.5 || (cfg.both && P.visBoth < 0.4)) {
    out = { ready: false, msg: "Full body not visible – move back / adjust the camera" };
  } else out = cfg.analyze(P);

  if (!out.ready) {
    color = "#f59e0b"; say(out.msg, "warn"); $("live").textContent = "";
  } else {
    color = out.issues.length ? "#ef4444" : "#22c55e";
    $("live").textContent = out.issues.length ? "⚠ " + out.issues.join(" • ") : "👍 Form looks good";
    $("angle").textContent = out.info;
    if (cfg.type === "plank") {
      if (out.issues.length === 0) { S.plankMs += dt; say("✅ Perfect plank – hold it!", "ok"); }
      else say("❌ " + out.issues[0] + " (timer paused)", "bad");
    } else repLogic(out.a, out.issues, cfg);
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

/* ---------- camera + model ---------- */
async function loadModel() {
  $("loading").textContent = "Loading AI model…";
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const opts = d => ({
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: d },
    runningMode: "VIDEO", numPoses: 1
  });
  try { landmarker = await PoseLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { landmarker = await PoseLandmarker.createFromOptions(fileset, opts("CPU")); }
}

async function start() {
  try {
    if (!landmarker) await loadModel();
    $("loading").textContent = "Starting camera…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    $("loading").style.display = "none";
    running = true; resetState();
    $("startBtn").textContent = "■ Stop";
    loop();
  } catch (e) {
    $("loading").style.display = "flex";
    $("loading").textContent = "Error: " + e.message + " (allow camera permission & use HTTPS)";
  }
}

function stop() {
  running = false;
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

/* ---------- events ---------- */
$("startBtn").onclick = () => running ? stop() : start();
$("resetBtn").onclick = resetState;
$("exercise").onchange = e => { current = e.target.value; resetState(); };
resetState();
