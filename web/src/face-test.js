// Face test bench: sliders, scripted demo, and a deterministic QA mode driven by URL params.
//   /face-test.html                          interactive
//   /face-test.html?qa=1&joy=0.9&view=front  deterministic still (no idle randomness), sets window.__ready
//   /face-test.html?seq=1                    live idle (used for the frame-sequence capture)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { AndroidFace, EMOTIONS } from './face/face.js';
import { createFaceStage } from './face/stage.js';

const params = new URLSearchParams(location.search);
const QA = params.has('qa');
if (QA) document.body.classList.add('qa');

const stageEl = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: QA });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = Number(params.get('exposure') || 1.0);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stageEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(24, window.innerWidth / window.innerHeight, 0.05, 20);

const face = new AndroidFace(scene, '/face/android.glb', { camera, seed: Number(params.get('seed') || 7) });
const stage = createFaceStage(renderer, scene, { target: face.root });

// camera framing: face anchor (between the eyes) near the upper third
const VIEWS = {
  front: { az: 0, el: 2, dist: 0.95 },
  three: { az: 32, el: 6, dist: 0.95 },
  side: { az: 80, el: 3, dist: 1.0 },
  close: { az: 0, el: 2, dist: 0.6 },
  low: { az: -12, el: -8, dist: 0.9 },
  back: { az: 150, el: 15, dist: 1.1 },
};
const lookAt = new THREE.Vector3(0, 0.005, 0.03);
function setView(name) {
  const v = VIEWS[name] || VIEWS.front;
  const az = v.az * Math.PI / 180, el = v.el * Math.PI / 180;
  camera.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)).multiplyScalar(v.dist).add(lookAt);
  camera.lookAt(lookAt);
}
setView(params.get('view') || 'front');

const controls = QA ? null : new OrbitControls(camera, renderer.domElement);
if (controls) { controls.target.copy(lookAt); controls.enableDamping = true; controls.minDistance = 0.3; controls.maxDistance = 2.5; }

// subtle bloom so the iris ring and seams glow
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.5, 0.82);
composer.addPass(bloom);
composer.addPass(new OutputPass());

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- UI
const state = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
let intensity = 1;
const sliders = {};
const sl = document.getElementById('sliders');
for (const e of EMOTIONS) {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<label>${e}</label><input type="range" min="0" max="1" step="0.01" value="0"><output>0.00</output>`;
  const inp = row.querySelector('input'), out = row.querySelector('output');
  inp.addEventListener('input', () => { state[e] = +inp.value; out.textContent = (+inp.value).toFixed(2); push(); });
  sliders[e] = { inp, out };
  sl.appendChild(row);
}
function setSliders(v) {
  for (const e of EMOTIONS) { state[e] = v[e] ?? 0; sliders[e].inp.value = state[e]; sliders[e].out.textContent = state[e].toFixed(2); }
  push();
}
const intEl = document.getElementById('intensity');
intEl.addEventListener('input', () => { intensity = +intEl.value; intEl.nextElementSibling.textContent = intensity.toFixed(2); push(); });
document.getElementById('iris').addEventListener('input', (ev) => face.setIrisColor(ev.target.value));
document.getElementById('writing').addEventListener('change', (ev) => face.setActivity({ writing: ev.target.checked }));
let gazeMode = 'camera';
const mouse = { x: 0, y: 0 };
addEventListener('pointermove', (ev) => { mouse.x = (ev.clientX / innerWidth) * 2 - 1; mouse.y = -(ev.clientY / innerHeight) * 2 + 1; });
document.getElementById('gaze').addEventListener('change', (ev) => { gazeMode = ev.target.value; });
function push() { face.setEmotion(state, { intensity }); }

const PRESETS = {
  'subtle joy': { joy: 0.3 }, 'broad joy': { joy: 1 }, sadness: { sadness: 0.8 }, fear: { fear: 0.85 },
  anger: { anger: 0.8 }, surprise: { surprise: 0.9 }, disgust: { disgust: 0.8 }, curiosity: { curiosity: 0.8 }, calm: { calm: 0.9 },
};
const pr = document.getElementById('presets');
for (const [k, v] of Object.entries(PRESETS)) {
  const b = document.createElement('button');
  b.textContent = k;
  b.onclick = () => { stopDemo(); setSliders(v); };
  pr.appendChild(b);
}
document.getElementById('reset').onclick = () => { stopDemo(); setSliders({}); };

// scripted demo: neutral -> subtle joy -> broad joy -> sadness -> fear -> anger -> curiosity -> calm
const DEMO = [
  ['neutral', {}, 3.0],
  ['subtle joy', { joy: 0.3 }, 3.5],
  ['broad joy', { joy: 0.95 }, 3.5],
  ['sadness', { sadness: 0.8 }, 4.5],
  ['fear', { fear: 0.85 }, 3.5],
  ['anger', { anger: 0.85 }, 4.0],
  ['curiosity', { curiosity: 0.8 }, 4.0],
  ['calm', { calm: 0.9 }, 5.0],
];
let demo = null;
const statusEl = document.getElementById('status');
function stopDemo() { if (demo) { clearTimeout(demo); demo = null; document.getElementById('demo').classList.remove('on'); statusEl.textContent = ''; } }
document.getElementById('demo').onclick = () => {
  if (demo) { stopDemo(); return; }
  document.getElementById('demo').classList.add('on');
  let i = 0;
  const step = () => {
    const [name, v, dur] = DEMO[i % DEMO.length];
    setSliders(v);
    statusEl.textContent = `demo: ${name}`;
    i++;
    demo = setTimeout(i < DEMO.length ? step : () => { stopDemo(); setSliders({}); }, dur * 1000);
  };
  step();
};

// ---------------------------------------------------------------- QA mode
await face.ready;
window.face = face;
window.scene = scene;
window.stage = stage;
if (QA) {
  const v = {};
  for (const e of EMOTIONS) if (params.has(e)) v[e] = +params.get(e);
  face.setOptions({ idle: false, blinks: false, micro: false, breathing: false });
  face.setEmotion(v, { intensity: Number(params.get('intensity') || 1) });
  if (params.has('iris')) face.setIrisColor('#' + params.get('iris'));
  if (params.has('writing')) face.setActivity({ writing: true });
  if (params.has('gx')) face.setGaze({ x: +params.get('gx'), y: +params.get('gy') });
  if (params.has('morph')) {
    // raw morph inspection: ?morph=jawOpen:1,mouthSmileLeft:0.5
    face.settle(0.1);
    const kv = params.get('morph').split(',').map((s) => s.split(':'));
    face._forced = Object.fromEntries(kv.map(([k, x]) => [k, +x]));
  }
  face.settle(Number(params.get('settle') || 3));
}
if (params.has('seq')) face.setEmotion({}, {});

const clock = new THREE.Clock();
let t = 0;
function frame() {
  const dt = clock.getDelta();
  t += dt;
  if (!QA) {
    if (gazeMode === 'mouse') face.setGaze({ x: mouse.x, y: mouse.y });
    else if (gazeMode === 'text') face.setGaze({ x: 0, y: -0.9 });
    else face.setGaze('camera');
    face.update(dt, t);
  } else if (face._forced) {
    for (const m of face.morphMeshes) for (const [name, i] of m.userData.map) m.morphTargetInfluences[i] = face._forced[name] || 0;
  }
  controls?.update();
  composer.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
setTimeout(() => { window.__ready = true; }, QA ? 300 : 1000);
