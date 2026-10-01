// BEHIND THE MASK -- porcelain android face runtime (three.js r18x).
//
//   import { AndroidFace } from './face/face.js';
//   const face = new AndroidFace(scene, '/face/android.glb', { camera });
//   await face.ready;
//   face.setEmotion({ joy: 0.6, curiosity: 0.2 }, { intensity: 1 });
//   face.setGaze('camera');                 // or a THREE.Vector3 (world) or {x, y} (NDC, -1..1)
//   face.setActivity({ writing: true });
//   face.setIrisColor('#7fe7ff');
//   // every frame:
//   face.update(dt, time);
//
// Expression pipeline:  emotion vector -> FACS action units (EMFACS-style prototypes, additive with
// soft caps) -> ARKit blendshape channels -> per-region onset/offset springs -> morph targets.
// Life layer: micro-expressions, blinks, saccades + fixation jitter, breathing, head noise / posture,
// silent mouthing while writing, per-side asymmetry. Everything random comes from a seeded PRNG.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

export const EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'calm', 'curiosity', 'surprise', 'disgust'];

// ------------------------------------------------------------------------------------------------
// small utilities
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Noise1D {
  // 1D gradient (Perlin) noise, seeded, output roughly in [-1, 1]
  constructor(rand) {
    this.g = new Float32Array(256);
    for (let i = 0; i < 256; i++) this.g[i] = rand() * 2 - 1;
  }
  at(x) {
    const i = Math.floor(x);
    const f = x - i;
    const a = this.g[i & 255] * f;
    const b = this.g[(i + 1) & 255] * (f - 1);
    const u = f * f * f * (f * (f * 6 - 15) + 10);
    return 2 * (a + (b - a) * u);
  }
  fbm(x, oct = 3) {
    let s = 0, amp = 1, norm = 0, fr = 1;
    for (let o = 0; o < oct; o++) { s += amp * this.at(x * fr + o * 19.7); norm += amp; amp *= 0.5; fr *= 2.03; }
    return s / norm;
  }
}

const clamp = (x, a = 0, b = 1) => (x < a ? a : x > b ? b : x);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const DEG = Math.PI / 180;

// exact critically-damped spring step
function spring(s, target, omega, dt) {
  const x0 = s.x - target;
  const v0 = s.v;
  const e = Math.exp(-omega * dt);
  const k = (v0 + omega * x0) * dt;
  s.x = target + (x0 + k) * e;
  s.v = (v0 - omega * k) * e;
}

// ------------------------------------------------------------------------------------------------
// Channels: ARKit names used by the asset (+ the split ICT channels). Each belongs to a region with
// its own onset/offset rise times (seconds, 10-90%). Brows and lids are fast, mouth slower.
const REGION_TIMES = {
  brow: [0.16, 0.42],
  lid: [0.12, 0.30],
  cheek: [0.42, 0.85],
  nose: [0.30, 0.65],
  mouth: [0.50, 0.95],
  jaw: [0.38, 0.70],
  look: [0.05, 0.05],
};
const CHANNELS = {
  browDownLeft: 'brow', browDownRight: 'brow', browInnerUpLeft: 'brow', browInnerUpRight: 'brow',
  browOuterUpLeft: 'brow', browOuterUpRight: 'brow',
  eyeBlinkLeft: 'lid', eyeBlinkRight: 'lid', eyeSquintLeft: 'lid', eyeSquintRight: 'lid', eyeWideLeft: 'lid', eyeWideRight: 'lid',
  cheekSquintLeft: 'cheek', cheekSquintRight: 'cheek', cheekRaiserLeft: 'cheek', cheekRaiserRight: 'cheek',
  cheekPuffLeft: 'cheek', cheekPuffRight: 'cheek',
  noseSneerLeft: 'nose', noseSneerRight: 'nose',
  mouthSmileLeft: 'mouth', mouthSmileRight: 'mouth', mouthFrownLeft: 'mouth', mouthFrownRight: 'mouth',
  mouthDimpleLeft: 'mouth', mouthDimpleRight: 'mouth', mouthStretchLeft: 'mouth', mouthStretchRight: 'mouth',
  mouthPressLeft: 'mouth', mouthPressRight: 'mouth', mouthLowerDownLeft: 'mouth', mouthLowerDownRight: 'mouth',
  mouthUpperUpLeft: 'mouth', mouthUpperUpRight: 'mouth', mouthPucker: 'mouth', mouthFunnel: 'mouth',
  mouthRollLower: 'mouth', mouthRollUpper: 'mouth', mouthShrugLower: 'mouth', mouthShrugUpper: 'mouth',
  mouthClose: 'mouth', mouthLeft: 'mouth', mouthRight: 'mouth',
  jawOpen: 'jaw', jawForward: 'jaw', jawLeft: 'jaw', jawRight: 'jaw',
};
// soft caps per channel (after additive mixing)
const CAPS = { jawOpen: 0.5, eyeBlinkLeft: 1, eyeBlinkRight: 1, eyeWideLeft: 0.9, eyeWideRight: 0.9, mouthSmileLeft: 0.95, mouthSmileRight: 0.95, mouthStretchLeft: 0.8, mouthStretchRight: 0.8 };

// Emotion prototypes (EMFACS). Each term: [channel (without side = both sides), weight, lo, hi]
// contributes weight * smoothstep(lo, hi, e) -- lo/hi let some AUs engage only at higher intensity.
// "L"/"R" suffix = one side only (asymmetric prototypes).
const PROTOS = {
  joy: [ // AU6 + AU12 (+ AU25/26, AU10 for broad joy)
    ['mouthSmile', 0.62, 0.0, 0.75],
    ['mouthSmile', 0.28, 0.5, 1.0],
    ['cheekSquint', 0.55, 0.05, 0.85],      // AU6 -- Duchenne component (scaled by `duchenne`)
    ['cheekRaiser', 0.25, 0.3, 1.0],
    ['eyeSquint', 0.30, 0.1, 0.9],
    ['mouthDimple', 0.10, 0.0, 0.6],
    ['browOuterUp', 0.06, 0.3, 1.0],
    ['mouthUpperUp', 0.22, 0.45, 1.0],
    ['mouthLowerDown', 0.16, 0.55, 1.0],
    ['jawOpen', 0.14, 0.55, 1.0],
  ],
  sadness: [ // AU1 + AU4 + AU15 (+ AU17), lids lowered
    ['browInnerUp', 0.75, 0.0, 0.8],
    ['browDown', 0.22, 0.0, 0.9],
    ['mouthFrown', 0.62, 0.0, 0.9],
    ['mouthLowerDown', 0.12, 0.3, 1.0],
    ['mouthShrugLower', 0.35, 0.2, 1.0],
    ['mouthPress', 0.10, 0.0, 1.0],
    ['eyeBlink', 0.24, 0.0, 1.0],
    ['eyeSquint', 0.10, 0.3, 1.0],
    ['cheekRaiser', 0.08, 0.5, 1.0],
  ],
  anger: [ // AU4 + AU5 + AU7 + AU23/24 (+ AU9 slight, AU17)
    ['browDown', 0.85, 0.0, 0.8],
    ['eyeWide', 0.34, 0.1, 0.9],
    ['eyeSquint', 0.55, 0.0, 0.8],
    ['mouthPress', 0.55, 0.0, 0.8],
    ['mouthRollLower', 0.12, 0.2, 1.0],
    ['mouthRollUpper', 0.10, 0.2, 1.0],
    ['mouthShrugLower', 0.22, 0.1, 1.0],
    ['noseSneer', 0.22, 0.2, 1.0],
    ['cheekSquint', 0.10, 0.3, 1.0],
    ['mouthUpperUp', 0.10, 0.6, 1.0],
  ],
  fear: [ // AU1 + AU2 + AU4 + AU5 + AU20 + AU26
    ['browInnerUp', 0.80, 0.0, 0.8],
    ['browOuterUp', 0.42, 0.0, 0.9],
    ['browDown', 0.26, 0.0, 0.9],
    ['eyeWide', 0.80, 0.0, 0.8],
    ['mouthStretch', 0.55, 0.1, 1.0],
    ['mouthLowerDown', 0.14, 0.3, 1.0],
    ['jawOpen', 0.12, 0.2, 1.0],
    ['mouthPress', 0.08, 0.0, 0.5],
  ],
  surprise: [ // AU1 + AU2 + AU5 + AU26
    ['browInnerUp', 0.70, 0.0, 0.7],
    ['browOuterUp', 0.85, 0.0, 0.8],
    ['eyeWide', 0.70, 0.0, 0.8],
    ['jawOpen', 0.30, 0.1, 1.0],
    ['mouthLowerDown', 0.10, 0.4, 1.0],
    ['mouthFunnel', 0.08, 0.3, 1.0],
  ],
  disgust: [ // AU9 + AU10 + AU15 (+ AU4, AU7, AU17)
    ['noseSneer', 0.62, 0.0, 0.8],
    ['mouthUpperUp', 0.45, 0.0, 0.9],
    ['mouthShrugUpper', 0.25, 0.1, 1.0],
    ['mouthFrown', 0.32, 0.1, 1.0],
    ['mouthShrugLower', 0.22, 0.2, 1.0],
    ['browDown', 0.32, 0.0, 0.9],
    ['eyeSquint', 0.32, 0.0, 0.9],
    ['cheekSquint', 0.18, 0.2, 1.0],
    ['noseSneerL', 0.10, 0.2, 1.0],        // disgust is often a little unilateral
    ['mouthUpperUpL', 0.08, 0.2, 1.0],
  ],
  calm: [ // relaxed lids, soft slight smile
    ['eyeBlink', 0.13, 0.0, 1.0],
    ['eyeSquint', 0.06, 0.0, 1.0],
    ['mouthSmile', 0.13, 0.0, 1.0],
    ['cheekSquint', 0.05, 0.2, 1.0],
    ['browDown', 0.03, 0.0, 1.0],
    ['mouthPress', 0.04, 0.0, 1.0],
  ],
  curiosity: [ // asymmetric brow raise, eyes wider, lips parted slightly (+ head tilt, see POSTURE)
    ['browOuterUpL', 0.46, 0.0, 0.8],
    ['browOuterUpR', 0.14, 0.0, 0.8],
    ['browInnerUp', 0.30, 0.0, 0.8],
    ['browDownR', 0.06, 0.2, 1.0],
    ['eyeWide', 0.25, 0.0, 0.8],
    ['jawOpen', 0.05, 0.1, 1.0],
    ['mouthLowerDown', 0.07, 0.1, 1.0],
    ['mouthSmile', 0.06, 0.2, 1.0],
  ],
};

// head posture per emotion: pitch (+ = chin up), yaw, roll (degrees), z lean (m, + = toward camera)
const POSTURE = {
  joy: { pitch: 2.5, yaw: 0, roll: 1.5, z: 0.002 },
  sadness: { pitch: -7.5, yaw: 0, roll: -1.5, z: 0.004 },
  anger: { pitch: -4.5, yaw: 0, roll: 0, z: 0.007 },
  fear: { pitch: 1.5, yaw: 0, roll: 0, z: -0.012 },
  surprise: { pitch: 3.0, yaw: 0, roll: 0, z: -0.008 },
  disgust: { pitch: 1.5, yaw: -5, roll: -2.5, z: -0.007 },
  calm: { pitch: -1.0, yaw: 0, roll: 1.5, z: 0 },
  curiosity: { pitch: 1.0, yaw: 2.5, roll: 7.0, z: 0.007 },
};

// arousal and valence (Russell) per emotion -- drives blink rate, breathing, saccades, pupils
const AROUSAL = { joy: 0.45, sadness: -0.35, anger: 0.8, fear: 0.9, calm: -0.8, curiosity: 0.35, surprise: 0.85, disgust: 0.3 };

// ------------------------------------------------------------------------------------------------
// shaders
const PORCELAIN_PARS = /* glsl */`
uniform sampler2D uSeamMap;
uniform float uSeamRange;   // mm encoded by G = 1
uniform float uSeamWidth;   // groove half width, mm
uniform float uSeamDepth;   // groove depth, metres
uniform float uSeamDark;
uniform vec3 uGlowColor;
uniform float uGlow;
uniform float uHasSeams;
uniform vec3 uSSS;
uniform float uWrap;
uniform vec2 uFade;         // object-space y (m) where the neck fades to black
varying vec2 vSeamUv;
varying vec3 vObjPos;
// Procedural panel seams, defined on the rest (un-morphed) head in object space (metres), so they ride along
// with the skin when it moves. Landmarks: eyes (+-0.0312, 0.0356), nose tip (0, 0.001), lips ~ y -0.025..-0.04,
// chin -0.085, brow ridge ~0.06, crown 0.124. Returns distance to the nearest seam in millimetres.
float segD(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
float ellD(vec2 p, vec2 c, vec2 r) { vec2 q = (p - c) / r; float k = length(q); return abs(k - 1.0) * min(r.x, r.y); }
float seamField(vec3 P) {
  vec2 p = P.xy; float ax = abs(P.x); vec2 pa = vec2(ax, P.y);
  float d = 1e3;
  if (P.y > 0.074) d = min(d, ax);                                                  // crown midline
  if (ax < 0.072) d = min(d, abs(P.y - (0.071 + 0.010 * (ax / 0.07) * (ax / 0.07))));  // brow band
  d = min(d, ellD(pa, vec2(0.0312, 0.0350), vec2(0.0245, 0.0165)));                  // eye rings
  d = min(d, segD(pa, vec2(0.0545, 0.0230), vec2(0.0590, -0.0420)));                 // cheek plates
  d = min(d, segD(pa, vec2(0.0590, -0.0420), vec2(0.0290, -0.0760)));                // jaw line
  if (P.y < -0.030) d = min(d, ellD(p, vec2(0.0, -0.030), vec2(0.0335, 0.0300)));     // chin plate
  if (P.y > -0.03 && P.y < 0.10) d = min(d, abs(ax - 0.0760));                       // temples
  if (P.z < 0.05) d = min(d, abs(P.z - 0.020) + max(0.0, P.y - 0.13));              // skull cap / nape ring
  return d * 1000.0;
}
float seamDist(vec2 uv) { return seamField(vObjPos); }
float seamH(vec2 uv) {
  float x = seamField(vObjPos) / uSeamWidth;
  float h = 1.0 - smoothstep(0.0, 1.0, x);   // rounded V groove with a soft shoulder
  return -uSeamDepth * h * h * uHasSeams;
}
vec3 seamPerturb(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDir) {
  vec3 vSigmaX = dFdx(surf_pos);
  vec3 vSigmaY = dFdy(surf_pos);
  vec3 R1 = cross(vSigmaY, surf_norm);
  vec3 R2 = cross(surf_norm, vSigmaX);
  float fDet = dot(vSigmaX, R1) * faceDir;
  vec3 vGrad = sign(fDet) * (dHdxy.x * R1 + dHdxy.y * R2);
  return normalize(abs(fDet) * surf_norm - vGrad);
}
`;

const EYE_PARS = /* glsl */`
uniform vec3 uIrisColor;
uniform float uIrisGlow;
uniform float uPupil;       // pupil radius as a fraction of the iris radius
uniform float uCorneaZ0;
uniform float uCorneaR;
uniform float uLimbusZ;
uniform float uLimbusR;
uniform float uIrisDepth;
uniform float uTime;
uniform float uActivity;
uniform vec3 uHeadUp;       // head "up" in eye-local space (for the lid occlusion band)
uniform float uLidTop;      // upper-lid height on the eyeball (cos of angle from head-up)
uniform float uLidBottom;
varying vec3 vEyeLocal;
varying vec3 vEyeCam;
float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2(1, 0)), u.x), mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), u.x), u.y); }
`;

// route a prototype channel name to ARKit channels: 'x' -> xLeft + xRight, 'xL' / 'xR' -> one side
function route(name, v, into) {
  const last = name[name.length - 1];
  if ((last === 'L' || last === 'R') && CHANNELS[name.slice(0, -1) + (last === 'L' ? 'Left' : 'Right')] !== undefined) {
    const c = name.slice(0, -1) + (last === 'L' ? 'Left' : 'Right');
    into[c] = (into[c] || 0) + v;
    return;
  }
  if (CHANNELS[name + 'Left'] !== undefined) {
    into[name + 'Left'] = (into[name + 'Left'] || 0) + v;
    into[name + 'Right'] = (into[name + 'Right'] || 0) + v;
    return;
  }
  if (CHANNELS[name] !== undefined) into[name] = (into[name] || 0) + v;
}

// ------------------------------------------------------------------------------------------------
export class AndroidFace {
  constructor(scene, url = '/face/android.glb', options = {}) {
    this.scene = scene;
    this.options = {
      camera: null,
      seed: 7,
      idle: true,          // blinks, saccades, look-aways, head noise, micro-expressions, breathing
      blinks: true,
      micro: true,
      breathing: true,
      textTarget: { x: 0, y: -0.8 },   // where the "text" is, for glances while writing (NDC or Vector3)
      castShadow: true,
      ...options,
    };
    this.camera = this.options.camera;
    this.root = new THREE.Group();
    this.root.name = 'AndroidFaceRoot';
    scene.add(this.root);

    this.rand = mulberry32(this.options.seed);
    this.noise = new Noise1D(this.rand);
    this.time = 0;

    this.emoTarget = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
    this.emo = Object.fromEntries(EMOTIONS.map((e) => [e, { x: 0, v: 0 }]));   // smoothed "felt" emotion
    this.emoPrev = { ...this.emoTarget };
    this.emoRiseT = Object.fromEntries(EMOTIONS.map((e) => [e, -10]));
    this.intensity = 1;
    this.duchenne = null;
    this.writing = false;
    this.writingAmt = { x: 0, v: 0 };

    this.ch = {};          // channel springs
    this.out = {};         // final channel values
    for (const c of Object.keys(CHANNELS)) { this.ch[c] = { x: 0, v: 0 }; this.out[c] = 0; }
    // per-channel fixed asymmetry (natural ~5-10% per side)
    this.asym = {};
    for (const c of Object.keys(CHANNELS)) {
      const side = c.endsWith('Left') ? 1 : c.endsWith('Right') ? -1 : 0;
      this.asym[c] = 1 + side * (0.03 + 0.05 * this.rand()) * (this.rand() < 0.5 ? 1 : -1);
    }

    this.micro = null;       // {emo, amp, t0}
    this.blink = { next: 1.2, t0: -10, amp: 1, dur: [0.08, 0.05, 0.17], second: false, pending: null };
    this.blinkVal = 0;

    this.gazeMode = 'camera';
    this.gazeTarget = new THREE.Vector3();
    this.gazeScreen = null;
    this.eye = {
      yaw: 0, pitch: 0,           // current (rad, relative to head rest)
      from: [0, 0], to: [0, 0], t0: -1, dur: 0.04, lastSacc: -1,
      offset: [0, 0],             // idle fixation offset (rad) -- eye-to-eye switching / look-aways
      offsetUntil: 0, nextShift: 0.8, lookAway: false,
      jitter: [0, 0],
    };
    this.glance = { active: false, until: 0, next: 2.5 };
    this.head = { pitch: { x: 0, v: 0 }, yaw: { x: 0, v: 0 }, roll: { x: 0, v: 0 }, z: { x: 0, v: 0 } };
    this.breath = { phase: 0 };
    this.mouthing = { t: 0, next: 0, target: {}, cur: {} };
    this.pupil = { x: 0.34, v: 0 };
    this.irisColor = new THREE.Color('#7fe7ff');
    this.irisGlow = 1.0;

    this._tmpV = new THREE.Vector3();
    this._tmpV2 = new THREE.Vector3();
    this._tmpQ = new THREE.Quaternion();
    this._tmpE = new THREE.Euler();
    this._tmpM = new THREE.Matrix4();

    this.ready = this._load(url).then(() => this);
  }

  // -------------------------------------------------------------------------------- loading
  async _load(url) {
    const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(url);
    this.gltf = gltf;
    const model = gltf.scene;
    this.root.add(model);
    this.model = model;
    this.morphMeshes = [];
    this.eyes = [];
    const pending = [];

    this.uniforms = {
      uSeamMap: { value: null }, uSeamRange: { value: 2.0 }, uSeamWidth: { value: 0.55 }, uSeamDepth: { value: 0.00035 },
      uSeamDark: { value: 0.28 }, uGlowColor: { value: this.irisColor.clone() }, uGlow: { value: 0.0 }, uHasSeams: { value: 1 },
      uSSS: { value: new THREE.Color(1.0, 0.62, 0.45).multiplyScalar(0.55) }, uWrap: { value: 0.45 },
      uFade: { value: new THREE.Vector2(-0.100, -0.165) },
    };

    model.traverse((o) => {
      if (o.isMesh) {
        const role = o.material.userData.role;
        if (role === 'porcelain') {
          const idx = o.material.userData.seamTexture;
          o.material = this._porcelainMaterial();
          if (idx !== undefined && !this._seamLoading) {
            this._seamLoading = true;
            pending.push(gltf.parser.getDependency('texture', idx).then((t) => {
              t.colorSpace = THREE.NoColorSpace;
              t.anisotropy = 8;
              this.uniforms.uSeamMap.value = t;
              this.uniforms.uHasSeams.value = 1;
              for (const m of this._porcelainMats) { m.aoMap = t; m.aoMapIntensity = 0.85; m.needsUpdate = true; }
            }));
          }
        } else if (role === 'cavity') {
          o.material = new THREE.MeshStandardMaterial({ color: 0x07080b, roughness: 0.7, metalness: 0.0 });
        } else if (role === 'teeth') {
          o.material = new THREE.MeshPhysicalMaterial({ color: 0xe9e6df, roughness: 0.32, clearcoat: 0.8, clearcoatRoughness: 0.12 });
        } else if (role === 'gums') {
          o.material = new THREE.MeshPhysicalMaterial({ color: 0x15181d, roughness: 0.42, clearcoat: 0.4, clearcoatRoughness: 0.2 });
        } else if (role === 'eye') {
          const ud = o.material.userData;
          o.material = this._eyeMaterial(ud);
          this.eyes.push({ node: o, mat: o.material, side: o.name.endsWith('R') ? 'R' : 'L' });
        }
        o.castShadow = this.options.castShadow;
        o.receiveShadow = true;
        o.frustumCulled = false;
        if (o.morphTargetDictionary) this.morphMeshes.push(o);
      }
      if (o.isBone && o.name === 'head') this.headBone = o;
      if (o.isBone && o.name === 'neck') this.neckBone = o;
    });
    // eyes: node names come from the glTF nodes (EyeL / EyeR); GLTFLoader may wrap them
    for (const e of this.eyes) {
      let n = e.node;
      while (n && !/^Eye[LR]$/.test(n.name)) n = n.parent;
      e.pivot = n || e.node;
      e.side = e.pivot.name.endsWith('R') ? 'R' : 'L';
      e.restQ = e.pivot.quaternion.clone();
      e.restAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(e.restQ);
    }
    this.eyes.sort((a, b) => (a.side < b.side ? -1 : 1));
    this.headRestQ = this.headBone.quaternion.clone();
    this.headRestP = this.headBone.position.clone();
    this.neckRestQ = this.neckBone.quaternion.clone();
    this.neckRestP = this.neckBone.position.clone();

    // morph index map per mesh
    for (const m of this.morphMeshes) {
      m.userData.map = Object.entries(m.morphTargetDictionary);
    }
    this.channelNames = new Set(this.morphMeshes.flatMap((m) => Object.keys(m.morphTargetDictionary)));

    // anchor between the eyes (for framing and gaze math)
    this.anchor = new THREE.Object3D();
    this.anchor.name = 'FaceAnchor';
    const eL = this.eyes[0].pivot.position, eR = this.eyes[1].pivot.position;
    this.anchor.position.copy(eL).add(eR).multiplyScalar(0.5);
    this.headBone.add(this.anchor);

    await Promise.all(pending);
    this.root.updateMatrixWorld(true);
  }

  _porcelainMaterial() {
    const m = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0.93, 0.905, 0.868),
      roughness: 0.4, metalness: 0,
      clearcoat: 1.0, clearcoatRoughness: 0.075,
      specularIntensity: 0.6,
      sheen: 0.25, sheenRoughness: 0.5, sheenColor: new THREE.Color(0.9, 0.93, 1.0),
    });
    const U = this.uniforms;
    m.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, U);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vSeamUv;\nvarying vec3 vObjPos;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSeamUv = uv;\nvObjPos = position;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + PORCELAIN_PARS)
        .replace('#include <color_fragment>', `#include <color_fragment>
          float sDist = seamDist(vSeamUv);
          float sStr = uHasSeams * smoothstep(-0.02, 0.03, vObjPos.z);  // fade seams out toward the back of the head
          float sFw = max(fwidth(sDist), 1e-4);
          float sCore = (1.0 - smoothstep(uSeamWidth * 0.42 - sFw, uSeamWidth * 0.42 + sFw, sDist)) * sStr;
          float sLip = (1.0 - smoothstep(0.0, uSeamWidth * 1.6 + sFw, sDist)) * sStr;
          diffuseColor.rgb *= mix(1.0, uSeamDark, sCore);
          diffuseColor.rgb *= mix(1.0, 0.9, sLip);
        `)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
          {
            float h0 = seamH(vSeamUv);
            vec2 dH = vec2(dFdx(h0), dFdy(h0));
            normal = seamPerturb(-vViewPosition, normal, dH, faceDirection);
          }
        `)
        .replace('#include <clearcoat_normal_fragment_begin>', '#include <clearcoat_normal_fragment_begin>\n#ifdef USE_CLEARCOAT\nclearcoatNormal = normal;\n#endif')
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          totalEmissiveRadiance += uGlowColor * uGlow * sCore;`)
        .replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment.replace(
          'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
          `{ float wNL = saturate((dot(geometryNormal, directLight.direction) + uWrap) / (1.0 + uWrap));
             vec3 wrapIrr = directLight.color * max(wNL - dotNL, 0.0) * uSSS;
             reflectedLight.directDiffuse += (irradiance + wrapIrr) * BRDF_Lambert( material.diffuseColor ); }`))
        .replace('#include <opaque_fragment>', `
          outgoingLight *= smoothstep(uFade.y, uFade.x, vObjPos.y);
          #include <opaque_fragment>`);
    };
    m.customProgramCacheKey = () => 'porcelain-v2';
    (this._porcelainMats ||= []).push(m);
    return m;
  }

  _eyeMaterial(ud) {
    const m = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, roughness: 0.08, metalness: 0,
      clearcoat: 1.0, clearcoatRoughness: 0.02, specularIntensity: 1.0, ior: 1.376,
    });
    const u = {
      uIrisColor: { value: this.irisColor }, uIrisGlow: { value: 1 }, uPupil: { value: 0.34 },
      uCorneaZ0: { value: ud.corneaZ0 }, uCorneaR: { value: ud.corneaR }, uLimbusZ: { value: ud.limbusZ }, uLimbusR: { value: ud.limbusR },
      uIrisDepth: { value: 0.0003 }, uTime: { value: 0 }, uActivity: { value: 0 },
      uHeadUp: { value: new THREE.Vector3(0, 1, 0) }, uLidTop: { value: 0.35 }, uLidBottom: { value: -0.45 },
    };
    m.userData.u = u;
    m.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, u);
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vEyeLocal;\nvarying vec3 vEyeCam;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvEyeLocal = position;\nvEyeCam = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + EYE_PARS)
        .replace('#include <color_fragment>', `#include <color_fragment>
          vec3 eP = vEyeLocal;
          vec3 eV = normalize(eP - vEyeCam);
          float eR = length(eP);
          vec3 eN = eP / eR;
          float limbusW = uLimbusR * 0.06;
          float corneaMask = smoothstep(uLimbusZ - limbusW * 0.2, uLimbusZ + limbusW, eP.z);
          // refract into the anterior chamber and hit the (slightly concave) iris plane
          vec3 cN = normalize(eP - vec3(0.0, 0.0, uCorneaZ0));
          vec3 rd = refract(eV, cN, 1.0 / 1.376);
          float irisZ = uLimbusZ - uIrisDepth;
          float tI = (irisZ - eP.z) / min(rd.z, -1e-3);
          vec2 q = eP.xy + rd.xy * tI;
          float rI = length(q) / uLimbusR;
          float ang = atan(q.y, q.x);
          // iris: dark glass with radial fibres, a luminous ring and a glowing pupil edge
          float fib = vnoise(vec2(ang * 18.0, rI * 3.0)) * 0.6 + vnoise(vec2(ang * 47.0, rI * 9.0)) * 0.4;
          vec3 irisBase = vec3(0.025, 0.03, 0.038) + uIrisColor * 0.05 * fib;
          float ring = smoothstep(0.50, 0.60, rI) * (1.0 - smoothstep(0.70, 0.80, rI));
          float ringFine = smoothstep(0.84, 0.87, rI) * (1.0 - smoothstep(0.89, 0.93, rI));
          float seg = 0.75 + 0.25 * smoothstep(0.2, 0.8, sin(ang * 24.0 - uTime * (0.6 + 2.4 * uActivity)) * 0.5 + 0.5);
          float pupEdge = smoothstep(uPupil - 0.02, uPupil + 0.02, rI) * (1.0 - smoothstep(uPupil + 0.03, uPupil + 0.09, rI));
          float pupil = 1.0 - smoothstep(uPupil - 0.015, uPupil + 0.015, rI);
          float limbal = smoothstep(0.86, 1.0, rI);
          vec3 irisCol = mix(irisBase, vec3(0.004), pupil);
          irisCol = mix(irisCol, vec3(0.012, 0.014, 0.018), limbal * 0.8);
          vec3 irisEmit = uIrisColor * uIrisGlow * (ring * (0.55 + 0.45 * fib) * seg + ringFine * 0.35 + pupEdge * 0.25) * (1.0 - pupil);
          // sclera: cool porcelain-glass white, darker toward the back
          vec3 scl = vec3(0.80, 0.83, 0.86) * mix(0.55, 1.0, smoothstep(-0.2, 0.8, eN.z));
          float outside = smoothstep(0.98, 1.04, rI);        // refracted ray missed the iris -> sclera behind cornea
          vec3 behind = mix(irisCol, scl, outside);
          vec3 eyeCol = mix(scl, behind, corneaMask);
          vec3 eyeEmit = irisEmit * corneaMask * (1.0 - outside);
          // soft occlusion under the lids (fixed to the head, not the eye)
          float hy = dot(eN, uHeadUp);
          float occ = smoothstep(uLidTop - 0.35, uLidTop + 0.05, hy) * 0.75 + (1.0 - smoothstep(uLidBottom - 0.05, uLidBottom + 0.3, hy)) * 0.45;
          occ = clamp(occ, 0.0, 0.85);
          eyeCol *= 1.0 - occ;
          eyeEmit *= 1.0 - occ * 0.6;
          diffuseColor.rgb = eyeCol;
        `)
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(0.22, 0.03, corneaMask);')
        .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += eyeEmit;');
    };
    m.customProgramCacheKey = () => 'android-eye-v1';
    return m;
  }

  // -------------------------------------------------------------------------------- public API
  setCamera(camera) { this.camera = camera; }

  setEmotion(values = {}, { intensity = 1, duchenne = null } = {}) {
    this.intensity = clamp(intensity, 0, 1.5);
    this.duchenne = duchenne;
    for (const e of EMOTIONS) {
      const v = clamp(values[e] ?? 0, 0, 1);
      const prev = this.emoTarget[e];
      if (v > prev) this.emoRiseT[e] = Math.max(this.emoRiseT[e], this.time - 0.001);
      this.emoTarget[e] = v;
    }
  }

  setGaze(target) {
    if (target === 'camera' || target == null) { this.gazeMode = 'camera'; return; }
    if (target.isVector3) { this.gazeMode = 'world'; this.gazeTarget.copy(target); return; }
    if (typeof target.x === 'number' && typeof target.y === 'number') { this.gazeMode = 'screen'; this.gazeScreen = { x: target.x, y: target.y }; }
  }

  setActivity({ writing = false } = {}) { this.writing = !!writing; }

  setIrisColor(color) {
    this.irisColor.set(color);
    if (this.uniforms) this.uniforms.uGlowColor.value.copy(this.irisColor);
  }

  setOptions(o) { Object.assign(this.options, o); }

  /** Fast-forward the dynamics (used by QA stills). */
  settle(seconds = 3, step = 1 / 60) {
    for (let t = 0; t < seconds; t += step) this.update(step, this.time + step);
  }

  // -------------------------------------------------------------------------------- simulation
  update(dt, time) {
    if (!this.headBone) return;
    dt = clamp(dt, 0, 0.1);
    this.time = time ?? this.time + dt;
    const t = this.time;
    const idle = this.options.idle;

    // ---- emotions: felt state follows target (fast rise, slower decay), surprise habituates
    const felt = {};
    let arousal = 0, total = 0;
    for (const e of EMOTIONS) {
      let target = this.emoTarget[e] * this.intensity;
      if (e === 'surprise') target *= lerp(1, 0.55, smooth(0.8, 3.0, t - this.emoRiseT[e]));
      const s = this.emo[e];
      spring(s, target, target > s.x ? 7 : 3.2, dt);
      let v = Math.max(0, s.x);
      if (idle) v *= 1 + 0.07 * this.noise.at(t * 0.23 + EMOTIONS.indexOf(e) * 13.1);   // apex is never static
      felt[e] = v;
      arousal += AROUSAL[e] * v;
      total += v;
      // micro-expression: a brief leak when an emotion target jumps up
      const jump = target - (this.emoPrev[e] ?? 0);
      if (this.options.micro && idle && jump > 0.22 && (!this.micro || t - this.micro.t0 > 0.6)) {
        this.micro = { emo: e, amp: clamp(0.35 + jump * 0.9, 0, 1), t0: t, dur: 0.15 + 0.15 * this.rand() };
      }
      this.emoPrev[e] = target;
    }
    this.felt = felt;
    this.arousal = clamp(arousal / Math.max(1, total * 0.9 + 0.2), -1, 1) * Math.min(1, total * 1.4);

    spring(this.writingAmt, this.writing ? 1 : 0, 4, dt);
    const W = clamp(this.writingAmt.x);

    // ---- action-unit targets
    const tgt = {};
    for (const c of Object.keys(CHANNELS)) tgt[c] = 0;
    const add = (name, v) => route(name, v, tgt);
    const applyProto = (e, amt, gain = 1) => {
      for (const [ch, w, lo, hi] of PROTOS[e]) {
        // lo = 0: ~linear engagement up to `hi`; lo > 0: this AU only joins at higher intensity
        let k = w * (lo === 0 ? clamp(amt / hi) * (1.5 - 0.5 * clamp(amt / hi)) : smooth(lo, hi, amt));
        if (e === 'joy' && (ch === 'cheekSquint' || ch === 'cheekRaiser' || ch === 'eyeSquint')) {
          const duch = this.duchenne ?? smooth(0.1, 0.55, amt);   // polite (AU12 only) at low joy -> Duchenne
          k *= 0.2 + 0.8 * duch;
        }
        add(ch, k * gain);
      }
    };
    for (const e of EMOTIONS) if (felt[e] > 1e-3) applyProto(e, felt[e]);

    // writing: concentration (slight brow knit / lid tension) + silent mouthing
    if (W > 0.01) {
      add('browDown', 0.06 * W * (0.6 + 0.4 * this.noise.at(t * 0.3 + 5)));
      add('eyeSquint', 0.05 * W);
      this._mouthing(dt, t, W, add);
    }

    // idle micro-motion of the face (never frozen)
    if (idle) {
      const n = (k, f) => this.noise.fbm(t * f + k * 7.3, 2);
      add('browInnerUp', 0.025 * Math.max(0, n(1, 0.21)));
      add('browOuterUpL', 0.02 * Math.max(0, n(2, 0.17)));
      add('browOuterUpR', 0.02 * Math.max(0, n(3, 0.19)));
      add('mouthSmile', 0.012 * n(4, 0.13));
      add('mouthPress', 0.03 * Math.max(0, n(5, 0.11)));
      add('mouthRollLower', 0.02 * Math.max(0, n(6, 0.09)));
      add('eyeSquint', 0.03 * Math.max(0, n(7, 0.15)));
      add('noseSneer', 0.012 * Math.max(0, n(8, 0.12)));
    }

    // ---- springs per region (asymmetric onset / offset), asymmetry, soft caps
    for (const c of Object.keys(CHANNELS)) {
      const [on, off] = REGION_TIMES[CHANNELS[c]];
      const s = this.ch[c];
      const target = Math.max(0, tgt[c] * this.asym[c]);
      const rising = target > s.x;
      spring(s, target, 3.36 / (rising ? on : off), dt);
    }

    // micro-expression flash (bypasses the slow springs): ~40 ms attack, short hold, ~120 ms decay
    const flash = {};
    if (this.micro) {
      const m = this.micro;
      const u = t - m.t0;
      const env = u < 0.04 ? u / 0.04 : u < 0.04 + m.dur * 0.4 ? 1 : Math.max(0, 1 - (u - 0.04 - m.dur * 0.4) / (m.dur * 0.6 + 0.08));
      if (env <= 0) this.micro = null;
      else {
        for (const [ch, w] of PROTOS[m.emo]) {
          const k = w * m.amp * env * (/brow|eye|nose/.test(ch) ? 1.0 : 0.6);
          route(ch, k, flash);
        }
      }
    }

    // ---- blinks
    this._blinks(dt, t);
    const B = this.blinkVal;

    for (const c of Object.keys(CHANNELS)) {
      let v = this.ch[c].x + (flash[c] || 0);
      const cap = CAPS[c] ?? 1;
      v = v <= 0 ? 0 : cap * (1 - Math.exp(-v / cap * 1.25)) / (1 - Math.exp(-1.25));   // soft cap, ~linear at small v
      this.out[c] = Math.min(v, cap);
    }
    // blink composes with the lid state: closes whatever is open
    for (const [side, k] of [['Left', 1], ['Right', 0.97]]) {
      const lid = this.out['eyeBlink' + side];
      const bk = B * k;
      this.out['eyeBlink' + side] = lid + (1 - lid) * bk;
      this.out['eyeWide' + side] *= 1 - bk;
      // blink-coupled brow motion (a slight dip during closure) and lower-lid lift
      this.out['browDown' + side] += 0.07 * bk;
      this.out['eyeSquint' + side] += 0.12 * bk;
      this.out['cheekSquint' + side] += 0.03 * bk;
    }

    // ---- gaze, head, breathing, pupils
    this._gaze(dt, t, W);
    this._head(dt, t, W);
    this._eyesAndLids(dt, t, W);

    // ---- apply morphs
    const outMap = this.out;
    outMap.browInnerUp = 0; outMap.cheekPuff = 0;   // combined channels unused (split ones are driven)
    for (const m of this.morphMeshes) {
      const inf = m.morphTargetInfluences;
      for (const [name, i] of m.userData.map) inf[i] = outMap[name] || 0;
    }
  }

  _mouthing(dt, t, W, add) {
    // silent, irregular syllable-rate lip motion -- not lip-sync, barely visible
    const M = this.mouthing;
    if (t >= M.next) {
      const pause = this.rand() < 0.16;
      const a = pause ? 0 : 0.35 + 0.65 * this.rand();
      const r = this.rand();
      M.target = pause ? {} :
        r < 0.35 ? { jawOpen: 0.035 * a, mouthLowerDown: 0.06 * a } :
        r < 0.55 ? { mouthPucker: 0.09 * a, mouthFunnel: 0.04 * a, jawOpen: 0.015 * a } :
        r < 0.75 ? { mouthPress: 0.12 * a, mouthRollLower: 0.05 * a } :
        r < 0.9 ? { mouthStretch: 0.06 * a, jawOpen: 0.02 * a, mouthUpperUp: 0.03 * a } :
        { mouthClose: 0.02 * a, mouthRollUpper: 0.06 * a };
      M.next = t + (pause ? 0.25 + 0.35 * this.rand() : 0.11 + 0.13 * this.rand());
    }
    for (const k of new Set([...Object.keys(M.cur), ...Object.keys(M.target)])) {
      const s = (M.cur[k] ||= { x: 0, v: 0 });
      spring(s, M.target[k] || 0, 26, dt);
      add(k, Math.max(0, s.x) * W);
    }
  }

  _blinks(dt, t) {
    const b = this.blink;
    const f = this.felt;
    if (!this.options.blinks || !this.options.idle) { this.blinkVal = 0; return; }
    if (t >= b.next && b.t0 < t - 0.3) {
      // start a blink
      b.t0 = t;
      b.amp = this.rand() < 0.14 ? 0.72 + 0.18 * this.rand() : 1;         // some incomplete blinks
      const slow = f.sadness * 0.5 + f.calm * 0.3;
      b.dur = [0.075 + 0.02 * this.rand() + 0.03 * slow, 0.03 + 0.04 * this.rand() + 0.08 * slow, 0.16 + 0.07 * this.rand() + 0.1 * slow];
      // rate: ~17/min at rest; rises with arousal, falls with calm and while concentrating on text
      const rate = clamp(17 * (1 + 0.75 * Math.max(0, this.arousal)) * (1 - 0.35 * f.calm) * (1 - 0.25 * clamp(this.writingAmt.x)) * (1 + 0.25 * Math.min(0, this.arousal)), 6, 38);
      const mean = 60 / rate;
      // log-normal-ish interval
      const g = Math.exp((this.rand() + this.rand() + this.rand() - 1.5) * 0.75);
      b.next = t + Math.max(0.7, mean * g);
      if (this.rand() < 0.12) b.pending = t + b.dur[0] + b.dur[1] + b.dur[2] * 0.6 + 0.05; // double blink
    }
    if (b.pending && t >= b.pending) { b.t0 = t; b.pending = null; b.amp = 0.85 + 0.15 * this.rand(); }
    const u = t - b.t0;
    const [c, h, o] = b.dur;
    let v = 0;
    if (u < c) v = Math.sin((u / c) * Math.PI / 2) ** 2;
    else if (u < c + h) v = 1;
    else if (u < c + h + o) { const k = (u - c - h) / o; v = 1 - (1 - (1 - k) * (1 - k)) ; v = v * v * (3 - 2 * v); }
    this.blinkVal = v * b.amp;
  }

  _screenToWorld(sx, sy, out) {
    const cam = this.camera;
    if (!cam) return out.set(sx * 0.3, sy * 0.3, 1).applyMatrix4(this.root.matrixWorld);
    this.anchor.getWorldPosition(this._tmpV2);
    const ndc = this._tmpV2.clone().project(cam);
    out.set(sx, sy, ndc.z).unproject(cam);
    return out;
  }

  _gaze(dt, t, W) {
    const E = this.eye;
    const f = this.felt;
    const idle = this.options.idle;
    // --- where do we want to look (world)?
    const tw = this._tmpV;
    if (this.gazeMode === 'world') tw.copy(this.gazeTarget);
    else if (this.gazeMode === 'screen') this._screenToWorld(this.gazeScreen.x, this.gazeScreen.y, tw);
    else if (this.camera) this.camera.getWorldPosition(tw);
    else { tw.set(0, 0, 1).applyMatrix4(this.anchor.matrixWorld); }

    // glances down at the text while writing
    const G = this.glance;
    if (idle && W > 0.5 && this.gazeMode === 'camera') {
      if (!G.active && t >= G.next) { G.active = true; G.until = t + 0.6 + 1.4 * this.rand(); }
      if (G.active && t >= G.until) { G.active = false; G.next = t + 1.8 + 3.5 * this.rand(); }
    } else if (G.active) G.active = false;
    if (G.active) {
      const tt = this.options.textTarget;
      if (tt.isVector3) tw.copy(tt); else this._screenToWorld(tt.x + 0.15 * this.noise.at(t * 0.7), tt.y, tw);
    }

    // desired angles relative to the head rest frame
    const headInv = this._tmpM.copy(this.headBone.matrixWorld).invert();
    const local = tw.clone().applyMatrix4(headInv);
    const eyeC = this.anchor.position;
    const dir = local.sub(eyeC);
    let yaw = Math.atan2(dir.x, dir.z);
    let pitch = Math.atan2(dir.y, Math.hypot(dir.x, dir.z));
    this.targetDist = Math.max(0.15, dir.length());

    // idle fixation pattern: eye-to-eye switching, occasional look-aways (thinking)
    if (idle) {
      if (t >= E.nextShift) {
        const r = this.rand();
        const awayP = 0.12 + 0.25 * f.sadness + 0.2 * f.disgust + 0.1 * f.fear - 0.08 * f.anger - 0.05 * f.joy + (W > 0.5 ? -0.05 : 0.05);
        if (r < awayP && !G.active) {
          const side = this.rand() < 0.5 ? -1 : 1;
          const down = f.sadness > 0.3 ? -1 : (this.rand() < 0.55 ? 1 : -1);
          E.offset = [side * (5 + 9 * this.rand()) * DEG, down * (3 + 6 * this.rand()) * DEG - f.sadness * 6 * DEG];
          E.nextShift = t + 0.5 + 1.3 * this.rand();
        } else {
          // tiny switches between the viewer's eyes / mouth
          E.offset = [(this.rand() - 0.5) * 1.8 * DEG, (this.rand() - 0.6) * 1.2 * DEG - f.sadness * 5 * DEG];
          const rate = 1 + 1.2 * f.fear + 0.8 * f.curiosity + 0.4 * f.surprise - 0.5 * f.calm - 0.4 * f.anger;
          E.nextShift = t + (0.5 + 1.8 * this.rand()) / Math.max(0.35, rate);
        }
      }
      yaw += E.offset[0];
      pitch += E.offset[1];
    } else {
      pitch -= f.sadness * 5 * DEG;
    }
    // ocular motor range
    yaw = clamp(yaw, -28 * DEG, 28 * DEG);
    pitch = clamp(pitch, -24 * DEG, 18 * DEG);
    this.gazeDesired = [yaw, pitch];

    // --- saccades (main sequence: D = 21 ms + 2.2 ms/deg), then fixation with jitter
    const dy = yaw - E.to[0], dp = pitch - E.to[1];
    const amp = Math.hypot(dy, dp) / DEG;
    const refractory = t - E.lastSacc > 0.12;
    if ((amp > 0.35 && refractory) || !idle) {
      if (!idle) { E.from = [yaw, pitch]; E.to = [yaw, pitch]; E.t0 = t - 1; }
      else {
        E.from = [E.yaw, E.pitch];
        E.to = [yaw, pitch];
        E.t0 = t;
        E.dur = 0.021 + 0.0022 * amp;
        E.lastSacc = t;
        if (amp > 9 && this.options.blinks && this.rand() < 0.35 && t - this.blink.t0 > 0.5) this.blink.next = t; // gaze-evoked blink
      }
    }
    const u = clamp((t - E.t0) / E.dur);
    const s = u * u * u * (u * (u * 6 - 15) + 10);   // minimum-jerk profile
    E.yaw = lerp(E.from[0], E.to[0], s);
    E.pitch = lerp(E.from[1], E.to[1], s);
    if (idle) {
      // fixational drift + tremor
      E.jitter[0] = 0.12 * DEG * this.noise.fbm(t * 1.7 + 40, 3);
      E.jitter[1] = 0.10 * DEG * this.noise.fbm(t * 1.9 + 80, 3);
    } else E.jitter = [0, 0];
  }

  _head(dt, t, W) {
    const f = this.felt;
    const H = this.head;
    const idle = this.options.idle;
    // posture from emotions
    let pitch = 0, yaw = 0, roll = 0, z = 0;
    for (const e of EMOTIONS) {
      const p = POSTURE[e];
      const k = f[e];
      pitch += p.pitch * k; yaw += p.yaw * k; roll += p.roll * k; z += p.z * k;
    }
    // writing: a slight downward inclination toward the text
    pitch -= 1.5 * W;
    // head follows large gaze offsets (eye-head coordination)
    const [gy, gp] = this.gazeDesired || [0, 0];
    const follow = (a) => Math.sign(a) * Math.max(0, Math.abs(a) - 6 * DEG) * 0.4;
    yaw += follow(gy) / DEG;
    pitch += follow(gp) / DEG * 0.7;
    // perlin micro-motion (amplitude grows with arousal, slower when calm)
    let breathP = 0, breathY = 0;
    if (idle) {
      const slow = 1 - 0.45 * f.calm;
      const amp = (0.7 + 0.8 * Math.max(0, this.arousal) + 0.3 * f.curiosity) * (1 - 0.4 * f.calm);
      pitch += amp * 1.1 * this.noise.fbm(t * 0.21 * slow + 100, 3);
      yaw += amp * 1.4 * this.noise.fbm(t * 0.17 * slow + 200, 3);
      roll += amp * 0.8 * this.noise.fbm(t * 0.19 * slow + 300, 3);
      // fear: small tremor
      pitch += f.fear * 0.25 * this.noise.at(t * 6.1 + 9);
      yaw += f.fear * 0.25 * this.noise.at(t * 5.3 + 19);
    }
    if (this.options.breathing) {
      // breathing: ~14/min at rest, slower when calm, faster when aroused; inhale shorter than exhale
      const rate = clamp(14 * (1 + 0.45 * this.arousal) * (1 - 0.3 * f.calm), 7, 24) / 60;
      this.breath.phase += dt * rate;
      const ph = this.breath.phase % 1;
      const b = ph < 0.4 ? Math.sin((ph / 0.4) * Math.PI / 2) : Math.cos(((ph - 0.4) / 0.6) * Math.PI / 2);
      this.breathVal = b * b;
      breathP = (this.breathVal - 0.5) * (0.55 + 0.3 * f.calm);
      breathY = this.breathVal * (0.0011 + 0.0006 * Math.max(0, this.arousal));
    }
    const wHead = 4.2 - 1.5 * f.calm + 1.5 * Math.max(0, this.arousal);
    spring(H.pitch, pitch, wHead, dt);
    spring(H.yaw, yaw, wHead, dt);
    spring(H.roll, roll, wHead * 0.8, dt);
    spring(H.z, z, 3.0, dt);

    // split rotation between the neck (35%) and head (65%) joints
    const e = this._tmpE;
    const P = (H.pitch.x + breathP) * DEG, Y = H.yaw.x * DEG, R = H.roll.x * DEG;
    e.set(-P * 0.35, Y * 0.35, -R * 0.35, 'YXZ');
    this.neckBone.quaternion.copy(this.neckRestQ).multiply(this._tmpQ.setFromEuler(e));
    e.set(-P * 0.65, Y * 0.65, -R * 0.65, 'YXZ');
    this.headBone.quaternion.copy(this.headRestQ).multiply(this._tmpQ.setFromEuler(e));
    this.neckBone.position.copy(this.neckRestP);
    this.neckBone.position.y += breathY * 0.6;
    this.neckBone.position.z += H.z.x * 0.3;
    this.headBone.position.copy(this.headRestP);
    this.headBone.position.z += H.z.x * 0.7;
    this.headBone.position.y += breathY * 0.4;
    this.root.updateMatrixWorld(true);
  }

  _eyesAndLids(dt, t, W) {
    const E = this.eye;
    const f = this.felt;
    const yaw = E.yaw + E.jitter[0];
    const pitch = E.pitch + E.jitter[1] - this.blinkVal * 2.5 * DEG;   // eyes dip slightly during blinks
    // aim each eye at the fixation point (vergence), expressed in head space
    const dist = this.targetDist || 0.6;
    const fix = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)).multiplyScalar(dist).add(this.anchor.position);
    for (const e of this.eyes) {
      const d = fix.clone().sub(e.pivot.position).normalize();
      // kappa: the optical (iris) axis sits ~4 deg temporal of the visual axis
      const kappa = (e.side === 'L' ? 1 : -1) * 4 * DEG;
      d.applyAxisAngle(new THREE.Vector3(0, 1, 0), kappa);
      const q = this._tmpQ.setFromUnitVectors(e.restAxis, d);
      e.pivot.quaternion.copy(q).multiply(e.restQ);
      // uniforms
      const u = e.mat.userData.u;
      u.uTime.value = t;
      u.uActivity.value = W;
      u.uPupil.value = this.pupil.x;
      u.uIrisGlow.value = this.irisGlow * (1 + 0.12 * W + 0.05 * Math.sin(t * 1.3));
      // head up in eye-local space
      const inv = e.pivot.quaternion.clone().invert();
      u.uHeadUp.value.set(0, 1, 0).applyQuaternion(inv);
      const side = e.side === 'L' ? 'Left' : 'Right';
      const blink = this.out['eyeBlink' + side], wide = this.out['eyeWide' + side], sq = this.out['eyeSquint' + side];
      u.uLidTop.value = 0.42 - 0.9 * blink + 0.28 * wide - 0.12 * sq + 0.35 * Math.min(0, Math.sin(pitch)) * 0;
      u.uLidBottom.value = -0.55 + 0.25 * sq + 0.15 * (this.out['cheekSquint' + side] || 0);
    }
    // eyelids follow gaze (ICT eyeLook* shapes deform the lids only)
    const up = Math.max(0, pitch) / (18 * DEG), down = Math.max(0, -pitch) / (24 * DEG);
    const yl = yaw / (28 * DEG);
    this.out.eyeLookUpLeft = this.out.eyeLookUpRight = clamp(up) * 0.9;
    this.out.eyeLookDownLeft = this.out.eyeLookDownRight = clamp(down) * 0.9;
    // subject's left eye looking to the subject's left (+yaw) = out
    this.out.eyeLookOutLeft = clamp(yl); this.out.eyeLookInLeft = clamp(-yl);
    this.out.eyeLookInRight = clamp(yl); this.out.eyeLookOutRight = clamp(-yl);
    // pupils: dilate with arousal / interest, constrict slightly with anger focus
    const pupT = 0.34 + 0.07 * Math.max(0, this.arousal) + 0.05 * f.curiosity + 0.04 * f.joy - 0.03 * f.anger + 0.02 * f.fear;
    spring(this.pupil, pupT, 2.5, dt);
  }

  getAnchorWorldPosition(out = new THREE.Vector3()) { return this.anchor.getWorldPosition(out); }

  dispose() {
    this.scene.remove(this.root);
    this.root.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
  }
}
