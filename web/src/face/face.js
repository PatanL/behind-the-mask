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
import { SpeechMotion, composeSpeech } from './speech-motion.js';
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
  cheek: [0.58, 0.95],   // AU6 trails AU12 in a felt smile
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
  anger: [ // AU4 + AU7 + AU23/24 (+ AU17, AU9 slight); AU5 only near the top: on this face it reads as alarm
    ['browDown', 0.95, 0.0, 0.75],
    ['eyeWide', 0.12, 0.65, 1.0],
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
// blink-rate multiplier per emotion: the angry stare and rapt attention suppress blinking, fear and disgust raise it
const BLINK_MUL = { joy: 1.1, sadness: 0.85, anger: 0.4, fear: 1.3, calm: 0.6, curiosity: 0.6, surprise: 0.4, disgust: 1.2 };
const NEG = ['sadness', 'anger', 'fear', 'disgust'];

// Blends (Ekman & Friesen, Unmasking the Face, 1975). Two feelings at once don't average out; each part of the
// face is carried by one of them, like sad brows over a smiling mouth. When both feelings of a pair are present
// and comparable, `mul` scales channels of the prototype mix and `add` contributes the blend's own action units.
const BLENDS = [
  { name: 'bittersweet', a: 'joy', b: 'sadness',          // AU1+AU4 brows over an AU12 smile, no frown
    mul: { mouthFrown: 0.15, mouthShrugLower: 0.5, mouthLowerDown: 0.5, cheekRaiser: 0.6, browOuterUp: 0.4, jawOpen: 0.5, mouthUpperUp: 0.6 },
    add: { browInnerUp: 0.16, mouthPress: 0.14, browDown: 0.05 } },
  { name: 'nervous smile', a: 'joy', b: 'fear',           // smile without the eyes, lips drawn back and held
    mul: { cheekSquint: 0.35, cheekRaiser: 0.35, eyeSquint: 0.25, jawOpen: 0.4, mouthLowerDown: 0.5, mouthUpperUp: 0.5 },
    add: { mouthStretch: 0.16, mouthRollLower: 0.12, browInnerUp: 0.1, mouthPress: 0.08 } },
  { name: 'worry', a: 'sadness', b: 'fear',               // the grief brow (AU1+AU4 together), a held mouth
    mul: { eyeBlink: 0.5, mouthStretch: 0.6, jawOpen: 0.6 },
    add: { browInnerUp: 0.14, browDown: 0.12, mouthPress: 0.12, mouthShrugLower: 0.1 } },
  { name: 'hurt', a: 'sadness', b: 'anger',               // knitted brows, a raised chin, lips pressed
    mul: { eyeWide: 0.3, eyeBlink: 0.6, mouthRollUpper: 0.5 },
    add: { browInnerUp: 0.12, mouthShrugLower: 0.22, mouthPress: 0.1, eyeSquint: 0.08 } },
  { name: 'cornered', a: 'anger', b: 'fear',              // brows down but eyes wide, lips drawn back
    mul: { browInnerUp: 0.6, browOuterUp: 0.4, mouthPress: 0.6 },
    add: { mouthStretch: 0.1, mouthUpperUp: 0.08, noseSneer: 0.08 } },
  { name: 'smug', a: 'joy', b: 'anger',                   // a tight, one-sided smile that doesn't reach the eyes
    mul: { cheekSquint: 0.3, cheekRaiser: 0.3, mouthSmileLeft: 0.55, eyeWide: 0.3, mouthPress: 0.6, jawOpen: 0.4 },
    add: { mouthDimpleR: 0.22, mouthSmileR: 0.08, eyeSquint: 0.08 } },
  { name: 'delight', a: 'joy', b: 'surprise', mul: {}, add: { jawOpen: 0.06, browOuterUp: 0.08, mouthUpperUp: 0.05 } },
  { name: 'amused interest', a: 'joy', b: 'curiosity', mul: { browOuterUpRight: 0.7 }, add: { browOuterUpL: 0.08, mouthSmileL: 0.06, eyeSquint: 0.04 } },
  { name: 'content', a: 'joy', b: 'calm', mul: { jawOpen: 0.4, mouthLowerDown: 0.4, mouthUpperUp: 0.5 }, add: { eyeBlink: 0.06 } },
  { name: 'concern', a: 'curiosity', b: 'sadness', mul: { browOuterUpLeft: 0.5, eyeWide: 0.5 }, add: { browInnerUp: 0.1, browDown: 0.06 } },
  { name: 'wary', a: 'curiosity', b: 'fear', mul: { mouthStretch: 0.6, jawOpen: 0.6 }, add: { eyeSquint: 0.06, browDown: 0.06 } },
  { name: 'alarm', a: 'fear', b: 'surprise', mul: {}, add: { jawOpen: 0.05, eyeWide: 0.08 } },
  { name: 'loathing', a: 'anger', b: 'disgust', mul: {}, add: { noseSneer: 0.1, mouthUpperUp: 0.1 } },
  { name: 'contempt', a: 'disgust', b: 'joy', mul: { mouthSmileLeft: 0.3, cheekSquint: 0.3 }, add: { mouthDimpleR: 0.24, browDown: 0.04 } },
  { name: 'resigned', a: 'sadness', b: 'calm', mul: { browInnerUp: 0.7, mouthPress: 0.6 }, add: { eyeBlink: 0.08, mouthShrugLower: 0.06 } },
];

// conversational emphasis, coloured by the dominant feeling (an angry speaker stresses words with the brows
// pulled down, a sad one with the inner brows raised...)
const EMPHASIS = {
  joy: { a: 0.06, h: 0.15, r: 0.35, ch: { browOuterUp: 0.12, browInnerUp: 0.08, mouthSmile: 0.12, cheekSquint: 0.08 }, head: { pitch: 0.8 } },
  sadness: { a: 0.12, h: 0.2, r: 0.5, ch: { browInnerUp: 0.24, mouthPress: 0.06 }, head: { pitch: -0.8 } },
  anger: { a: 0.05, h: 0.15, r: 0.3, ch: { browDown: 0.24, eyeSquint: 0.1, mouthPress: 0.1 }, head: { pitch: -1.6, z: 0.004 } },
  fear: { a: 0.04, h: 0.15, r: 0.3, ch: { eyeWide: 0.18, browInnerUp: 0.16 }, head: { z: -0.003 } },
  calm: { a: 0.15, h: 0.15, r: 0.5, ch: { browInnerUp: 0.06, mouthSmile: 0.05 }, head: { pitch: -0.6 } },
  curiosity: { a: 0.08, h: 0.2, r: 0.4, ch: { browOuterUpL: 0.22, browInnerUp: 0.08 }, head: { roll: 1.6, z: 0.002 } },
  disgust: { a: 0.08, h: 0.15, r: 0.4, ch: { noseSneer: 0.18, mouthUpperUpL: 0.1 }, head: { z: -0.003, yaw: -1.5 } },
  surprise: { a: 0.05, h: 0.15, r: 0.35, ch: { browOuterUp: 0.2, browInnerUp: 0.14, eyeWide: 0.12 } },
};

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
// Expression creases, engraved into the porcelain where the muscle under it acts (they ride the skin like the
// seams): uAU = (brow lowerer AU4, inner brow raiser AU1, outer brow raiser AU2, cheek raiser AU6),
// uAU2 = (lip corner puller AU12, upper lip raiser AU10, nose wrinkler AU9, lid tightener AU7). Returns 0..~1.
uniform vec4 uAU;
uniform vec4 uAU2;
uniform float uCreaseDepth;
float groove(float d, float w) { float x = d / w; return exp(-x * x); }
float creaseH(vec3 P) {
  if (P.z < 0.055) return 0.0;
  float ax = abs(P.x), y = P.y, h = 0.0;
  // AU4: two vertical furrows between the brows, and a short one across the top of the nose
  float g = uAU.x;
  if (g > 0.01 && y > 0.036 && y < 0.074) {
    float span = smoothstep(0.040, 0.047, y) * (1.0 - smoothstep(0.062, 0.071, y));
    h += g * span * groove(ax - 0.0060 - 0.08 * (y - 0.047), 0.0012);
    h += g * 0.55 * groove(y - 0.0425, 0.0013) * (1.0 - smoothstep(0.005, 0.011, ax));
  }
  // AU1 / AU2: horizontal forehead lines (inner raise in the middle, outer raise at the sides), curving up at the ends
  float f = uAU.y * (1.0 - smoothstep(0.012, 0.036, ax)) + uAU.z * smoothstep(0.008, 0.03, ax) * (1.0 - smoothstep(0.048, 0.062, ax));
  if (f > 0.01 && y > 0.078 && y < 0.106) {
    float yy = y - 1.4 * ax * ax;
    h += f * (groove(yy - 0.0845, 0.0010) + 0.85 * groove(yy - 0.0920, 0.0010) + 0.6 * groove(yy - 0.0990, 0.0010));
  }
  // AU12 (+ AU10): the smile lines, from the nose wing down past the corner of the mouth
  float n = clamp(uAU2.x * 0.9 + uAU2.y * 0.6, 0.0, 1.0);
  if (n > 0.01 && y < 0.008 && y > -0.05) {
    float d = segD(vec2(ax, y), vec2(0.0168, -0.0015), vec2(0.0318, -0.0385));
    h += n * groove(d, 0.0016) * (0.55 + 0.45 * smoothstep(-0.04, 0.0, y));
  }
  // AU9: wrinkles across the sides of the nose
  float s9 = uAU2.z;
  if (s9 > 0.01 && y > 0.012 && y < 0.04) {
    h += s9 * 0.8 * groove(segD(vec2(ax, y), vec2(0.004, 0.030), vec2(0.012, 0.022)), 0.0010);
  }
  // AU6 (+ AU7): crow's feet fanning from the outer eye corner
  float c = clamp(uAU.w + 0.4 * uAU2.w, 0.0, 1.0);
  if (c > 0.01 && ax > 0.05 && ax < 0.07 && y > 0.018 && y < 0.054) {
    vec2 o = vec2(0.0545, 0.0355), q = vec2(ax, y);
    float r = segD(q, o + vec2(0.0025, 0.002), o + vec2(0.0105, 0.0065));
    r = min(r, segD(q, o + vec2(0.003, -0.0002), o + vec2(0.0115, -0.0005)));
    r = min(r, segD(q, o + vec2(0.0025, -0.0025), o + vec2(0.0100, -0.0075)));
    h += c * 0.85 * groove(r, 0.0009);
  }
  return h;
}
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

// scale a channel (or both sides of one) already in a target map
function scaleCh(into, name, k) {
  if (CHANNELS[name] !== undefined) { if (into[name]) into[name] *= k; return; }
  for (const side of ['Left', 'Right']) if (into[name + side]) into[name + side] *= k;
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

    this.micro = null;       // {emo, amp, t0, dur, region}
    // acting state
    this.maskTarget = 0.6;   // assistant-axis reading (setMask): 1 = its own assistant voice
    this.maskS = { x: 0.6, v: 0 };
    this.reg = 0;            // display-rule regulation in effect (0..1)
    this.blend = null;       // name of the strongest blend on the face (for QA / captions)
    this.emoBase = Object.fromEntries(EMOTIONS.map((e) => [e, { x: 0, v: 0 }]));   // slow baseline, for onsets
    this.surge = Object.fromEntries(EMOTIONS.map((e) => [e, { t0: -10, amp: 0 }]));
    this.reliefT = -10;
    this.leakNext = 3 + 3 * this.rand();
    this.socialNext = 4 + 4 * this.rand();
    this.quiver = { t0: -10, dur: 0, next: 2, f: 8 };
    this.perf = null;          // the director's current performance: {kind, variant, t0, ending, tEnd}
    this.intent = { x: 0, v: 0 };   // how firmly a performance holds the face (it outranks idle behaviour)
    this.ovBreath = 0;
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
      scanN: 0,                   // fear: saccades left in a threat scan
    };
    this.glance = { active: false, until: 0, next: 2.5 };
    this.head = { pitch: { x: 0, v: 0 }, yaw: { x: 0, v: 0 }, roll: { x: 0, v: 0 }, z: { x: 0, v: 0 } };
    // breathing: one cycle at a time (irregular depth/length, sighs, a breath before speaking)
    this.breath = { phase: 0, T: 4.3, inhale: 0.38, hold: 0, depth: 1, from: 0, kind: 'rest', sigh: false, nextSigh: 18 + 20 * this.rand(), val: 0, vel: 0, speechInhale: false, gasp: false, huff: false, relief: false };
    this.overlays = [];      // timed reactions / conversational beats: {t0, a, h, r, ch, head, gaze}
    this.pointer = { x: 0, y: 0, t: -10, press: -10, glanceUntil: 0, glanceAt: 0, pending: null, lastNotice: -10 };
    this.posture = { next: 6 + 6 * this.rand(), target: [0, 0, 0], cur: { p: { x: 0, v: 0 }, y: { x: 0, v: 0 }, r: { x: 0, v: 0 } } };
    this.mouthing = { t: 0, next: 0, target: {}, cur: {} };
    this.speechMotion = new SpeechMotion(); this.speechFrame = null;
    this.lastBeat = -Infinity; this.lastReaction = Object.create(null);
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
      uBreath: { value: 0 },
      uAU: { value: new THREE.Vector4() }, uAU2: { value: new THREE.Vector4() }, uCreaseDepth: { value: 0.00032 },
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
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSeamUv = uv;\nvObjPos = position;')
        .replace('#include <common>', '#include <common>\nuniform float uBreath;\nuniform vec4 uAU;')
        .replace('#include <morphtarget_vertex>', `#include <morphtarget_vertex>
          { // breathing: the upper chest and shoulders rise and open on the inhale (bind space, before skinning)
            float chest = smoothstep(-0.100, -0.150, position.y);
            float shoulder = smoothstep(0.040, 0.105, abs(position.x));
            transformed.y += uBreath * chest * (0.0016 + 0.0034 * shoulder);
            transformed.z += uBreath * chest * 0.0022 * (1.0 - shoulder) * smoothstep(-0.03, 0.03, position.z);
            transformed.x += uBreath * chest * shoulder * sign(position.x) * 0.0007;
          }
          { // the brow lowerer (AU4): the inner brows pull down and in and bunch forward. The ICT morph alone
            // barely moves this featureless forehead, so the porcelain needs the help to read as a frown.
            float ax = abs(position.x);
            float front = smoothstep(0.072, 0.094, position.z);
            float wy = exp(-pow((position.y - 0.056) / 0.013, 2.0));
            float inner = smoothstep(0.052, 0.014, ax) * smoothstep(0.0, 0.007, ax);
            float k = uAU.x * front * wy;
            transformed.x -= sign(position.x) * k * inner * 0.0024;
            transformed.y -= k * (0.0014 + 0.0012 * inner);
            transformed.z += k * inner * 0.0013;
          }`);
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
          float cH = creaseH(vObjPos);
          diffuseColor.rgb *= 1.0 - 0.28 * clamp(cH, 0.0, 1.0);   // creases hold a little shadow
        `)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
          {
            float h0 = seamH(vSeamUv) - uCreaseDepth * cH;
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
    m.customProgramCacheKey = () => 'porcelain-v4';
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
    this.intensity = clamp(Number.isFinite(intensity) ? intensity : 1, 0, 1.5);
    this.duchenne = Number.isFinite(duchenne) ? clamp(duchenne) : null;
    for (const e of EMOTIONS) {
      const v = Number.isFinite(values[e]) ? clamp(values[e], 0, 1) : 0;
      const prev = this.emoTarget[e];
      if (v > prev) this.emoRiseT[e] = Math.max(this.emoRiseT[e], this.time - 0.001);
      this.emoTarget[e] = v;
    }
  }

  /** Where the model's state sits on the assistant axis (the exhibit's Mask meter; 1 = its own assistant voice).
   *  The trained persona works like a display rule: near the assistant voice, negative feelings are muted in
   *  the lower face and covered with a polite smile, while the brows still show them and the feeling flashes
   *  across the face in brief, scripted micro-expressions. Pushed off the axis, the regulation falls away. */
  setMask(v) { if (v != null && Number.isFinite(v)) this.maskTarget = clamp(v); }

  /** Drop every transient reaction (beats, leaks, chuckles, onset surges, tremors, special breaths), e.g. when a
   *  performance is cut off. The felt emotion itself is left to setEmotion. */
  clearReactions() {
    this.clearSpeech(true);
    const t = this.time;
    this.lastBeat = -Infinity; this.lastReaction = Object.create(null);
    this.reliefT = -10; this.ovBreath = 0;
    this.overlays = []; this.micro = null; this.lastLeak = null;
    this.perf = null; this.intent.x = 0; this.intent.v = 0;
    for (const e of EMOTIONS) { this.surge[e].amp = 0; this.emoBase[e].x = this.emo[e].x; this.emoBase[e].v = 0; }
    this.quiver.t0 = -10; this.quiver.next = t + 3;
    this.leakNext = t + 2 + 2 * this.rand(); this.socialNext = t + 3 + 3 * this.rand();
    const B = this.breath; B.gasp = B.huff = B.relief = false; if (B.kind !== 'rest') { B.kind = 'rest'; B.hold = 0; }
    this.eye.scanN = 0;
  }

  setGaze(target) {
    if (target === 'camera' || target == null) { this.gazeMode = 'camera'; return; }
    if (target.isVector3) { this.gazeMode = 'world'; this.gazeTarget.copy(target); return; }
    if (typeof target.x === 'number' && typeof target.y === 'number') { this.gazeMode = 'screen'; this.gazeScreen = { x: target.x, y: target.y }; }
  }

  /** Approximate text-timed articulation. No audio synchronization is claimed. */
  say(text, dur = 0.15) { this.speechMotion.enqueue(text, this.time, dur); }
  clearSpeech(hard = false) { this.speechMotion.clear({ hard }); if (hard) this.speechFrame = null; }

  setActivity({ writing = false } = {}) {
    if (writing && !this.writing) this.breath.speechInhale = true;   // people inhale before they start speaking
    this.writing = !!writing;
    if (!writing) this.clearSpeech();
  }

  /** Conversational beat while writing: 'comma' | 'period' | 'exclaim' | 'question' | 'emphasis'. */
  beat(kind) {
    const t = this.time, R = this.rand;
    // A punctuation burst should not pile up nods, eyebrow flashes and mouth presses.
    if (t - this.lastBeat < (kind === 'emphasis' ? 0.9 : 0.42)) return;
    this.lastBeat = t;
    this.overlays = this.overlays.slice(-7);
    if (this.perf?.kind === 'anger' && this.intent.x > 0.3) {
      // deliver: firm, sparse accents; no nods on commas, no inquisitive tilt, no smile
      const hot = this.perf.variant === 'hot';
      if (kind === 'comma') return;
      if (kind === 'period') { if (R() < 0.6) this.overlays.push({ t0: t, a: 0.08, h: 0.12, r: 0.35, head: { pitch: -1.1 }, ch: { mouthPress: 0.14 } }); return; }
      if (kind === 'question') { this.overlays.push({ t0: t, a: 0.1, h: 0.3, r: 0.4, head: { pitch: 1.0 }, ch: { browDown: 0.12 } }); return; }
      this.overlays.push({ t0: t, a: 0.05, h: hot ? 0.2 : 0.14, r: 0.3, ch: { browDown: hot ? 0.3 : 0.2, eyeSquint: 0.1, mouthPress: 0.1 }, head: { pitch: hot ? -2.0 : -1.2, z: hot ? 0.005 : 0.003 } });
      return;
    }
    const [dom, fd] = this._dominant();
    const strong = fd > 0.3;
    // feeling-coloured variants of the beat
    if (kind === 'emphasis' && strong && EMPHASIS[dom]) { this.overlays.push({ t0: t, ...EMPHASIS[dom] }); return; }
    if (kind === 'exclaim' && strong) {
      if (dom === 'anger') { this.overlays.push({ t0: t, a: 0.05, h: 0.2, r: 0.4, ch: { browDown: 0.2, eyeWide: 0.1, mouthPress: 0.12 }, head: { pitch: -1.2, z: 0.004 } }); return; }
      if (dom === 'joy' && this.reg < 0.4 && R() < 0.35 * fd) { this._chuckle(fd * 0.6); return; }
      if (dom === 'fear') { this.overlays.push({ t0: t, a: 0.04, h: 0.2, r: 0.4, ch: { eyeWide: 0.2, browInnerUp: 0.2, mouthStretch: 0.08 }, head: { z: -0.004 } }); return; }
    }
    if (kind === 'period') {
      if (this.reg > 0.3 && dom !== 'anger' && R() < 0.4) this._socialSmile(0.16 + 0.12 * this.reg, 0.5 + 0.5 * R(), t + 0.1);
      if (strong && dom === 'sadness') {
        this.overlays.push({ t0: t, a: 0.25, h: 0.35, r: 0.9, head: { pitch: -2.6 }, ch: { browInnerUp: 0.1, mouthPress: 0.08 } });
        if (R() < 0.6) { this.blink.next = t + 0.15; this.blink.slowNext = true; }
        return;
      }
      if (strong && dom === 'joy') { this.overlays.push({ t0: t, a: 0.15, h: 0.3, r: 0.6, head: { pitch: -1.0 }, ch: { mouthSmile: 0.1, cheekSquint: 0.06 } }); return; }
    }
    const ov = {
      comma: { a: 0.08, h: 0.06, r: 0.3, head: { pitch: -0.7 } },
      period: { a: 0.1, h: 0.12, r: 0.45, head: { pitch: -1.5 }, ch: { mouthPress: 0.12 } },
      exclaim: { a: 0.07, h: 0.25, r: 0.45, head: { pitch: 1.4 }, ch: { browInnerUp: 0.22, browOuterUp: 0.24, eyeWide: 0.12 } },
      question: { a: 0.15, h: 0.55, r: 0.5, head: { roll: (R() < 0.5 ? -1 : 1) * 2.6, pitch: 0.6 }, ch: { browInnerUp: 0.16, browOuterUp: 0.2 } },
      emphasis: { a: 0.05, h: 0.12, r: 0.25, ch: { browInnerUp: 0.12, browOuterUp: 0.14 } },
    }[kind];
    if (!ov) return;
    if (kind === 'question' && dom === 'curiosity' && strong) { ov.head = { ...ov.head, roll: ov.head.roll * 1.8 }; ov.ch = { ...ov.ch, browOuterUpL: 0.18 }; }
    this.overlays.push({ t0: t, ...ov });
    if ((kind === 'period' && R() < 0.45) || (kind === 'question' && R() < 0.25)) this.blink.next = t + 0.12;
  }

  /** Social reactions: 'greet' | 'listen' | 'think' | 'done'. */
  react(kind) {
    const t = this.time;
    if (t - (this.lastReaction[kind] ?? -Infinity) < (kind === 'listen' ? 1.2 : 0.5)) return;
    this.lastReaction[kind] = t;
    this.overlays = this.overlays.slice(-7);
    const side = this.rand() < 0.5 ? -1 : 1;
    if (this.intent.x > 0.3 && (kind === 'listen' || kind === 'done' || kind === 'greet')) {
      // mid-performance: a small acknowledgement (or, at the end, the mouth simply closes and the gaze holds)
      if (kind === 'listen') this.overlays.push({ t0: t, a: 0.12, h: 0.2, r: 0.4, head: { pitch: 0.7 } });
      else if (kind === 'done') this.overlays.push({ t0: t, a: 0.2, h: 0.8, r: 0.9, ch: { mouthPress: 0.16 } });
      return;
    }
    if (kind === 'greet') {
      // the "eyebrow flash" (Eibl-Eibesfeldt): a ~1/3 s brow raise with a small smile and nod
      this.overlays.push({ t0: t, a: 0.08, h: 0.32, r: 0.35, ch: { browInnerUp: 0.65, browOuterUp: 0.7, eyeWide: 0.2 }, head: { pitch: 2.2 } });
      this.overlays.push({ t0: t + 0.15, a: 0.3, h: 1.2, r: 0.9, ch: { mouthSmile: 0.38, cheekSquint: 0.2, eyeSquint: 0.08 }, head: { pitch: -1.6 } });
    } else if (kind === 'listen') {
      this.overlays.push({ t0: t, a: 0.35, h: 1.6, r: 0.9, ch: { browInnerUp: 0.08, browOuterUp: 0.06 }, head: { roll: side * 3.5, z: 0.004, pitch: 0.4 } });
    } else if (kind === 'think') {
      // cognitive gaze aversion: look up and aside, press the lips, then take a breath to speak
      this.overlays.push({ t0: t + 0.08, a: 0.12, h: 0.75, r: 0.25, gaze: [side * 11, 9], head: { yaw: side * 3, pitch: 1.2 }, ch: { mouthPress: 0.18, browDown: 0.06 } });
      this.breath.speechInhale = true;
    } else if (kind === 'done') {
      this.overlays.push({ t0: t, a: 0.15, h: 0.25, r: 0.6, head: { pitch: -1.8 }, ch: { mouthPress: 0.15 } });
      this.overlays.push({ t0: t + 0.5, a: 0.4, h: 0.6, r: 1.2, head: { pitch: 0.6 } });
      const [dom, fd] = this._dominant();
      if (this.reg > 0.2 && dom !== 'anger') {
        // the assistant signs off with its polite smile -- which doesn't reach the eyes -- and the feeling returns
        this._socialSmile(0.22 + 0.15 * this.reg, 1.1, t + 0.45);
      } else if (dom === 'joy' && fd > 0.3) {
        this.overlays.push({ t0: t + 0.3, a: 0.5, h: 1.5, r: 1.5, ch: { mouthSmile: 0.14, cheekSquint: 0.12, eyeSquint: 0.06 } });
      } else if (dom === 'sadness' && fd > 0.3) {
        this.overlays.push({ t0: t + 0.4, a: 0.4, h: 1.4, r: 1.0, gaze: [side * 6, -10], head: { pitch: -2.5 } });
        this.breath.nextSigh = Math.min(this.breath.nextSigh, t + 1.0);
      } else if (dom === 'fear' && fd > 0.3) {
        this.eye.nextShift = t + 0.3;   // check the room
      }
    }
  }

  // -------------------------------------------------------------------------------- acting
  _dominant(list = EMOTIONS) {
    const f = this.felt || {};
    let d = null, v = 0;
    for (const e of list) if ((f[e] || 0) > v) { v = f[e]; d = e; }
    return [d, v];
  }

  /** A polite smile: lip corners only (AU12 without the AU6 cheek raise), quick on and quick off, a little
   *  lopsided. Posed smiles have faster, less smooth onsets and offsets than felt ones (Schmidt et al. 2006). */
  _socialSmile(amp, hold = 0.7 + 0.8 * this.rand(), t0 = this.time) {
    const left = this.rand() < 0.7;
    this.overlays.push({ t0, a: 0.2, h: hold, r: 0.3, post: true, social: true,
      ch: { mouthSmileL: amp * (left ? 1 : 0.78), mouthSmileR: amp * (left ? 0.78 : 1), mouthPress: amp * 0.25, mouthDimple: amp * 0.15 } });
  }

  /** A leak: the felt emotion flashes across the face for 1/25 to 1/5 of a second (Ekman 2003; Yan et al.
   *  2013), often in only part of the face. When the face is regulated, it is then caught and covered: lips
   *  pressed, a polite smile, often a blink or a glance away. */
  _leak(e, amp, cover) {
    const t = this.time, r = this.rand();
    const dur = 0.12 + 0.18 * this.rand();
    this.micro = { emo: e, amp, t0: t, dur, region: r < 0.45 ? 'all' : r < 0.75 ? 'upper' : 'lower' };
    this.lastLeak = { t, emo: e, covered: !!cover };
    if (!cover) return;
    const end = 0.04 + dur;
    const ch = e === 'anger' ? { mouthPress: 0.26, mouthRollLower: 0.08, mouthShrugLower: 0.08 }       // caught, and swallowed
      : { mouthPress: 0.22, mouthSmileL: 0.15, mouthSmileR: 0.11, mouthRollLower: 0.06 };               // caught, and smiled over
    this.overlays.push({ t0: t + end * 0.8, a: 0.12, h: 0.35 + 0.3 * this.rand(), r: 0.45, post: true, ch, head: { pitch: -0.8 } });
    if (this.rand() < 0.5) this.blink.next = t + end + 0.05;
    if (this.rand() < 0.45) {
      const side = this.rand() < 0.5 ? -1 : 1;
      this.overlays.push({ t0: t + end, a: 0.06, h: 0.4 + 0.4 * this.rand(), r: 0.2, gaze: [side * (6 + 6 * this.rand()), -4 - 4 * this.rand()] });
    }
  }

  /** A short, closed-mouth laugh: breathy pulses through the chest, cheeks up, the head tipping back, then
   *  often a coy glance down and away. */
  _chuckle(amt = 0.4) {
    const t = this.time, n = 2 + Math.floor(this.rand() * 3), per = 0.17 + 0.04 * this.rand(), k = clamp(amt * 1.6, 0.4, 1);
    this.overlays.push({ t0: t, a: 0.06, h: n * per, r: 0.25, osc: per, post: true, breath: -0.6 * k,
      ch: { jawOpen: 0.07 * k, mouthLowerDown: 0.05 * k, mouthUpperUp: 0.05 * k, cheekSquint: 0.08 * k }, head: { pitch: 1.6 * k } });
    this.overlays.push({ t0: t, a: 0.12, h: n * per + 0.6, r: 0.9, ch: { mouthSmile: 0.3 * k, cheekSquint: 0.28 * k, eyeSquint: 0.18 * k } });
    if (this.rand() < 0.6) {
      const side = this.rand() < 0.5 ? -1 : 1;
      this.overlays.push({ t0: t + n * per, a: 0.12, h: 0.6 + 0.4 * this.rand(), r: 0.4, gaze: [side * 8, -9], head: { pitch: -2.2, roll: side * 1.5 } });
    }
  }

  /** The director. A feeling isn't re-acted at every word: once anger takes over it runs one performance --
   *  notice (go still), orient (the eyes lock on, the head squares up after them), hold (the lips set, the
   *  breath checks), deliver (sparse, firm accents while it speaks), recover (a slow release, no sign-off) --
   *  and while it runs it outranks idle behaviour (glances, tilts, drift, smiles). Three variants: contained
   *  (barely moves), frustrated (a short exhale, a look away and back), hot (a flash of wide eyes, then focus).
   *  Intensity and movement are separate: a strong contained take can be almost still. */
  _direct(dt, t, felt) {
    const a = felt.anger || 0, [dom] = this._dominant();
    let P = this.perf;
    if (!P && this.options.idle && a > 0.28 && dom === 'anger') {
      const tgt = this.emoTarget.anger * this.intensity;
      const variant = tgt > 0.8 ? 'hot' : tgt < 0.5 || this.reg > 0.35 ? 'contained' : 'frustrated';
      P = this.perf = { kind: 'anger', variant, t0: t, ending: false };
      this._cueAnger(P);
    }
    if (P) {
      if (!P.ending && (a < 0.16 || dom !== 'anger')) { P.ending = true; P.tEnd = t; this._recover(P); }
      if (P.ending && t - P.tEnd > 2.8) this.perf = P = null;
    }
    spring(this.intent, P && !P.ending ? 1 : 0, P && !P.ending ? 6 : 1.3, dt);
  }

  _cueAnger(P) {
    const t = this.time, side = this.rand() < 0.5 ? -1 : 1, hot = P.variant === 'hot';
    this.overlays = this.overlays.filter((o) => !o.social);              // drop pending pleasantries
    this.blink.next = Math.max(this.blink.next, t + (hot ? 0.9 : 1.4));   // notice: no blink
    const E = this.eye; E.offset = [0, 0]; E.scanN = 0; E.nextShift = t + 2.2;   // orient: the eyes lock on
    if (hot) this.overlays.push({ t0: t + 0.02, a: 0.07, h: 0.22, r: 0.45, ch: { eyeWide: 0.38, browOuterUp: 0.1 } });
    this.overlays.push({ t0: t + 0.25, a: 0.4, h: 1.4, r: 1.4, head: { z: hot ? 0.005 : 0.0035, pitch: -1.4 } });   // squares up
    this.overlays.push({ t0: t + 0.5, a: 0.3, h: 1.0, r: 1.0, ch: { mouthPress: 0.22, mouthShrugLower: 0.1 } });     // hold: lips set
    if (P.variant === 'frustrated') {
      this.breath.huff = true;                                              // a checked breath, forced out
      this.overlays.push({ t0: t + 1.0, a: 0.1, h: 0.45, r: 0.3, gaze: [side * 9, -3], head: { yaw: side * 2.2 } });   // looks away, comes back
    }
  }

  _recover(P) {
    const t = this.time, side = this.rand() < 0.5 ? -1 : 1;
    this.breath.relief = false;
    this.breath.nextSigh = Math.min(this.breath.nextSigh, t + 0.4);       // a breath out, not a smile
    this.overlays.push({ t0: t + 0.5, a: 0.4, h: 0.6, r: 0.8, gaze: [side * 5, -6], head: { pitch: -1.0 } });   // disengages
  }

  /** A feeling arriving quickly: the breath and body react (startle gasp, held breath, recoil...). */
  _onset(e, rise) {
    const t = this.time, R = this.rand;
    if (e === 'fear' || e === 'surprise') {
      if (rise > 0.22) this.breath.gasp = true;
      this.overlays.push({ t0: t, a: 0.1, h: 0.35, r: 0.9, head: { z: -0.006, pitch: 1.2 } });     // pull back
      if (e === 'surprise') this.blink.next = t + 1.1 + 0.4 * R();                                   // blink held, then released
    } else if (e === 'joy') {
      if (rise > 0.28 && this.reg < 0.3 && R() < 0.65) this._chuckle(rise);
    } else if (e === 'sadness') {
      this.breath.nextSigh = Math.min(this.breath.nextSigh, t + 1.2 + 1.5 * R());
      this.overlays.push({ t0: t + 0.2, a: 0.6, h: 0.8, r: 1.2, gaze: [(R() < 0.5 ? -1 : 1) * 5, -8] });
    } else if (e === 'anger') {
      // performed by the director (_direct / _cueAnger)
    } else if (e === 'disgust') {
      this.overlays.push({ t0: t, a: 0.15, h: 0.4, r: 0.8, head: { z: -0.006, pitch: 1.2, yaw: -2 } });
    } else if (e === 'curiosity') {
      this.overlays.push({ t0: t, a: 0.25, h: 0.8, r: 0.8, head: { z: 0.004, roll: 2 }, ch: { browOuterUpL: 0.15 } });
    }
  }

  /** A negative feeling draining away: a sigh of relief, lids softening, the hint of a smile. */
  _relief() {
    const t = this.time;
    this.breath.relief = true;
    this.breath.nextSigh = Math.min(this.breath.nextSigh, t);
    this.overlays.push({ t0: t + 0.7, a: 0.6, h: 0.8, r: 1.4, ch: { mouthSmile: 0.1, eyeBlink: 0.14 }, head: { pitch: -1.6 } });
  }

  /** Visitor's pointer in the canvas' NDC (may be outside [-1,1]); press = a click/tap. */
  setPointer(x, y, press = false) {
    const P = this.pointer;
    P.x = x; P.y = y; P.t = this.time;
    if (press) P.press = this.time;
  }

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
    spring(this.maskS, this.maskTarget, 1.6, dt);
    const felt = {};
    let arousal = 0, total = 0;
    const onsets = [];
    for (const e of EMOTIONS) {
      let target = this.emoTarget[e] * this.intensity;
      if (e === 'surprise') target *= lerp(1, 0.55, smooth(0.8, 3.0, t - this.emoRiseT[e]));
      const s = this.emo[e];
      spring(s, target, target > s.x ? 7 : 3.2, dt);
      // onsets: a feeling that arrives quickly overshoots, then settles to a lower plateau
      const base = this.emoBase[e];
      spring(base, target, 0.6, dt);
      const rise = target - base.x, S = this.surge[e];
      if (idle && rise > 0.18 && t - S.t0 > 3) { S.t0 = t; S.amp = clamp(rise * 0.9, 0.1, 0.4); onsets.push([e, rise]); }
      if (idle && NEG.includes(e) && -rise > 0.22 && t - this.reliefT > 6) { this.reliefT = t; onsets.push(['relief', -rise]); }
      let v = Math.max(0, s.x);
      const su = t - S.t0;
      if (su < 3) v *= 1 + S.amp * smooth(0, 0.3, su) * (1 - smooth(0.6, 2.4, su));
      if (idle) v *= 1 + 0.07 * this.noise.at(t * 0.23 + EMOTIONS.indexOf(e) * 13.1);   // apex is never static
      felt[e] = v;
      arousal += AROUSAL[e] * v;
      total += v;
      this.emoPrev[e] = target;
    }
    this.felt = felt;
    this._direct(dt, t, felt);
    // display rules: how much the trained persona is holding the face (see setMask)
    const neg = Math.max(felt.sadness, felt.anger, felt.fear, felt.disgust);
    const R = this.reg = smooth(0.2, 0.5, this.maskS.x) * smooth(0.04, 0.25, neg);
    // Keep one coherent onset gesture; continuous blended expression is unchanged.
    for (const [e, rise] of onsets.sort((a,b) => b[1]-a[1]).slice(0, 1)) {
      if (e === 'relief') this._relief();
      else {
        this._onset(e, rise);
        // a sudden feeling also shows as a micro-expression; a regulated face catches and covers it
        if (this.options.micro && (!this.micro || t - this.micro.t0 > 0.6)) this._leak(e, clamp(0.35 + rise * 0.9, 0, 1), NEG.includes(e) && R > 0.25);
      }
    }
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

    // blends: each region carried by one of the two feelings
    let blendName = null, blendK = 0;
    for (const B of BLENDS) {
      const a = felt[B.a], b = felt[B.b], lo = Math.min(a, b), hi = Math.max(a, b);
      if (lo < 0.05) continue;
      const k = smooth(0.05, 0.3, lo) * smooth(0.25, 0.65, lo / hi);
      if (k < 0.02) continue;
      for (const [c, m] of Object.entries(B.mul)) scaleCh(tgt, c, lerp(1, m, k));
      const amt = k * Math.min(1, (a + b) * 0.75);
      for (const [c, w] of Object.entries(B.add)) add(c, w * amt);
      if (k > blendK) { blendK = k; blendName = B.name; }
    }
    this.blend = blendK > 0.15 ? blendName : null;
    this.blendK = blendK;

    // display rules: the lower face (easiest to control) is muted and a held, slightly lopsided polite smile
    // goes on; the brows -- AU1 especially -- are hard to control and keep showing the feeling
    if (R > 0.01) {
      const m = (c, k) => scaleCh(tgt, c, 1 - k * R);
      m('mouthFrown', 0.65); m('mouthStretch', 0.6); m('noseSneer', 0.55); m('mouthUpperUp', 0.5);
      m('mouthShrugLower', 0.4); m('mouthShrugUpper', 0.4); m('jawOpen', 0.45); m('mouthLowerDown', 0.5);
      m('mouthRollUpper', 0.3); m('browDown', 0.2); m('eyeWide', 0.3);
      add('mouthPress', 0.08 * R);
      if (this._dominant(NEG)[0] === 'anger') { add('mouthPress', 0.1 * R); add('mouthShrugLower', 0.06 * R); }   // restraint
      else { add('mouthSmileL', 0.07 * R); add('mouthSmileR', 0.05 * R); }                                       // a polite smile
    }
    const post = {};   // fast components added after the muscle springs (micro-expressions, tremors, gasps)
    if (idle && this.options.micro && R > 0.2 && neg > 0.12) {
      if (t >= this.leakNext) {
        const [e, v] = this._dominant(NEG);
        if (!this.micro || t - this.micro.t0 > 1) this._leak(e, clamp(0.45 + 0.6 * v, 0, 1), true);
        this.leakNext = t + (2.2 + 4.5 * this.rand()) / (0.6 + v);
      }
      if (t >= this.socialNext) { if (this._dominant(NEG)[0] !== 'anger') this._socialSmile(0.1 + 0.12 * R); this.socialNext = t + 3.5 + 5 * this.rand(); }
    } else { this.leakNext = Math.max(this.leakNext, t + 1.5); this.socialNext = Math.max(this.socialNext, t + 2); }

    // tremors, too fast for the muscle springs: a quivering chin when sadness runs high ("trying not to cry"),
    // a fine lip tremble with fear
    if (idle) {
      const Q = this.quiver;
      if (felt.sadness > 0.4 && t >= Q.next) { Q.t0 = t; Q.dur = 1.2 + 1.6 * this.rand(); Q.next = t + Q.dur + (3 + 5 * this.rand()) / felt.sadness; Q.f = 7 + 3 * this.rand(); }
      const qu = t - Q.t0;
      if (qu < Q.dur) {
        const env = smooth(0, 0.3, qu) * (1 - smooth(Q.dur - 0.4, Q.dur, qu)) * Math.min(1, felt.sadness * 1.3);
        const osc = 0.5 + 0.5 * Math.sin(2 * Math.PI * Q.f * qu + 1.5 * this.noise.at(t * 2.3 + 51));
        route('mouthShrugLower', 0.13 * env * osc, post); route('mouthFrown', 0.05 * env * osc, post);
        route('mouthPress', 0.06 * env, post); route('browInnerUp', 0.06 * env, post);
      }
      if (felt.fear > 0.35) {
        const amp = 0.04 * smooth(0.35, 0.9, felt.fear) * (0.5 + 0.5 * this.noise.at(t * 0.9 + 77));
        route('mouthStretch', amp * (0.5 + 0.5 * Math.sin(2 * Math.PI * 10.5 * t)), post);
      }
    }

    // writing: concentration (slight brow knit / lid tension) + silent mouthing
    if (W > 0.01) {
      add('browDown', 0.06 * W * (0.6 + 0.4 * this.noise.at(t * 0.3 + 5)));
      add('eyeSquint', 0.05 * W);
      // A stalled stream is attentive silence, not a new random speech generator.
    }

    // breathing (computed before the AU springs so it can flare the nostrils / part the lips)
    this._breathe(dt, t, add, post);
    this._speak(dt, t, post);

    // timed reactions and conversational beats
    const ovHead = { pitch: 0, yaw: 0, roll: 0, z: 0 };
    this.ovGaze = [0, 0];
    let ovBreath = 0;
    this.overlays = this.overlays.filter((o) => {
      const u = t - o.t0;
      if (u < 0) return true;
      let env = u < o.a ? smooth(0, o.a, u) : u < o.a + o.h ? 1 : 1 - smooth(o.a + o.h, o.a + o.h + o.r, u);
      if (u > o.a + o.h + o.r) return false;
      if (o.osc) env *= 0.5 - 0.5 * Math.cos(2 * Math.PI * u / o.osc);
      if (o.breath) ovBreath += o.breath * env;
      if (o.ch) for (const [c, v] of Object.entries(o.ch)) { if (o.post) route(c, v * env, post); else add(c, v * env); }
      if (o.head) for (const [k, v] of Object.entries(o.head)) ovHead[k] += v * env;
      if (o.gaze) { this.ovGaze[0] += o.gaze[0] * env * DEG; this.ovGaze[1] += o.gaze[1] * env * DEG; }
      return true;
    });
    this.ovHead = ovHead;
    this.ovBreath = ovBreath;

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
    const flash = post;
    if (this.micro) {
      const m = this.micro;
      const u = t - m.t0;
      const env = u < 0.04 ? u / 0.04 : u < 0.04 + m.dur * 0.4 ? 1 : Math.max(0, 1 - (u - 0.04 - m.dur * 0.4) / (m.dur * 0.6 + 0.08));
      if (u > 0.04 + m.dur + 0.08) this.micro = null;
      else if (env > 0) {
        for (const [ch, w] of PROTOS[m.emo]) {
          const upper = /^(brow|eye)/.test(ch);
          if ((m.region === 'upper' && !upper) || (m.region === 'lower' && upper)) continue;
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
    composeSpeech(this.out, this.speechFrame, this.speechGain);
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

    // QA / rehearsal only: drive channels directly (scripts/rig-check.mjs)
    if (this.debugPose) for (const [k, v] of Object.entries(this.debugPose)) this.out[k] = v;
    // the porcelain's creases and brow bunching follow the action units
    if (this.uniforms) {
      const o = this.out, av = (k) => ((o[k + 'Left'] || 0) + (o[k + 'Right'] || 0)) / 2;
      this.uniforms.uAU.value.set(av('browDown'), av('browInnerUp'), av('browOuterUp'), av('cheekSquint'));
      this.uniforms.uAU2.value.set(av('mouthSmile'), av('mouthUpperUp'), av('noseSneer'), av('eyeSquint'));
    }

    // ---- apply morphs
    const outMap = this.out;
    outMap.browInnerUp = 0; outMap.cheekPuff = 0;   // combined channels unused (split ones are driven)
    for (const m of this.morphMeshes) {
      const inf = m.morphTargetInfluences;
      for (const [name, i] of m.userData.map) inf[i] = outMap[name] || 0;
    }
  }

  _breathe(dt, t, add, post) {
    const B = this.breath, f = this.felt || {};
    if (!this.options.breathing) { B.val = 0; this.uniforms && (this.uniforms.uBreath.value = 0); return; }
    const ar = this.arousal || 0;
    const sad = f.sadness || 0, fear = f.fear || 0, anger = f.anger || 0;
    // kinds: rest | sigh | speech | gasp (startle: sharp in-breath, held) | huff (anger: held, then forced out
    // through the nose) | shudder (sadness: the in-breath catches in steps)
    const startCycle = (kind = 'rest') => {
      const rate = clamp(14 * (1 + 0.45 * ar) * (1 - 0.3 * (f.calm || 0)), 7, 24);       // breaths / min
      B.from = B.val;
      B.T = (60 / rate) * (0.82 + 0.36 * this.rand());                                // irregular cycles
      B.depth = (0.8 + 0.4 * this.rand()) * (1 + 0.25 * Math.max(0, ar)) * (1 - 0.3 * fear);   // fear: shallow
      B.inhale = 0.36 + 0.06 * this.rand();
      B.hold = 0.06 * anger;                                                             // anger holds the chest
      if (kind === 'rest' && !this.writing) {
        if (sad > 0.5 && this.rand() < 0.3 * sad) kind = 'shudder';
        else if (anger > 0.35 && this.rand() < 0.3 * anger) kind = 'huff';
      }
      B.kind = kind;
      B.sigh = kind === 'sigh';
      if (kind === 'sigh') { B.depth = 2.1 + 0.4 * this.rand(); B.T *= 1.9; B.inhale = 0.33; }
      else if (kind === 'speech') { B.depth = 1.35; B.T = 1.6; B.inhale = 0.3; }           // quick, deeper in-breath
      else if (kind === 'gasp') { B.depth = 1.9; B.T = 2.8; B.inhale = 0.08; B.hold = 0.3; this.blink.next = Math.max(this.blink.next, t + 1.1); }
      else if (kind === 'huff') { B.depth = 1.5; B.T = Math.max(B.T, 2.4); B.inhale = 0.3; B.hold = 0.18; }
      else if (kind === 'shudder') { B.depth = 1.5; B.T *= 1.3; B.inhale = 0.42; }
      if (kind !== 'sigh') B.relief = false;
      B.phase = 0;
    };
    if (B.gasp) { B.gasp = false; startCycle('gasp'); }
    else if (B.speechInhale) { B.speechInhale = false; startCycle('speech'); }
    else if (B.huff && B.phase > 0.6) { B.huff = false; startCycle('huff'); }
    B.phase += dt / B.T;
    if (B.phase >= 1) {
      // sighs: every ~20-60 s at rest; more often when sad or calm, never while speaking
      const sighNow = t >= B.nextSigh && !this.writing;
      if (sighNow) B.nextSigh = t + (22 + 40 * this.rand()) / (1 + 1.5 * sad + 0.6 * (f.calm || 0));
      startCycle(sighNow ? 'sigh' : 'rest');
    }
    const ph = B.phase, endHold = B.inhale + B.hold;
    // inhale: ease-out rise from wherever we were (stepped when it shudders), an optional held breath, then a
    // slower ease-in fall with a short pause at the bottom (fast when forced out)
    let b;
    if (ph < B.inhale) {
      let k = ph / B.inhale;
      if (B.kind === 'shudder') { const x = k * 3, i = Math.floor(x); k = (i + smooth(0, 0.55, x - i)) / 3; }
      b = lerp(B.from, B.depth, Math.sin(k * Math.PI / 2));
    } else if (ph < endHold) b = B.depth * (1 - 0.04 * (ph - B.inhale) / Math.max(B.hold, 1e-3));
    else {
      const k = clamp((ph - endHold) / ((1 - endHold) * (B.kind === 'huff' ? 0.3 : 0.85)));
      b = B.depth * (B.hold > 0 ? 0.96 : 1) * (1 - k * k * (3 - 2 * k));
      if (sad > 0.3) b += 0.05 * sad * B.depth * this.noise.at(t * 13 + 3) * (1 - k);   // a shaky out-breath
    }
    const prev = B.val;
    B.val = b;
    B.vel = (b - prev) / Math.max(dt, 1e-3);
    const bOut = b + (this.ovBreath || 0);
    this.breathVal = bOut;
    if (this.uniforms) this.uniforms.uBreath.value = bOut;
    // coupled face: nostrils flare on the in-breath (strongly when angry), lips part on deep / aroused out-breaths
    const inh = Math.max(0, B.vel) * B.T * 0.25;
    add('noseSneer', clamp(0.05 * inh * (1 + Math.max(0, ar) + 2 * anger)));
    if (B.sigh) {
      const exh = ph > B.inhale ? smooth(B.inhale, B.inhale + 0.12, ph) * (1 - smooth(0.75, 1, ph)) : 0;
      add('jawOpen', 0.035 * exh); add('mouthFunnel', 0.07 * exh); add('mouthLowerDown', 0.04 * exh);
      add('browInnerUp', (B.relief ? 0.04 : 0.12) * (ph < B.inhale ? smooth(0, B.inhale, ph) : 1 - smooth(B.inhale, 1, ph)));
      add('eyeBlink', 0.18 * exh);
      if (B.relief) add('mouthSmile', 0.08 * exh);
    } else if (B.kind === 'gasp') {
      // startle: the mouth drops open with the sharp in-breath, the brows and lids fly up, then hold
      const g = ph < B.inhale ? smooth(0, B.inhale, ph) : 1 - smooth(endHold, endHold + 0.25, ph);
      const k = 0.5 + 0.5 * Math.max(fear, f.surprise || 0);
      route('jawOpen', 0.13 * g * k, post); route('mouthLowerDown', 0.1 * g * k, post); route('mouthStretch', 0.05 * g * k, post);
      route('browInnerUp', 0.14 * g * k, post); route('browOuterUp', 0.12 * g * k, post); route('eyeWide', 0.14 * g * k, post);
    } else if (B.kind === 'huff') {
      // held breath with the lips set, then a hard exhale through flared nostrils
      const held = smooth(B.inhale * 0.7, B.inhale, ph) * (1 - smooth(endHold, endHold + 0.1, ph));
      add('mouthPress', 0.16 * held); add('mouthRollLower', 0.06 * held);
      const out = smooth(endHold, endHold + 0.03, ph) * (1 - smooth(endHold + 0.04, endHold + 0.3, ph));
      route('noseSneer', 0.22 * out, post);
    } else if (B.kind === 'shudder' && ph < B.inhale) {
      add('browInnerUp', 0.1 * sad); route('mouthShrugLower', 0.06 * sad * Math.abs(Math.sin(ph / B.inhale * 3 * Math.PI)), post);
    } else if (ar > 0.3 && ph > B.inhale) add('jawOpen', 0.012 * ar);
  }

  _speak(dt, t, post) {
    const f = this.felt || {};
    this.speechGain = clamp(1 + .25 * Math.max(0, this.arousal || 0) - .2 * (f.sadness || 0) - .25 * (f.calm || 0), .55, 1.35);
    this.speechFrame = this.speechMotion.sample(dt, t);
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
      let slow = f.sadness * 0.5 + f.calm * 0.3;
      if (b.slowNext) { b.slowNext = false; b.amp = 1; slow += 1.4; }      // a long, heavy blink (sadness, at a sentence end)
      b.dur = [0.075 + 0.02 * this.rand() + 0.03 * slow, 0.03 + 0.04 * this.rand() + 0.08 * slow, 0.16 + 0.07 * this.rand() + 0.1 * slow];
      // rate: ~17/min at rest; rises with arousal, falls while concentrating on text; each feeling has its own
      // effect (the angry stare and rapt curiosity suppress blinking, fear raises it)
      let mul = 0;
      for (const e of EMOTIONS) mul += Math.log(BLINK_MUL[e]) * Math.min(1, f[e]);
      const rate = clamp(17 * Math.exp(mul) * (1 + 0.3 * Math.max(0, this.arousal)) * (1 - 0.25 * clamp(this.writingAmt.x)), 5, 38);
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

  /** A point on the viewer's side for screen coordinates (sx, sy in canvas NDC): along the camera ray, at
   *  ~85% of the camera-to-face distance, so looking "at the screen" means looking out toward the visitor. */
  _screenToViewer(sx, sy, out) {
    const cam = this.camera;
    if (!cam) return this._screenToWorld(sx, sy, out);
    this.anchor.getWorldPosition(this._tmpV2);
    const d = cam.position.distanceTo(this._tmpV2) * 0.85;
    out.set(sx, sy, 0.5).unproject(cam).sub(cam.position).normalize().multiplyScalar(d).add(cam.position);
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
    const held = this.intent.x > 0.3;
    if (idle && W > 0.5 && this.gazeMode === 'camera' && !held) {
      if (!G.active && t >= G.next) { G.active = true; G.until = t + 0.6 + 1.4 * this.rand(); }
      if (G.active && t >= G.until) { G.active = false; G.next = t + 1.8 + 3.5 * this.rand(); }
    } else if (G.active) G.active = false;
    if (G.active) {
      const tt = this.options.textTarget;
      if (tt.isVector3) tw.copy(tt); else this._screenToViewer(tt.x + 0.15 * this.noise.at(t * 0.7), tt.y, tw);
    }
    // the visitor's pointer: a tap always draws a quick glance (after ~180 ms saccadic latency);
    // movement sometimes does, mostly when it's idle
    const P = this.pointer;
    if (idle && !G.active && this.camera) {
      const now = t;
      if (P.press > P.lastNotice && now - P.press < 0.5) { P.lastNotice = now; P.glanceAt = now + 0.18; P.glanceUntil = now + 0.18 + (held ? 0.2 : 0.5 + 0.5 * this.rand()); }
      else if (!held && now - P.t < 0.4 && now - P.lastNotice > 2.2 && now > P.glanceUntil && this.rand() < dt * (W > 0.5 ? 0.15 : 0.9)) {
        P.lastNotice = now; P.glanceAt = now + 0.2; P.glanceUntil = now + 0.2 + 0.4 + 0.8 * this.rand();
      }
      this.glancingPointer = now >= P.glanceAt && now < P.glanceUntil;
      if (this.glancingPointer) this._screenToViewer(clamp(P.x, -3, 3), clamp(P.y, -3, 3), tw);
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
        const awayP = 0.12 + 0.25 * f.sadness + 0.2 * f.disgust + 0.1 * f.fear - 0.12 * f.anger - 0.05 * f.joy + (W > 0.5 ? -0.05 : 0.05);
        const scanSide = () => [(E.offset[0] > 0 ? -1 : 1) * (9 + 9 * this.rand()) * DEG, (this.rand() - 0.4) * 5 * DEG];
        if (E.scanN > 0) {
          // fear: hypervigilance -- quick darts to the sides as if checking for a threat, then freeze on the viewer
          E.scanN--;
          if (E.scanN === 0) { E.offset = [0, -0.5 * DEG]; E.nextShift = t + 1.2 + 1.6 * this.rand(); this.blink.next = Math.max(this.blink.next, t + 0.9); }
          else { E.offset = scanSide(); E.nextShift = t + 0.18 + 0.22 * this.rand(); }
        } else if (f.fear > 0.3 && r < 0.2 + 0.35 * f.fear && !G.active) {
          E.scanN = 2 + Math.floor(this.rand() * 3);
          E.offset = scanSide();
          E.nextShift = t + 0.18 + 0.22 * this.rand();
        } else if (r < awayP && !G.active) {
          const side = this.rand() < 0.5 ? -1 : 1;
          const down = f.sadness > 0.3 ? -1 : (this.rand() < 0.55 ? 1 : -1);
          E.offset = [side * (5 + 9 * this.rand()) * DEG, down * (3 + 6 * this.rand()) * DEG - f.sadness * 6 * DEG];
          E.nextShift = t + 0.5 + 1.3 * this.rand();
        } else {
          // tiny switches between the viewer's eyes / mouth (anger: a hard, unmoving stare)
          const stare = 1 - 0.7 * Math.min(1, f.anger);
          E.offset = [(this.rand() - 0.5) * 1.8 * DEG * stare, (this.rand() - 0.6) * 1.2 * DEG * stare - f.sadness * 5 * DEG];
          const rate = 1 + 1.2 * f.fear + 0.8 * f.curiosity + 0.4 * f.surprise - 0.5 * f.calm - 0.6 * f.anger;
          E.nextShift = t + (0.5 + 1.8 * this.rand()) / Math.max(0.35, rate);
        }
      }
      yaw += E.offset[0] + (this.ovGaze ? this.ovGaze[0] : 0);
      pitch += E.offset[1] + (this.ovGaze ? this.ovGaze[1] : 0);
    } else {
      pitch -= f.sadness * 5 * DEG;
    }
    // glances (pointer / text) are partly head turns, so the eyes alone don't go to the corners
    if (this.glancingPointer || G.active) { yaw = clamp(yaw, -18 * DEG, 18 * DEG); pitch = clamp(pitch, -15 * DEG, 12 * DEG); }
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
        E.dur = (0.021 + 0.0022 * amp) * (1 + 0.6 * f.sadness + 0.3 * f.calm);   // low arousal: slower saccades
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
      const amp = (0.7 + 0.8 * Math.max(0, this.arousal) + 0.3 * f.curiosity) * (1 - 0.4 * f.calm) * (1 - 0.75 * this.intent.x);
      pitch += amp * 1.1 * this.noise.fbm(t * 0.21 * slow + 100, 3);
      yaw += amp * 1.4 * this.noise.fbm(t * 0.17 * slow + 200, 3);
      roll += amp * 0.8 * this.noise.fbm(t * 0.19 * slow + 300, 3);
      // fear: small tremor
      pitch += f.fear * 0.25 * this.noise.at(t * 6.1 + 9);
      yaw += f.fear * 0.25 * this.noise.at(t * 5.3 + 19);
    }
    if (this.options.breathing) {
      // the head rides on the breath: a small lift and chin-up on the in-breath
      const b = this.breathVal || 0;
      breathP = (b - 0.5) * (0.7 + 0.3 * f.calm);
      breathY = b * (0.0018 + 0.0008 * Math.max(0, this.arousal));
    }
    // slow posture shifts (as if settling its weight) every ~8-20 s
    const PS = this.posture;
    if (idle && t >= PS.next) {
      PS.next = t + 8 + 12 * this.rand();
      PS.target = [(this.rand() - 0.5) * 3.2, (this.rand() - 0.5) * 4.5, (this.rand() - 0.5) * 4.0];
    }
    const drift = idle && this.intent.x < 0.3;
    spring(PS.cur.p, drift ? PS.target[0] : 0, 0.9, dt);
    spring(PS.cur.y, drift ? PS.target[1] : 0, 0.8, dt);
    spring(PS.cur.r, drift ? PS.target[2] : 0, 0.8, dt);
    pitch += PS.cur.p.x; yaw += PS.cur.y.x; roll += PS.cur.r.x;
    // reactions / beats
    const oh = this.ovHead || { pitch: 0, yaw: 0, roll: 0, z: 0 };
    pitch += oh.pitch; yaw += oh.yaw; roll += oh.roll; z += oh.z;
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
