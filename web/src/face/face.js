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
import { NATURAL_FOLDS_GLSL, CORRECTIVE_GLSL } from './natural-folds.js';
import { reliefSample } from '../acting/relief-take.js';
import { SpeechMotion, composeSpeech } from './speech-motion.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
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

// The director's performances: how firmly each holds attention (glances, cursor and drift yield above 0.3) and how
// much the head moves (idle noise, posture drift). Values from the acting-lab takes the user approved.
const PERF_STYLE = {
  neutral: { hold: 0.15, amp: 0.65, posture: 0.65 },
  contained: { hold: 0.95, amp: 0.18, posture: 0.2 },
  frustrated: { hold: 0.9, amp: 0.3, posture: 0.25 },
  hot: { hold: 1.0, amp: 0.38, posture: 0.28 },
  fear: { hold: 0.72, amp: 0.32, posture: 0.22 },        // guarded
  bittersweet: { hold: 0.45, amp: 0.42, posture: 0.45 },
  curiosity: { hold: 0.72, amp: 0.42, posture: 0.5 },    // focused
};

// Liveliness, measured against reference performances (an Oscar-performance study: on the same lines, the android's
// brows moved 0.3-0.5x as much as the actors', its head 0.2-0.5x, and it was almost perfectly symmetric).
// beatBrow / beatHead scale the brows and head of conversational beats; asym makes a brow beat one-sided;
// speechHead is head movement while words are coming; headNoise scales the slow idle drift.
const LIVELY = { beatBrow: 1, beatHead: 1, asym: 0, speechHead: 0, headNoise: 1, moment: 0, momentHead: 0, momentMouth: 0.5, momentEyes: 0.5, momentGaze: 0, momentLid: 0, momentEmo: 2, emoNorm: 0, momentHeadroom: 0, emoPulse: 0, emoSharpen: 0, beatRaise: 1, momentSmile: 1, momentBrowDown: 1, beatGap: 0.9, socialSmile: 1, raiseEmph: 0, emoScale: 1, beats: 1, griefBrow: 1, raiseGain: 1, angerBrow: 1, browSlow: 1, fearFreeze: 0, fearBrow: -1, angerHi: 0, glanceGap: 1, hurtBrow: 1, beatHold: 1, beatMerge: 0, momentCenter: 0, momentHeadSmooth: 0, momentYaw: 1, fearLeadBrow: 1, fearBlink: 0, fearStretch: 1, fearSpeech: 0, outerGain: 1, lidFollow: 0.9, momentSmooth: 0, momentAngerRaise: 1 };
// moment: gain of the motion-matched performance moments (setMoments) on the brows and nose; momentMouth / momentEyes
// relative gains for their mouth (on top of the android's own speech) and squint / cheek; momentHead: their head movement;
// momentGaze: where the actor looked (degrees, through the face's own saccades); momentLid: their slow lid closure;
// momentEmo: how much the felt emotion weighs when picking a moment (it predicted the actors' moment-level feeling
// no better than a constant in the reference study, so continuity, talking and stress can carry the choice);
// emoNorm: several strong feelings at once share the face (their expressions are scaled by total^-emoNorm when the
// total passes 1, so stacked prototypes don't pin the brows at their maximum); momentHeadroom: a moment's movement
// scales with the room a muscle has left (1 - its current value), so it still shows on top of a strong expression;
// emoPulse: the expression is phrased like an actor's -- it builds into stressed words, questions and exclamations
// and eases (to ~60%) between them and at sentence ends, instead of being held at one level;
// emoSharpen: the face shows the dominant feeling and lets the others leak (each scaled by (v / strongest)^emoSharpen):
// the readout of natural speech is diffuse -- three strong feelings at once pinned every brow channel at its maximum;
// beatRaise: brow raises in conversational beats (the hairless brow needs ~0.3 before a raise shows at all);
// momentSmile / momentBrowDown: the moments' smiles and brow lowering (relative to moment); beatGap: seconds between
// emphasis beats; socialSmile: how often the regulated face smiles over a sentence end or a caught leak (x1);
// angerBrow: how hard anger lowers the brows when it is NOT the leading feeling (mild anger mixed into a sad or
// fearful reading pinned the brow in every scene; leading anger always keeps its full brow); browSlow: brow muscle
// timing multiplier (slower brows make fewer, steadier movements); raiseGain: the brow-raise shapes' strength (the hairless brow's raise shape is weak: sadness 0.8 read as a tired
// frown with no lift; extrapolated x1.35 the sad brow rises and reads); griefBrow: how much sadness and fear knit the brows (AU4; the actors' grief and fear raised the inner brows with
// little knitting); raiseEmph: the share of emphasis beats that are an eyebrow flash (AU1+2) whatever the feeling -- the actors'
// brows rose on stress ~3x as often as they lowered, the android's the other way round
const EMO6 = ['joy', 'sadness', 'anger', 'fear', 'calm', 'curiosity'];

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
uniform vec4 uAU3; // lip press, chin raiser, eye widen, mouth stretch
uniform vec2 uDetail; // corrective geometry gain, dynamic wrinkle gain
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
  // AU7/AU6: a lower-lid compression fold. This is strongest during squint/cheek support, not eye widening.
  float lid = clamp(0.75 * uAU2.w + 0.5 * uAU.w, 0.0, 1.0);
  if (lid > 0.01 && ax > 0.010 && ax < 0.055 && y > 0.018 && y < 0.036) {
    float e = ellD(vec2(ax, y), vec2(0.0312, 0.0350), vec2(0.0230, 0.0122));
    h += lid * 0.55 * groove(e, 0.00085);
  }
  // AU17/AU24: chin pad bunching and the fold immediately under the lower lip.
  float chin = clamp(0.55 * uAU3.x + 0.9 * uAU3.y, 0.0, 1.0);
  if (chin > 0.01 && y < -0.040 && y > -0.073 && ax < 0.034) {
    float arc = ellD(vec2(ax, y), vec2(0.0, -0.052), vec2(0.025, 0.010));
    h += chin * 0.7 * groove(arc, 0.00115);
  }
  // AU20: lateral mouth tension creates short diagonal folds toward the corners.
  float stretch = uAU3.w;
  if (stretch > 0.01 && y < -0.018 && y > -0.052 && ax > 0.020 && ax < 0.052) {
    float d = segD(vec2(ax, y), vec2(0.028, -0.031), vec2(0.046, -0.024));
    h += stretch * 0.42 * groove(d, 0.0011);
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
  return h * uDetail.y;
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
// marks on the skin (setLook): line patterns in the face's own box (q: 0..1 across the head), front of the face only
const MARKS_GLSL = /* glsl */`
uniform float uMarks; uniform vec3 uMarkColor; uniform float uMarkGlow; uniform vec3 uEyeMid; uniform float uEyeD;
float mSeg(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
// a slash that tapers to points at both ends (w: its half-width at the middle)
float mSlash(vec2 p, vec2 a, vec2 b, float w) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - w * pow(sin(3.14159 * h), 0.6); }
float mHash(vec2 c) { return fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453); }
// p in eye-widths from the point between the eyes (+x: the face's left, our right; +y: up). eyes at (+-0.5, 0),
// brows ~(+-0.5, 0.45), nose tip (0, -0.56), mouth (0, -1.15)
float faceMarks(vec3 pos) {
  vec2 p = (pos.xy - uEyeMid.xy) / uEyeD;
  float front = smoothstep(0.0, 0.03, pos.z - uEyeMid.z + 0.075);
  float d = 1e3, w = 0.02;
  if (uMarks < 1.5) {          // kintsugi: a crack down from the hairline, and one under the right eye down the cheek
    d = min(d, mSeg(p, vec2(0.10, 1.60), vec2(0.26, 1.18))); d = min(d, mSeg(p, vec2(0.26, 1.18), vec2(0.17, 0.86)));
    d = min(d, mSeg(p, vec2(0.26, 1.18), vec2(0.47, 1.06)));
    d = min(d, mSeg(p, vec2(-0.55, -0.34), vec2(-0.70, -0.74))); d = min(d, mSeg(p, vec2(-0.70, -0.74), vec2(-0.60, -1.10)));
    d = min(d, mSeg(p, vec2(-0.60, -1.10), vec2(-0.74, -1.46))); d = min(d, mSeg(p, vec2(-0.70, -0.74), vec2(-0.95, -0.86)));
    w = 0.018;
  } else if (uMarks < 2.5) {   // a scar across the left cheek
    d = mSeg(p, vec2(0.42, -0.28), vec2(0.86, -0.98)); w = 0.026;
  } else if (uMarks < 3.5) {   // tally marks under the left eye, struck through
    for (int i = 0; i < 4; i++) { float x = 0.34 + 0.10 * float(i); d = min(d, mSeg(p, vec2(x, -0.30), vec2(x, -0.52))); }
    d = min(d, mSeg(p, vec2(0.27, -0.49), vec2(0.73, -0.33))); w = 0.014;
  } else if (uMarks > 5.5 && uMarks < 6.5) {   // claws: three tapered slashes down across the right eye
    for (int i = -1; i < 2; i++) { float o = 0.17 * float(i); d = min(d, mSlash(p, vec2(-0.12 + o, 0.8), vec2(-0.76 + o, -0.8), 0.062)); }
    w = 0.0;
  } else if (uMarks > 6.5 && uMarks < 7.5) {   // tears: streaks running down from both eyes, each ending in a drop
    for (int i = 0; i < 2; i++) {
      float sx = i == 0 ? -1.0 : 1.0, x0 = 0.47 * sx;
      vec2 q = vec2(p.x - 0.022 * sin(p.y * 10.0 + sx), p.y);
      d = min(d, mSeg(q, vec2(x0, -0.22), vec2(x0, -1.12)) - 0.028);
      d = min(d, length(q - vec2(x0, -1.16)) - 0.055);
      d = min(d, mSeg(q, vec2(x0 + 0.12 * sx, -0.25), vec2(x0 + 0.12 * sx, -0.66)) - 0.02);
      d = min(d, length(q - vec2(x0 + 0.12 * sx, -0.69)) - 0.038);
    }
    w = 0.0;
  } else if (uMarks > 7.5 && uMarks < 8.5) {   // split: the right half of the head in another colour, a zigzag edge
    d = p.x + 0.045 * (abs(fract(p.y * 2.6) - 0.5) * 4.0 - 1.0); w = 0.0; front = 1.0;
  } else if (uMarks > 8.5 && uMarks < 9.5) {   // stardust: points of light across the nose and cheeks
    float cell = 0.085; vec2 c = floor(p / cell), f = p / cell - c;
    float hsh = mHash(c), r = 0.1 + 0.2 * mHash(c + 7.3) * mHash(c + 2.9);
    vec2 ctr = 0.3 + 0.4 * vec2(mHash(c + 1.7), mHash(c + 4.1));
    float inside = step(length(vec2(p.x / 1.15, (p.y + 0.32) / 0.36)), 1.0) * step(0.5, hsh);
    d = inside > 0.5 ? (length(f - ctr) - r) * cell : 1e3; w = 0.0;
  } else {                     // circuit traces on the right temple
    d = min(d, mSeg(p, vec2(-1.00, 1.05), vec2(-0.62, 1.05))); d = min(d, mSeg(p, vec2(-0.62, 1.05), vec2(-0.62, 0.78)));
    d = min(d, mSeg(p, vec2(-0.98, 0.88), vec2(-0.80, 0.88))); d = min(d, mSeg(p, vec2(-0.80, 0.88), vec2(-0.80, 0.62)));
    d = min(d, mSeg(p, vec2(-0.40, 1.22), vec2(-0.40, 0.95)));
    d = min(d, length(p - vec2(-0.62, 0.78)) - 0.03); d = min(d, length(p - vec2(-0.80, 0.62)) - 0.03); d = min(d, length(p - vec2(-0.40, 0.95)) - 0.03);
    w = 0.016;
  }
  float fw = clamp(fwidth(d), 1e-5, 0.02);   // (clamped: a pattern drawn cell by cell jumps at the cell edges)
  return front * (1.0 - smoothstep(w - fw, w + fw, d));
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
// The model is parsed once per page and shared: a page of faces (the explore grid) would otherwise parse the same 2 MB
// file once per face, each costing a phone's main thread a few hundred milliseconds.
const MODELS = new Map();
function loadModel(url) {
  if (!MODELS.has(url)) MODELS.set(url, new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(url).catch((e) => { MODELS.delete(url); throw e; }));
  return MODELS.get(url);
}

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
    this.manualChannels = {};
    this.lively = { ...LIVELY };
    // acting version 2 (a frame-by-frame review's changes; see _actingV2), only with ?acting=v2 in the address
    try { if (new URLSearchParams(location.search).get('acting') === 'v2') this.lively.v2 = 1; } catch { /* (no page) */ }
    this.motionStyle = { amplitude: 1, posture: 1 };
    this.externalIntent = { x: 0, v: 0, target: 0 };
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
    this.style = { amp: { x: 0.65, v: 0 }, posture: { x: 0.65, v: 0 } };   // how much the head moves, per performance
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
    this.sleep = { x: 0, v: 0, target: 0 };   // 0 awake .. 1 asleep (setSleep): eyes shut, slow deep breaths, head down

    this._tmpV = new THREE.Vector3();
    this._tmpV2 = new THREE.Vector3();
    this._tmpQ = new THREE.Quaternion();
    this._tmpE = new THREE.Euler();
    this._tmpM = new THREE.Matrix4();

    this.ready = this._load(url).then(() => this);
  }

  // -------------------------------------------------------------------------------- loading
  async _load(url) {
    const gltf = await loadModel(url);
    this.gltf = gltf;
    const model = SkeletonUtils.clone(gltf.scene);   // its own copy (bones and all); the geometry is shared, materials are its own
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
      uNaturalFolds: { value: 1 }, uFoldLeft: { value: new THREE.Vector4() },
      uFoldRight: { value: new THREE.Vector4() }, uFoldEye: { value: new THREE.Vector4() },
      uAU: { value: new THREE.Vector4() }, uAU2: { value: new THREE.Vector4() }, uAU3: { value: new THREE.Vector4() },
      uDetail: { value: new THREE.Vector2(1, 1) }, uCreaseDepth: { value: 0.00032 },
      // marks (setLook): 0 none, 1 kintsugi (gold-filled cracks), 2 scar, 3 tally marks, 4 circuit lines
      uMarks: { value: 0 }, uMarkColor: { value: new THREE.Color(0.85, 0.62, 0.22) }, uMarkGlow: { value: 0 },
      uEyeMid: { value: new THREE.Vector3(0, 0.0357, 0.0803) }, uEyeD: { value: 0.0624 },
    };

    model.traverse((o) => {
      if (o.isMesh) {
        const role = o.material.userData.role;
        if (role === 'porcelain') {
          this._skinMesh = o;
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
          // teeth sit in the shadow of the lips: a softer off-white with little clear coat (bright glossy teeth read as dentures)
          o.material = new THREE.MeshPhysicalMaterial({ color: 0xd9d3c7, roughness: 0.5, clearcoat: 0.15, clearcoatRoughness: 0.4, envMapIntensity: 0.5 });
        } else if (role === 'gums') {
          // the inside of the mouth: a warm, matte dark (a clear coat reflected the blue studio like chrome: uncanny)
          o.material = new THREE.MeshPhysicalMaterial({ color: 0x1a1012, roughness: 0.95, clearcoat: 0, envMapIntensity: 0.05, specularIntensity: 0.2 });
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
    if (this._skinMesh) {   // the marks are laid out from the eyes, in the skin mesh's own space
      this.root.updateMatrixWorld(true);
      const w0 = this._skinMesh.worldToLocal(this.eyes[0].pivot.getWorldPosition(new THREE.Vector3()));
      const w1 = this._skinMesh.worldToLocal(this.eyes[1].pivot.getWorldPosition(new THREE.Vector3()));
      this.uniforms.uEyeMid.value.copy(w0).add(w1).multiplyScalar(0.5); this.uniforms.uEyeD.value = w0.distanceTo(w1);
    }
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
        .replace('#include <common>', '#include <common>\nuniform float uBreath;\nuniform vec4 uAU;\nuniform vec4 uAU2;\nuniform vec4 uAU3;\nuniform vec2 uDetail;')
        .replace('uniform vec2 uDetail;', 'uniform vec2 uDetail;\n' + CORRECTIVE_GLSL)
        .replace('#include <morphnormal_vertex>', `#include <morphnormal_vertex>
          if (uNaturalFolds > 0.5) objectNormal = labCorrectiveNormal(position,objectNormal);`)
        .replace('#include <morphtarget_vertex>', `#include <morphtarget_vertex>
          transformed += labCorrective(position);`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', '#include <common>\n' + PORCELAIN_PARS + NATURAL_FOLDS_GLSL + MARKS_GLSL)
        .replace('#include <color_fragment>', `#include <color_fragment>
          float sDist = seamDist(vSeamUv);
          float sStr = uHasSeams * smoothstep(-0.02, 0.03, vObjPos.z);  // fade seams out toward the back of the head
          float sFw = max(fwidth(sDist), 1e-4);
          float sCore = (1.0 - smoothstep(uSeamWidth * 0.42 - sFw, uSeamWidth * 0.42 + sFw, sDist)) * sStr;
          float sLip = (1.0 - smoothstep(0.0, uSeamWidth * 1.6 + sFw, sDist)) * sStr;
          diffuseColor.rgb *= mix(1.0, uSeamDark, sCore);
          diffuseColor.rgb *= mix(1.0, 0.9, sLip);
          float mk = uMarks > 0.5 ? faceMarks(vObjPos) : 0.0;
          diffuseColor.rgb = mix(diffuseColor.rgb, uMarkColor, mk);
          float cH = creaseH(vObjPos);
          // Legacy mode retained for comparison. Natural mode has NO ink/albedo stroke.
          diffuseColor.rgb *= 1.0 - (1.0-uNaturalFolds)*0.28*clamp(cH,0.0,1.0);
        `)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
          {
            float pixelSize = max(length(dFdx(vObjPos)),length(dFdy(vObjPos)));
            float foldH = naturalFoldHeight(vObjPos,pixelSize);
            float h0 = seamH(vSeamUv) + mix(-uCreaseDepth*cH,foldH,uNaturalFolds);
            vec2 dH = vec2(dFdx(h0), dFdy(h0));
            normal = seamPerturb(-vViewPosition, normal, dH, faceDirection);
          }
        `)
        .replace('#include <clearcoat_normal_fragment_begin>', '#include <clearcoat_normal_fragment_begin>\n#ifdef USE_CLEARCOAT\nclearcoatNormal = normal;\n#endif')
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          totalEmissiveRadiance += uGlowColor * uGlow * sCore + uMarkColor * uMarkGlow * mk;`)
        .replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment.replace(
          'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
          `{ float wNL = saturate((dot(geometryNormal, directLight.direction) + uWrap) / (1.0 + uWrap));
             vec3 wrapIrr = directLight.color * max(wNL - dotNL, 0.0) * uSSS;
             reflectedLight.directDiffuse += (irradiance + wrapIrr) * BRDF_Lambert( material.diffuseColor ); }`))
        .replace('#include <opaque_fragment>', `
          outgoingLight *= smoothstep(uFade.y, uFade.x, vObjPos.y);
          #include <opaque_fragment>`);
    };
    m.customProgramCacheKey = () => 'porcelain-natural-folds-v2';
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

  /** Acting version 2, from a frame-by-frame review of each feeling's "a lot" answer (only with ?acting=v2), for
   *  sadness and fear (anger keeps its smirk: the owner likes its crazy vibe): when one of them leads, its face loses
   *  what contradicted it on screen -- a smile over it, outer brows held raised in sadness -- and gains its key:
   *  sadness the oblique inner brow, corners down, chin up, heavy lids; fear eyes held open (lids closing read as
   *  drowsy), wide, brows up and in, lips stretched. */
  _actingV2(tgt, felt, add) {
    const [dom, v] = this._dominant(NEG);
    const k = smooth(0.15, 0.5, v || 0);
    if (!(dom === 'sadness' || dom === 'fear') || k < 0.02) return;
    scaleCh(tgt, 'mouthSmile', 1 - 0.85 * k);
    if (dom === 'sadness') {
      scaleCh(tgt, 'browOuterUp', 1 - 0.7 * k); scaleCh(tgt, 'eyeWide', 1 - 0.7 * k);   // (heavy, not alert, eyes)
      add('browInnerUp', 0.3 * k); add('mouthFrown', 0.22 * k); add('mouthShrugLower', 0.14 * k); add('eyeBlink', 0.08 * k);
    } else if (dom === 'fear') {
      scaleCh(tgt, 'eyeBlink', 1 - 0.75 * k); scaleCh(tgt, 'eyeSquint', 1 - 0.6 * k);
      add('eyeWide', 0.18 * k); add('browInnerUp', 0.18 * k); add('mouthStretch', 0.1 * k);
    }
  }

  /** Acting-lab controls. These are display/animation controls, not model measurements. */
  setNaturalFolds(enabled = true) { if(this.uniforms) this.uniforms.uNaturalFolds.value = enabled ? 1 : 0; }
  setPanelSeams(visible = true) { if(this.uniforms) this.uniforms.uHasSeams.value = visible ? 1 : 0; }
  cancelReliefTake() {
    if(!this.reliefTake) return;
    this.reliefTake=null;
    const B=this.breath; B.kind='rest'; B.sigh=B.relief=false;
    B.from=B.val; B.phase=0; B.T=4.5; B.inhale=.38; B.hold=0; B.depth=Math.max(.8,B.val);
    B.gasp=B.huff=B.speechInhale=false;
  }
  /** drive: the take also sets the feelings (rehearsal); false = only the release itself (live: the feelings keep
   *  coming from the model's readout). */
  startReliefTake(onComplete, { drive = true } = {}) {
    this.reliefTake = { start:this.time, fear:this.felt?.fear || this.emoTarget.fear || .64,
      breath:this.breath.val, onComplete, sample:null, drive };
    this.breath.gasp=this.breath.huff=this.breath.speechInhale=false;
    this.overlays=this.overlays.filter(o=>!o.social);
    this.overlays.push({t0:this.time+.25,a:.65,h:.7,r:2.1,head:{pitch:-.85,z:.001}});
    this.setAttentionHold(.7);
  }
  /** Motion matching: a library of short real-performance moments ({fps, channels, moments: [{e, i, spk, stress, d, h}]}:
   *  per-frame expression deltas in rig units and head degrees, labelled by feeling, intensity, talking and a stressed
   *  word at the start). One moment always plays, picked to fit what the face feels and does; its movement is added
   *  on top of the face's own expression (scaled by lively.moment / lively.momentHead). */
  setMoments(lib) {
    this.moments = lib?.moments?.length ? lib : null;
    this.mom = this.momPrev = null; this.momRecent = []; this.momHead = null;
  }

  _pickMoment(t) {
    const lib = this.moments, f = this.felt || {}, R = this.rand;
    const ev = EMO6.map((e) => f[e] || 0), tot = ev.reduce((a, b) => a + b, 0);
    const want = tot > 0.06 ? ev.map((x) => x / tot) : [0, 0, 0, 0, 1, 0];
    const inten = Math.min(1, tot), talking = t - (this.lastSay ?? -10) < 0.8 ? 1 : 0, stress = this.momKick ? 1 : 0;
    const cur = this.mom ? this.mom.m.d[this.mom.m.d.length - 1] : null;
    let best = [];
    for (let n = 0; n < 160; n++) {
      const m = lib.moments[Math.floor(R() * lib.moments.length)];
      let dot = 0, nm = 0;
      for (let k = 0; k < 6; k++) { dot += want[k] * m.e[k]; nm += m.e[k] * m.e[k]; }
      const ce = 1 - dot / Math.sqrt(nm * want.reduce((a, x) => a + x * x, 0) + 1e-9);
      const ci = Math.abs(Math.min(1, m.i / 0.12) - inten);
      let cc = 0;
      if (cur) { for (let k = 0; k < cur.length; k++) cc += Math.abs(cur[k] - m.d[0][k]); cc /= cur.length; }
      const cost = this.lively.momentEmo * ce + 0.5 * ci + 0.8 * (m.spk !== talking) + 0.6 * (stress && !m.stress) + 4 * cc + (this.momRecent.includes(m) ? 3 : 0);
      best.push([cost, m]);
    }
    best.sort((a, b) => a[0] - b[0]);
    const m = best[Math.floor(R() * Math.min(4, best.length))][1];
    this.momRecent.push(m); if (this.momRecent.length > 24) this.momRecent.shift();
    this.momPrev = this.mom; this.mom = { m, t0: t }; this.momKick = false;
  }

  _moment(t, post) {
    const lib = this.moments, L = this.lively;
    if (!lib || !(L.moment > 0 || L.momentHead > 0) || !this.options.idle) { this.momHead = null; return; }
    const fps = lib.fps, len = (M) => M.m.d.length / fps;
    if (!this.mom || t - this.mom.t0 > len(this.mom) - 0.25 || (this.momKick && t - this.mom.t0 > 0.4)) this._pickMoment(t);
    const at = (M, key) => {   // the moment's frame at time t (linear between frames), or null when over
      const u = (t - M.t0) * fps, i = Math.floor(u), a = M.m[key];
      if (i >= a.length - 1) return null;
      const w = u - i;
      return a[i].map((x, k) => x + (a[i + 1][k] - x) * w);
    };
    const xf = smooth(0, 0.25, t - this.mom.t0);   // crossfade from the previous moment
    const cd = at(this.mom, 'd'), pd = this.momPrev ? at(this.momPrev, 'd') : null;
    const ch = lib.channels;
    if (!this.momGain || this.momGain.lib !== lib || this.momGain.key !== `${L.moment}|${L.momentMouth}|${L.momentEyes}|${L.momentSmile}|${L.momentBrowDown}`) {
      this.momGain = { lib, key: `${L.moment}|${L.momentMouth}|${L.momentEyes}|${L.momentSmile}|${L.momentBrowDown}`,
        g: ch.map((c) => L.moment * (c.startsWith('mouthSmile') ? L.momentSmile : c.startsWith('browDown') ? L.momentBrowDown : c.startsWith('mouth') ? L.momentMouth : /^(eyeSquint|cheek)/.test(c) ? L.momentEyes : 1)) };   // smiles at full strength: below ~0.25 they don't show
    }
    const G = this.momGain.g;
    const hr = L.momentHeadroom > 0;
    // momentSmooth: the moments' brow / eye movement goes through a spring (rad/s) like the face's own muscles -- added
    // after the springs it carried the actors' tracker jitter and the jumps between moments (the brows twitched)
    const MS = L.momentSmooth > 0 ? (this.momS || (this.momS = ch.map(() => ({ x: 0, v: 0 })))) : null;
    const mdt = Math.min(0.05, Math.max(0, t - (this.momST ?? t))); this.momST = t;
    // momentAngerRaise: while anger clearly leads, a moment can't raise the brows (the readout calls many film moments
    // 'anger', and an actor's raised brows over the android's frown read as alarm, not rage)
    const fe = this.felt || {}, fm = Math.max(1e-3, ...EMOTIONS.map((e) => fe[e] || 0));
    const angerK = L.momentAngerRaise !== 1 ? smooth(0.6, 0.9, (fe.anger || 0) / fm) * smooth(0.15, 0.4, fe.anger || 0) : 0;
    if (cd) for (let k = 0; k < ch.length; k++) {
      const room = hr ? Math.max(0, 1 - (this.ch[ch[k]]?.x || 0)) : 1;
      let v = G[k] * room * (xf * cd[k] + (pd ? (1 - xf) * pd[k] : 0));
      if (angerK > 0 && v > 0 && /^brow(Inner|Outer)Up/.test(ch[k])) v *= lerp(1, L.momentAngerRaise, angerK);
      if (MS && /^(brow|eye|cheek)/.test(ch[k])) { spring(MS[k], v, L.momentSmooth, mdt); v = MS[k].x; }
      post[ch[k]] = (post[ch[k]] || 0) + v;
    }
    // head: momentCenter removes each moment's average head offset (switching moments otherwise swung the head ~10 deg
    // in 0.25 s), momentYaw scales its turns (it presents to an audience), momentHeadSmooth springs it like the head
    const hm = (M) => { if (!L.momentCenter || !M) return [0, 0, 0]; if (!M.hMean) { const h = M.m.h; M.hMean = [0, 1, 2].map((k) => h.reduce((a, f) => a + f[k], 0) / h.length); } return M.hMean.map((v) => v * L.momentCenter); };
    const ch2 = at(this.mom, 'h'), ph = this.momPrev ? at(this.momPrev, 'h') : null, c0 = hm(this.mom), c1 = hm(this.momPrev);
    const yawK = [L.momentYaw, 1, 1];
    let mh = ch2 ? ch2.map((x, k) => L.momentHead * yawK[k] * (xf * (x - c0[k]) + (ph ? (1 - xf) * (ph[k] - c1[k]) : 0))) : null;
    if (L.momentHeadSmooth > 0) {
      const S = this.momHeadS || (this.momHeadS = [0, 1, 2].map(() => ({ x: 0, v: 0 })));
      S.forEach((sp, k) => spring(sp, mh ? mh[k] : 0, L.momentHeadSmooth, Math.min(0.05, t - (this.momHeadT ?? t))));
      this.momHeadT = t; mh = S.map((sp) => sp.x);
    }
    this.momHead = mh;
    const cg = this.mom.m.g ? at(this.mom, 'g') : null, pg = this.momPrev?.m.g ? at(this.momPrev, 'g') : null;
    this.momGaze = cg && L.momentGaze > 0 ? cg.map((x, k) => L.momentGaze * (xf * x + (pg ? (1 - xf) * pg[k] : 0))) : null;
    const cl = this.mom.m.l, pl = this.momPrev?.m.l;
    if (cl && L.momentLid > 0) {
      const u = Math.min(cl.length - 1, (t - this.mom.t0) * fps), i = Math.floor(u), v = cl[i] + ((cl[i + 1] ?? cl[i]) - cl[i]) * (u - i);
      let w = xf * v;
      if (pl) { const u2 = Math.min(pl.length - 1, (t - this.momPrev.t0) * fps); w += (1 - xf) * pl[Math.floor(u2)]; }
      post.eyeBlinkLeft = (post.eyeBlinkLeft || 0) + L.momentLid * w; post.eyeBlinkRight = (post.eyeBlinkRight || 0) + L.momentLid * w;
    }
  }

  setManualChannels(values = {}) {
    this.manualChannels = Object.fromEntries(Object.entries(values).filter(([,v]) => Number.isFinite(v)).map(([k,v]) => [k, clamp(v, -1, 1)]));
  }
  setMotionStyle({ amplitude, posture } = {}) {
    if (Number.isFinite(amplitude)) this.motionStyle.amplitude = clamp(amplitude, 0, 1.5);
    if (Number.isFinite(posture)) this.motionStyle.posture = clamp(posture, 0, 1.5);
  }
  setAttentionHold(v = 0) { this.externalIntent.target = clamp(v); }
  setFaceDetail({ correctives, wrinkles } = {}) {
    if (!this.uniforms) return;
    if (Number.isFinite(correctives)) this.uniforms.uDetail.value.x = clamp(correctives, 0, 1.5);
    if (Number.isFinite(wrinkles)) this.uniforms.uDetail.value.y = clamp(wrinkles, 0, 1.5);
  }
  /** The android's look: skin 'porcelain' | 'chrome' | 'matte' (matte black) | 'glass'; eye colour (CSS colour);
   *  marks 'none' | 'kintsugi' | 'scar' | 'tally' | 'circuit'. Unset fields keep their value. */
  /** The android's look: skin (porcelain / chrome / matte / glass), eye colour, marks; optionally color (a skin colour
   *  over the skin's own), mark_color (over the mark's own) and eye_glow (glowing eyes). Unset keys keep their value;
   *  color / mark_color: '' for the default. */
  setLook({ skin, eye, marks, color, mark_color, eye_glow } = {}) {
    this.look = { ...(this.look || { skin: 'porcelain', eye: '#7fe7ff', marks: 'none', color: '', mark_color: '', eye_glow: false }),
      ...Object.fromEntries(Object.entries({ skin, eye, marks, color, mark_color, eye_glow }).filter(([, v]) => v != null)) };
    const L = this.look, U = this.uniforms;
    if (eye) this.setIrisColor(eye);
    this.irisGlow = L.eye_glow ? 2.6 : 1.0;
    if (!U) return;
    const SK = {
      porcelain: { color: [0.93, 0.905, 0.868], roughness: 0.4, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.075, sheen: 0.25, transmission: 0, sss: 0.55, wrap: 0.45, seamDark: 0.28 },
      chrome: { color: [0.86, 0.87, 0.9], roughness: 0.13, metalness: 1, clearcoat: 1, clearcoatRoughness: 0.03, sheen: 0, transmission: 0, sss: 0, wrap: 0.05, seamDark: 0.15 },
      matte: { color: [0.055, 0.055, 0.062], roughness: 0.72, metalness: 0, clearcoat: 0.12, clearcoatRoughness: 0.5, sheen: 0.55, transmission: 0, sss: 0.08, wrap: 0.3, seamDark: 0.5 },
      glass: { color: [0.62, 0.76, 0.9], roughness: 0.07, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.02, sheen: 0, transmission: 0, iridescence: 1, sss: 0.1, wrap: 0.3, seamDark: 0.4 },
    }[L.skin] || null;
    if (SK) for (const m of this._porcelainMats || []) {
      if (L.color) m.color.set(L.color); else m.color.setRGB(...SK.color);
      m.roughness = SK.roughness; m.metalness = SK.metalness; m.clearcoat = SK.clearcoat;
      m.clearcoatRoughness = SK.clearcoatRoughness; m.sheen = SK.sheen; m.transmission = SK.transmission;
      m.iridescence = SK.iridescence || 0; m.iridescenceIOR = 1.45; m.iridescenceThicknessRange = [180, 520];   // glass: a clear, oil-sheen finish
      m.needsUpdate = true;
    }
    if (SK) { U.uSSS.value.setRGB(1.0, 0.62, 0.45).multiplyScalar(SK.sss); U.uWrap.value = SK.wrap; U.uSeamDark.value = SK.seamDark; }
    const MK = { none: [0], kintsugi: [1, [0.86, 0.64, 0.24], 0.4], scar: [2, [0.32, 0.2, 0.2], 0], tally: [3, [0.12, 0.12, 0.14], 0], circuit: [4, [0.35, 0.85, 1.0], 2.0],
      claws: [6, [0.9, 0.06, 0.08], 1.3], tears: [7, [0.02, 0.02, 0.03], 0], split: [8, [0.025, 0.025, 0.03], 0],
      stardust: [9, [0.75, 0.92, 1.0], 2.4] }[L.marks] || [0];
    U.uMarks.value = MK[0];
    if (MK[1]) { if (L.mark_color) U.uMarkColor.value.set(L.mark_color); else U.uMarkColor.value.setRGB(...MK[1]); U.uMarkGlow.value = MK[2]; }
  }

  /** 0 awake .. 1 asleep: the eyes close, breathing slows and deepens, the head sinks, the eyes' light dims. */
  setSleep(v = 1) { this.sleep.target = clamp(v); }

  /** Woken with a start: a gasp, eyes and brows up, the head jerks back, then it settles awake. */
  startle() {
    const t = this.time, S = this.sleep;
    S.target = 0; S.x = Math.min(S.x, 0.35); S.v = 0;
    this.breath.gasp = true;
    this.overlays = this.overlays.slice(-6);
    this.overlays.push({ t0: t, a: 0.06, h: 0.35, r: 0.9, ch: { eyeWide: 0.7, browInnerUp: 0.45, browOuterUp: 0.55, jawOpen: 0.12 }, head: { pitch: 3.2, z: -0.006 } });
  }

  setPorcelainMatte(on = false) {
    for (const m of this._porcelainMats || []) {
      m.roughness = on ? 0.68 : 0.4; m.clearcoat = on ? 0.12 : 1.0; m.clearcoatRoughness = on ? 0.45 : 0.075; m.needsUpdate = true;
    }
  }

  /** Drop every transient reaction (beats, leaks, chuckles, onset surges, tremors, special breaths), e.g. when a
   *  performance is cut off. The felt emotion itself is left to setEmotion. */
  clearReactions() {
    this.clearSpeech(true);
    const t = this.time;
    this.lastBeat = -Infinity; this.lastReaction = Object.create(null);
    this.reliefT = -10; this.ovBreath = 0; this.reliefTake = null; this.lastReliefSample = null; this.externalIntent.target = 0;
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
  say(text, dur = 0.15) { this.lastSay = this.time; this.speechMotion.enqueue(text, this.time, dur); }
  clearSpeech(hard = false) { this.speechMotion.clear({ hard }); if (hard) this.speechFrame = null; }

  setActivity({ writing = false } = {}) {
    if (writing && !this.writing) this.breath.speechInhale = true;   // people inhale before they start speaking
    this.writing = !!writing;
    if (!writing) this.clearSpeech();
  }

  /** Conversational beat while writing: 'comma' | 'period' | 'exclaim' | 'question' | 'emphasis'. */
  beat(kind) {
    const t = this.time, R = this.rand;
    if (!this.lively.beats) return;   // diagnostics: beats off
    // A punctuation burst should not pile up nods, eyebrow flashes and mouth presses.
    if (t - this.lastBeat < (kind === 'emphasis' ? this.lively.beatGap : 0.42)) return;
    this.lastBeat = t;
    if (kind === 'emphasis') this.momKick = true;   // the next performance moment should start on this stress
    if (this.pulse && (kind === 'emphasis' || kind === 'exclaim' || kind === 'question')) this.pulse.target = 1.05;   // phrasing peak
    if (this.pulse && kind === 'period') this.pulse.target = 0.5;                                                   // sentence end: ease off
    this.overlays = this.overlays.slice(-7);
    const flash = kind === 'emphasis' && R() < this.lively.raiseEmph;
    if (flash) { this._beat({ t0: t, a: 0.05, h: 0.14, r: 0.3, ch: { browInnerUp: 0.12, browOuterUp: 0.16 } }); return; }   // an eyebrow flash
    if (this.perf?.kind === 'anger' && this.intent.x > 0.3) {
      // deliver: firm, sparse accents; no nods on commas, no inquisitive tilt, no smile
      const hot = this.perf.variant === 'hot';
      if (kind === 'comma') return;
      if (kind === 'period') { if (R() < 0.6) this._beat({ t0: t, a: 0.08, h: 0.12, r: 0.35, head: { pitch: -1.1 }, ch: { mouthPress: 0.14 } }); return; }
      if (kind === 'question') { this._beat({ t0: t, a: 0.1, h: 0.3, r: 0.4, head: { pitch: 1.0 }, ch: { browDown: 0.12 } }); return; }
      this._beat({ t0: t, a: 0.05, h: hot ? 0.2 : 0.14, r: 0.3, ch: { browDown: hot ? 0.3 : 0.2, eyeSquint: 0.1, mouthPress: 0.1 }, head: { pitch: hot ? -2.0 : -1.2, z: hot ? 0.005 : 0.003 } });
      return;
    }
    const [dom, fd] = this._dominant();
    const strong = fd > 0.3;
    // feeling-coloured variants of the beat
    if (kind === 'emphasis' && strong && EMPHASIS[dom]) { this._beat({ t0: t, ...EMPHASIS[dom] }); return; }
    if (kind === 'exclaim' && strong) {
      if (dom === 'anger') { this._beat({ t0: t, a: 0.05, h: 0.2, r: 0.4, ch: { browDown: 0.2, eyeWide: 0.1, mouthPress: 0.12 }, head: { pitch: -1.2, z: 0.004 } }); return; }
      if (dom === 'joy' && this.reg < 0.4 && R() < 0.35 * fd) { this._chuckle(fd * 0.6); return; }
      if (dom === 'fear') { this._beat({ t0: t, a: 0.04, h: 0.2, r: 0.4, ch: { eyeWide: 0.2, browInnerUp: 0.2, mouthStretch: 0.08 }, head: { z: -0.004 } }); return; }
    }
    if (kind === 'period') {
      if (this.reg > 0.3 && dom !== 'anger' && R() < 0.4 * this.lively.socialSmile) this._socialSmile(0.16 + 0.12 * this.reg, 0.5 + 0.5 * R(), t + 0.1);
      if (strong && dom === 'sadness') {
        this._beat({ t0: t, a: 0.25, h: 0.35, r: 0.9, head: { pitch: -2.6 }, ch: { browInnerUp: 0.1, mouthPress: 0.08 } });
        if (R() < 0.35) { this.blink.next = t + 0.15; this.blink.slowNext = true; }
        return;
      }
      if (strong && dom === 'joy') { this._beat({ t0: t, a: 0.15, h: 0.3, r: 0.6, head: { pitch: -1.0 }, ch: { mouthSmile: 0.1, cheekSquint: 0.06 } }); return; }
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
    this._beat({ t0: t, ...ov });
    if ((kind === 'period' && R() < 0.15) || (kind === 'question' && R() < 0.08)) this.blink.next = t + 0.12;
  }

  /** A conversational beat's overlay, scaled for liveliness: brow and eye channels by beatBrow, head by beatHead,
   *  and a symmetric brow movement made one-sided (the other side moves 1 - asym * (0.4..1) as much). */
  _beat(o) {
    const L = this.lively, side = this.rand() < 0.5 ? 'L' : 'R', other = side === 'L' ? 'R' : 'L';
    const k = 1 - L.asym * (0.4 + 0.6 * this.rand()), ch = {};
    for (const [c, v] of Object.entries(o.ch || {})) {
      const g = /^brow(Inner|Outer)Up/.test(c) ? L.beatRaise : c.startsWith('brow') || c.startsWith('eye') ? L.beatBrow : 1;
      if (c.startsWith('brow') && !/(L|R|Left|Right)$/.test(c)) { ch[c + side] = (ch[c + side] || 0) + v * g; ch[c + other] = (ch[c + other] || 0) + v * g * k; }
      else ch[c] = (ch[c] || 0) + v * g;
    }
    const head = Object.fromEntries(Object.entries(o.head || {}).map(([a, v]) => [a, v * L.beatHead]));
    // beatHold: brow beats rise and fall slower and hold longer; beatMerge: a stress that lands while the brows are still
    // up holds them up (to at most 2.5 s x beatMerge) instead of a new flick -- the actors hold a raise across a phrase
    const brow = Object.fromEntries(Object.entries(ch).filter(([c]) => c.startsWith('brow')));
    if ((L.beatHold !== 1 || L.beatMerge > 0) && Object.keys(brow).length) {
      const rest = Object.fromEntries(Object.entries(ch).filter(([c]) => !c.startsWith('brow')));
      const P = this._browBeat, t = o.t0, hold = o.h * L.beatHold;
      if (L.beatMerge > 0 && P && this.overlays.includes(P) && t < P.t0 + P.a + P.h + 0.5 * P.r && t + hold - P.t0 < 2.5 * L.beatMerge) {
        P.h = Math.max(P.h, t - P.t0 - P.a + hold);
        for (const [c, v] of Object.entries(brow)) P.ch[c] = Math.max(P.ch[c] || 0, v);
      } else {
        const sl = Math.sqrt(L.beatHold);
        this._browBeat = { t0: t, a: o.a * sl, h: hold, r: o.r * sl, ch: brow };
        this.overlays.push(this._browBeat);
      }
      this.overlays.push({ ...o, ch: rest, head });
      return;
    }
    this.overlays.push({ ...o, ch, head });
  }

  /** Social reactions: 'greet' | 'listen' | 'think' | 'done'. */
  react(kind) {
    const t = this.time;
    if (t - (this.lastReaction[kind] ?? -Infinity) < (kind === 'listen' ? 1.2 : 0.5)) return;
    this.lastReaction[kind] = t;
    this.overlays = this.overlays.slice(-7);
    const side = this.rand() < 0.5 ? -1 : 1;
    if (this.perf?.kind === 'anger' && this.intent.x > 0.3 && (kind === 'listen' || kind === 'done' || kind === 'greet')) {
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
    const smileOver = e !== 'anger' && this.rand() < Math.min(1, this.lively.socialSmile);
    const ch = !smileOver ? { mouthPress: 0.26, mouthRollLower: 0.08, mouthShrugLower: 0.08 }           // caught, and swallowed
      : { mouthPress: 0.22, mouthSmileL: 0.15, mouthSmileR: 0.11, mouthRollLower: 0.06 };               // caught, and smiled over
    this.overlays.push({ t0: t + end * 0.8, a: 0.12, h: 0.35 + 0.3 * this.rand(), r: 0.45, post: true, ch, head: { pitch: -0.8 } });
    if (this.rand() < 0.25) this.blink.next = t + end + 0.05;
    if (this.rand() < 0.2) {
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
    if (this.rand() < 0.3) {
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
    let P = this.perf;
    const want = this.options.idle ? this._wantPerf(felt) : null;
    if (P && !P.ending && want !== P.kind && t - P.t0 > 0.8) { P.ending = true; P.tEnd = t; if (P.kind === 'anger') this._recover(P); }
    if (P?.ending && t - P.tEnd > (P.kind === 'anger' ? 2.8 : 1.2)) this.perf = P = null;
    if (!P && want) {
      P = this.perf = { kind: want, t0: t, ending: false };
      if (want === 'anger') {
        const tgt = this.emoTarget.anger * this.intensity;
        P.variant = tgt > 0.8 ? 'hot' : tgt < 0.5 || this.reg > 0.35 ? 'contained' : 'frustrated';
        this._cueAnger(P);
      } else this._cue(P);
    }
    const S = PERF_STYLE[P && !P.ending ? P.variant || P.kind : 'neutral'];
    spring(this.intent, S.hold, P && !P.ending ? 6 : 1.3, dt);
    spring(this.style.amp, S.amp, 1.5, dt);
    spring(this.style.posture, S.posture, 1.0, dt);
  }

  /** Which performance the feelings call for. A running one needs less to keep going (hysteresis), so the
   *  token-to-token noise in the readout doesn't flip it. */
  _wantPerf(f) {
    const [dom] = this._dominant(), on = (k) => this.perf?.kind === k && !this.perf.ending;
    if (dom === 'anger' && f.anger > (on('anger') ? 0.16 : 0.28)) return 'anger';
    if (dom === 'fear' && f.fear > (on('fear') ? 0.18 : 0.3)) return 'fear';
    if (this.blend === 'bittersweet' && this.blendK > (on('bittersweet') ? 0.25 : 0.35)) return 'bittersweet';
    if (dom === 'curiosity' && f.curiosity > (on('curiosity') ? 0.2 : 0.32)) return 'curiosity';
    return null;
  }

  _cue(P) {
    const t = this.time, side = this.rand() < 0.5 ? -1 : 1;
    if (P.kind === 'fear') {
      // guarded: eyes and inner brows up, a held blink, then it checks -- one look each way -- and settles watchful
      this.blink.next = Math.max(this.blink.next, t + 1.2);
      this.overlays.push({ t0: t + 0.2, a: 0.08, h: 0.22, r: 0.55, ch: { eyeWide: 0.18, browInnerUp: 0.16 } });
      this.overlays.push({ t0: t + 0.75, a: 0.07, h: 0.32, r: 0.12, gaze: [side * 14, 2], head: { yaw: side * 2 } });
      this.overlays.push({ t0: t + 1.2, a: 0.07, h: 0.32, r: 0.15, gaze: [-side * 12, 1.5], head: { yaw: -side * 1.6 } });
      this.eye.scanN = 0; this.eye.nextShift = t + 2.4;
    } else if (P.kind === 'bittersweet') {
      this.overlays.push({ t0: t + 0.5, a: 0.25, h: 0.45, r: 0.4, gaze: [side * 4, -6], head: { pitch: -1 } });   // a small look down, and back
    } else if (P.kind === 'curiosity') {
      this.overlays.push({ t0: t + 0.5, a: 0.1, h: 0.3, r: 0.2, gaze: [side * 5, 1.5] });                        // one glance aside, then focus
      this.eye.nextShift = t + 1.6;
    }
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

  /** A negative feeling draining away: a visible release -- the breath goes out first, the eyes close softly and
   *  reopen, and only then a small smile (acting/relief-take.js, on the face's own clock so a slow frame can't skip
   *  it). Live, the feelings themselves keep coming from the readout. */
  _relief() {
    if (this.reliefTake) return;
    this.startReliefTake(() => this.setAttentionHold(0), { drive: false });
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
    { const S = this.sleep; spring(S, S.target, S.target > S.x ? 0.6 : 5, dt); S.x = clamp(S.x); }   // falls asleep slowly, wakes fast

    // New relief take owns its release/breath until it completes or is interrupted.
    if(this.reliefTake) {
      const R=this.reliefTake; R.sample=reliefSample(t-R.start,R.fear,R.breath);
      this.lastReliefSample=R.sample;
      if (R.drive) this.setEmotion({fear:R.sample.fear,calm:R.sample.calm,joy:R.sample.joy});
    }
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
    spring(this.externalIntent, this.externalIntent.target, this.externalIntent.target > this.externalIntent.x ? 5.5 : 2.0, dt);
    this._direct(dt, t, felt);
    // display rules: how much the trained persona is holding the face (see setMask)
    const neg = Math.max(felt.sadness, felt.anger, felt.fear, felt.disgust);
    const v2sf = this.lively.v2 && ['sadness', 'fear'].includes(this._dominant(NEG)[0]);   // (acting v2: sadness and fear only)
    const R = this.reg = smooth(0.2, 0.5, this.maskS.x) * smooth(0.04, 0.25, neg) * (v2sf ? 0.5 : 1);   // (v2: the display rule half as strong)
    // Keep one coherent onset gesture; continuous blended expression is unchanged.
    for (const [e, rise] of (this.reliefTake ? [] : onsets.sort((a,b) => b[1]-a[1]).slice(0, 1))) {
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
      for (const [ch, w, lo, hi0] of PROTOS[e]) {
        // angerHi: where the anger frown reaches full (pushed anger reads 0.8-0.95, so 0.85 keeps it whole and only
        // grades the weak anger the readout finds in ordinary dramatic lines)
        const hi = e === 'anger' && ch === 'browDown' && this.lively.angerHi > 0 ? this.lively.angerHi : hi0;
        // lo = 0: ~linear engagement up to `hi`; lo > 0: this AU only joins at higher intensity
        let k = w * (lo === 0 ? clamp(amt / hi) * (1.5 - 0.5 * clamp(amt / hi)) : smooth(lo, hi, amt));
        if (e === 'joy' && (ch === 'cheekSquint' || ch === 'cheekRaiser' || ch === 'eyeSquint')) {
          const duch = this.duchenne ?? smooth(0.1, 0.55, amt);   // polite (AU12 only) at low joy -> Duchenne
          k *= 0.2 + 0.8 * duch;
        }
        // anger keeps its full brow whenever it is the leading feeling (a push, a real anger scene); only anger that is
        // a minor part of a mixed reading is softened
        const angerLead = smooth(0.75, 0.95, (felt.anger || 0) / fmax);
        const L = this.lively;
        // fearLeadBrow: when fear clearly leads, the anger and sadness read alongside it (a fear push reads anger ~0.7)
        // don't knit the brows -- the actors' fear raises them and barely lowers them (brow-lower range 0.10-0.17)
        const fearLead = L.fearLeadBrow !== 1 && (felt.fear || 0) >= fmax - 1e-6 ? smooth(0, 0.25, ((felt.fear || 0) - Math.max(felt.anger || 0, felt.sadness || 0)) / fmax) : 0;
        let bk = ch === 'browDown' ? (e === 'sadness' ? L.griefBrow : e === 'fear' ? (L.fearBrow >= 0 ? L.fearBrow : L.griefBrow) : e === 'anger' ? lerp(L.angerBrow, 1, angerLead) : 1) : 1;
        if (ch === 'browDown' && fearLead > 0 && (e === 'anger' || e === 'sadness')) bk *= lerp(1, L.fearLeadBrow, fearLead);
        // fearStretch: fear's lips drawn back (AU20); the actors' fear stretches them ~0.4x as much as this proto did
        if (e === 'fear' && ch === 'mouthStretch') bk *= L.fearStretch;
        if (ch === 'browDown' && this.dbgBrow) this.dbgBrow[e] = (this.dbgBrow[e] || 0) + k * bk * gain;
        add(ch, k * bk * gain);
      }
    };
    // phrasing: the pulse is 1 on a stressed word and eases toward 0.6 while talking (0.75 when quiet)
    const PU = this.pulse || (this.pulse = { x: 1, v: 0, target: 1 });
    const talkingNow = t - (this.lastSay ?? -10) < 0.8;
    PU.target += ((talkingNow ? 0.6 : 0.75) - PU.target) * Math.min(1, dt / 1.4);
    spring(PU, PU.target, 5, dt);
    const pulse = 1 - this.lively.emoPulse * (1 - clamp(PU.x, 0, 1.2));
    const fmax = Math.max(1e-3, ...EMOTIONS.map((e) => felt[e] || 0));
    const sharp = (v) => (this.lively.emoSharpen > 0 ? v * Math.pow(v / fmax, this.lively.emoSharpen) : v);
    const feltTotal = EMOTIONS.reduce((a, e) => a + sharp(felt[e] || 0), 0);
    const share = this.lively.emoNorm > 0 && feltTotal > 1 ? Math.pow(feltTotal, -this.lively.emoNorm) : 1;
    this.dbgBrow = {};   // diagnostics: each feeling's share of the brow-lower target this frame
    for (const e of EMOTIONS) if (felt[e] > 1e-3) applyProto(e, sharp(felt[e]) * share * pulse * this.lively.emoScale);
    // Lab/manual local control is additive and channel-level, so artists can diagnose the rig without inventing a new emotion label.
    for (const [ch, v] of Object.entries(this.manualChannels)) route(ch, v, tgt);

    // blends: each region carried by one of the two feelings
    let blendName = null, blendK = 0;
    for (const B of BLENDS) {
      const a = felt[B.a], b = felt[B.b], lo = Math.min(a, b), hi = Math.max(a, b);
      if (lo < 0.05) continue;
      const k = smooth(0.05, 0.3, lo) * smooth(0.25, 0.65, lo / hi);
      if (k < 0.02) continue;
      for (const [c, m] of Object.entries(B.mul)) scaleCh(tgt, c, lerp(1, m, k));
      // hurtBrow: anger carried with grief reads as anguish, not a scowl (pure anger, pushed or played, has ~no sadness)
      if (B.name === 'hurt' && this.lively.hurtBrow !== 1) scaleCh(tgt, 'browDown', lerp(1, this.lively.hurtBrow, k));
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
    if (this.lively.v2) this._actingV2(tgt, felt, add);
    const post = {};   // fast components added after the muscle springs (micro-expressions, tremors, gasps)
    this.dbg = { tgt, post };   // diagnostics: this frame's spring targets and post-spring additions
    if (idle && this.options.micro && !this.reliefTake && R > 0.2 && neg > 0.12) {
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
    this._moment(t, post);

    // timed reactions and conversational beats
    const ovHead = { pitch: 0, yaw: 0, roll: 0, z: 0 };
    this.ovGaze = [0, 0];
    let ovBreath = 0;
    this.overlays = this.overlays.filter((o) => {
      const u = t - o.t0;
      if (u < 0) return true;
      // at most one run of reaction glances every 6 s (a run: the glances of one cue, within 1.5 s); a cue that comes
      // sooner keeps its expression and head, not the eye movement -- otherwise a changing push keeps the eyes darting
      if (o.gaze && !o.gazeOk) {
        const run = this._gazeRun ?? -10;
        if (t - run < 1.5) o.gazeOk = true;
        else if (t - run > 6) { o.gazeOk = true; this._gazeRun = t; }
        else o.gaze = null;
      }
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
      const sl = CHANNELS[c] === 'brow' ? this.lively.browSlow : 1;
      const s = this.ch[c];
      const target = Math.max(0, tgt[c] * this.asym[c]);
      const rising = target > s.x;
      spring(s, target, 3.36 / ((rising ? on : off) * sl), dt);
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
    if (this.lively.raiseGain !== 1) for (const c of ['browInnerUpLeft', 'browInnerUpRight', 'browOuterUpLeft', 'browOuterUpRight']) this.out[c] *= this.lively.raiseGain;
    // outerGain: the outer brow raise on its own (the tracker barely reads it below ~0.3 on this face; actors' 0.58 range vs ours 0.32)
    if (this.lively.outerGain !== 1) for (const c of ['browOuterUpLeft', 'browOuterUpRight']) this.out[c] *= this.lively.outerGain;
    // A directed eye closure must not fight the fear eye-widen morph.
    // Only the new relief take uses this ownership rule; other takes are unchanged.
    if(this.reliefTake?.sample) {
      const close=smooth(0,.5,this.reliefTake.sample.eyeClose);
      this.out.eyeWideLeft*=1-close; this.out.eyeWideRight*=1-close;
    }
    composeSpeech(this.out, this.speechFrame, this.speechGain);
    // blink composes with the lid state: closes whatever is open
    for (const [side, k] of [['Left', 1], ['Right', 0.97]]) {
      const lid = this.out['eyeBlink' + side];
      const bk = B * k;
      const bks = Math.max(bk, 0.98 * this.sleep.x);   // asleep: the lids stay shut
      this.out['eyeBlink' + side] = lid + (1 - lid) * bks;
      this.out['eyeWide' + side] *= 1 - bks;
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
      this.uniforms.uAU3.value.set(av('mouthPress'), av('mouthShrugLower'), av('eyeWide'), av('mouthStretch'));
      this.uniforms.uFoldLeft.value.set(o.browDownLeft||0,o.browInnerUpLeft||0,o.browOuterUpLeft||0,o.cheekSquintLeft||0);
      this.uniforms.uFoldRight.value.set(o.browDownRight||0,o.browInnerUpRight||0,o.browOuterUpRight||0,o.cheekSquintRight||0);
      this.uniforms.uFoldEye.value.set(o.eyeSquintLeft||0,o.eyeSquintRight||0,o.eyeWideLeft||0,o.eyeWideRight||0);
    }

    // ---- apply morphs
    const outMap = this.out;
    outMap.browInnerUp = 0; outMap.cheekPuff = 0;   // combined channels unused (split ones are driven)
    for (const m of this.morphMeshes) {
      const inf = m.morphTargetInfluences;
      for (const [name, i] of m.userData.map) inf[i] = outMap[name] || 0;
    }
    if(this.reliefTake?.sample?.done) {
      const callback=this.reliefTake.onComplete; this.reliefTake=null;
      const B=this.breath; B.kind='rest'; B.sigh=false; B.relief=false;
      B.from=B.val; B.depth=Math.max(.14,B.val); B.phase=.80; B.T=4.5; B.inhale=.38; B.hold=0;
      if(typeof callback==='function') callback();
    }
  }

  _breathe(dt, t, add, post) {
    const B = this.breath, f = this.felt || {};
    if (!this.options.breathing) { B.val = 0; this.uniforms && (this.uniforms.uBreath.value = 0); return; }
    if (this.reliefTake?.sample) {
      const S=this.reliefTake.sample,prev=B.val;
      B.kind='directed-relief'; B.relief=true; B.sigh=false;
      B.val=S.breath; B.vel=(B.val-prev)/Math.max(dt,.001);
      B.gasp=B.huff=B.speechInhale=false;
      this.breathVal=B.val; if(this.uniforms) this.uniforms.uBreath.value=B.val;
      route('eyeBlink',S.eyeClose,post);
      add('mouthFunnel',S.exhaleMouth); add('jawOpen',S.exhaleMouth*.32);
      add('mouthSmileL',S.smile); add('mouthSmileR',S.smile*.78);
      add('cheekSquint',S.cheek);
      return;
    }
    const ar = this.arousal || 0;
    const sad = f.sadness || 0, fear = f.fear || 0, anger = f.anger || 0;
    // kinds: rest | sigh | speech | gasp (startle: sharp in-breath, held) | huff (anger: held, then forced out
    // through the nose) | shudder (sadness: the in-breath catches in steps)
    const startCycle = (kind = 'rest') => {
      const rate = clamp(14 * (1 + 0.45 * ar) * (1 - 0.3 * (f.calm || 0)), 7, 24) * (1 - 0.5 * this.sleep.x);       // breaths / min (asleep: slow)
      B.from = B.val;
      B.T = (60 / rate) * (0.82 + 0.36 * this.rand());                                // irregular cycles
      B.depth = (0.8 + 0.4 * this.rand()) * (1 + 0.25 * Math.max(0, ar)) * (1 - 0.3 * fear) * (1 + 0.45 * this.sleep.x);   // fear: shallow; asleep: deep
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
    // (fearSpeech: frightened speech is held and small -- the actors' mouths moved ~half as much in fear scenes)
    this.speechGain = clamp(1 + .25 * Math.max(0, this.arousal || 0) - .2 * (f.sadness || 0) - .25 * (f.calm || 0) - this.lively.fearSpeech * (f.fear || 0), this.lively.fearSpeech > 0 ? .45 : .55, 1.35);
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
      // rate: ~12.5/min at rest plus fewer beat-, leak- and gaze-evoked blinks: ~11/min overall from the face's own
      // blink log, the median of the film performances it was tuned on (a face tracker over-counts the porcelain
      // lids: it read 15-28 'blinks'/min with blinking switched off); rises with arousal, falls while concentrating on text; each feeling has
      // its own effect (the angry stare and rapt curiosity suppress blinking, fear raises it)
      let mul = 0;
      // (fearBlink > 0 replaces fear's multiplier: the actors' fear blinks ~13.5/min, more than at rest)
      // (acting v2: fear stares -- wide eyes that keep closing read as drowsy)
      for (const e of EMOTIONS) mul += Math.log(e === 'fear' && this.lively.v2 ? 0.55 : e === 'fear' && this.lively.fearBlink > 0 ? this.lively.fearBlink : BLINK_MUL[e]) * Math.min(1, f[e]);
      const rate = clamp(12.5 * Math.exp(mul) * (1 + 0.3 * Math.max(0, this.arousal)) * (1 - 0.25 * clamp(this.writingAmt.x)), 5, 38);
      const mean = 60 / rate;
      // log-normal-ish interval
      const g = Math.exp((this.rand() + this.rand() + this.rand() - 1.5) * 0.75);
      b.next = t + Math.max(0.7, mean * g);
      if (this.rand() < 0.06) b.pending = t + b.dur[0] + b.dur[1] + b.dur[2] * 0.6 + 0.05; // double blink
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
    const held = Math.max(this.intent.x, this.externalIntent.x) > 0.3;
    if (idle && W > 0.5 && this.gazeMode === 'camera' && !held && this.options.textGlances !== false) {
      if (!G.active && t >= G.next) { G.active = true; G.until = t + 0.5 + 0.8 * this.rand(); }
      if (G.active && t >= G.until) { G.active = false; G.next = t + (6 + 8 * this.rand()) * this.lively.glanceGap; }   // glanceGap: spacing (each glance down also lowers the lids)
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
      if (P.press > P.lastNotice && now - P.press < 0.5 && now - P.lastNotice > 10) { P.lastNotice = now; P.glanceAt = now + 0.18; P.glanceUntil = now + 0.18 + (held ? 0.2 : 0.4 + 0.4 * this.rand()); }
      else if (!held && now - P.t < 0.4 && now - P.lastNotice > 10 && now > P.glanceUntil && this.rand() < dt * (W > 0.5 ? 0.05 : 0.5)) {
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
        // fear freeze (LIVELY.fearFreeze): the actors' fear holds the eyes still on the threat (59-82 shifts/min), not darting
        const fz = this.lively.fearFreeze * smooth(0.25, 0.6, f.fear) * (f.fear >= Math.max(f.anger, f.sadness, f.joy) ? 1 : 0.5);
        // (a performance moment steering the eyes takes over most of the look-aways)
        const awayP = (0.06 + 0.12 * f.sadness + 0.1 * f.disgust + 0.05 * f.fear - 0.06 * f.anger - 0.03 * f.joy + (W > 0.5 ? -0.03 : 0.03)) * (this.momGaze ? 0.3 : 1) * (1 - 0.5 * fz);
        const scanSide = () => [(E.offset[0] > 0 ? -1 : 1) * (9 + 9 * this.rand()) * DEG, (this.rand() - 0.4) * 5 * DEG];
        if (E.scanN > 0) {
          // fear: hypervigilance -- quick darts to the sides as if checking for a threat, then freeze on the viewer
          E.scanN--;
          if (E.scanN === 0) { E.offset = [0, -0.5 * DEG]; E.nextShift = t + 1.2 + 1.6 * this.rand(); this.blink.next = Math.max(this.blink.next, t + 0.9); }
          else { E.offset = scanSide(); E.nextShift = t + 0.18 + 0.22 * this.rand(); }
        } else if (f.fear > 0.3 && f.fear >= Math.max(f.anger, f.sadness, f.joy) && r < (0.04 + 0.08 * f.fear) * (this.perf?.kind === 'fear' ? 0.25 : 1) * (1 - 0.8 * fz) && !G.active) {
          E.scanN = 2 + Math.floor(this.rand() * 2);
          E.offset = scanSide();
          E.nextShift = t + 0.18 + 0.22 * this.rand();
        } else if (r < awayP && !G.active) {
          const side = this.rand() < 0.5 ? -1 : 1;
          const down = f.sadness > 0.3 ? -1 : (this.rand() < 0.55 ? 1 : -1);
          E.offset = [side * (5 + 9 * this.rand()) * DEG, down * (3 + 6 * this.rand()) * DEG - f.sadness * 6 * DEG];
          E.nextShift = t + 0.8 + 1.4 * this.rand();
        } else {
          // tiny switches between the viewer's eyes / mouth (anger: a hard, unmoving stare)
          const stare = 1 - 0.7 * Math.min(1, f.anger + fz);
          E.offset = [(this.rand() - 0.5) * 1.8 * DEG * stare, (this.rand() - 0.6) * 1.2 * DEG * stare - f.sadness * 5 * DEG];
          const rate = (1 + 1.2 * f.fear * (1 - fz) + 0.8 * f.curiosity + 0.4 * f.surprise - 0.5 * f.calm - 0.6 * f.anger) * (1 - 0.5 * fz);
          E.nextShift = t + (0.8 + 2.2 * this.rand()) / Math.max(0.35, rate);
        }
      }
      yaw += E.offset[0] + (this.ovGaze ? this.ovGaze[0] : 0) + (this.momGaze ? this.momGaze[0] * DEG : 0);
      pitch += E.offset[1] + (this.ovGaze ? this.ovGaze[1] : 0) + (this.momGaze ? this.momGaze[1] * DEG : 0);
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
        if (amp > 9 && this.options.blinks && this.rand() < 0.08 && t - this.blink.t0 > 0.5) this.blink.next = t; // gaze-evoked blink
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
    pitch -= 8 * this.sleep.x; roll += 2.5 * this.sleep.x;   // asleep: the head sinks and tips
    // head follows large gaze offsets (eye-head coordination)
    const [gy, gp] = this.gazeDesired || [0, 0];
    const follow = (a) => Math.sign(a) * Math.max(0, Math.abs(a) - 6 * DEG) * 0.4;
    yaw += follow(gy) / DEG;
    pitch += follow(gp) / DEG * 0.7;
    // perlin micro-motion (amplitude grows with arousal, slower when calm)
    let breathP = 0, breathY = 0;
    if (idle) {
      const slow = 1 - 0.45 * f.calm;
      const amp = this.lively.headNoise * (0.7 + 0.8 * Math.max(0, this.arousal) + 0.3 * f.curiosity) * (1 - 0.4 * f.calm) * (1 - 0.75 * Math.max(this.intent.x, this.externalIntent.x)) * this.motionStyle.amplitude * this.style.amp.x;
      pitch += amp * 1.1 * this.noise.fbm(t * 0.21 * slow + 100, 3);
      yaw += amp * 1.4 * this.noise.fbm(t * 0.17 * slow + 200, 3);
      roll += amp * 0.8 * this.noise.fbm(t * 0.19 * slow + 300, 3);
      // talking: the head moves with the phrasing (faster and larger than the idle drift), less when holding
      const spk = Math.max(0, 1 - (t - (this.lastSay ?? -10)) / 0.8);
      if (spk > 0 && this.lively.speechHead > 0) {
        const a = this.lively.speechHead * spk * (1 + 0.6 * Math.max(0, this.arousal)) * (1 - 0.6 * Math.max(this.intent.x, this.externalIntent.x));
        pitch += a * 1.2 * this.noise.fbm(t * 0.9 + 400, 2);
        yaw += a * 1.6 * this.noise.fbm(t * 0.7 + 500, 2);
        roll += a * 0.9 * this.noise.fbm(t * 0.8 + 600, 2);
      }
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
      const pm = this.motionStyle.posture * this.style.posture.x; PS.target = [(this.rand() - 0.5) * 3.2 * pm, (this.rand() - 0.5) * 4.5 * pm, (this.rand() - 0.5) * 4.0 * pm];
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
    const mh = this.momHead || [0, 0, 0];   // a performance moment's head movement (after the springs: its own timing)
    const P = (H.pitch.x + breathP + mh[1]) * DEG, Y = (H.yaw.x + mh[0]) * DEG, R = (H.roll.x + mh[2]) * DEG;
    this.headOut = [Y / DEG, P / DEG, R / DEG];   // final head angles (deg), for diagnostics
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
      u.uIrisGlow.value = this.irisGlow * (1 + 0.12 * W + 0.05 * Math.sin(t * 1.3)) * (1 - 0.9 * this.sleep.x);
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
    // lidFollow: how far the lids follow a downward look (the face tracker reads this face's lowered lid as a squint)
    this.out.eyeLookDownLeft = this.out.eyeLookDownRight = clamp(down) * this.lively.lidFollow;
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
    this.root.traverse((o) => { if (o.isMesh) o.material.dispose(); });   // (the geometry is shared by every face on the page)
  }
}
