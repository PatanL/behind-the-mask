// Many live android faces on one page (the explore grid, the hero). Each card holds its own canvas, a picture frame
// that scrolls with the page like any element; one offscreen WebGL renderer draws each face in turn and copies it into
// its frame. Only frames on screen are drawn; asleep androids are drawn less often (they barely move).
import * as THREE from 'three';
import { AndroidFace } from '../../src/face/face.js';
import { createStudioEnvironment } from '../../src/face/stage.js';
import { Governor, startTier } from '../../src/quality.js';

const MAX = 640;   // a frame's largest side in pixels at full quality (the offscreen canvas is this big)

function lights(scene, target) {
  const g = new THREE.Group();
  const aim = new THREE.Object3D(); aim.position.set(0, 0.02, 0.05); g.add(aim);
  const add = (l, p) => { l.position.set(...p); l.target = aim; g.add(l); };
  add(new THREE.SpotLight(0xffe3c8, 7.5, 0, Math.PI / 9, 0.85, 2), [-0.75, 0.75, 1.15]);
  add(new THREE.DirectionalLight(0x9fb6e0, 0.35), [1.2, 0.1, 1.0]);
  add(new THREE.DirectionalLight(0x8fdcff, 2.2), [1.1, 0.55, -1.2]);
  add(new THREE.DirectionalLight(0xffa86a, 0.9), [-1.2, 0.25, -1.0]);
  add(new THREE.DirectionalLight(0xdfe7ff, 0.25), [0.1, 1.5, 0.2]);
  target.add(g);
}

export class FaceWall {
  constructor() {
    const off = document.createElement('canvas');
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas: off, antialias: true, alpha: true, powerPreference: 'high-performance' }));
    r.setPixelRatio(1);
    r.setSize(MAX, MAX, false);
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.9;
    r.setScissorTest(true);
    r.setClearColor(0x000000, 0);
    this.env = createStudioEnvironment(r);
    this.slots = new Map();
    this.clock = new THREE.Clock();
    // smooth before sharp: smaller frames, then each face drawn every other / third frame, when frames run late
    this.max = MAX; this.every = 1; this.n = 0;
    this.gov = new Governor({ name: 'faces', start: startTier() ? 1 : 0, apply: (q) => { this.max = q.max; this.every = q.every; r.setSize(q.max, q.max, false); },
      tiers: [{ max: MAX, every: 1 }, { max: 512, every: 1 }, { max: 400, every: 2 }, { max: 320, every: 3 }] });
    this.gov.arm(4000);
    const loop = () => { if (this.stopped) return; this.render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  /** A live face framed in element `el` (a canvas is added inside it). Returns the slot ({face, ready}). */
  attach(el, { look, seed = 1 } = {}) {
    if (this.slots.has(el)) return this.slots.get(el);
    const frame = document.createElement('canvas');
    frame.className = 'face-frame';
    el.appendChild(frame);
    const scene = new THREE.Scene();
    scene.environment = this.env; scene.environmentIntensity = 0.45;
    const camera = new THREE.PerspectiveCamera(22, 1, 0.05, 20);
    camera.position.set(0, 0.03, 0.82); camera.lookAt(0, 0.0, 0.03);
    const face = new AndroidFace(scene, `${import.meta.env.BASE_URL}face/android.glb`, { camera, seed, castShadow: false, textTarget: { x: 0, y: -0.9 } });
    const slot = { el, frame, ctx: frame.getContext('2d'), scene, camera, face, look, ready: false, last: 0, asleep: false };
    face.ready.then(async () => {
      lights(scene, face.root);
      face.setGaze('camera');
      face.setFaceDetail({ wrinkles: 2 });
      if (look) face.setLook(look);
      // its shaders compile in the background before it's first drawn (compiling on first draw froze a phone for ~2 s)
      await Promise.race([this.renderer.compileAsync(scene, camera).catch(() => {}), new Promise((r) => setTimeout(r, 3000))]);
      slot.ready = true;
    }).catch(() => {});
    this.slots.set(el, slot);
    return slot;
  }

  /** Opening a coin from the grid: its card's live face carries on in the coin page's stage (framed like it) while
   *  the full-quality face gets ready there, so there's no frozen moment. -> the slot */
  retarget(from, to) {
    const s = this.slots.get(from);
    if (!s) return null;
    this.slots.delete(from);
    s.el = to; s.stage = true;
    this.slots.set(to, s);
    s.frame.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
    to.prepend(s.frame);   // under the stage's own canvas, which fades in over it
    if (this.max < 1024) { this.max = 1024; this.renderer.setSize(1024, 1024, false); }   // one big face now, not a grid
    return s;
  }

  /** Done (the page moved on): no more drawing, and the GPU context is let go. */
  stop() {
    for (const el of [...this.slots.keys()]) this.detach(el);
    this.stopped = true;
    try { this.renderer.dispose(); this.renderer.forceContextLoss(); } catch { /* */ }
  }

  detach(el) {
    const s = this.slots.get(el);
    if (!s) return;
    this.slots.delete(el);
    s.frame.remove();
    try { s.face.dispose(); } catch { /* */ }
  }

  render() {
    this.gov.frame(performance.now());
    const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime, r = this.renderer, dpr = Math.min(devicePixelRatio || 1, 2), MX = this.max;
    let i = 0; this.n++;
    for (const s of this.slots.values()) {
      if (!s.ready || !s.el.isConnected) continue;
      const b = s.el.getBoundingClientRect();
      if (b.bottom < -50 || b.top > innerHeight + 50 || b.width < 4) continue;
      if (this.every > 1 && (this.n + i++) % this.every) continue;   // round robin: this face waits a frame
      if (t - s.last < (s.asleep ? 1 / 8 : 0)) continue;   // asleep: a few frames a second is plenty
      s.face.update(Math.min(0.2, t - (s.lastT ?? t) || dt)); s.lastT = t; s.last = t;
      const k = Math.min(1, MX / Math.max(b.width * dpr, b.height * dpr));
      const w = Math.round(b.width * dpr * k), h = Math.round(b.height * dpr * k);
      if (s.frame.width !== w || s.frame.height !== h) { s.frame.width = w; s.frame.height = h; }
      s.camera.aspect = w / h;
      if (s.stage) {   // framed like the coin page's stage (see stage.js resize), so the hand-over doesn't jump
        const tall = w / h < 0.8, fit = tall ? 1.0 : 0.87;
        s.camera.fov = 24; s.camera.position.set(0, 0.035, fit + 0.03); s.camera.lookAt(0, tall ? -0.05 : -0.04, 0.03);
      }
      s.camera.updateProjectionMatrix();
      r.setViewport(0, 0, w, h); r.setScissor(0, 0, w, h); r.clear();
      r.render(s.scene, s.camera);
      // the rendered corner (bottom-left in GL terms is the canvas's bottom rows) into this card's frame
      s.ctx.clearRect(0, 0, w, h);
      s.ctx.drawImage(r.domElement, 0, MX - h, w, h, 0, 0, w, h);
    }
  }
}
