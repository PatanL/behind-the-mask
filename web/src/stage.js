// The android on stage: renderer, camera, bloom, the porcelain face, and a glow that takes the push's colour.
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { AndroidFace } from './face/face.js';
import { createFaceStage } from './face/stage.js';

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.9;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(24, 1, 0.05, 20);
    this.lookAt = new THREE.Vector3(0, -0.012, 0.03);
    this.camera.position.set(0, 0.035, 0.98).add(new THREE.Vector3(0, 0.005, 0.03));
    this.camera.lookAt(this.lookAt);
    this.face = new AndroidFace(this.scene, `${import.meta.env.BASE_URL}face/android.glb`, { camera: this.camera, seed: 11, textTarget: { x: 0.9, y: -0.2 } });
    this.lights = createFaceStage(r, this.scene, { target: this.face.root, envIntensity: 0.45 });
    this.lights.key.intensity = 9;   // the stage defaults blow the porcelain out under our tone mapping
    // an aura behind the head that takes the colour of the push
    const auraMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uColor: { value: new THREE.Color('#9fb4ff') }, uAmt: { value: 0.0 }, uTime: { value: 0 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: `uniform vec3 uColor; uniform float uAmt; uniform float uTime; varying vec2 vUv;
        void main(){ vec2 p = vUv - 0.5; float r = length(p);
          float wob = 0.03 * sin(atan(p.y, p.x) * 5.0 + uTime * 0.8);
          float g = pow(smoothstep(0.34, 0.06 + wob, r), 1.6);
          gl_FragColor = vec4(uColor * g * (0.02 + 0.12 * uAmt), 1.0); }`,
    });
    this.aura = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.9), auraMat);
    this.aura.position.set(0, 0.03, -0.22);
    this.scene.add(this.aura);
    this.auraTarget = { color: new THREE.Color('#9fb4ff'), amt: 0 };
    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.35, 0.5, 0.82);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.clock = new THREE.Clock();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.resize();
    this.face.setGaze('camera');
    // the visitor's pointer, in this canvas' NDC (beyond +-1 when it's over the text or the controls)
    const toNdc = (e) => { const r = canvas.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1)]; };
    addEventListener('pointermove', (e) => { const [x, y] = toNdc(e); this.face.setPointer(x, y, false); }, { passive: true });
    addEventListener('pointerdown', (e) => { const [x, y] = toNdc(e); this.face.setPointer(x, y, true); }, { passive: true });
    const loop = () => { this.render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  get ready() { return this.face.ready; }

  resize() {
    const w = this.canvas.clientWidth || 400, h = this.canvas.clientHeight || 500;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    // keep the whole head in frame on tall and wide stages
    // frame head and shoulders; tall (phone) stages step back a little further
    const fit = w / h < 0.8 ? 1.02 : w / h < 1.15 ? 0.8 : 0.9;   // square centre stage: closer, the face is the star
    this.camera.position.set(0, 0.035, fit + 0.03);
    this.camera.lookAt(this.lookAt);
    this.camera.updateProjectionMatrix();
  }

  setPushColor(hex, amount) {
    this.auraTarget.color.set(hex); this.auraTarget.amt = amount;
    if (this.face.uniforms) this.face.uniforms.uGlowColor.value.set(hex);
  }
  /** Seam glow 0..1 (driven by the measured signal while it writes). */
  setGlow(v) { this.glowTarget = v; }

  render() {
    const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime;
    if (!this.paused) this.face.update(dt);   // the face keeps its own monotonic time (QA can pause and settle())
    const u = this.aura.material.uniforms;
    u.uColor.value.lerp(this.auraTarget.color, Math.min(1, dt * 2));
    u.uAmt.value += (this.auraTarget.amt - u.uAmt.value) * Math.min(1, dt * 1.5);
    u.uTime.value = t;
    if (this.face.uniforms) { const g = this.face.uniforms.uGlow; g.value += ((this.glowTarget ?? 0.15) * 3.2 - g.value) * Math.min(1, dt * 3); }
    this.composer.render();
  }
}
