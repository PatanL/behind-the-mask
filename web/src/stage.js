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
    this.lookAt = new THREE.Vector3(0, 0.0, 0.03);
    this.camera.position.set(0, 0.035, 0.98).add(new THREE.Vector3(0, 0.005, 0.03));
    this.camera.lookAt(this.lookAt);
    this.face = new AndroidFace(this.scene, '/face/android.glb', { camera: this.camera, seed: 11, textTarget: { x: 0.9, y: -0.2 } });
    this.lights = createFaceStage(r, this.scene, { target: this.face.root });
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
    const fit = w / h < 0.8 ? 1.12 : 0.98;
    this.camera.position.set(0, 0.04, fit + 0.03);
    this.camera.lookAt(this.lookAt);
    this.camera.updateProjectionMatrix();
  }

  setPushColor(hex, amount) { this.auraTarget.color.set(hex); this.auraTarget.amt = amount; }

  render() {
    const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime;
    this.face.update(dt, t);
    const u = this.aura.material.uniforms;
    u.uColor.value.lerp(this.auraTarget.color, Math.min(1, dt * 2));
    u.uAmt.value += (this.auraTarget.amt - u.uAmt.value) * Math.min(1, dt * 1.5);
    u.uTime.value = t;
    this.composer.render();
  }
}
