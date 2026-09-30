// Recommended lighting for the porcelain android face: warm key, cool fill, cyan + amber rims,
// and a procedural studio environment (soft boxes on a deep blue-black room) for the glaze reflections.
//
//   import { createFaceStage } from './face/stage.js';
//   const stage = createFaceStage(renderer, scene, { target: face.root });
//
// All light positions are relative to `target` (the face root) and scale with `scale`.
import * as THREE from 'three';

function panel(scene, { pos, size, color, intensity, lookAt = [0, 0, 0] }) {
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(size[0], size[1]),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity), side: THREE.DoubleSide }),
  );
  m.position.set(...pos);
  m.lookAt(...lookAt);
  scene.add(m);
  return m;
}

export function createStudioEnvironment(renderer) {
  const env = new THREE.Scene();
  // deep blue-black room with a slight vertical gradient
  const room = new THREE.Mesh(
    new THREE.SphereGeometry(10, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {},
      vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: 'varying vec3 vP; void main(){ float y = normalize(vP).y; vec3 c = mix(vec3(0.006,0.008,0.014), vec3(0.018,0.026,0.045), smoothstep(-0.4,0.9,y)); gl_FragColor = vec4(c,1.0); }',
    }),
  );
  env.add(room);
  // big warm soft box, upper left front (key)
  panel(env, { pos: [-3.2, 2.6, 3.4], size: [3.2, 3.6], color: 0xffe0c2, intensity: 3.2 });
  // cool strip, right rear (rim) and left rear
  panel(env, { pos: [4.0, 1.0, -3.2], size: [0.7, 5.5], color: 0x9ad8ff, intensity: 4.5 });
  panel(env, { pos: [-4.2, 0.6, -2.8], size: [0.6, 4.5], color: 0xffb27a, intensity: 1.6 });
  // soft overhead
  panel(env, { pos: [0.4, 5.0, 0.6], size: [3.5, 2.0], color: 0xdce6ff, intensity: 0.7 });
  // faint cool fill, right front
  panel(env, { pos: [4.0, 0.2, 3.0], size: [2.4, 2.4], color: 0x8ea8d8, intensity: 0.45 });
  // very faint warm bounce from below (the text screen)
  panel(env, { pos: [0, -3.5, 2.5], size: [4, 1.2], color: 0xffd9b0, intensity: 0.18 });
  const pmrem = new THREE.PMREMGenerator(renderer);
  const rt = pmrem.fromScene(env, 0.035);
  pmrem.dispose();
  env.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
  return rt.texture;
}

export function createFaceStage(renderer, scene, { target = null, scale = 1, shadows = true, envIntensity = 0.85 } = {}) {
  scene.environment = createStudioEnvironment(renderer);
  scene.environmentIntensity = envIntensity;
  const group = new THREE.Group();
  group.name = 'FaceStageLights';
  scene.add(group);
  const s = scale;
  const aim = new THREE.Object3D();
  aim.position.set(0, 0.02 * s, 0.05 * s);
  group.add(aim);

  // Key: warm, upper left, slightly soft (spot, casts the nose / brow / lid shadows)
  const key = new THREE.SpotLight(0xffe3c8, 30, 0, Math.PI / 9, 0.85, 2);
  key.position.set(-0.75 * s, 0.75 * s, 1.15 * s);
  key.target = aim;
  if (shadows) {
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.00012;
    key.shadow.normalBias = 0.0012 * s;
    key.shadow.radius = 5;
    key.shadow.camera.near = 0.8 * s;
    key.shadow.camera.far = 2.4 * s;
  }
  group.add(key);

  // Fill: cool, right front, low
  const fill = new THREE.DirectionalLight(0x9fb6e0, 0.35);
  fill.position.set(1.2 * s, 0.1 * s, 1.0 * s);
  fill.target = aim;
  group.add(fill);

  // Rims: cyan from right rear, amber from left rear
  const rimCool = new THREE.DirectionalLight(0x8fdcff, 2.2);
  rimCool.position.set(1.1 * s, 0.55 * s, -1.2 * s);
  rimCool.target = aim;
  group.add(rimCool);
  const rimWarm = new THREE.DirectionalLight(0xffa86a, 0.9);
  rimWarm.position.set(-1.2 * s, 0.25 * s, -1.0 * s);
  rimWarm.target = aim;
  group.add(rimWarm);

  // faint top light so the skull reads
  const top = new THREE.DirectionalLight(0xdfe7ff, 0.25);
  top.position.set(0.1 * s, 1.5 * s, 0.2 * s);
  top.target = aim;
  group.add(top);

  if (target) {
    target.add(group);
  }
  return { group, key, fill, rimCool, rimWarm, top, aim };
}
