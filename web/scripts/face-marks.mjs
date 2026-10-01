import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 600, height: 600 } });
await page.goto('http://127.0.0.1:5190/?made=1&attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
console.log(await page.evaluate(() => {
  const { stage, THREE } = window.__btm; const f = stage.face; const out = [];
  const skin = f.morphMeshes.reduce((m, o) => (!m || o.geometry.attributes.position.count > m.geometry.attributes.position.count ? o : m), null);
  const P = skin.geometry.attributes.position; const v = new THREE.Vector3();
  let nose = null, chin = null, top = null; const bb = new THREE.Box3().setFromBufferAttribute(P);
  // mouth: vertices near x=0 in front with a gap: find the lip line as the y of the maximum z-dip near x=0 below the nose
  const mid = [];
  for (let i = 0; i < P.count; i++) { v.fromBufferAttribute(P, i); if (!nose || v.z > nose.z) nose = v.clone(); if (Math.abs(v.x) < 0.003) mid.push(v.clone()); if (!top || v.y > top.y) top = v.clone(); }
  mid.sort((a, b) => b.y - a.y);
  const front = mid.filter((p) => p.z > nose.z - 0.05);
  chin = front.reduce((m, p) => (p.y < m.y ? p : m), front[0]);
  const eyes = f.eyes.map((e) => { const w = new THREE.Vector3(); e.pivot.getWorldPosition(w); skin.worldToLocal(w); return w; });
  out.push('bbox ' + ['min', 'max'].map((k) => bb[k].toArray().map((x) => x.toFixed(4)).join(',')).join(' | '));
  out.push('nose ' + nose.toArray().map((x) => x.toFixed(4)));
  out.push('chin ' + chin.toArray().map((x) => x.toFixed(4)));
  out.push('top ' + top.toArray().map((x) => x.toFixed(4)));
  out.push('eyes ' + eyes.map((e) => e.toArray().map((x) => x.toFixed(4)).join(',')).join(' | '));
  // midline profile: y -> z (front-most) in 5mm bins, to find lips/brow
  const bins = {}; for (const p of front) { const k = Math.round(p.y * 200) / 200; if (!(k in bins) || p.z > bins[k]) bins[k] = p.z; }
  out.push('profile ' + Object.entries(bins).sort((a, b) => b[0] - a[0]).map(([y, z]) => `${(+y).toFixed(3)}:${z.toFixed(3)}`).join(' '));
  return out.join('\n');
}));
await browser.close();
