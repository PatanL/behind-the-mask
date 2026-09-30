import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 600, height: 600 } });
await page.goto('http://127.0.0.1:5190/?attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
console.log(await page.evaluate(() => {
  const { stage, THREE } = window.__btm; const out = [];
  stage.face.root.traverse((o) => {
    if (!o.isMesh || !o.name.startsWith('Skin')) return;
    const g = o.geometry, P = g.attributes.position, N = g.attributes.normal, I = g.index;
    let neg = 0, deg = 0, negx = 0, tris = I.count / 3, posX = 0, negXside = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3(), vn = new THREE.Vector3();
    for (let t = 0; t < I.count; t += 3) {
      const i0 = I.getX(t), i1 = I.getX(t + 1), i2 = I.getX(t + 2);
      a.fromBufferAttribute(P, i0); b.fromBufferAttribute(P, i1); c.fromBufferAttribute(P, i2);
      n.subVectors(b, a).cross(c.clone().sub(a));
      if (n.lengthSq() < 1e-16) { deg++; continue; }
      vn.fromBufferAttribute(N, i0).add(new THREE.Vector3().fromBufferAttribute(N, i1)).add(new THREE.Vector3().fromBufferAttribute(N, i2));
      const cx = (a.x + b.x + c.x) / 3;
      if (cx > 0) posX++;
      if (n.dot(vn) < 0) { neg++; negx += cx; if (cx < 0) negXside++; }
    }
    const bb = new THREE.Box3().setFromBufferAttribute(P);
    out.push(`${o.name}: tris ${tris} degenerate ${deg} normal-disagree ${neg} (mean x ${(negx / Math.max(1, neg)).toFixed(3)}, on -x ${negXside}) tris with x>0: ${posX} bbox x ${bb.min.x.toFixed(3)}..${bb.max.x.toFixed(3)} indexType ${I.array.constructor.name} maxIndex ${Math.max(...I.array)} verts ${P.count}`);
  });
  return out.join('\n');
}));
await browser.close();
