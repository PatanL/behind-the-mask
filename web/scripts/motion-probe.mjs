import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.goto('http://127.0.0.1:5190/?attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
// frame strip around the greeting (fires ~1.4 s after load)
const frames = [];
for (let i = 0; i < 12; i++) { await page.waitForTimeout(160); frames.push(await page.screenshot({ clip: { x: 160, y: 120, width: 300, height: 340 } })); }
const fs = await import('node:fs');
frames.forEach((b, i) => fs.writeFileSync(`../runs/mf_${String(i).padStart(2, '0')}.png`, b));
// numeric trace for 25 s with a pointer sweep and a tap
const trace = await page.evaluate(async () => {
  const f = window.__btm.stage.face; const out = [];
  const t0 = performance.now();
  for (let k = 0; k < 100; k++) {
    await new Promise((r) => setTimeout(r, 250));
    if (k === 40) f.setPointer(2.2, -1.6, true);
    if (k > 60 && k < 70) f.setPointer(-1 + (k - 60) * 0.3, 0.2, false);
    if (k === 80) f.react('think');
    out.push([((performance.now() - t0) / 1000).toFixed(1), f.breathVal.toFixed(2), f.breath.sigh ? 'S' : '', (f.head.pitch.x).toFixed(2), (f.eye.yaw / (Math.PI / 180)).toFixed(1), (f.eye.pitch / (Math.PI / 180)).toFixed(1), f.blinkVal.toFixed(2)].join(' '));
  }
  return out;
});
console.log('t breath sigh headPitch eyeYaw eyePitch blink');
console.log(trace.filter((_, i) => i % 2 === 0).join('\n'));
console.log('errors', errs);
await browser.close();
