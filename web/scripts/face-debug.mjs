// Render the exhibit's face under code variants: node scripts/face-debug.mjs  (variants below)
import { chromium } from 'playwright';
const VARIANTS = JSON.parse(process.env.VARIANTS || JSON.stringify({
  x_fixed: '',
  y_dim: 's.lights.key.intensity = 9; s.scene.environmentIntensity = 0.45;',
  z_joy: 's.face.setEmotion({ joy: 0.6 }); s.face.settle(2);',
  zz_sad: 's.face.setEmotion({ joy: 0, sadness: 0.6 }); s.face.settle(2);',
}));
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto('http://127.0.0.1:5190/?attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
await page.waitForTimeout(3000);
for (const [k, code] of Object.entries(VARIANTS)) {
  await page.evaluate((code) => { const s = window.__btm.stage; eval(code); }, code);
  await page.waitForTimeout(900);
  await page.screenshot({ path: `../runs/fd_${k}.png`, clip: { x: 40, y: 80, width: 520, height: 600 } });
}
await browser.close();
