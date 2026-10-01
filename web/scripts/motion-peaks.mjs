import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
await page.goto('http://127.0.0.1:5190/?attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
await page.waitForTimeout(3500);
const shots = {
  rest: 'f.options.idle = false; f.settle(1.5);',
  greet_flash: 'f.options.idle = true; f.react("greet"); f.settle(0.30);',
  greet_smile: 'f.settle(0.9);',
  think: 'f.settle(3); f.react("think"); f.settle(0.55);',
  breath_out: 'f.settle(3); f.options.breathing = true; f.breath.phase = 0.97; f.breath.depth = 1.2; f.settle(0.05);',
  breath_in: 'f.breath.speechInhale = true; f.settle(0.55);',
  sigh: 'f.breath.nextSigh = 0; f.breath.phase = 0.99; f.settle(0.05); f.settle(f.breath.T * 0.5);',
};
for (const [k, code] of Object.entries(shots)) {
  await page.evaluate((code) => { const f = window.__btm.stage.face; window.__btm.stage.render = window.__btm.stage.render; eval(code); }, code);
  await page.waitForTimeout(60);
  await page.screenshot({ path: `../runs/mp_${k}.png`, clip: { x: 40, y: 75, width: 513, height: 575 } });
}
await browser.close();
