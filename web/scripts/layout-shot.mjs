// Screenshots of the one-screen layout: live and ready-made, desktop and phone -> ../runs/lay_*.png
import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const errs = [];
for (const [tag, W, H, mob] of [['desk', 1440, 900, false], ['phone', 390, 844, true]]) {
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: mob ? 2 : 1, isMobile: mob, hasTouch: mob });
  page.on('pageerror', (e) => errs.push(`${tag}: ${e.message}`));
  await page.goto('http://127.0.0.1:5190/?attract=999');
  await page.waitForFunction(() => document.querySelectorAll('#speech .tok').length > 30, null, { timeout: 120000 }).catch(() => errs.push(`${tag}: no live words`));
  await page.screenshot({ path: `../runs/lay_${tag}_live.png` });
  const before = await page.evaluate(() => document.querySelector('#live-buttons').getBoundingClientRect().top);
  await page.waitForTimeout(4000);
  const after = await page.evaluate(() => document.querySelector('#live-buttons').getBoundingClientRect().top);
  console.log(tag, 'live buttons moved by', Math.round(after - before), 'px');
  await page.click('#mode-made');
  await page.waitForTimeout(400);
  await page.click('.orb[data-e="sadness"]');
  await page.click('#go');
  await page.waitForTimeout(6000);
  await page.screenshot({ path: `../runs/lay_${tag}_made.png` });
  await page.waitForFunction(() => !window.__btm.state.playing, null, { timeout: 90000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: `../runs/lay_${tag}_made_end.png` });
  await page.close();
}
console.log('errors:', errs.length ? errs : 'none');
await browser.close();
