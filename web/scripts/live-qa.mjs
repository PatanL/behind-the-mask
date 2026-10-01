// Live mode, end to end: two visitors open ?live=1; one holds Joy, then Sadness; the other taps Fear.
// Screenshots of visitor A to ../runs/live_*.png.   URL=http://127.0.0.1:5190 node scripts/live-qa.mjs
import { chromium } from 'playwright';
const URL0 = (process.env.URL || 'http://127.0.0.1:5190') + '/?live=1&attract=999' + (process.env.Q ? '&' + process.env.Q : '');
const W = +(process.env.W || 1440), H = +(process.env.H || 900), tag = process.env.TAG || 'desk';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const A = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: tag === 'phone' ? 2 : 1, isMobile: tag === 'phone', hasTouch: tag === 'phone' });
const B = await browser.newPage({ viewport: { width: 800, height: 700 } });
const errs = [];
for (const p of [A, B]) p.on('pageerror', (e) => errs.push(e.message));
await A.goto(URL0); await B.goto(URL0);
await A.waitForSelector('#live:not([hidden])', { timeout: 30000 });
const shot = (n) => A.screenshot({ path: `../runs/live_${tag}_${n}.png` });
// wait for a story to start
await A.waitForFunction(() => document.querySelectorAll('#speech .tok').length > 3, null, { timeout: 120000 });
await shot('0_start');
const hold = async (page, b, ms) => {
  const el = page.locator(`.tapb[data-b="${b}"]`);
  const box = await el.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down(); await page.waitForTimeout(ms); await page.mouse.up();
};
await Promise.all([hold(A, 'joy', 5000), (async () => { for (let k = 0; k < 6; k++) { await B.locator('.tapb[data-b="fear"]').click(); await B.waitForTimeout(500); } })()]);
await shot('1_joy_fear');
await A.waitForTimeout(2500); await shot('2_after');
await hold(A, 'sadness', 6000);
await shot('3_sadness');
const pushed = await A.evaluate(() => ({ toks: document.querySelectorAll('#speech .tok').length, viewers: document.querySelector('#crowd-viewers').textContent, power: document.querySelector('#crowd-power').textContent, caption: document.querySelector('#caption').textContent }));
console.log(pushed);
// wait for the end of the story and the "nobody pushed" card
await A.waitForSelector('#live-plain:not([hidden])', { timeout: 120000 }).catch(() => console.log('no end card'));
await A.waitForTimeout(800);
await shot('4_end');
await A.screenshot({ path: `../runs/live_${tag}_full.png`, fullPage: true });
// a word from the live story: the decision inspector
const tok = A.locator('#speech .tok.pushed').first();
if (await tok.count()) { await tok.click(); await A.waitForTimeout(400); await shot('5_inspect'); }
console.log('errors:', errs.length ? errs : 'none');
await browser.close();
