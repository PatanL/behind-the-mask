// End-to-end flow test: pick a question + feeling via clicks, play, inspect a pushed word. Screenshots to runs/.
import { chromium } from 'playwright';
const W = +(process.env.W || 1440), H = +(process.env.H || 900), tag = process.env.TAG || 'desk';
const q = process.env.QQ || 'Do you have feelings?', feel = process.env.FEEL || 'Sadness', amt = process.env.AMT || 'A lot';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: tag === 'phone' ? 2 : 1, isMobile: tag === 'phone', hasTouch: tag === 'phone' });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));
await page.goto('http://127.0.0.1:5190/?attract=999');
await page.waitForSelector('.chip');
await page.getByRole('button', { name: q }).click();
await page.locator('.orb', { hasText: feel }).click();
if (!['No push', 'Mood swing'].includes(feel)) await page.getByRole('radio', { name: amt }).click();
await page.screenshot({ path: `../runs/flow_${tag}_0.png` });
await page.locator('#go').click();
await page.waitForTimeout(2500);
await page.screenshot({ path: `../runs/flow_${tag}_1.png` });
await page.waitForFunction(() => !window.__btm.state.playing, null, { timeout: 90000 });
await page.waitForTimeout(800);
const pushed = page.locator('.tok.pushed').first();
if (await pushed.count()) { await pushed.click(); await page.waitForTimeout(400); }
await page.screenshot({ path: `../runs/flow_${tag}_2.png` });
await page.screenshot({ path: `../runs/flow_${tag}_full.png`, fullPage: true });
console.log('pushed words:', await page.locator('.tok.pushed').count(), 'tokens:', await page.locator('.tok').count(), 'errors:', errs.length ? errs : 'none');
await browser.close();
