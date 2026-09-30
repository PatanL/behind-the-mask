// node scripts/app-shot.mjs "<query>" [waitMs] [out.png] [w] [h] -> screenshot of the exhibit page
import { chromium } from 'playwright';
const [q = '', wait = '6000', out = '../runs/app.png', w = '1440', h = '900'] = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[${m.type()}]`, m.text().slice(0, 300)); });
await page.goto(`http://127.0.0.1:5190/?${q}`);
await page.waitForTimeout(+wait);
await page.screenshot({ path: out, fullPage: process.env.FULL === '1' });
console.log('saved', out);
await browser.close();
