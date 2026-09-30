// Screenshot the face test page and the exhibit stage for comparison.
import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const out = process.env.OUT || '../runs/face-look';
for (const [name, url, w, h] of [['test', 'http://127.0.0.1:5190/face-test.html', 800, 800], ['app', 'http://127.0.0.1:5190/?attract=999', 1440, 900]]) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  page.on('pageerror', (e) => console.log(name, 'pageerror', e.message));
  await page.goto(url);
  await page.waitForTimeout(7000);
  await page.screenshot({ path: `${out}-${name}.png` });
  await page.close();
}
await browser.close();
