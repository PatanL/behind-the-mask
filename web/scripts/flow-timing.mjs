// How evenly do the live words appear on screen? Records when each new word span appears, for N seconds.
import { chromium } from 'playwright';
const secs = +(process.argv[2] || 40);
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
await page.goto(process.env.URL || 'http://127.0.0.1:5190/?attract=99999');
await page.waitForFunction(() => document.querySelectorAll('#speech .tok').length > 5, null, { timeout: 120000 });
const t = await page.evaluate((secs) => new Promise((done) => {
  const times = [];
  new MutationObserver((ms) => { for (const m of ms) for (const n of m.addedNodes) if (n.classList?.contains('tok')) times.push(performance.now()); })
    .observe(document.querySelector('#speech'), { childList: true, subtree: true });
  setTimeout(() => done(times), secs * 1000);
}), secs);
const gaps = t.slice(1).map((x, i) => x - t[i]).filter((g) => g > 5).sort((a, b) => a - b);
const q = (p) => Math.round(gaps[Math.min(gaps.length - 1, Math.floor(p * gaps.length))]);
console.log(`words on screen: ${(t.length / secs).toFixed(1)}/s; gap ms median ${q(0.5)} p90 ${q(0.9)} p99 ${q(0.99)} max ${Math.round(gaps.at(-1))}; gaps over 400 ms: ${gaps.filter((g) => g > 400).length}`);
await browser.close();
