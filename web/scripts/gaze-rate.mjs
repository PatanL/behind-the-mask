// How much the eyes move: large saccades (> 3°) per minute, by cause, and the share of time at the subtitles / pointer,
// per ready-made performance and in live (with a tap every 1.5 s).   URL=http://127.0.0.1:4340 node scripts/gaze-rate.mjs
import { chromium } from 'playwright';
const URL0 = process.env.URL || 'http://100.97.32.64:4340';
const SECS = +(process.env.SECS || 20);
const CASES = (process.env.CASES || 'none|none,joy|lot,sadness|lot,fear|lot,anger|lot,live').split(',');
const b = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const DEG = Math.PI / 180;
for (const c of CASES) {
  const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
  const live = c === 'live';
  const [e, l] = c.split('|');
  await p.goto(live ? `${URL0}/` : `${URL0}/?q=how-was-your-day&e=${e}&l=${l}&autoplay=1`);
  await p.waitForFunction(() => window.__btm?.stage?.face?.eye && document.querySelectorAll('#speech .tok').length > 2, null, { timeout: 60000 });
  const t0 = Date.now(); let lastTap = 0;
  const samples = [];
  while (Date.now() - t0 < SECS * 1000) {
    if (live && Date.now() - lastTap > 1500) { lastTap = Date.now(); await p.locator('.tapb[data-b="sadness"]').click(); }
    samples.push(await p.evaluate(() => { const f = window.__btm.stage.face, E = f.eye; return [E.to[0], E.to[1], f.glance.active ? 1 : 0, f.glancingPointer ? 1 : 0, E.offset[0], E.offset[1], f.ovGaze?.[0] || 0, f.ovGaze?.[1] || 0, E.scanN]; }));
    await p.waitForTimeout(40);
  }
  let sacc = 0, away = 0, text = 0, ptr = 0; const why = {};
  for (let i = 1; i < samples.length; i++) {
    const a = samples[Math.max(0, i - 4)], b = samples[i];
    const [y0, p0] = samples[i - 1], [y1, p1, g, gp] = b;
    if (Math.hypot(y1 - y0, p1 - p0) > 3 * DEG) {
      sacc++;
      const k = a[2] !== b[2] ? 'text' : a[3] !== b[3] ? 'pointer' : (b[8] > 0 || a[8] > 0) ? 'scan'
        : Math.hypot(b[4] - a[4], b[5] - a[5]) > 2 * DEG ? 'idle' : Math.hypot(b[6] - a[6], b[7] - a[7]) > 1.5 * DEG ? 'reaction' : 'other';
      why[k] = (why[k] || 0) + 1;
    }
    text += g; ptr += gp;
  }
  const n = samples.length, mins = SECS / 60;
  const per = Object.entries(why).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${Math.round(v / mins)}`).join(', ');
  console.log(`${c.padEnd(12)} big saccades/min ${(sacc / mins).toFixed(0).padStart(3)} (${per}) | at subtitles ${(100 * text / n).toFixed(0)}% | at pointer ${(100 * ptr / n).toFixed(0)}%`);
  await p.close();
}
await b.close();
