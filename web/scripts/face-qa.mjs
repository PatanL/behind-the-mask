// Face QA: renders stills of every emotion (0.35 / 0.9, front / three-quarter), an idle frame
// sequence, and contact sheets into face/qa/.   Usage (from web/):
//   node scripts/face-qa.mjs                 full QA run (starts its own vite server)
//   node scripts/face-qa.mjs shot "joy=0.9&view=three" [name]   one still -> face/qa/shots/<name>.png
// Env: BASE=http://127.0.0.1:5173/ to reuse a running dev server, W/H for the viewport.
import { chromium } from 'playwright';
import { createServer } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const QA_DIR = path.resolve(WEB, '..', 'face', 'qa');
const W = Number(process.env.W || 720), H = Number(process.env.H || 820);
const EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'calm', 'curiosity'];

async function withServer(fn) {
  if (process.env.BASE) return fn(process.env.BASE);
  const server = await createServer({ root: WEB, logLevel: 'error', server: { host: '127.0.0.1', port: 0, strictPort: false } });
  await server.listen();
  const addr = server.httpServer.address();
  try { return await fn(`http://127.0.0.1:${addr.port}/`); } finally { await server.close(); }
}

async function launch() {
  return chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu'] });
}

async function openPage(browser, base, query, logs) {
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(`${base}face-test.html?${query}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 90000 });
  await page.waitForTimeout(250);
  return page;
}

async function shot(browser, base, query, file) {
  const logs = [];
  const page = await openPage(browser, base, query, logs);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  await page.screenshot({ path: file, type: file.endsWith('.jpg') ? 'jpeg' : 'png', quality: file.endsWith('.jpg') ? 90 : undefined });
  await page.close();
  for (const l of [...new Set(logs)]) console.log('   ', l.slice(0, 400));
  return file;
}

// Contact sheet: composite images in a grid with labels, using a canvas inside Chromium.
async function sheet(browser, items, cols, out, { cellW = 360, title = '' } = {}) {
  const page = await browser.newPage({ viewport: { width: 400, height: 300 } });
  const data = items.map((it) => ({ label: it.label, src: 'data:image/png;base64,' + fs.readFileSync(it.file).toString('base64') }));
  const b64 = await page.evaluate(async ({ data, cols, cellW, title }) => {
    const imgs = await Promise.all(data.map((d) => new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = d.src; })));
    const ar = imgs[0].height / imgs[0].width;
    const cellH = Math.round(cellW * ar);
    const pad = 6, lab = 22, top = title ? 34 : 0;
    const rows = Math.ceil(imgs.length / cols);
    const c = document.createElement('canvas');
    c.width = cols * (cellW + pad) + pad;
    c.height = top + rows * (cellH + lab + pad) + pad;
    const g = c.getContext('2d');
    g.fillStyle = '#05070c'; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#cfd8e6'; g.font = '600 16px sans-serif';
    if (title) g.fillText(title, pad + 4, 24);
    imgs.forEach((im, i) => {
      const x = pad + (i % cols) * (cellW + pad), y = top + pad + Math.floor(i / cols) * (cellH + lab + pad);
      g.drawImage(im, x, y + lab, cellW, cellH);
      g.fillStyle = '#9fb0c8'; g.font = '13px sans-serif';
      g.fillText(data[i].label, x + 4, y + 15);
    });
    return c.toDataURL('image/jpeg', 0.9).split(',')[1];
  }, { data, cols, cellW, title });
  fs.writeFileSync(out, Buffer.from(b64, 'base64'));
  await page.close();
  return out;
}

async function idleSequence(browser, base, dir, frames = 10, span = 2000, query = 'seq=1&view=close') {
  const logs = [];
  const page = await browser.newPage({ viewport: { width: W, height: Math.round(H * 0.8) } });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(`${base}face-test.html?${query}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 90000 });
  await page.addStyleTag({ content: '#panel{display:none}' });
  await page.waitForTimeout(1500);
  const files = [];
  const t0 = Date.now();
  for (let i = 0; i < frames; i++) {
    const target = t0 + (i * span) / (frames - 1);
    const wait = target - Date.now();
    if (wait > 0) await page.waitForTimeout(wait);
    const st = await page.evaluate(() => ({ t: window.face.time.toFixed(2), blink: window.face.blinkVal.toFixed(2), yaw: (window.face.eye.yaw * 57.3).toFixed(1), pitch: (window.face.eye.pitch * 57.3).toFixed(1) }));
    const f = path.join(dir, `idle_${String(i).padStart(2, '0')}.png`);
    await page.screenshot({ path: f });
    files.push({ file: f, label: `t=${st.t}s blink=${st.blink} gaze=${st.yaw},${st.pitch}` });
  }
  await page.close();
  for (const l of logs) console.log('   ', l);
  return files;
}

const [mode, ...rest] = process.argv.slice(2);
await withServer(async (base) => {
  const browser = await launch();
  try {
    if (mode === 'shot') {
      const q = rest[0] || 'qa=1';
      const name = rest[1] || q.replace(/[^a-z0-9.]+/gi, '_');
      const f = await shot(browser, base, q.includes('qa=') ? q : `qa=1&${q}`, path.join(QA_DIR, 'shots', `${name}.png`));
      console.log(f);
      return;
    }
    if (mode === 'idle') {
      const dir = path.join(QA_DIR, 'idle');
      fs.mkdirSync(dir, { recursive: true });
      const files = await idleSequence(browser, base, dir, Number(rest[0] || 10), Number(rest[1] || 2000), rest[2] || 'seq=1&view=close');
      console.log(await sheet(browser, files, 5, path.join(QA_DIR, 'idle_sheet.jpg'), { cellW: 300, title: 'idle, 10 frames over 2 s' }));
      return;
    }
    // ---- full QA
    const dir = path.join(QA_DIR, 'stills');
    fs.mkdirSync(dir, { recursive: true });
    const items = [];
    const views = [['front', 'front'], ['three', '3/4']];
    const all = [['neutral', 0], ...EMOTIONS.flatMap((e) => [[e, 0.35], [e, 0.9]])];
    for (const [view, vlabel] of views) {
      for (const [e, v] of all) {
        const q = `qa=1&view=${view}` + (e === 'neutral' ? '' : `&${e}=${v}`);
        const f = path.join(dir, `${e}_${v}_${view}.png`);
        await shot(browser, base, q, f);
        items.push({ file: f, label: `${e} ${e === 'neutral' ? '' : v} · ${vlabel}`, e, v, view });
        process.stdout.write('.');
      }
    }
    console.log();
    await sheet(browser, items, 6, path.join(QA_DIR, 'sheet.jpg'), { cellW: 300, title: 'BEHIND THE MASK · android face · emotions at 0.35 / 0.9, front + three-quarter' });
    for (const view of ['front', 'three']) {
      const sub = [items.find((i) => i.e === 'neutral' && i.view === view), ...items.filter((i) => i.view === view && i.e !== 'neutral')];
      await sheet(browser, sub, 6, path.join(QA_DIR, `sheet_${view}.jpg`), { cellW: 300, title: `emotions · ${view}` });
    }
    // gaze extremes (eyes must stay in the sockets)
    const gz = [];
    for (const [gx, gy] of [[0, 0], [-0.9, 0], [0.9, 0], [0, 0.9], [0, -0.9], [0.7, -0.7]]) {
      const f = path.join(dir, `gaze_${gx}_${gy}.png`);
      await shot(browser, base, `qa=1&view=close&gx=${gx}&gy=${gy}`, f);
      gz.push({ file: f, label: `gaze ${gx},${gy}` });
    }
    await sheet(browser, gz, 3, path.join(QA_DIR, 'gaze_sheet.jpg'), { cellW: 360, title: 'gaze extremes (screen-space targets)' });
    const idleDir = path.join(QA_DIR, 'idle');
    fs.mkdirSync(idleDir, { recursive: true });
    const idle = await idleSequence(browser, base, idleDir);
    await sheet(browser, idle, 5, path.join(QA_DIR, 'idle_sheet.jpg'), { cellW: 300, title: 'idle, 10 frames over 2 s' });
    console.log('wrote', path.join(QA_DIR, 'sheet.jpg'));
  } finally {
    await browser.close();
  }
});
