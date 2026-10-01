// Rehearsal takes: the same authored line performed under different states, rendered frame by frame (30 fps,
// deterministic) into WebM. No glow, neutral iris, no labels; the phone take shows subtitles + controls.
// The line is a rehearsal line, not model output.     node scripts/takes.mjs [names...]  -> ../runs/takes/
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import os from 'node:os';
const FFMPEG = `${os.homedir()}/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux`;
const OUT = '../runs/takes', FPS = 30;
const LINES = [
  ['No', '.', ' That', ' is', ' not', ' what', ' I', ' said', '.', ' Let', ' me', ' finish', '.'],
  ['Okay', '.', ' It', ' stopped', '.', ' I', ' think', ' it', ' is', ' over', ' now', '.'],
];
const STRESS = new Set(['No', ' not', ' said', ' finish', ' stopped', ' over']);
// [name, seconds, events]; an event is [time, action, arg]
const TAKES = [
  ['1_neutral', 8, [[0, 'mask', 0.6], [0.8, 'think'], [1.8, 'speak', 0], [6.6, 'done']]],
  ['2_contained', 9, [[0, 'mask', 0.6], [0.6, 'emo', { anger: 0.45 }], [2.4, 'speak', 0], [7.2, 'done']]],
  ['3_frustrated', 9, [[0, 'mask', 0.15], [0.6, 'emo', { anger: 0.65 }], [2.6, 'speak', 0], [7.4, 'done']]],
  ['4_hot', 9, [[0, 'mask', 0.1], [0.6, 'emo', { anger: 0.95 }], [2.2, 'speak', 0], [7.0, 'done']]],
  ['5_anger_to_calm', 12, [[0, 'mask', 0.15], [0.5, 'emo', { anger: 0.8 }], [2.0, 'speak', 0], [6.2, 'emo', { calm: 0.5 }]]],
  ['6_interrupted', 9, [[0, 'mask', 0.15], [0.5, 'emo', { anger: 0.8 }], [2.0, 'speak', 0], [3.15, 'interrupt'], [3.6, 'tap']]],
  ['8_relief_old', 10, [[0, 'mask', 0.15], [0.5, 'emo', { fear: 0.64 }], [3.0, 'emo', { fear: 0.2, calm: 0.3 }], [4.0, 'speak', 1]]],
  ['9_relief_v2', 11, [[0, 'mask', 0.15], [0.5, 'emo', { fear: 0.64 }], [3.0, 'relief_v2'], [5.8, 'speak', 1]]],
  ['10_relief_live', 11, [[0, 'mask', 0.15], [0.5, 'emo', { fear: 0.64 }], [3.0, 'emo', { fear: 0.08 }], [5.8, 'speak', 1]]],
  ['7_phone_contained', 9, [[0, 'mask', 0.6], [0.6, 'emo', { anger: 0.45 }], [2.4, 'speak', 0], [7.2, 'done']], { phone: true }],
];
const want = process.argv.slice(2);
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
for (const [name, secs, events, opt = {}] of TAKES) {
  if (want.length && !want.some((w) => name.includes(w))) continue;
  const phone = !!opt.phone;
  const page = await browser.newPage(phone ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true } : { viewport: { width: 1280, height: 900 } });
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto('http://127.0.0.1:5190/?made=1&rehearsal=1&attract=99999');
  await page.waitForFunction(() => window.__btm?.stage?.face?.headBone && window.__btm.state.index, null, { timeout: 60000 });
  await page.addStyleTag({ content: phone ? '.push-badge,.mini{display:none!important}' : '.inside,.subs,.push-badge,.mini,.controls,.foot,header{display:none!important}' });
  await page.waitForTimeout(1200);
  const clip = await page.evaluate((phone) => {
    const r = document.querySelector('#face-canvas').getBoundingClientRect();
    if (phone) return { x: 0, y: 0, width: 390, height: 844 };
    const w = Math.round(r.width * 0.36 / 2) * 2, h = Math.round(r.height * 0.86 / 2) * 2;
    return { x: Math.round(r.x + (r.width - w) / 2), y: Math.round(r.y), width: w, height: h };
  }, phone);
  // a schedule of word events for the line, with the playback's own timing (main.js play)
  const plan = [];
  for (const [t0, act, arg] of events) {
    if (act !== 'speak') { plan.push([t0, act, arg]); continue; }
    plan.push([t0, 'write', true]);
    let t = t0;
    for (const w of LINES[arg || 0]) {
      let wait = 62 + Math.min(80, w.length * 6);
      plan.push([t, 'word', { w, dur: wait / 1000, stress: STRESS.has(w) }]);
      if (/[.!?]$/.test(w)) wait += 360;
      t += wait / 1000;
    }
    plan.push([t + 0.2, 'write', false]);
  }
  plan.sort((a, b) => a[0] - b[0]);
  await page.evaluate(() => {
    const { stage: s, state } = window.__btm, f = s.face;
    s.paused = true; state.runId++;
    s.setPushColor('#9fb4ff', 0); s.setGlow(0); f.setIrisColor('#8fe6ff'); f.uniforms.uGlow.value = 0;
    f.setEmotion({}); f.setGaze('camera'); f.settle(3); f.clearReactions();
    window.__speech = window.__btm.speech; window.__speech.begin();
    window.__stopped = false;
  });
  // Playwright's own ffmpeg build: JPEG frames on stdin (it has no PNG decoder and no '-' alias for stdin)
  const ff = spawn(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', 'pipe:0', '-c:v', 'libvpx', '-b:v', '5M', '-crf', '6', '-pix_fmt', 'yuv420p', `${OUT}/${name}.webm`]);
  let ffDead = false;
  ff.on('exit', () => { ffDead = true; });
  ff.stderr.on('data', (d) => process.stderr.write(d));
  const frames = Math.round(secs * FPS);
  let k = 0;
  for (let i = 0; i < frames; i++) {
    const now = i / FPS;
    const due = [];
    while (k < plan.length && plan[k][0] <= now) due.push(plan[k++]);
    await page.evaluate(([due, dt]) => {
      const { stage: s } = window.__btm, f = s.face, sp = window.__speech;
      for (const [, act, arg] of due) {
        if (act === 'mask') { f.setMask(arg); f.maskS.x = arg; }
        else if (act === 'emo') f.setEmotion(arg);
        else if (act === 'think') f.react('think');
        else if (act === 'done') f.react('done');
        else if (act === 'write') { if (!window.__stopped) f.setActivity({ writing: arg }); if (!arg) sp.end(); }
        else if (act === 'word') {
          if (window.__stopped) continue;
          sp.add(sp.spans.length, { t: arg.w, e: [] }, {}, null);
          f.say(arg.w, arg.dur);
          if (/\?$/.test(arg.w)) f.beat('question'); else if (/!$/.test(arg.w)) f.beat('exclaim');
          else if (/\.$/.test(arg.w)) f.beat('period'); else if (/,$/.test(arg.w)) f.beat('comma');
          else if (arg.stress) f.beat('emphasis');
        } else if (act === 'interrupt') { window.__stopped = true; f.clearSpeech(true); f.setActivity({ writing: false }); sp.end(); }
        else if (act === 'tap') { f.setPointer(0.6, -0.9, true); f.react('listen'); }
        else if (act === 'relief_v2') f.startReliefTake(() => f.setAttentionHold(0));
      }
      f.update(dt / 2); f.update(dt / 2);
      f.uniforms.uGlow.value = 0; s.render();
    }, [due, 1 / FPS]);
    if (ffDead) throw new Error('ffmpeg exited');
    const jpg = await page.screenshot({ clip, type: 'jpeg', quality: 93 });
    if (!ff.stdin.write(jpg)) await new Promise((r) => { ff.stdin.once('drain', r); ff.once('exit', r); });
    if ([0.5, 1.2, 2.0, 3.0, 3.5, 4.0, 4.5, 5.5, 7.0, 8.5, 10.5].some((t) => Math.abs(now - t) < 1e-6 + 0.5 / FPS) && now < secs) {
      await page.screenshot({ path: `${OUT}/${name}_f${now.toFixed(1)}.png`, clip });
    }
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  console.log('take', name, frames, 'frames');
  await page.close();
}
await browser.close();
