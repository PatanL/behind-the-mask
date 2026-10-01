// Contact sheets of the face's acting: steady expressions, blends, masking, and timed sequences (leak + cover,
// chuckle, startle gasp, anger huff). The stage is paused and the face stepped deterministically.
//   node scripts/acting-qa.mjs [steady|seq|all]     -> ../runs/acting/*.png (+ sheet_*.png via python)
import { chromium } from 'playwright';
import fs from 'node:fs';
const which = process.argv[2] || 'all';
const OUT = '../runs/acting';
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto('http://127.0.0.1:5190/?made=1&rehearsal=1&attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
await page.waitForTimeout(2500);
const box = await page.evaluate(() => { const r = document.querySelector('#face-canvas').getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
const clip = { x: box.x + box.width * 0.08, y: box.y + box.height * 0.02, width: box.width * 0.84, height: box.height * 0.9 };

await page.evaluate(() => {
  const s = window.__btm.stage, f = s.face;
  s.paused = true;
  window.__reset = (seed = 1) => {
    f.setEmotion({}); f.setMask(0.6); f.maskS.x = 0.6; f.overlays = []; f.micro = null; f.setGaze('camera');
    f.options.idle = true; f.options.blinks = true; f.options.micro = true;
    f.settle(5);
    f.overlays = []; f.micro = null; f.blink.next = f.time + 99; f.blinkVal = 0;
  };
  window.__frame = () => { s.render(); };
  window.__probe = () => {
    const o = f.out, r = (x) => +(x || 0).toFixed(2);
    return { reg: r(f.reg), blend: f.blend, smile: r((o.mouthSmileLeft + o.mouthSmileRight) / 2), frown: r((o.mouthFrownLeft + o.mouthFrownRight) / 2),
      innerUp: r((o.browInnerUpLeft + o.browInnerUpRight) / 2), browDown: r((o.browDownLeft + o.browDownRight) / 2), cheek: r((o.cheekSquintLeft + o.cheekSquintRight) / 2),
      press: r((o.mouthPressLeft + o.mouthPressRight) / 2), stretch: r((o.mouthStretchLeft + o.mouthStretchRight) / 2), jaw: r(o.jawOpen), wide: r((o.eyeWideLeft + o.eyeWideRight) / 2), breath: r(f.breathVal) };
  };
});

async function shot(name, label) {
  await page.evaluate(() => window.__frame());
  await page.screenshot({ path: `${OUT}/${name}.png`, clip });
  const p = await page.evaluate(() => window.__probe());
  fs.appendFileSync(`${OUT}/probe.txt`, `${name.padEnd(28)} ${label.padEnd(26)} ${JSON.stringify(p)}\n`);
}

const STEADY = [
  ['s01_neutral', 'no push', '', 0.6],
  ['s02_joy', 'joy (lot)', 'joy: 0.65', 0.52],
  ['s03_sad_masked', 'sadness, masked', 'sadness: 0.55', 0.6],
  ['s04_sad_half', 'sadness lot, mask .38', 'sadness: 0.72', 0.38],
  ['s05_sad_raw', 'sadness, unmasked', 'sadness: 0.8', 0.12],
  ['s06_anger', 'anger, unmasked', 'anger: 0.8', 0.2],
  ['s07_fear', 'fear, unmasked', 'fear: 0.8', 0.2],
  ['s08_bittersweet', 'joy + sadness', 'joy: 0.5, sadness: 0.55', 0.15],
  ['s09_nervous', 'joy + fear', 'joy: 0.55, fear: 0.45', 0.2],
  ['s10_worry', 'sadness + fear', 'sadness: 0.65, fear: 0.45', 0.2],
  ['s11_smug', 'anger + joy (toomuch)', 'anger: 1, joy: 0.45, calm: 0.4', 0.0],
  ['s12_calm', 'calm', 'calm: 0.75', 0.55],
  ['s13_curious', 'curiosity', 'curiosity: 0.7', 0.55],
  ['s14_anger_masked', 'anger, masked', 'anger: 0.45', 0.55],
];
const SEQ = [
  // [name, label, setup, trigger, frame times]
  ['q1_leak', 'masked sadness: leak, cover', 'f.setEmotion({ sadness: 0.5 }); f.setMask(0.6); f.maskS.x = 0.6; f.settle(4); f.overlays = []; f.leakNext = f.time + 99; f.socialNext = f.time + 99; f.blink.next = f.time + 99;',
    'f.blink.next = f.time + 99; f._leak("sadness", 0.95, true); f.blink.next = f.time + 99;', [0.0, 0.07, 0.15, 0.3, 0.5, 0.8, 1.3]],
  ['q2_chuckle', 'joy: chuckle', 'f.setEmotion({ joy: 0.6 }); f.setMask(0.5); f.settle(4); f.overlays = [];', 'f._chuckle(0.6);', [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.3]],
  ['q3_gasp', 'fear onset: gasp', 'f.setMask(0.2); f.maskS.x = 0.2; f.settle(1);', 'f.setEmotion({ fear: 0.8 });', [0.0, 0.1, 0.25, 0.5, 0.9, 1.5, 3.0]],
  ['q4_huff', 'anger: huff', 'f.setEmotion({ anger: 0.7 }); f.setMask(0.2); f.maskS.x = 0.2; f.settle(5); f.overlays = [];', 'f.breath.huff = true; f.breath.phase = 0.7;', [0.0, 0.5, 1.0, 1.4, 1.6, 1.8, 2.4]],
  ['q5_relief', 'sadness drains: relief', 'f.setEmotion({ sadness: 0.8 }); f.setMask(0.2); f.maskS.x = 0.2; f.settle(6); f.overlays = [];', 'f.setEmotion({ sadness: 0.1 });', [0.2, 0.8, 1.4, 2.0, 2.8, 3.6, 4.6]],
  ['q6_social', 'polite closing smile', 'f.setEmotion({ sadness: 0.45 }); f.setMask(0.6); f.maskS.x = 0.6; f.settle(4); f.overlays = []; f.leakNext = f.time + 99; f.socialNext = f.time + 99;', 'f.react("done");', [0.0, 0.5, 0.75, 1.0, 1.6, 2.2, 3.0]],
];

fs.writeFileSync(`${OUT}/probe.txt`, '');
if (which === 'steady' || which === 'all') {
  for (const [name, label, emo, mask] of STEADY) {
    await page.evaluate(([emo, mask]) => {
      const f = window.__btm.stage.face; window.__reset();
      f.setMask(mask); f.maskS.x = mask;
      f.setEmotion(eval(`({${emo}})`));
      f.settle(3.5); f.overlays = []; f.micro = null; f.blink.next = f.time + 99; f.blinkVal = 0; f.settle(1.2);
    }, [emo, mask]);
    await shot(name, label);
  }
}
if (which === 'seq' || which === 'all') {
  for (const [name, label, setup, trig, times] of SEQ) {
    await page.evaluate(([setup, trig]) => { const f = window.__btm.stage.face; window.__reset(); eval(setup); eval(trig); }, [setup, trig]);
    let at = 0;
    for (const [i, tt] of times.entries()) {
      await page.evaluate((d) => { if (d > 0) window.__btm.stage.face.settle(d, 1 / 120); }, tt - at);
      at = tt;
      await shot(`${name}_${i}`, `${label} +${tt}s`);
    }
  }
}
await browser.close();
