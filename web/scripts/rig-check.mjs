// Rig check: each anger-relevant channel alone, then combined, on the glossy porcelain and on a diagnostic matte
// material (to tell weak geometry from shine washing it out). No glow, neutral iris, fixed camera, no UI.
//   node scripts/rig-check.mjs   -> ../runs/rig/*.png + sheet
import { chromium } from 'playwright';
const POSES = [
  ['neutral', {}],
  ['browDown 1', { browDownLeft: 1, browDownRight: 1 }],
  ['browDown + innerUp .35', { browDownLeft: 1, browDownRight: 1, browInnerUpLeft: 0.35, browInnerUpRight: 0.35 }],
  ['eyeSquint 1', { eyeSquintLeft: 1, eyeSquintRight: 1 }],
  ['eyeWide .5', { eyeWideLeft: 0.5, eyeWideRight: 0.5 }],
  ['mouthPress 1', { mouthPressLeft: 1, mouthPressRight: 1 }],
  ['press .6 close .3 roll .3', { mouthPressLeft: 0.6, mouthPressRight: 0.6, mouthClose: 0.3, mouthRollLower: 0.3, mouthRollUpper: 0.2 }],
  ['noseSneer .6', { noseSneerLeft: 0.6, noseSneerRight: 0.6 }],
  ['brows up (AU1+2) .8', { browInnerUpLeft: 0.8, browInnerUpRight: 0.8, browOuterUpLeft: 0.6, browOuterUpRight: 0.6 }],
  ['smile + cheeks .8', { mouthSmileLeft: 0.8, mouthSmileRight: 0.8, cheekSquintLeft: 0.7, cheekSquintRight: 0.7, eyeSquintLeft: 0.3, eyeSquintRight: 0.3 }],
  ['anger 0.8 (as acted)', null],
];
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
await page.goto('http://127.0.0.1:5190/?made=1&attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
await page.addStyleTag({ content: '.inside,.subs,.push-badge,.mini,.controls,.foot,header{display:none!important}' });
await page.waitForTimeout(1500);
const box = await page.evaluate(() => { const r = document.querySelector('#face-canvas').getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
const clip = { x: box.x + box.w * 0.34, y: box.y + box.h * 0.12, width: box.w * 0.32, height: box.h * 0.6 };
await page.evaluate(() => {
  const s = window.__btm.stage, f = s.face;
  s.paused = true; window.__btm.state.runId++;
  f.options.idle = false; f.options.blinks = false; f.options.breathing = false; f.options.micro = false;
  s.setPushColor('#9fb4ff', 0); s.setGlow(0); f.uniforms.uGlow.value = 0; f.setIrisColor('#8fe6ff');
  f.setGaze('camera'); f.setMask(0); f.maskS.x = 0;
  window.__matte = (on) => { for (const m of f._porcelainMats) { m.userData.orig ||= { r: m.roughness, c: m.clearcoat }; m.roughness = on ? 0.7 : m.userData.orig.r; m.clearcoat = on ? 0 : m.userData.orig.c; m.needsUpdate = true; } };
});
let n = 0;
for (const mat of ['gloss', 'matte']) {
  await page.evaluate((m) => window.__matte(m === 'matte'), mat);
  for (const [name, pose] of POSES) {
    await page.evaluate(([pose]) => {
      const s = window.__btm.stage, f = s.face;
      f.debugPose = null; f.setEmotion(pose ? {} : { anger: 0.8 }); f.overlays = []; f.micro = null;
      f.settle(3); f.debugPose = pose && Object.keys(pose).length ? pose : (pose ? {} : null); f.settle(0.05);
      f.uniforms.uGlow.value = 0; s.render();
    }, [pose]);
    await page.screenshot({ path: `../runs/rig/${mat}_${String(n % POSES.length).padStart(2, '0')}.png`, clip });
    n++;
  }
}
await browser.close();
