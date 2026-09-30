import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true, args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 600, height: 600 } });
await page.goto('http://127.0.0.1:5190/?attract=999');
await page.waitForFunction(() => window.__btm?.stage?.face?.headBone, null, { timeout: 60000 });
await page.waitForTimeout(2500);
console.log(await page.evaluate(() => {
  const f = window.__btm.stage.face, U = f.uniforms;
  const mats = []; f.gltf.parser.json.materials.forEach((m) => mats.push(`${m.name} extras=${JSON.stringify(m.extras || {})}`));
  const img = U.uSeamMap.value?.image;
  return `hasSeams=${U.uHasSeams.value} map=${!!U.uSeamMap.value} img=${img ? img.width + 'x' + img.height : 'none'} textures=${(f.gltf.parser.json.textures || []).length} images=${(f.gltf.parser.json.images || []).length}\n` + mats.join('\n');
}));
await browser.close();
