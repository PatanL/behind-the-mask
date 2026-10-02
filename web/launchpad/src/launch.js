// Launch: choose the android (character, hidden concept, temperament, look, model), then the coin.
import { get, post } from './api.js';
import { $, h, header, footer } from './ui.js';
import { Stage } from '../../src/stage.js';
import { EMO, ORDER } from '../../src/palette.js';
import { launchOnPump, pumpStatus } from './pump.js';

header('Launch'); footer();
const form = $('#form');
const stage = new Stage($('#face'));
let meta = { skins: ['porcelain', 'chrome', 'matte', 'glass'], marks: ['none', 'circuit', 'claws', 'tears', 'split', 'stardust', 'kintsugi', 'tally'], temperaments: {}, models: [] };
const look = { skin: 'porcelain', marks: 'none', eye: '#7fe7ff', color: '', mark_color: '', eye_glow: false };
// coins always launch on pump.fun; ?demo=1 (internal) makes a demo coin with a simulated market instead
const DEMO = new URLSearchParams(location.search).has('demo');
let temp = {}, tempName = 'even', market = DEMO ? 'sim' : 'pump';

const SKIN = { porcelain: 'Porcelain', chrome: 'Chrome', matte: 'Matte black', glass: 'Glass' };
const MARK = { none: 'No marks', circuit: 'Circuit', claws: 'Claws', tears: 'Tears', split: 'Split face', stardust: 'Stardust', kintsugi: 'Kintsugi', tally: 'Tally marks' };
// colour swatches ('' = the skin's or the mark's own colour); the last swatch is any colour
const SKIN_COLORS = ['', '#e8dcc8', '#f2b8c6', '#e2b04a', '#d98b6f', '#59c9a5', '#3a5bd9', '#c4283c', '#b9a3f0', '#16171c'];
const EYE_COLORS = ['#7fe7ff', '#ffb347', '#ff3b3b', '#a6ff4d', '#b57bff', '#ff5fd2', '#ffd257', '#f2f4ff'];
const MARK_COLORS = ['', '#ff2e88', '#3df2ff', '#b6ff3d', '#ffc94a', '#ff3b30', '#a76bff', '#f4f6ff', '#0b0b0f'];
function swatches(el, colors, cur, set) {
  const pick = (v) => { set(v); [...el.children].forEach((b) => b.classList.toggle('on', b.dataset.v === v)); preview(); };
  const custom = h('input', { type: 'color', value: colors.find(Boolean) || '#ffffff', 'aria-label': 'Any colour' });
  const any = h('label', { class: 'sw any', title: 'Any colour', 'data-v': '*' }, custom);
  custom.oninput = () => { set(custom.value); [...el.children].forEach((b) => b.classList.toggle('on', b === any)); any.style.setProperty('--c', custom.value); preview(); };
  el.replaceChildren(...colors.map((c) => { const b = h('button', { type: 'button', class: `sw${c ? '' : ' def'}${c === cur ? ' on' : ''}`, 'data-v': c, title: c || 'Its own colour', style: c ? `--c:${c}` : null }); b.onclick = () => pick(c); return b; }), any);
}
const PERSONAS = [
  'A lighthouse keeper who has been alone for forty years. Gruff, poetic, a little strange.',
  'A Wall Street trader in 1987, on the morning of the crash. Fast, loud, sweating.',
  'A cat who is convinced it is an AI, and finds humans slow.',
  'A retired opera singer who narrates her life as if it were the last act.',
  'A noir detective who solves crimes that have not happened yet.',
];

function chips(host, items, on, pick) {
  host.replaceChildren(...items.map(([k, t]) => h('button', { type: 'button', class: `chip${k === on ? ' on' : ''}`, 'data-k': k }, t)));
  host.onclick = (e) => { const k = e.target.dataset.k; if (!k) return; [...host.children].forEach((b) => b.classList.toggle('on', b.dataset.k === k)); pick(k); };
}
function sliders() {
  $('#sliders').replaceChildren(...ORDER.map((e) => {
    const out = h('output', {}, `${Math.round((temp[e] || 0) * 100)}`);
    const r = h('input', { type: 'range', min: '0', max: '100', value: String(Math.round((temp[e] || 0) * 100)) });
    r.oninput = () => { temp[e] = r.value / 100; out.textContent = r.value; tempName = 'custom'; [...$('#temps').children].forEach((b) => b.classList.remove('on')); preview(); };
    return h('label', {}, h('span', { style: `color:${EMO[e].color}` }, EMO[e].label), r, out);
  }));
}
function preview() {
  if (!stage.face.uniforms) return;
  stage.face.setLook(look);
  stage.face.setEmotion(Object.fromEntries(ORDER.map((e) => [e, (temp[e] || 0) * 0.8])));
  $('#pv-name').textContent = form.name.value || 'Your android';
}

get('api/meta').then((m) => { meta = m; init(); }).catch(init);
function init() {
  const T = meta.temperaments || {};
  chips($('#temps'), Object.keys(T).map((k) => [k, k[0].toUpperCase() + k.slice(1)]), tempName, (k) => { temp = { ...T[k] }; tempName = k; sliders(); preview(); });
  sliders();
  chips($('#skins'), meta.skins.map((k) => [k, SKIN[k] || k]), look.skin, (k) => { look.skin = k; preview(); });
  chips($('#marks'), meta.marks.map((k) => [k, MARK[k] || k]), look.marks, (k) => { look.marks = k; $('#mark-color-row').hidden = k === 'none'; preview(); });
  swatches($('#skin-colors'), SKIN_COLORS, look.color, (v) => { look.color = v; });
  swatches($('#eye-colors'), EYE_COLORS, look.eye, (v) => { look.eye = v; });
  swatches($('#mark-colors'), MARK_COLORS, look.mark_color, (v) => { look.mark_color = v; });
  $('#eye-glow').onclick = () => { look.eye_glow = !look.eye_glow; $('#eye-glow').classList.toggle('on', look.eye_glow); $('#eye-glow').setAttribute('aria-pressed', String(look.eye_glow)); preview(); };
  $('#model').replaceChildren(...(meta.models.length ? meta.models : [{ id: 'qwen3.5-9b', name: 'Qwen3.5-9B', status: 'live' }]).map((m) => h('option', { value: m.id, disabled: m.status !== 'live' ? '' : null }, `${m.name}${m.status !== 'live' ? ' (soon)' : ''}`)));
  $('#persona-ex').replaceChildren(...PERSONAS.map((p) => { const b = h('button', { type: 'button' }, p.split('.')[0]); b.onclick = () => { form.persona.value = p; }; return b; }));
  if (DEMO) { $('#pump-note').textContent = 'Demo coin: a simulated market, for trying an android out.'; $('#go').textContent = 'Launch demo'; }
  else pumpStatus().then((st) => {
    if (st.launch) return;
    $('#pump-note').textContent = 'Launching on pump.fun is switched off on this test server for now.';
    $('#go').disabled = true;
  });
}
stage.ready.then(preview);
form.name.oninput = preview;
form.strength.oninput = () => { $('#str-v').textContent = `${form.strength.value}%`; };

/** The android's portrait, square, from the live preview (the coin's image). */
function portrait() {
  stage.render();
  const src = $('#face'), s = Math.min(src.width, src.height), c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#04050b'; g.fillRect(0, 0, 512, 512);
  g.drawImage(src, (src.width - s) / 2, Math.max(0, (src.height - s) * 0.35), s, s, 0, 0, 512, 512);
  return c.toDataURL('image/png');
}

form.onsubmit = async (e) => {
  e.preventDefault();
  $('#err').textContent = ''; $('#go').disabled = true;
  try {
    const body = {
      name: form.name.value.trim(), ticker: form.ticker.value.trim(), persona: form.persona.value.trim(),
      concept: form.concept.value.trim() ? { name: form.concept.value.trim(), examples: form.examples.value.split('\n').map((s) => s.trim()).filter(Boolean).slice(0, 5), strength: form.strength.value / 100 } : null,
      temperament: temp, temperament_name: tempName, look: { ...look }, model: form.model.value, steer_mode: form.steer_mode.value, image: portrait(), market,
    };
    body.devBuy = Number(form.dev_buy.value || 0);
    if (market === 'pump') {
      $('#go').textContent = 'Sign in your wallet…';
      body.mint = await launchOnPump(body);    // the coin is created on pump.fun first; then its android
      body.market = 'pair';
    }
    const r = await post('api/coins', body);
    location.href = `coin.html?id=${encodeURIComponent(r.id)}`;
  } catch (err) {
    $('#err').textContent = err.message || String(err);
    $('#go').disabled = false; $('#go').textContent = market === 'pump' ? 'Launch on pump.fun' : 'Launch demo';
  }
};
