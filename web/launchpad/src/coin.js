// One coin's android, live: its face and words, its chart, steering it, asking it things.
import { get, post, wsUrl, visitorId, api } from './api.js';
import { $, h, header, footer, sol, pct, left, feelBar, STATUS, inUsd } from './ui.js';
import { Performer } from './feel.js';
import { Stage } from '../../src/stage.js';
import { Speech } from '../../src/speech.js';
import { FeelWheel } from '../../src/wheel.js';
import { EMO, ORDER } from '../../src/palette.js';
import * as wallet from './wallet.js';

header(new URLSearchParams(location.search).get('id') ? 'Explore' : 'Live'); footer();
const Q = new URLSearchParams(location.search);
let id = Q.get('id') || Q.get('mint');   // a coin's page by id, or by its mint (its pump.fun website)
const DEMO = Q.has('demo');   // simulated markets are shown only in demo mode
const cid = visitorId();
const stage = new Stage($('#face'));
let perf = null, info = null, ws = null, series = [], queue = [], text = [];
stage.ready.then(() => { perf = new Performer(stage.face); if (info?.mu) perf.reset(info.mu); if (info) look(info); });

function look(d) {
  if (!stage.face.uniforms) return;
  stage.face.setLook(d.look || {});
  stage.face.setSleep(d.status === 'asleep' ? 1 : 0);
}

// ---- the words, revealed at a speaking pace, as the exhibit's subtitles (tinted by the feeling they carry)
const speech = new Speech($('#speech'));
const wheel = new FeelWheel($('#wheel'));   // the exhibit's "how it feels inside"
speech.begin();
let wi = 0, asked = null;
function label() {
  const el = $('#speech-label');
  if (!info) return;
  el.textContent = info.status === 'asleep' ? 'Asleep' : info.status === 'listening' ? 'Listening · it speaks again in a moment'
    : asked ? `Answering a visitor: “${asked}”` : `Live · ${info.name}`;
}
function showWord(w) {
  if (perf) perf.word(w);
  const shown = perf?.shown || {};
  const top = ORDER.reduce((a, e) => ((shown[e] || 0) > (shown[a] || 0) ? e : a), ORDER[0]);
  speech.add(wi++, w, shown, (shown[top] || 0) > 0.15 ? top : null);
  wheel.setState(shown); wheel.setFocus((shown[top] || 0) > 0.15 ? top : null);
  if (perf) {
    feelBar($('#feelbar'), shown);
    $('#feel-label').textContent = shown[top] > 0.12 ? `Mostly ${EMO[top].label.toLowerCase()}` : 'Calm, as itself';
  }
}
setInterval(() => { const w = queue.shift(); if (w) showWord(w); }, 155);

// ---- the chart
function chart() {
  const c = $('#chart'), r = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  c.width = W * r; c.height = H * r;
  const g = c.getContext('2d'); g.scale(r, r); g.clearRect(0, 0, W, H);
  if (series.length < 2) return;
  const ps = series.map((s) => s[1]), lo = Math.min(...ps), hi = Math.max(...ps), sp = hi - lo || hi * 0.01 || 1;
  const t0 = series[0][0], t1 = series[series.length - 1][0] || t0 + 1;
  const X = (t) => ((t - t0) / Math.max(1, t1 - t0)) * (W - 4) + 2, Y = (p) => H - 6 - ((p - lo) / sp) * (H - 14);
  const up = ps[ps.length - 1] >= ps[0];
  const grad = g.createLinearGradient(0, 0, 0, H); grad.addColorStop(0, up ? 'rgba(94,230,168,0.25)' : 'rgba(255,107,107,0.25)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.beginPath(); series.forEach(([t, p], i) => (i ? g.lineTo(X(t), Y(p)) : g.moveTo(X(t), Y(p))));
  g.strokeStyle = up ? '#5ee6a8' : '#ff6b6b'; g.lineWidth = 1.6; g.stroke();
  g.lineTo(X(t1), H); g.lineTo(X(t0), H); g.closePath(); g.fillStyle = grad; g.fill();
}

// ---- steering
const BTNS = [...ORDER.map((e) => [e, EMO[e].label, EMO[e].color]), ['unmask', 'Off script', EMO.unmask.color]];
const pending = {};
function buildTaps(d) {
  const host = $('#taps');
  if (host.childElementCount) return;
  const all = d.concept ? [...BTNS, ['concept', `More ${d.concept}`, '#9fb4ff']] : BTNS;
  for (const [b, label, color] of all) {
    const el = h('button', { class: `tapb${b === 'unmask' || b === 'concept' ? ' wide' : ''}`, 'data-b': b, style: `--c:${color}` }, label);
    el.onpointerdown = (e) => { e.preventDefault(); pending[b] = (pending[b] || 0) + 1; el.style.setProperty('--heat', '1'); setTimeout(() => el.style.setProperty('--heat', '0'), 220); };
    host.append(el);
  }
  $('#mix').append(...all.map(([b, , color]) => h('i', { 'data-b': b, style: `background:${color};width:0` })));
}
setInterval(() => {
  if (!ws || ws.readyState !== 1 || !Object.keys(pending).length) return;
  ws.send(JSON.stringify({ type: 'taps', c: { ...pending } }));
  for (const k of Object.keys(pending)) delete pending[k];
}, 250);
const MODE = { everyone: 'Everyone steers it', holders: 'Holders steer it', weighted: 'Holders steer it, by holdings' };

// ---- questions (one line per coin)
// ---- holders: a signed message proves the wallet; the server reads its balance of this coin
let nonceMsg = null, walletAddr = null;
function walletNote(d) {
  $('#wallet').hidden = d.steer_mode === 'everyone';   // only coins steered by their holders need a wallet
  if (walletAddr) return;
  $('#wallet-s').textContent = d.steer_mode === 'everyone' ? 'Everyone’s taps count. Holders can connect to show they hold.' : `Only holders’ taps count${d.steer_mode === 'weighted' ? ', weighted by how much they hold' : ''}. Connect a wallet that holds $${d.ticker}.`;
}
$('#wallet-b').onclick = async () => {
  try {
    const w = await wallet.connect(); const addr = wallet.address;
    const msg = new TextEncoder().encode(nonceMsg.replace('{address}', addr));
    const signature = await w.signMessage(msg);
    ws.send(JSON.stringify({ type: 'wallet', address: addr, signature: btoa(String.fromCharCode(...signature)) }));
  } catch (e) { $('#wallet-s').textContent = e.message || String(e); }
};
let mine = null, askGap = 30, nextAskAt = 0, askNote = null, lastState = null, askTimer = 0;
$('#ask').onsubmit = (e) => { e.preventDefault(); const q = $('#ask-in').value.trim(); if (q && ws?.readyState === 1) ws.send(JSON.stringify({ type: 'ask', q })); };
function askView() {
  const L = lastState || {}, qids = L.qids || [];
  const pos = mine ? qids.indexOf(mine.id) : -1, answering = !!mine && L.asking_id === mine.id;
  if (mine && (pos >= 0 || answering)) mine.seen = true; else if (mine?.seen) mine = null;
  const card = !!mine;
  $('#ask').hidden = card; $('#ask-card').hidden = !card; $('#ask-card').classList.toggle('now', answering);
  if (card) {
    $('#ask-pill').textContent = answering ? 'Answering now' : pos === 0 ? 'Next' : pos > 0 ? `#${pos + 1} in line` : 'Sent';
    $('#ask-q').textContent = `“${mine.q}”`;
    $('#ask-why').textContent = answering ? 'It’s answering you. Ask another when it’s done.' : 'One at a time. You can ask another once it’s answered.';
  }
  const wait = Math.ceil((nextAskAt - Date.now()) / 1000), cooling = !card && wait > 0;
  $('#ask-in').disabled = $('#ask-b').disabled = cooling;
  $('#ask-b').textContent = cooling ? `${wait}s` : 'Ask';
  const line = $('#ask-line'), note = askNote && Date.now() < askNote.until;
  line.classList.toggle('warn', !!note);
  line.textContent = note ? askNote.text : cooling ? `You can ask again in ${wait} s.` : card ? '' : L.qn ? `Next: “${L.qs?.[0] || ''}”${L.qn > 1 ? ` · ${L.qn} in line` : ''}` : 'Questions are answered in order, after its current thought.';
  clearTimeout(askTimer); if (cooling || note) askTimer = setTimeout(askView, 1000);
}

// ---- state
function apply(d) {
  const first = !info;
  info = d;
  document.title = `${d.name} ($${d.ticker}) · Steer AI`;
  $('#name').textContent = d.name; $('#ticker').textContent = `$${d.ticker}`;
  const st = $('#status'); st.textContent = STATUS[d.status] || d.status; st.className = `status ${d.status === 'asleep' ? '' : d.status === 'losing it' ? 'losing' : 'awake'}`;
  $('#sub').textContent = `${d.viewers || 1} watching${d.featured ? ' · the platform’s own android' : ''}`;
  const real = d.market !== 'sim' || DEMO;
  for (const el of document.querySelectorAll('.head .price, .head .chart, .head .kv')) el.hidden = !real;
  $('#nocoin').hidden = real;
  $('#nocoin').textContent = `$${d.ticker} hasn't launched yet.`;
  $('#price').textContent = d.price ? `${d.price.toExponential(3)} SOL` : '—';
  const ch = $('#chg'); ch.textContent = `${pct(d.change_5m)} 5m`; ch.className = `mono ${d.change_5m >= 0 ? 'up' : 'down'}`;
  $('#mcap').textContent = inUsd(d.mcap_sol, d.sol_usd); $('#vol').textContent = inUsd(d.volume_sol, d.sol_usd);
  $('#curve').textContent = `${Math.round((d.curve || 0) * 100)}%`; $('#left').textContent = left(d.time_left);
  $('#demo').hidden = d.market !== 'sim' || !DEMO;
  $('#sleep-note').hidden = d.status !== 'asleep';
  label();
  $('#mode').textContent = MODE[d.steer_mode] || '';
  walletNote(d);
  buildTaps(d);
  const mix = d.mix || {};
  const tot = Object.values(mix).reduce((a, b) => a + b, 0) || 1;
  [...$('#mix').children].forEach((i) => { i.style.width = `${((mix[i.dataset.b] || 0) / Math.max(1, tot)) * 100}%`; });
  $('#why').textContent = 'Its temperament and everyone’s taps set its mood.';
  if (first) {
    $('#persona').textContent = d.persona;
    const T = d.temperament_mix || {}, tt = Object.entries(T).filter(([, v]) => v > 0).map(([e, v]) => `${EMO[e]?.label || e} ${Math.round(v * 100)}%`).join(', ') || 'even';
    const c = d.concept_full;
    $('#char').replaceChildren(
      h('dt', {}, 'Hidden obsession'), h('dd', {}, c ? `${c.name} (${Math.round(c.strength * 100)}%)` : 'nothing hidden'),
      h('dt', {}, 'Temperament'), h('dd', {}, d.temperament ? `${d.temperament} · ${tt}` : tt),
      h('dt', {}, 'Look'), h('dd', {}, `${d.look.skin}, ${d.look.marks === 'none' ? 'unmarked' : d.look.marks}`, h('span', { style: `display:inline-block;width:10px;height:10px;border-radius:50%;background:${d.look.eye};margin-left:8px;vertical-align:middle` })));
    if (perf && d.mu) perf.reset(d.mu);
    look(d);
  }
  if (perf && d.mu && !perf.muSet) { perf.mu = d.mu; perf.muSet = true; }
  lastState = d; askView();
}

function connect() {
  ws = new WebSocket(wsUrl(`api/ws/coin/${encodeURIComponent(id)}`) + `?cid=${cid}`);
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'state') { const was = info?.status; apply(m); if (was && was !== m.status) onStatus(was, m.status); }
    else if (m.type === 'series') { series = m.series; chart(); }
    else if (m.type === 'nonce') nonceMsg = m.message;
    else if (m.type === 'wallet_ok') { walletAddr = m.address; $('#wallet-b').hidden = true; $('#wallet-s').textContent = `${m.address.slice(0, 4)}…${m.address.slice(-4)} · holds ${m.holds ? m.holds.toLocaleString() : 'none'} · your taps count ×${m.weight.toFixed(1)}`; }
    else if (m.type === 'wallet_err') $('#wallet-s').textContent = m.why;
    else if (m.type === 'trade') { series.push([m.t, m.price]); if (series.length > 400) series.shift(); chart(); }
    else if (m.type === 'begin') { asked = m.asked || null; label(); speech.continueLine(); }
    else if (m.type === 'w') queue.push(...m.w);
    else if (m.type === 'wake') { stage.face.setSleep(0); if (m.startle) stage.face.startle(); }
    else if (m.type === 'sleep') { stage.face.setSleep(1); stage.face.setEmotion({}); wheel.reset(); }
    else if (m.type === 'rest') { if (info) { info.status = 'listening'; label(); } }   // between turns to speak: awake, quiet
    else if (m.type === 'ask_ok' || m.type === 'ask_mine') { mine = { id: m.id, q: m.q }; $('#ask-in').value = ''; askNote = null; nextAskAt = Date.now() + 1000 * (m.type === 'ask_ok' ? m.gap || askGap : m.wait || 0); askView(); }
    else if (m.type === 'ask_err') { if (m.why === 'wait') nextAskAt = Date.now() + 1000 * (m.wait || 0); else askNote = { text: m.why, until: Date.now() + 6000 }; askView(); }
  };
  ws.onclose = () => setTimeout(connect, 2500);
}
function onStatus(was, now) {
  if (was === 'asleep' && now !== 'asleep') { stage.face.setSleep(0); stage.face.startle(); }
  if (now === 'asleep') { stage.face.setSleep(1); stage.face.setEmotion({}); }
}
for (const b of document.querySelectorAll('#demo .btn')) b.onclick = () => post(`api/coins/${encodeURIComponent(id)}/trade`, { side: b.dataset.side, sol: Number(b.dataset.sol), cid }).catch((e) => alert(e.message));
addEventListener('resize', chart);

if (!id) location.replace('./');   // no coin named: the home page's android
(id ? Promise.resolve(id) : new Promise(() => {}))
  .then(() => get(`api/coins/${encodeURIComponent(id)}`)).then((d) => {
  id = d.id || id;   // opened by its mint (the pump.fun website): the live socket and trades use its id
  apply(d); series = d.series || []; chart();
  connect();
}).catch(() => { $('#name').textContent = 'No such coin'; });
