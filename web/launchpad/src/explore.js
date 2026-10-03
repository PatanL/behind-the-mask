// Explore: the platform's own android live, then every coin's android -- awake ones live, asleep ones breathing.
import './style.css';
import { get, wsUrl, imgUrl } from './api.js';
import { $, h, header, footer, sol, pct, left, feelBar, STATUS, inUsd } from './ui.js';
import { FaceWall } from './faces.js';
import { faceFrom, Performer } from './feel.js';
import { ORDER, EMO } from '../../src/palette.js';

header('Explore'); footer();
const wall = new FaceWall();
let coins = [], sort = localStorage.getItem('lp-sort') || 'new', filter = 'all';
const cards = new Map();   // id -> {el, slot, lift, lastLine}
// opening a card: hand its look to the coin page, so its face appears in it straight away
document.addEventListener('click', (e) => {
  const a = e.target.closest?.('a.card'); if (!a) return;
  const c = cards.get(a.dataset.id); if (!c?.coin?.look) return;
  try { sessionStorage.setItem(`lp-look-${c.coin.id}`, JSON.stringify(c.coin.look)); } catch { /* */ }
  // and its face as it is right now, shown on the coin page until the live face is ready there
  try { const f = a.querySelector('canvas.face-frame'); if (f && f.width > 0) sessionStorage.setItem(`lp-shot-${c.coin.id}`, f.toDataURL('image/jpeg', 0.85)); } catch { /* */ }
  if (e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;   // (a new tab: an ordinary link)
  e.preventDefault();
  openCoin(a, c).catch(() => { location.href = a.href; });
}, true);

// ---- opening a coin without leaving the page: its page's markup and code come in here (fetched ahead of time), and
// the card's live face moves into its stage and keeps going until the coin page's own full-quality face fades in
// over it. The address is the coin's (a reload or a shared link is the ordinary coin page); back reloads explore.
let torn = false, exploreWs = null, opened = false;
const coinPage = new Promise((r) => setTimeout(r, 1500)).then(async () => {
  const html = await (await fetch(`${import.meta.env.BASE_URL}coin`)).text();
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const script = doc.querySelector('script[type="module"][src]')?.getAttribute('src');
  const css = [...doc.querySelectorAll('link[rel="stylesheet"]')].map((l) => l.getAttribute('href'));
  const link = (attrs) => document.head.append(Object.assign(document.createElement('link'), attrs));
  for (const l of doc.querySelectorAll('link[rel="modulepreload"]')) link({ rel: 'modulepreload', href: l.getAttribute('href'), crossOrigin: 'anonymous' });
  if (script) link({ rel: 'modulepreload', href: script, crossOrigin: 'anonymous' });
  for (const href of css) link({ rel: 'preload', as: 'style', href });
  return { body: [...doc.body.children].filter((n) => n.tagName !== 'SCRIPT'), script, css, title: doc.title };
}).catch(() => null);

async function openCoin(a, c) {
  const page = await Promise.race([coinPage, new Promise((r) => setTimeout(() => r(null), 1200))]);
  if (!page?.script || opened) { location.href = a.href; return; }
  opened = true;
  await Promise.all(page.css.map((href) => (document.querySelector(`link[rel="stylesheet"][href="${href}"]`) ? null
    : new Promise((res) => { const l = Object.assign(document.createElement('link'), { rel: 'stylesheet', href }); l.onload = l.onerror = res; document.head.append(l); }))));
  history.pushState({ soft: true }, '', a.href);
  const live = c.slot ? c.box : null;
  if (live) try { sessionStorage.removeItem(`lp-shot-${c.coin.id}`); } catch { /* */ }   // (the live face carries over instead)
  // explore stops: its sockets, its observer, every other face
  torn = true;
  try { exploreWs?.close(); sx.ws?.close(); } catch { /* */ }
  io.disconnect();
  for (const el of [...wall.slots.keys()]) if (el !== live) wall.detach(el);
  document.querySelector('main').replaceWith(...page.body.map((n) => document.importNode(n, true)));
  document.title = page.title;
  scrollTo(0, 0);
  const stage = document.querySelector('#stage');
  if (live && stage) wall.retarget(live, stage); else wall.stop();
  addEventListener('coin-face-shown', () => wall.stop(), { once: true });
  await import(/* @vite-ignore */ page.script);   // the coin page's own code, on this page
}
addEventListener('popstate', () => { if (opened) location.reload(); });

// ---- $STEER at the top: the home page's android, live (a light feed of its words and readouts)
const sx = { slot: wall.attach($('#sx-box'), { look: { skin: 'porcelain', eye: '#7fe7ff' }, seed: 11 }), perf: null, words: [], queue: [], timer: 0, plain: [], L: null };
sx.slot.face.ready.then(() => { sx.perf = new Performer(sx.slot.face); });
// the readout's label order (the live server's; the same as the ready-made answers' index)
fetch(`${import.meta.env.BASE_URL}performances/index.json`).then((r) => r.json()).then((j) => { sx.L = j.labels; }).catch(() => {});
const inOrder = (v) => ORDER.map((e) => { const k = sx.L ? sx.L.indexOf(e) : ORDER.indexOf(e); return k >= 0 ? v?.[k] ?? 0 : 0; });
function sxWord(w) {
  if (torn) return;
  // like the home page: its readout over the turn's unpushed baseline (the mean of b so far)
  if (w.b) sx.plain.push(inOrder(w.b));
  const n = sx.plain.length, mu = ORDER.map((_, k) => (n ? sx.plain.reduce((a, x) => a + x[k], 0) / n : 0));
  let shown = null;
  if (sx.perf && w.e) { sx.perf.mu = mu; shown = sx.perf.word({ t: w.t, e: inOrder(w.e) }); }
  sx.words.push(w.t); if (sx.words.length > 60) sx.words.splice(0, sx.words.length - 60);
  $('#sx-say').replaceChildren(h('span', {}, sx.words.join('').trim()));   // the newest words at the bottom, older ones fade out above
  if (shown) {
    feelBar($('#sx-feel'), shown);
    const top = ORDER.reduce((a, k) => (shown[k] > (shown[a] || 0) ? k : a), ORDER[0]);
    $('#sx-feel-l').textContent = shown[top] > 0.15 ? `feeling ${EMO[top].label.toLowerCase()}` : 'even';
  }
}
function sxPlay() {   // the light feed arrives in batches: say them at speaking pace
  if (sx.timer) return;
  const step = () => { const w = sx.queue.shift(); if (!w) { sx.timer = 0; return; } sxWord(w); sx.timer = setTimeout(step, Math.min(260, 2400 / Math.max(6, sx.queue.length))); };
  step();
}
function sxLive() {
  let ws;
  try { ws = new WebSocket(`${wsUrl('live/ws')}?cid=x-${Math.random().toString(36).slice(2, 10)}&lite=1`); } catch { return; }
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'live_begin') { sx.words = []; sx.queue = []; sx.plain = []; $('#sx-say').textContent = ''; $('#sx-status').textContent = 'Live'; }
    else if (m.type === 'ws' || m.type === 'wc') { sx.queue.push(...m.w); sxPlay(); }
    else if (m.type === 'crowd') $('#sx-meta').textContent = `${m.viewers} watching${m.power > 0.05 ? ` · being pushed ${Math.round(m.power * 100)}%` : ''}`;
    else if (m.type === 'full') $('#sx-status').textContent = 'Busy';
  };
  sx.ws = ws;
  ws.onclose = () => { if (!torn) setTimeout(sxLive, 5000); };
}
sxLive();

const SORTS = [['new', 'Newest'], ['mcap', 'Market cap'], ['emotional', 'Most emotional'], ['volume', 'Volume']];
$('#sorts').append(...SORTS.map(([k, t]) => h('button', { class: `chip${k === sort ? ' on' : ''}`, 'data-k': k }, t)));
$('#sorts').onclick = (e) => { const k = e.target.dataset.k; if (!k) return; sort = k; try { localStorage.setItem('lp-sort', k); } catch {} [...$('#sorts').children].forEach((b) => b.classList.toggle('on', b.dataset.k === k)); render(); };
$('#filters').onclick = (e) => { const k = e.target.dataset.k; if (!k) return; filter = k; render(); };

// faces attach while their card is on screen (a phone can't hold dozens of live faces)
const io = new IntersectionObserver((ents) => {
  for (const en of ents) {
    const id = en.target.dataset.id, c = cards.get(id);
    if (!c) continue;
    if (en.isIntersecting && !c.slot && (c.coin.status !== 'asleep' || !c.coin.img)) attachFace(c);
    else if (!en.isIntersecting && c.slot) { wall.detach(c.box); c.slot = null; c.box.querySelector('.portrait')?.removeAttribute('hidden'); }
  }
}, { rootMargin: '200px' });

function attachFace(c) {
  c.box.querySelector('.portrait')?.setAttribute('hidden', '');   // the live face replaces its still portrait
  c.slot = wall.attach(c.box, { look: c.coin.look, seed: [...c.coin.id].reduce((a, ch) => a + ch.charCodeAt(0), 0) });
  c.slot.asleep = c.coin.status === 'asleep';
  c.slot.face.ready.then(() => c.slot && c.slot.face.setSleep(c.coin.status === 'asleep' ? 1 : 0));
}

function sorted() {
  const key = { new: (c) => -c.created, mcap: (c) => -c.mcap_sol, emotional: (c) => -c.intensity, volume: (c) => -c.volume_sol }[sort];
  return coins.filter((c) => !c.featured && (filter === 'all' || c.status !== 'asleep')).sort((a, b) => key(a) - key(b));
}

function card(c) {
  const box = h('div', { class: 'face-box' });
  if (c.img) box.append(h('img', { class: 'portrait', src: imgUrl(c.img), alt: '' }));
  const el = h('a', { class: 'card', href: `coin?id=${encodeURIComponent(c.id)}`, 'data-id': c.id },
    box, h('span', { class: 'status' }),
    h('div', { class: 'body' },
      h('div', { class: 'who' }, h('b', {}, c.name), h('span', { class: 'tk mono' }, `$${c.ticker}`)),
      h('div', { class: 'line' }), h('div', { class: 'feelbar' }),
      h('div', { class: 'nums mono' }, h('div', {}, 'Mcap', h('b', { class: 'mcap' })), h('div', {}, '5 min', h('b', { class: 'chg' })), h('div', {}, 'Awake for', h('b', { class: 'tl' })))));
  return { el, box, coin: c, slot: null, lift: ORDER.map(() => 0), lastLine: '', said: 0 };
}

// the end of what it's saying, starting at a word (not mid-word)
function tail(line, n = 130) {
  const t = (line || '…').trim();
  if (t.length <= n) return t;
  const cut = t.slice(-n), sp = cut.indexOf(' ');
  return `…${sp > 0 && sp < 25 ? cut.slice(sp + 1) : cut}`;
}
function update(c, d) {
  const prev = c.coin.status;
  c.coin = d;
  const st = c.el.querySelector('.status');
  st.textContent = STATUS[d.status] || d.status;
  st.className = `status ${d.status === 'asleep' ? '' : d.status === 'losing it' ? 'losing' : 'awake'}`;
  c.el.classList.toggle('asleep', d.status === 'asleep');
  c.el.querySelector('.line').textContent = d.status === 'asleep' ? 'Asleep. A trade will wake it.' : tail(d.line);
  const real = d.market !== 'sim';
  c.el.querySelector('.mcap').textContent = real ? inUsd(d.mcap_sol, solUsd) : 'not yet';
  const ch = c.el.querySelector('.chg'); ch.textContent = real ? pct(d.change_5m) : '—'; ch.className = `chg ${d.change_5m >= 0 ? 'up' : 'down'}`;
  c.el.querySelector('.tl').textContent = left(d.time_left);
  // feelings: the readout above its own baseline, through the exhibit's decoder
  if (d.e && d.mu) ORDER.forEach((_, k) => { c.lift[k] += (d.e[k] - d.mu[k] - c.lift[k]) * 0.35; });
  const shown = faceFrom(c.lift);
  feelBar(c.el.querySelector('.feelbar'), shown);
  if (c.slot?.ready) {
    const f = c.slot.face;
    f.setEmotion(d.status === 'asleep' ? {} : shown);
    if (prev === 'asleep' && d.status !== 'asleep') { f.startle(); c.slot.asleep = false; }
    else if (d.status === 'asleep') { f.setSleep(1); c.slot.asleep = true; }
    // mouth the new words since the last update, spread over the second
    if (d.status !== 'asleep' && d.line !== c.lastLine) {
      const fresh = d.line.startsWith(c.lastLine.slice(-60)) ? d.line.slice(d.line.indexOf(c.lastLine.slice(-60)) + c.lastLine.slice(-60).length) : d.line.slice(-24);
      const words = fresh.match(/\S+\s*/g) || [];
      words.slice(-8).forEach((w, i) => setTimeout(() => c.slot && f.say(w, 0.12), i * (900 / Math.max(1, words.length))));
    }
  } else if (!c.slot && d.status !== 'asleep' && prev === 'asleep') {
    const r = c.el.getBoundingClientRect(); if (r.bottom > 0 && r.top < innerHeight) attachFace(c);
  }
  c.lastLine = d.line || '';
}

function render() {
  [...$('#filters').children].length || $('#filters').append(h('button', { class: 'chip', 'data-k': 'all' }, 'All'), h('button', { class: 'chip', 'data-k': 'awake' }, 'Awake'));
  [...$('#filters').children].forEach((b) => b.classList.toggle('on', b.dataset.k === filter));
  const list = sorted(), grid = $('#grid');
  const want = new Set(list.map((c) => c.id));
  for (const [id, c] of cards) if (!want.has(id)) { io.unobserve(c.el); if (c.slot) wall.detach(c.box); c.el.remove(); cards.delete(id); }
  list.forEach((d, i) => {
    let c = cards.get(d.id);
    if (!c) { c = card(d); cards.set(d.id, c); io.observe(c.el); }
    if (grid.children[i] !== c.el) grid.insertBefore(c.el, grid.children[i] || null);
    update(c, d);
  });
  if (!list.length) grid.replaceChildren(h('p', { style: 'color:var(--ink3)' }, filter === 'awake' ? 'Nobody is awake right now.' : 'No coins yet. Launch the first one.'));
}

// ---- data
let solUsd = null;   // SOL's dollar price, from the server
function apply(m) {
  coins = m.coins;
  const s = m.stats;
  if (s.sol_usd) solUsd = s.sol_usd;
  $('#stats').replaceChildren(...[[s.coins, 'coins'], [s.awake, 'awake'], [s.compute_sol.toFixed(3), 'SOL compute'], [inUsd(s.volume_sol, solUsd), 'traded']]
    .map(([v, k]) => h('div', {}, h('b', {}, String(v)), h('span', {}, k))));
  const lately = coins.filter((c) => c.status !== 'asleep').slice(0, 8);
  $('#lately').replaceChildren(h('span', { class: 'lbl' }, 'awake now'), ...lately.map((c) => h('span', {}, h('b', {}, `$${c.ticker}`), ` ${pct(c.change_5m)} · ${(STATUS[c.status] || c.status).toLowerCase()}`)));
  render();
}
// the live socket sends the full list first, then changes: no separate request
let listed = false;
setTimeout(() => { if (!listed) $('#stats').replaceChildren(h('div', {}, h('b', {}, 'offline'), h('span', {}, 'the androids are resting'))); }, 6000);
function connect() {
  const ws = new WebSocket(wsUrl('api/ws/explore'));
  exploreWs = ws;
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'coins') { listed = true; apply(m); }
    else if (m.type === 'delta') {   // only what changed since the last update (absolute values)
      const by = new Map(coins.map((c) => [c.id, c]));
      for (const d of m.c) { const c = by.get(d.id); if (c) by.set(d.id, { ...c, ...d }); }
      for (const c of m.add) by.set(c.id, c);
      for (const id of m.gone) by.delete(id);
      apply({ coins: [...by.values()], stats: m.stats });
    }
  };
  ws.onclose = () => { if (!torn) setTimeout(connect, 3000); };
}
connect();
