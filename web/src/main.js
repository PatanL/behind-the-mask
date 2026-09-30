import '@fontsource/fraunces/400.css';
import '@fontsource/fraunces/500.css';
import '@fontsource/fraunces/400-italic.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/600.css';
import { EMO, ORDER, LEVELS } from './palette.js';
import { loadIndex, loadPerformance, performanceId, pushInfo } from './data.js';
import { Speech } from './speech.js';
import { Spine } from './spine.js';
import { Stage } from './stage.js';
import { HOW_STEPS } from './how.js';
import { FeelMap } from './feelmap.js';

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const state = { index: null, qid: null, emotion: 'joy', level: 'lot', playing: false, doc: null, runId: 0, lastTouch: Date.now(), attract: false };
const GAIN = Number(params.get('gain') || 1.4);

const stage = new Stage($('#face-canvas'));
const spine = new Spine($('#spine'), $('#meters'));
const feelmap = new FeelMap($('#feelmap'));
const speech = new Speech($('#speech'), { onPick: (i, el) => inspect(i, el) });

// ------------------------------------------------------------------ controls
function buildControls() {
  const qs = $('#questions');
  let lastGroup = null;
  for (const q of state.index.questions) {
    if (q.group !== lastGroup) {
      const g = document.createElement('div'); g.className = 'chip-group'; g.textContent = q.group; qs.appendChild(g); lastGroup = q.group;
    }
    const b = document.createElement('button');
    b.className = 'chip'; b.textContent = q.text; b.dataset.q = q.id; b.setAttribute('aria-pressed', 'false');
    b.onclick = () => { touch(); state.qid = q.id; refresh(); };
    qs.appendChild(b);
  }
  const fs = $('#feelings');
  const feel = (key, cls = '') => {
    const b = document.createElement('button');
    b.className = `orb ${cls}`; b.dataset.e = key; b.style.setProperty('--c', EMO[key].color);
    b.innerHTML = `<span class="ball"></span><span>${EMO[key].label}</span>`;
    b.onclick = () => { touch(); state.emotion = key; if (key === 'none' || key === 'swing') state.level = key; else if (!['little', 'lot', 'toomuch'].includes(state.level)) state.level = 'lot'; refresh(); };
    fs.appendChild(b);
  };
  ORDER.forEach((e) => feel(e));
  feel('none', 'none');
  feel('swing', 'swing');
  const am = $('#amount');
  for (const [k, label] of LEVELS) {
    const b = document.createElement('button');
    b.textContent = label; b.dataset.l = k; b.setAttribute('role', 'radio');
    b.onclick = () => { touch(); state.level = k; refresh(); };
    am.appendChild(b);
  }
  $('#go').onclick = () => { touch(); play(); };
  $('#again').onclick = () => { touch(); $('#deck').scrollIntoView({ behavior: 'smooth', block: 'start' }); };
  $('#base-push').onchange = () => renderBase();
}

function refresh() {
  document.querySelectorAll('.chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.q === state.qid)));
  document.querySelectorAll('.orb').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.e === state.emotion)));
  const fixed = state.emotion === 'none' || state.emotion === 'swing';
  document.querySelectorAll('.amount button').forEach((b) => { b.disabled = fixed; b.setAttribute('aria-checked', String(!fixed && b.dataset.l === state.level)); });
  const col = EMO[state.emotion]?.color || EMO.none.color;
  document.documentElement.style.setProperty('--push', state.emotion === 'none' ? '#9fb4ff' : state.emotion === 'swing' ? '#ffffff' : col);
  const go = $('#go');
  go.disabled = !state.qid || state.playing;
  go.classList.toggle('busy', state.playing);
  go.querySelector('span').textContent = state.playing ? 'Speaking' : 'Let it speak';
  $('#go-note').textContent = !state.qid ? 'Choose a question first.' : state.emotion === 'none' ? 'No push: the assistant exactly as it was trained.'
    : state.emotion === 'swing' ? 'Starts with one feeling, then we switch it mid-sentence.' : `We'll add the ${EMO[state.emotion].label.toLowerCase()} direction inside it while it writes.`;
}

// ------------------------------------------------------------------ playback
function pickSwing(q) {
  const keys = Object.keys(q.performances).filter((k) => k.endsWith('|swing'));
  return keys[Math.floor(Math.random() * keys.length)];
}

async function play(opts = {}) {
  if (!state.qid) return;
  const q = state.index.questions.find((x) => x.id === state.qid);
  let key = `${state.emotion}|${state.level}`;
  if (state.emotion === 'none') key = 'none|none';
  if (state.emotion === 'swing') key = pickSwing(q);
  const pid = q.performances[key];
  if (!pid) return;
  const run = ++state.runId;
  state.playing = true; refresh();
  $('#reveals').hidden = true;
  if (!state.attract) document.querySelector(window.innerWidth < 1100 ? '.stage' : '#app').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('#speech-hint').hidden = true;
  closeInspect();
  const doc = await loadPerformance(pid);
  if (run !== state.runId) return;
  state.doc = doc;
  const steered = doc.streams.steered.tokens;
  const plain = doc.streams.plain.tokens;
  // baseline: the model's own readout on the unpushed answer to the same question
  const mu = ORDER.map((_, k) => (plain.length ? plain.reduce((s, t) => s + t.e[k], 0) / plain.length : 0));
  const [e1, e2] = doc.emotion.includes('>') ? doc.emotion.split('>') : [doc.emotion, null];
  const focus0 = e1 === 'none' ? null : e1;
  let focus = focus0;
  setPush(focus, doc.level, e2 ? 'swing-start' : null);
  $('#speech-label').textContent = `“${doc.question}”`;
  $('#speech').classList.toggle('overdrive', doc.level === 'toomuch');
  $('#push-badge').classList.toggle('warn', doc.level === 'toomuch');
  speech.begin();
  feelmap.reset(); feelmap.setFocus(focus);
  stage.face.setActivity({ writing: true });
  const ema = Object.fromEntries(ORDER.map((e) => [e, 0]));
  for (let i = 0; i < steered.length; i++) {
    if (run !== state.runId) return;
    const tok = steered[i];
    let swingMark = false;
    if (e2 && doc.swing_at != null && i === doc.swing_at) { focus = e2; setPush(focus, 'lot', 'swing'); feelmap.setFocus(focus); swingMark = true; }
    const delta = {};
    ORDER.forEach((e, k) => { delta[e] = Math.max(0, Math.min(1, (tok.e[k] - mu[k]) / GAIN)); ema[e] += (delta[e] - ema[e]) * 0.28; });
    speech.add(i, tok, ema, focus, { swingMark });
    spine.pulse();
    spine.setMeters(ema);
    feelmap.setState(ema);
    stage.face.setEmotion(ema, { intensity: doc.level === 'toomuch' ? 1.25 : 1.0 });
    const t = tok.t;
    let wait = 62 + Math.min(80, t.length * 6);
    if (/[.!?]\s*$/.test(t)) wait += 360; else if (/[,;:]\s*$/.test(t)) wait += 170; else if (t.includes('\n')) wait += 260;
    await sleep(wait / (opts.fast ? 3 : 1));
  }
  speech.end();
  stage.face.setActivity({ writing: false });
  state.playing = false; refresh();
  // the feeling lingers, then relaxes
  // meters settle on the answer's average lift; the face relaxes but keeps a trace of the feeling
  const avg = Object.fromEntries(ORDER.map((e, k) => [e, Math.max(0, Math.min(1, steered.reduce((s, t) => s + (t.e[k] - mu[k]), 0) / Math.max(1, steered.length) / GAIN))]));
  setTimeout(() => { if (run === state.runId && !state.playing) { stage.face.setEmotion(Object.fromEntries(ORDER.map((e) => [e, avg[e] * 0.6]))); spine.setMeters(avg); feelmap.setState(avg); } }, 2600);
  showReveals(doc);
}

function setPush(emotion, level, swing) {
  const badge = $('#push-badge');
  spine.setPush(emotion);
  if (!emotion) {
    badge.hidden = false; badge.textContent = 'No push · as trained';
    stage.setPushColor('#9fb4ff', 0.1); stage.face.setIrisColor('#8fe6ff');
    return;
  }
  const lv = { little: 'a little', lot: 'a lot', toomuch: 'way too much', swing: '' }[level] ?? '';
  badge.hidden = false;
  badge.textContent = swing === 'swing' ? `Switched to ${EMO[emotion].label}!` : `Pushing ${EMO[emotion].label.toLowerCase()}${lv ? ' · ' + lv : ''}`;
  document.documentElement.style.setProperty('--push', EMO[emotion].color);
  stage.setPushColor(EMO[emotion].color, level === 'little' ? 0.35 : level === 'toomuch' ? 1 : 0.7);
  stage.face.setIrisColor(EMO[emotion].color);
}

// ------------------------------------------------------------------ reveals
function textOf(tokens) { return tokens.map((t) => t.t).join(''); }
function esc(s) { return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])).replace(/\n/g, '<br>'); }

function showReveals(doc) {
  const st = doc.streams.steered.tokens, pl = doc.streams.plain.tokens;
  // where do the two answers first differ?
  let split = -1;
  for (let i = 0; i < Math.max(st.length, pl.length); i++) { if (!st[i] || !pl[i] || st[i].t !== pl[i].t) { split = i; break; } }
  const plainEl = $('#plain-text');
  if (doc.emotion === 'none') {
    plainEl.innerHTML = esc(textOf(pl));
    $('#plain-foot').textContent = 'No push this time, so this is identical to what you just read. Pick a feeling to see the difference.';
  } else if (split < 0) {
    plainEl.innerHTML = esc(textOf(pl));
    $('#plain-foot').textContent = 'At this strength the push did not change a single word.';
  } else {
    plainEl.innerHTML = `${esc(textOf(pl.slice(0, split)))}<span class="split">${esc(pl[split]?.t || '')}</span>${esc(textOf(pl.slice(split + 1)))}`;
    const words = textOf(st.slice(0, split)).trim().split(/\s+/).filter(Boolean).length;
    $('#plain-foot').textContent = split === 0 ? 'The two answers differ from the very first word.' : `The two answers were identical for ${words} word${words === 1 ? '' : 's'}, then split at the highlighted word.`;
  }
  $('#base-frame').textContent = doc.base_frame;
  renderBase();
  // decisions
  const infos = st.map((t, i) => ({ i, t, info: pushInfo(t) })).filter((x) => x.info);
  const pushed = infos.filter((x) => x.info.pushed);
  const statEl = $('#decision-stat');
  if (doc.emotion === 'none') {
    statEl.innerHTML = `${st.length}<small>word-pieces written, each picked from ~150,000 options.</small>`;
  } else {
    statEl.innerHTML = `${pushed.length} of ${st.length}<small>choices were tipped by the push (the chosen word became at least 2.5× likelier).</small>`;
  }
  const bars = $('#decision-bars'); bars.innerHTML = '';
  infos.filter((x) => x.info.shown).sort((a, b) => b.info.ratio - a.info.ratio).slice(0, 4).forEach(({ i, t, info }) => {
    const r = document.createElement('div'); r.className = 'dbar';
    const was = info.cfExact ? `${(info.cfP * 100).toFixed(info.cfP < 0.1 ? 1 : 0)}%` : `under ${(info.cfP * 100).toFixed(1)}%`;
    r.innerHTML = `<span><b>“${esc(t.t.trim())}”</b> ${(t.p * 100).toFixed(0)}% with the push, ${was} without</span><span>${info.ratio > 99 ? '99+' : info.ratio.toFixed(0)}×</span>`;
    r.onclick = () => { const s = speech.spans[i]; s?.scrollIntoView({ block: 'nearest' }); inspect(i, s); };
    bars.appendChild(r);
  });
  $('#speech-hint').hidden = doc.emotion === 'none';
  $('#reveals').hidden = false;
  if (!state.attract && window.innerWidth < 1100) setTimeout(() => $('#reveals').scrollIntoView({ behavior: 'smooth', block: 'start' }), 900);
}

function renderBase() {
  const doc = state.doc; if (!doc) return;
  const which = $('#base-push').checked ? 'base_steered' : 'base';
  const s = doc.streams[which];
  $('#base-push').parentElement.style.display = doc.emotion === 'none' ? 'none' : '';
  const text = s.text.replace(/\n\s*Q?:?\s*$/, '').trim();
  $('#base-text').innerHTML = s.safe === false || !text ? '<em>(The base model wrote something we don’t show in this exhibit, or nothing at all. Base models are unfiltered.)</em>' : esc(text);
}

// ------------------------------------------------------------------ word inspector
function inspect(i, el) {
  const doc = state.doc; if (!doc || !el) return;
  const st = doc.streams.steered.tokens, tok = st[i];
  const box = $('#inspect');
  const before = textOf(st.slice(Math.max(0, i - 10), i));
  $('#inspect-context').innerHTML = `…${esc(before)}<b>${esc(tok.t)}</b>`;
  const color = EMO[doc.emotion.split('>')[0]]?.color || '#9fb4ff';
  const fill = (host, alts, chosen, c) => {
    host.innerHTML = '';
    for (const [t, p] of alts) {
      const row = document.createElement('div'); row.className = `brow${t === chosen ? ' chosen' : ''}`; row.style.setProperty('--c', c);
      row.innerHTML = `<div class="lbl"><i style="width:${Math.max(2, p * 100)}%"></i><span>${esc(t.replace(/^ /, '·')) || '·'}</span></div><div class="pct">${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%</div>`;
      host.appendChild(row);
    }
  };
  fill($('#bars-steered'), tok.a || [], tok.t, color);
  fill($('#bars-plain'), tok.cf || [], tok.t, '#aab0c6');
  const info = pushInfo(tok);
  $('#inspect-foot').textContent = !info ? '' : doc.emotion === 'none' ? 'No push here, so both columns are the same model.'
    : info.pushed ? `The push made “${tok.t.trim()}” about ${info.ratio > 99 ? '100' : info.ratio.toFixed(0)}× more likely. Without it, the favourite was “${info.cfTop.trim()}”.`
    : 'Here the push barely changed the odds: both versions would likely have chosen the same word.';
  box.hidden = false;
  const r = el.getBoundingClientRect(), w = box.offsetWidth, h = box.offsetHeight;
  let x = Math.min(window.innerWidth - w - 12, Math.max(12, r.left + r.width / 2 - w / 2));
  let y = r.bottom + 12; if (y + h > window.innerHeight - 12) y = Math.max(12, r.top - h - 12);
  box.style.left = `${x}px`; box.style.top = `${y}px`;
}
function closeInspect() { $('#inspect').hidden = true; document.querySelectorAll('.tok.sel').forEach((x) => x.classList.remove('sel')); }
$('#inspect-x').onclick = closeInspect;
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeInspect(); $('#how').hidden = true; } });

// ------------------------------------------------------------------ how it works
let howI = 0;
function renderHow() {
  const s = HOW_STEPS[howI];
  $('#how-steps').innerHTML = `<div class="how-step"><h2>${s.title}</h2>${s.body}<div class="viz">${typeof s.viz === 'function' ? s.viz(state.index?.map) : s.viz || ''}</div></div>`;
  $('#how-dots').innerHTML = HOW_STEPS.map((_, k) => `<i class="${k === howI ? 'on' : ''}"></i>`).join('');
  $('#how-prev').style.visibility = howI ? 'visible' : 'hidden';
  $('#how-next').textContent = howI === HOW_STEPS.length - 1 ? 'Try it' : 'Next';
}
$('#open-how').onclick = () => { touch(); howI = 0; renderHow(); $('#how').hidden = false; };
$('#how-x').onclick = () => { $('#how').hidden = true; };
$('#how-prev').onclick = () => { howI = Math.max(0, howI - 1); renderHow(); };
$('#how-next').onclick = () => { if (howI < HOW_STEPS.length - 1) { howI++; renderHow(); } else $('#how').hidden = true; };

// ------------------------------------------------------------------ attract mode (kiosk)
function touch() { state.lastTouch = Date.now(); if (state.attract) { state.attract = false; $('#caption').textContent = ''; } }
['pointerdown', 'keydown'].forEach((ev) => addEventListener(ev, () => { state.lastTouch = Date.now(); if (state.attract) { state.attract = false; state.runId++; state.playing = false; $('#caption').textContent = ''; refresh(); } }, true));
const ATTRACT_AFTER = Number(params.get('attract') || 45) * 1000;
setInterval(async () => {
  if (state.playing || !state.index || Date.now() - state.lastTouch < ATTRACT_AFTER) return;
  state.attract = true;
  const qs = state.index.questions;
  const q = qs[Math.floor(Math.random() * qs.length)];
  const emos = [...ORDER, 'swing', 'none'];
  state.qid = q.id; state.emotion = emos[Math.floor(Math.random() * emos.length)];
  state.level = state.emotion === 'none' || state.emotion === 'swing' ? state.emotion : ['little', 'lot', 'lot', 'toomuch'][Math.floor(Math.random() * 4)];
  refresh();
  $('#caption').textContent = 'Demo · tap anywhere to take over';
  await play();
}, 2500);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ boot
(async () => {
  state.index = await loadIndex();
  spine.configure(state.index.n_layers, state.index.layer);
  if (state.index.map) feelmap.setMap(state.index.map);
  buildControls();
  state.qid = params.get('q') || null;
  if (params.get('e')) state.emotion = params.get('e');
  if (params.get('l')) state.level = params.get('l');
  refresh();
  setPush(null);
  $('#push-badge').hidden = true;
  if (params.get('autoplay')) play({ fast: !!params.get('fast') });
  window.__btm = { state, play, stage };
})();
