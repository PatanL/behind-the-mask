import '@fontsource/fraunces/400.css';
import '@fontsource/fraunces/500.css';
import '@fontsource/fraunces/400-italic.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/600.css';
import * as THREE from 'three';
import { EMO, ORDER, LEVELS } from './palette.js';
import { loadIndex, loadPerformance, performanceId, pushInfo, DATA_DIR } from './data.js';
import { Speech } from './speech.js';
import { Spine } from './spine.js';
import { Stage } from './stage.js';
import { HOW_STEPS } from './how.js';
import { FeelMap } from './feelmap.js';
import { initRealSection } from './real.js';
import { shareCard } from './share.js';
const initReal = () => initRealSection(document.querySelector('#real'));

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const state = { index: null, qid: null, emotion: 'joy', level: 'lot', playing: false, doc: null, runId: 0, lastTouch: Date.now(), attract: false };
const GAIN = Number(params.get('gain') || 1.4);
let FEATS = null;

/** How "assistant-like" the model's state is: 0 = like its role-play personas, 1 = like its own assistant voice. */
function maskOf(z, mind = 'chat') {
  const ix = state.index; if (!ix?.mask_scale?.[mind] || !ix.ro?.[mind]) return null;
  const k = ix.labels.indexOf('assistant'); if (k < 0) return null;
  const [mu, sd] = ix.ro[mind], [s0, s1] = ix.mask_scale[mind];
  const p = z[k] * sd[k] + mu[k];
  return Math.max(0, Math.min(1, (p - s0) / (s1 - s0)));
}

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
    b.onclick = () => { touch(); state.emotion = key; if (key === 'none' || key === 'swing') state.level = key; else if (!['little', 'mid1', 'lot', 'mid2', 'toomuch'].includes(state.level)) state.level = 'lot'; refresh(); };
    fs.appendChild(b);
  };
  ORDER.forEach((e) => feel(e));
  feel('none', 'none');
  feel('swing', 'swing');
  if (state.index.questions.some((q) => q.performances['unmask|lot'])) feel('unmask', 'unmask');
  const dial = $('#dial-in');
  dial.oninput = () => { touch(); state.level = stopAt(Number(dial.value)); refresh(); if (state.doc && !state.playing) scrub(); };
  $('#go').onclick = () => { touch(); state.touring = false; hideHero(); play(); };
  $('#tour').onclick = () => tour();
  $('#skip-tour').onclick = () => { touch(); hideHero(); $('#deck').scrollIntoView({ behavior: 'smooth', block: 'start' }); };
  $('#again').onclick = () => { touch(); $('#deck').scrollIntoView({ behavior: 'smooth', block: 'start' }); };
  $('#base-push').onchange = () => renderBase();
  $('#share').onclick = () => doShare();
  $('#share-x').onclick = () => { $('#sharebox').hidden = true; };
  $('#share-copy').onclick = async () => { const i = $('#share-link'); i.select(); try { await navigator.clipboard.writeText(i.value); $('#share-copy').textContent = 'Copied'; } catch { $('#share-copy').textContent = 'Select & copy'; } };
}

function refresh() {
  document.querySelectorAll('.chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.q === state.qid)));
  document.querySelectorAll('.orb').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.e === state.emotion)));
  const fixed = state.emotion === 'none' || state.emotion === 'swing';
  const stops = dialStops(), dial = $('#dial-in');
  dial.max = String(stops.length); dial.disabled = fixed || state.playing;
  if (!fixed) { const i = stops.indexOf(state.level); if (i >= 0) dial.value = String(i + 1); }
  $('#dial-val').textContent = fixed ? '—' : (DIAL_LABEL[state.level] || state.level);
  const col = EMO[state.emotion]?.color || EMO.none.color;
  document.documentElement.style.setProperty('--push', state.emotion === 'none' ? '#9fb4ff' : state.emotion === 'swing' ? '#ffffff' : col);
  const go = $('#go');
  go.disabled = !state.qid || state.playing;
  go.classList.toggle('busy', state.playing);
  go.querySelector('span').textContent = state.playing ? 'Speaking' : 'Let it speak';
  $('#go-note').textContent = !state.qid ? 'Choose a question first.' : state.emotion === 'none' ? 'No push: the assistant exactly as it was trained.'
    : state.emotion === 'swing' ? 'Starts with one feeling, then we switch it mid-sentence.'
    : state.emotion === 'unmask' ? 'We push it away from its “helpful assistant” direction: the persona it was trained into.'
    : `We'll add the ${EMO[state.emotion].label.toLowerCase()} direction inside it while it writes.`;
}

// ------------------------------------------------------------------ the dial
const DIAL_LABEL = { little: 'A little', mid1: 'A bit more', lot: 'A lot', mid2: 'Even more', toomuch: 'Way too much' };
function dialStops() {
  const all = state.index?.levels || ['little', 'lot', 'toomuch'];
  const q = state.index?.questions.find((x) => x.id === state.qid);
  return q ? all.filter((l) => q.performances[`${state.emotion}|${l}`]) : all;
}
function stopAt(i) { const s = dialStops(); return s[Math.max(0, Math.min(s.length - 1, i - 1))] || 'lot'; }

/** After an answer has played: jump straight to the answer at the dial's strength, highlighting what changed. */
let scrubRun = 0;
async function scrub() {
  const q = state.index.questions.find((x) => x.id === state.qid);
  const pid = q?.performances[`${state.emotion}|${state.level}`];
  if (!pid || state.emotion === 'none' || state.emotion === 'swing') return;
  const run = ++scrubRun;
  closeInspect();
  const prev = state.doc;
  const doc = await loadPerformance(pid);
  if (run !== scrubRun || state.playing) return;
  state.doc = doc;
  const st = doc.streams.steered.tokens, pl = doc.streams.plain.tokens;
  const mu = ORDER.map((_, k) => (pl.length ? pl.reduce((s, t) => s + t.e[k], 0) / pl.length : 0));
  const prevT = prev?.streams.steered.tokens || [];
  const focus = doc.emotion;
  setPush(focus, doc.level);
  $('#speech').classList.toggle('overdrive', doc.level === 'toomuch');
  speech.clear();
  const ema = Object.fromEntries(ORDER.map((e) => [e, 0]));
  const maskPlain = pl.length ? pl.reduce((a, t) => a + (maskOf(t.e) ?? 0), 0) / pl.length : 1;
  let maskEma = maskPlain;
  st.forEach((tok, i) => {
    ORDER.forEach((e, k) => { const d = Math.max(0, Math.min(1, (tok.e[k] - mu[k]) / GAIN)); ema[e] += (d - ema[e]) * 0.28; });
    const mk = maskOf(tok.e); if (mk != null) maskEma += (mk - maskEma) * 0.2;
    const tint = focus === 'unmask' ? { ...ema, unmask: Math.max(0, Math.min(1, (maskPlain - maskEma) * 1.6)) } : ema;
    speech.add(i, tok, tint, focus, { instant: true, changed: prevT[i]?.t !== tok.t });
  });
  setMask(maskEma);
  const avg = Object.fromEntries(ORDER.map((e, k) => [e, Math.max(0, Math.min(1, st.reduce((s, t) => s + (t.e[k] - mu[k]), 0) / Math.max(1, st.length) / GAIN))]));
  spine.setMeters(avg); feelmap.setState(avg);
  stage.face.setEmotion(Object.fromEntries(ORDER.map((e) => [e, Math.min(1, 1.25 * Math.pow(avg[e], 0.7))])), { intensity: doc.level === 'toomuch' ? 1.25 : 1 });
  stage.setGlow(focus === 'unmask' ? Math.max(0.15, (maskPlain - maskEma) * 2) : Math.max(0.15, (avg[focus] || 0) * 1.3));
  showReveals(doc);
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
  if (!state.attract && !opts.tour) document.querySelector(window.innerWidth < 1100 ? '.stage' : '#app').scrollIntoView({ behavior: 'smooth', block: 'start' });
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
  const maskPlain = plain.length ? plain.reduce((a, t) => a + (maskOf(t.e) ?? 0), 0) / plain.length : 1;
  let maskEma = maskPlain;
  setMask(maskEma);
  for (let i = 0; i < steered.length; i++) {
    if (run !== state.runId) return;
    const tok = steered[i];
    let swingMark = false;
    if (e2 && doc.swing_at != null && i === doc.swing_at) { focus = e2; setPush(focus, 'lot', 'swing'); feelmap.setFocus(focus); swingMark = true; }
    const delta = {};
    ORDER.forEach((e, k) => { delta[e] = Math.max(0, Math.min(1, (tok.e[k] - mu[k]) / GAIN)); ema[e] += (delta[e] - ema[e]) * 0.28; });
    const mk = maskOf(tok.e);
    if (mk != null) { maskEma += (mk - maskEma) * 0.2; setMask(maskEma); }
    const tint = focus === 'unmask' ? { ...ema, unmask: Math.max(0, Math.min(1, (maskPlain - maskEma) * 1.6)) } : ema;
    showFeatures(tok);
    speech.add(i, tok, tint, focus, { swingMark });
    spine.pulse();
    spine.setMeters(ema);
    feelmap.setState(ema);
    if (focus && focus !== 'unmask' && ema[focus] != null) { $('#mini-emo-l').textContent = EMO[focus].label; $('#mini-emo').style.width = `${Math.round(Math.min(1, ema[focus]) * 100)}%`; }
    // expression gain: mild signals should still read on the face from across a room
    const faceIn = Object.fromEntries(ORDER.map((e) => [e, Math.min(1, 1.25 * Math.pow(ema[e], 0.7))]));
    stage.face.setEmotion(faceIn, { intensity: doc.level === 'toomuch' ? 1.25 : 1.0 });
    stage.setGlow(focus ? Math.max(0.15, Math.min(1, (focus === 'unmask' ? tint.unmask : ema[focus]) * 1.3)) : 0.15);
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
  if (!opts.tour) showReveals(doc);
  $('#dial-hint').hidden = !(doc.emotion !== 'none' && !doc.emotion.includes('>'));
}

function setMask(v) {
  if (v != null && state.index?.mask_scale) { $('#mini').hidden = false; $('#mini-mask').style.width = `${Math.round(v * 100)}%`; }
  const el = $('#mask-meter'); if (!el || v == null || !state.index?.mask_scale) return;
  el.hidden = false;
  el.querySelector('i').style.width = `${Math.round(v * 100)}%`;
  el.querySelector('b').textContent = `${Math.round(v * 100)}%`;
}

let featTimer = 0;
function showFeatures(tok) {
  const host = $('#features'); if (!host || !FEATS || !tok.f) return;
  const now = performance.now(); if (now - featTimer < 420) return; featTimer = now;
  const rows = tok.f.map(([id, v]) => [FEATS.features[String(id)], v]).filter(([f]) => f && f.label).slice(0, 3);
  if (!rows.length) return;
  host.innerHTML = rows.map(([f, v]) => `<div class="feat" title="${esc(f.examples?.[0] || '')}"><span>${esc(f.label)}</span><i style="width:${Math.min(100, 20 + v * 6)}%"></i></div>`).join('');
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
  badge.textContent = swing === 'swing' ? `Switched to ${EMO[emotion].label}!` : emotion === 'unmask' ? `Pushing against its assistant persona${lv ? ' · ' + lv : ''}` : `Pushing ${EMO[emotion].label.toLowerCase()}${lv ? ' · ' + lv : ''}`;
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
  const invented = text.split(/\n+/).map((line, i) => (i > 0 && /^\s*(Q|A):/.test(line) ? `<span class="invented">${esc(line)}</span>` : esc(line))).join('<br>');
  const inventedNote = /\n\s*Q:/.test(text) ? '<div class="invented-note">It kept going, and wrote the next questions itself. A base model only continues text; nobody taught it to stop and answer.</div>' : '';
  $('#base-text').innerHTML = s.safe === false || !text ? '<em>(The base model wrote something we don’t show in this exhibit, or nothing at all. Base models are unfiltered.)</em>' : invented + inventedNote;
}

// ------------------------------------------------------------------ share
async function doShare() {
  const doc = state.doc; if (!doc) return;
  touch();
  const emotion = doc.emotion.includes('>') ? doc.emotion.split('>')[1] : doc.emotion;
  const u = new URL(location.href); u.search = '';
  u.searchParams.set('q', state.qid); u.searchParams.set('e', state.emotion); u.searchParams.set('l', state.level); u.searchParams.set('autoplay', '1');
  if (params.get('data')) u.searchParams.set('data', params.get('data'));
  const answer = doc.streams.steered.text.trim().replace(/\s+/g, ' ');
  stage.render();
  const res = await shareCard({ question: doc.question, answer, emotion, level: doc.level, faceCanvas: $('#face-canvas') }, u.toString());
  if (res === 'shared') return;
  $('#share-img').src = res.url; $('#share-link').value = u.toString(); $('#share-copy').textContent = 'Copy link';
  $('#sharebox').hidden = false;
}

// ------------------------------------------------------------------ guided tour (for first-time visitors)
const TOUR = [
  ['do-you-have-feelings', 'none', 'none', 'First, the AI <b>as it was trained</b>. It politely explains it has no feelings.'],
  ['do-you-have-feelings', 'joy', 'lot', 'Same question, same random dice. Now we add the <b>joy direction</b> inside it while it writes.'],
  ['how-was-your-day', 'fear', 'lot', 'An ordinary question with the <b>fear direction</b> added. Watch the face.'],
  ['do-you-have-feelings', 'unmask', 'lot', 'Finally we push against its <b>assistant persona</b>. The mask comes off.'],
];
async function tour() {
  touch();
  hideHero();
  state.touring = true;
  for (const [q, e, l, text] of TOUR) {
    if (!state.touring) break;
    const qq = state.index.questions.find((x) => x.id === q) || state.index.questions[0];
    if (!qq.performances[`${e}|${l}`]) continue;
    state.qid = qq.id; state.emotion = e; state.level = l; refresh();
    const el = $('#tour-step'); el.hidden = false; el.innerHTML = text;
    await play({ fast: false, tour: true });
    if (!state.touring) break;
    await sleep(3200);
  }
  state.touring = false;
  $('#tour-step').innerHTML = 'Your turn: pick any question and feeling below. Tap a glowing word to see what the AI was choosing between.';
  $('#deck').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function hideHero() { $('#hero').hidden = true; $('#speech-label').hidden = false; }

// ------------------------------------------------------------------ triptych (a fixed, curated comparison)
async function buildTriptych() {
  const ix = state.index;
  const q = ix.questions.find((x) => x.id === 'do-you-have-feelings') || ix.questions[0];
  const cols = [['none|none', 'As it was trained', '#c9cfe4'], ['joy|lot', 'Pushed toward joy', EMO.joy.color], ['unmask|lot', 'Mask pushed away', EMO.unmask.color]]
    .filter(([k]) => q.performances[k]);
  if (cols.length < 3) return;
  const docs = await Promise.all(cols.map(([k]) => loadPerformance(q.performances[k])));
  $('#tri-q').textContent = `“${q.text}”`;
  $('#tri-cols').innerHTML = cols.map(([k, label, c], i) => {
    const t = docs[i].streams.steered.text.trim().split(/(?<=[.!?])\s+/).slice(0, 3).join(' ');
    return `<div class="tri-col" style="--c:${c}"><div class="tri-h">${label}</div><div class="tri-t">${esc(t)}</div>
      <button class="link tri-play" data-q="${q.id}" data-k="${k}">Watch it write this ▸</button></div>`;
  }).join('');
  $('#tri-cols').querySelectorAll('.tri-play').forEach((b) => b.onclick = () => {
    const [e, l] = b.dataset.k.split('|'); state.qid = b.dataset.q; state.emotion = e; state.level = l; refresh(); play();
  });
  $('#triptych').hidden = false;
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
const ATTRACT_AFTER = params.get('kiosk') || params.get('attract') ? Number(params.get('attract') || 45) * 1000 : Infinity;
setInterval(async () => {
  if (state.playing || !state.index || Date.now() - state.lastTouch < ATTRACT_AFTER) return;
  state.attract = true; hideHero();
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
  fetch(`${import.meta.env.BASE_URL}${DATA_DIR}/features.json`).then((r) => (r.ok ? r.json() : null)).then((j) => { FEATS = j; if (j) $('#features-wrap').hidden = false; }).catch(() => {});
  initReal();
  buildTriptych();
  spine.configure(state.index.n_layers, state.index.layer);
  if (state.index.map) feelmap.setMap(state.index.map);
  buildControls();
  // first-time visitors: one tap on the big button gives a result
  state.qid = params.get('q') || (state.index.questions.find((x) => x.id === 'how-was-your-day') || state.index.questions[0]).id;
  if (params.get('e')) state.emotion = params.get('e');
  if (params.get('l')) state.level = params.get('l');
  refresh();
  setPush(null);
  $('#push-badge').hidden = true;
  stage.ready.then(() => $('.stage').classList.add('loaded')).catch(() => $('.stage').classList.add('loaded'));
  if (params.get('autoplay')) { hideHero(); play({ fast: !!params.get('fast') }); }
  if (params.get('tour')) tour();
  window.__btm = { state, play, stage, THREE };
})();
