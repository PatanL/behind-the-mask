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
import { FeelWheel } from './wheel.js';
import { shareCard } from './share.js';
import { createLive, liveStatus } from './live.js';

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const state = { index: null, qid: null, emotion: 'joy', level: 'lot', playing: false, doc: null, runId: 0, lastTouch: Date.now(), attract: false, mode: null };
const GAIN = Number(params.get('gain') || 1.4);
let FEATS = null;
let DECODER = null;   // ${DATA_DIR}/decoder.json: readout lift -> the feeling the face shows (scripts/fit-decoder.mjs)

// What the face shows. The raw readout has cross-talk (the anger direction also rises on fear and sadness
// pushes), so the face reads it through a linear decoder fitted on the pre-computed pushes. `lift` is the
// smoothed, signed readout minus the unpushed answer's mean.
function faceFrom(lift) {
  let v;
  if (DECODER) {
    const { M, floor, gain } = DECODER;
    v = ORDER.map((_, j) => Math.max(0, (ORDER.reduce((s, _e, i) => s + lift[i] * M[i][j], 0) - floor) * gain[j]));
  } else v = ORDER.map((_, k) => Math.max(0, lift[k] / (GAIN * 4)));
  // a small dead zone, then a gain so that mild signals still read from across a room
  return Object.fromEntries(ORDER.map((e, j) => [e, Math.min(1, 1.3 * Math.pow(Math.max(0, v[j] - 0.04) / 0.96, 0.8))]));
}
// smoothed signed lift over a token list (as the face would have seen it by the end)
function liftOf(tokens, mu, upto = tokens.length) {
  const rate = DECODER?.rate ?? 0.2, l = ORDER.map(() => 0);
  for (let i = 0; i < upto; i++) ORDER.forEach((_, k) => { l[k] += (tokens[i].e[k] - mu[k] - l[k]) * rate; });
  return l;
}
// mean signed lift (for the lingering after-state)
const meanLift = (tokens, mu) => ORDER.map((_, k) => (tokens.length ? tokens.reduce((s, t) => s + t.e[k] - mu[k], 0) / tokens.length : 0));

/** How "assistant-like" the model's state is: 0 = like its role-play personas, 1 = like its own assistant voice. */
function maskOf(z, mind = 'chat') {
  const ix = state.index; if (!ix?.mask_scale?.[mind] || !ix.ro?.[mind]) return null;
  const k = ix.labels.indexOf('assistant'); if (k < 0) return null;
  const [mu, sd] = ix.ro[mind], [s0, s1] = ix.mask_scale[mind];
  const p = z[k] * sd[k] + mu[k];
  return Math.max(0, Math.min(1, (p - s0) / (s1 - s0)));
}

const stage = new Stage($('#face-canvas'));
const spine = new Spine($('#spine'));
const wheel = new FeelWheel($('#wheel'));
const speech = new Speech($('#speech'), { onPick: (i, el) => inspect(i, el) });

// ------------------------------------------------------------------ controls
function buildControls() {
  const sel = $('#q-select');
  let group = null;
  for (const q of state.index.questions) {
    if (q.group !== group?.label) { group = document.createElement('optgroup'); group.label = q.group; sel.appendChild(group); }
    const o = document.createElement('option'); o.value = q.id; o.textContent = q.text; group.appendChild(o);
  }
  sel.onchange = () => { touch(); state.qid = sel.value; refresh(); stage.face.react('listen'); };
  const fs = $('#feelings');
  const feel = (key, cls = '') => {
    const b = document.createElement('button');
    b.className = `orb ${cls}`; b.dataset.e = key; b.style.setProperty('--c', EMO[key].color);
    b.innerHTML = `<span class="ball"></span><span>${key === 'unmask' ? 'Unmask' : key === 'swing' ? 'Swing' : EMO[key].label}</span>`;
    b.title = EMO[key].label;
    b.onclick = () => { touch(); stage.face.react('listen'); state.emotion = key; if (key === 'none' || key === 'swing') state.level = key; else if (!['little', 'mid1', 'lot', 'mid2', 'toomuch'].includes(state.level)) state.level = 'lot'; refresh(); };
    fs.appendChild(b);
  };
  ORDER.forEach((e) => feel(e));
  feel('none', 'none');
  feel('swing', 'swing');
  if (state.index.questions.some((q) => q.performances['unmask|lot'])) feel('unmask', 'unmask');
  const dial = $('#dial-in');
  dial.oninput = () => { touch(); state.level = stopAt(Number(dial.value)); refresh(); if (state.doc && !state.playing) scrub(); };
  $('#go').onclick = () => { touch(); play(); };
  $('#share').onclick = () => doShare();
  $('#share-x').onclick = () => { $('#sharebox').hidden = true; };
  $('#share-copy').onclick = async () => { const i = $('#share-link'); i.select(); try { await navigator.clipboard.writeText(i.value); $('#share-copy').textContent = 'Copied'; } catch { $('#share-copy').textContent = 'Select & copy'; } };
}

function refresh() {
  if (state.qid) $('#q-select').value = state.qid;
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
  if (state.playing || state.answered) return;
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
  const shown = faceFrom(liftOf(st, mu));
  wheel.setState(shown); wheel.setFocus(focus === 'none' ? null : focus);
  stage.face.setEmotion(shown, { intensity: doc.level === 'toomuch' ? 1.15 : 1 });
  stage.setGlow(glowFor(focus, shown, maskPlain - maskEma));
  afterAnswer(doc);
}

// ------------------------------------------------------------------ playback
function pickSwing(q) {
  const keys = Object.keys(q.performances).filter((k) => k.endsWith('|swing'));
  return keys[Math.floor(Math.random() * keys.length)];
}

async function* arrayFeed(arr) { for (const x of arr) yield x; }
const meanE = (toks, k) => (toks.length ? toks.reduce((s, t) => s + t.e[k], 0) / toks.length : 0);

/** Perform one ready-made answer (state.qid / emotion / level). Live mode runs in live.js. */
async function play(opts = {}) {
  if (state.mode !== 'made') setMode('made');
  const run = ++state.runId;
  if (!state.qid) return;
  const q = state.index.questions.find((x) => x.id === state.qid);
  let key = `${state.emotion}|${state.level}`;
  if (state.emotion === 'none') key = 'none|none';
  if (state.emotion === 'swing') key = pickSwing(q);
  const pid = q.performances[key];
  if (!pid) return;
  state.playing = true; state.answered = false; refresh();
  $('#go-note').textContent = '';
  $('#share').hidden = true;
  const doc = await loadPerformance(pid);
  if (run !== state.runId) return;
  const feed = arrayFeed(doc.streams.steered.tokens);
  closeInspect();
  state.doc = doc;
  const plain = doc.streams.plain.tokens;
  // baseline: the model's own readout on the unpushed answer
  const fullMu = ORDER.map((_, k) => meanE(plain, k));
  const muNow = () => fullMu;
  const [e1, e2] = doc.emotion.includes('>') ? doc.emotion.split('>') : [doc.emotion, null];
  let focus = e1 === 'none' ? null : e1;
  setPush(focus, doc.level, e2 ? 'swing-start' : null);
  $('#speech-label').textContent = `“${doc.question}”`;
  $('#speech').classList.toggle('overdrive', doc.level === 'toomuch');
  $('#push-badge').classList.toggle('warn', doc.level === 'toomuch');
  speech.begin();
  wheel.reset(); wheel.setFocus(focus);
  // a moment to "think": glance aside, press the lips, breathe in, then start writing
  stage.face.react('think');
  await sleep(opts.fast ? 250 : 950);
  if (run !== state.runId) return;
  stage.face.setActivity({ writing: true });
  let lastEmph = -10, i = -1;
  const ema = Object.fromEntries(ORDER.map((e) => [e, 0]));
  const lift = ORDER.map(() => 0), liftRate = DECODER?.rate ?? 0.2;
  const maskPlainNow = () => (plain.length ? plain.reduce((a, t) => a + (maskOf(t.e) ?? 0), 0) / plain.length : 1);
  const maskFixed = maskPlainNow();
  let maskEma = maskPlainNow();
  setMask(maskEma);
  for await (const tok of feed) {
    i++;
    if (run !== state.runId) return;
    let swingMark = false;
    if (e2 && doc.swing_at != null && i === doc.swing_at) { focus = e2; setPush(focus, 'lot', 'swing'); wheel.setFocus(focus); swingMark = true; }
    const mu = muNow(), maskPlain = maskFixed ?? maskPlainNow();
    ORDER.forEach((e, k) => { const d = Math.max(0, Math.min(1, (tok.e[k] - mu[k]) / GAIN)); ema[e] += (d - ema[e]) * 0.28; lift[k] += (tok.e[k] - mu[k] - lift[k]) * liftRate; });
    const mk = maskOf(tok.e);
    if (mk != null) { maskEma += (mk - maskEma) * 0.2; setMask(maskEma); }
    const tint = focus === 'unmask' ? { ...ema, unmask: Math.max(0, Math.min(1, (maskPlain - maskEma) * 1.6)) } : ema;
    showFeatures(tok);
    speech.add(i, tok, tint, focus, { swingMark });
    spine.pulse();
    const shown = faceFrom(lift);
    wheel.setState(shown);
    if (focus && focus !== 'unmask') { $('#mini-emo-l').textContent = EMO[focus].label; $('#mini-emo').style.width = `${Math.round(Math.min(1, shown[focus]) * 100)}%`; }
    stage.face.setEmotion(shown, { intensity: doc.level === 'toomuch' ? 1.15 : 1.0 });
    stage.setGlow(glowFor(focus, shown, maskPlain - maskEma));
    const t = tok.t;
    if (beatFor(tok, i, lastEmph)) lastEmph = i;
    let wait = 62 + Math.min(80, t.length * 6);
    stage.face.say(t, wait / 1000 / (opts.fast ? 3 : 1));   // mouth the word while it appears
    if (/[.!?]\s*$/.test(t)) wait += 360; else if (/[,;:]\s*$/.test(t)) wait += 170; else if (t.includes('\n')) wait += 260;
    await sleep(wait / (opts.fast ? 3 : 1));
  }
  speech.end();
  stage.face.setActivity({ writing: false });
  stage.face.react('done');
  state.playing = false; refresh();
  // the feeling lingers, then relaxes
  // meters settle on the answer's average lift; the face relaxes but keeps a trace of the feeling
  const mu = muNow();
  const st = doc.streams.steered.tokens;
  const after = faceFrom(meanLift(st, mu));
  setTimeout(() => { if (run === state.runId && !state.playing) { stage.face.setEmotion(Object.fromEntries(ORDER.map((e) => [e, after[e] * 0.6]))); wheel.setState(after); } }, 2600);
  afterAnswer(doc);
}

/** Conversational beats from the text as it is written; returns true for an emphasis beat. */
function beatFor(tok, i, lastEmph) {
  const t = tok.t;
  if (/\?\s*$/.test(t)) stage.face.beat('question');
  else if (/!\s*$/.test(t)) stage.face.beat('exclaim');
  else if (/[.]\s*$/.test(t) && !/\.\.\s*$/.test(t)) stage.face.beat('period');
  else if (/[,;:—]\s*$/.test(t)) stage.face.beat('comma');
  else if (pushInfo(tok)?.shown && i - lastEmph > 6) { stage.face.beat('emphasis'); return true; }
  return false;
}

// ------------------------------------------------------------------ live: everyone steers one story
let live = null, liveLastEmph = -10, livePushKey = '';
function setupLive() {
  live = createLive({
    $, stage, speech, spine, wheel, faceFrom, maskOf, setMask, showFeatures, glowFor,
    labels: () => state.index.labels,
    setDoc: (doc) => { if (state.mode === 'live') state.doc = doc; },
    beat: (tok, i) => { if (i < liveLastEmph) liveLastEmph = -10; if (beatFor(tok, i, liveLastEmph)) liveLastEmph = i; },
    onFull: (n) => {
      // the live room is full: a ready-made answer instead, and the Live tab is offered again in a minute
      setMode('made');
      $('#go-note').textContent = `Live is full right now (${n} watching). Try a ready-made answer, then Live again in a minute.`;
    },
    onPush: (lead, p) => {
      const lv = p < 0.35 ? 'little' : p < 0.7 ? 'lot' : 'toomuch', key = `${lead}|${lv}`;
      if (key === livePushKey) return;
      livePushKey = key;
      setPush(lead, lv);
      $('#push-badge').textContent = lead ? `Live · everyone is pushing ${lead === 'unmask' ? 'off the mask' : EMO[lead].label.toLowerCase()}` : 'Live · nobody pushing yet';
    },
  });
  $('#mode-live').onclick = () => { touch(); setMode('live'); };
  $('#mode-made').onclick = () => { touch(); setMode('made'); };
  $('#new-topic').onclick = () => { touch(); live.newTopic(); };
}

/** The two ways to use it. Both share the stage; only the controls under it change (same place, same size). */
function setMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === 'live') live.leave();
  state.mode = mode;
  state.runId++; state.playing = false; state.doc = null;
  closeInspect();
  speech.clear(); $('#speech-label').textContent = '';
  $('#live').hidden = mode !== 'live'; $('#made').hidden = mode !== 'made';
  $('#mode-live').setAttribute('aria-pressed', String(mode === 'live')); $('#mode-made').setAttribute('aria-pressed', String(mode === 'made'));
  stage.face.setActivity({ writing: false });
  livePushKey = '';
  setPush(null);
  if (mode === 'live') live.enter();
  else { state.answered = false; refresh(); $('#push-badge').hidden = true; }
}

function setMask(v) {
  if (v != null) stage.face.setMask?.(v);   // the trained persona acts as a display rule on the face
  if (v != null && state.index?.mask_scale) { $('#mini').hidden = false; $('#mini-mask').style.width = `${Math.round(v * 100)}%`; }
  const el = $('#mask-meter'); if (!el || v == null || !state.index?.mask_scale) return;
  el.hidden = false;
  el.querySelector('.val').style.strokeDashoffset = String(100 - Math.round(v * 100));
  el.querySelector('b').textContent = `${Math.round(v * 100)}%`;
}

/** The glow (seams, eyes, aura): how strongly the pushed feeling reads inside, as the face shows it. With no
 *  push it rests low; "unmask" glows with how far the state has moved off the assistant voice. */
function glowFor(focus, shown, maskDrop) {
  if (!focus || focus === 'none') return 0.08;
  const v = focus === 'unmask' ? Math.max(0, Math.min(1, maskDrop * 1.6)) : shown[focus] || 0;
  return Math.max(0.08, Math.min(1, v * 1.2));
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

// ------------------------------------------------------------------ text helpers
function textOf(tokens) { return tokens.map((t) => t.t).join(''); }
function esc(s) { return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])).replace(/\n/g, '<br>'); }

/** After an answer: the last lines stay; the status line says what you can do with them. */
function afterAnswer(doc) {
  const fixed = doc.emotion === 'none' || doc.emotion.includes('>');
  state.answered = true;
  $('#go-note').textContent = fixed ? 'Tap a word to see what else it might have said.' : 'Tap a word to see what else it might have said, or drag the strength.';
  $('#share').hidden = !!state.attract;
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

// ------------------------------------------------------------------ word inspector
function inspect(i, el) {
  const doc = state.doc; if (!doc || !el) return;
  const st = doc.streams.steered.tokens, tok = st[i];
  if (tok && !tok.cf && state.mode === 'live') {   // live words carry no alternatives until someone looks
    live.alts(i).then((r) => { if (r && state.doc === doc) { tok.a = r.a; tok.cf = r.cf; inspect(i, el); } });
    return;
  }
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
function touch() { state.lastTouch = Date.now(); state.attract = false; }
['pointerdown', 'keydown'].forEach((ev) => addEventListener(ev, () => { state.lastTouch = Date.now(); if (state.attract) { state.attract = false; state.runId++; state.playing = false; refresh(); } }, true));
const ATTRACT_AFTER = params.get('kiosk') || params.get('attract') ? Number(params.get('attract') || 45) * 1000 : Infinity;
setInterval(async () => {
  if (state.playing || state.mode === 'live' || !state.index || Date.now() - state.lastTouch < ATTRACT_AFTER) return;
  state.attract = true;
  const qs = state.index.questions;
  const q = qs[Math.floor(Math.random() * qs.length)];
  const emos = [...ORDER, 'swing', 'none'];
  state.qid = q.id; state.emotion = emos[Math.floor(Math.random() * emos.length)];
  state.level = state.emotion === 'none' || state.emotion === 'swing' ? state.emotion : ['little', 'lot', 'lot', 'toomuch'][Math.floor(Math.random() * 4)];
  refresh();
  await play();
  if (state.attract) $('#go-note').textContent = 'Demo · tap anywhere to take over';
}, 2500);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ boot
(async () => {
  state.index = await loadIndex();
  fetch(`${import.meta.env.BASE_URL}${DATA_DIR}/decoder.json`).then((r) => (r.ok ? r.json() : null)).then((j) => { DECODER = j; }).catch(() => {});
  fetch(`${import.meta.env.BASE_URL}${DATA_DIR}/features.json`).then((r) => (r.ok ? r.json() : null)).then((j) => { FEATS = j; if (j) $('#features-wrap').hidden = false; }).catch(() => {});
  spine.configure(state.index.n_layers, state.index.layer);
  buildControls();
  setupLive();
  state.qid = params.get('q') || (state.index.questions.find((x) => x.id === 'do-you-have-feelings') || state.index.questions[0]).id;
  if (params.get('e')) state.emotion = params.get('e');
  if (params.get('l')) state.level = params.get('l');
  stage.ready.then(() => $('.stage').classList.add('loaded')).catch(() => $('.stage').classList.add('loaded'));
  // straight into the demo: the live AI if its server is up, otherwise a ready-made answer, already playing
  const ls = params.get('autoplay') || params.get('made') ? null : await liveStatus();
  $('#mode-live').hidden = !ls;
  if (ls) setMode('live');
  else { setMode('made'); if (!params.get('rehearsal')) stage.ready.then(() => setTimeout(() => play({ fast: !!params.get('fast') }), 600)).catch(() => {}); }
  setInterval(async () => { const s = await liveStatus(); $('#mode-live').hidden = !s; $('#mode-live-n').textContent = s?.viewers ? `· ${s.viewers}` : ''; }, 15000);
  window.__btm = { state, play, stage, speech, THREE };
})();
