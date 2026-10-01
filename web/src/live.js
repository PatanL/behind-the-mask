// Live: the AI talks about itself, without stopping, and everyone on the page steers how it feels together
// (server/live.py). Visitors only press feeling buttons (and "New topic"); the server mixes everyone's recent taps
// into the push and streams the monologue, word by word, to everyone.
import { EMO, ORDER } from './palette.js';

const BUTTONS = [...ORDER, 'unmask'];
const BASE = import.meta.env.BASE_URL;
// where the live server is: the page's own host by default; another host (the Spark, through a tunnel) when the
// page itself is served from a CDN such as GitHub Pages -- set VITE_LIVE_URL at build time
const LIVE = (import.meta.env.VITE_LIVE_URL || '').replace(/\/+$/, '');
const wsUrl = () => (LIVE ? `${LIVE.replace(/^http/, 'ws')}/live/ws` : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${BASE}live/ws`) + `?cid=${cid()}`;
function cid() {
  // per tab (two tabs are two visitors), kept across reloads so a reconnect replaces the old socket
  try { let v = sessionStorage.getItem('btm-cid'); if (!v) { v = Math.random().toString(36).slice(2, 12); sessionStorage.setItem('btm-cid', v); } return v; }
  catch { return Math.random().toString(36).slice(2, 12); }
}

/** Is the live server up? -> {viewers, ready, story} or null */
export async function liveStatus() {
  try {
    const r = await fetch(LIVE ? `${LIVE}/live/status` : `${BASE}live/status`, { cache: 'no-store' });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

/**
 * ctx: { $, stage, speech, spine, wheel, faceFrom, maskOf, setMask, showFeatures, glowFor, labels: () => string[],
 *        setDoc(doc), beat(tok, i) }
 */
export function createLive(ctx) {
  const { $, stage, speech } = ctx;
  let ws = null, on = false, story = null, lastListen = 0, retry = 0, refused = 0, shownTalk = null, topic = '', viewers = 0, lite = false;
  // taps go out in batches (counts per feeling, every 0.4 s): a tenth of the messages, the same push
  const counts = {};
  let tapTimer = 0;
  const flushTaps = () => {
    tapTimer = 0;
    if (!Object.keys(counts).length || !ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({ type: 'taps', c: { ...counts } }));
    for (const k of Object.keys(counts)) delete counts[k];
  };
  // the word inspector asks for a word's alternatives only when someone taps it
  const altWait = new Map();

  // ---- buttons (press and hold to keep pushing)
  const host = $('#live-buttons');
  for (const b of BUTTONS) {
    const el = document.createElement('button');
    el.className = `tapb${b === 'unmask' ? ' unmask' : ''}`; el.dataset.b = b; el.style.setProperty('--c', EMO[b].color);
    const label = b === 'unmask' ? '<span class="long">Take off the mask</span><span class="short">Unmask</span>' : EMO[b].label;
    el.innerHTML = `<span class="ball"></span><span class="tl">${label}</span><span class="ring"></span>`;
    el.setAttribute('aria-label', `Push ${EMO[b].label.toLowerCase()}`);
    let hold = 0;
    const stop = () => { clearInterval(hold); hold = 0; el.classList.remove('held'); };
    el.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      tap(b, el);
      el.classList.add('held');
      clearInterval(hold); hold = setInterval(() => tap(b, el), 190);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach((t) => el.addEventListener(t, stop));
    el.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); tap(b, el); } });
    host.appendChild(el);
  }
  const bar = $('#crowd-bar');
  bar.innerHTML = BUTTONS.map((b) => `<i data-b="${b}" style="--c:${EMO[b].color}"></i>`).join('');

  function tap(b, el) {
    if (!ws || ws.readyState !== 1) return;
    counts[b] = (counts[b] || 0) + 1;
    if (!tapTimer) tapTimer = setTimeout(flushTaps, 400);
    el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
    navigator.vibrate?.(6);
    const now = performance.now();
    if (now - lastListen > 2500) { stage.face.react('listen'); lastListen = now; }
  }

  // ---- the story as it arrives
  const labels = () => ctx.labels();
  let lift, maskEma, maskPlain, i;
  const ema = Object.fromEntries(ORDER.map((e) => [e, 0]));
  // one turn of the monologue; a turn that continues the talk on screen carries on in the same subtitle box
  function begin(msg) {
    const cont = !!msg.continues && shownTalk === msg.talk;
    shownTalk = msg.talk;
    story = { id: msg.id, topic: msg.topic, steered: [], plain: [], ended: false };
    lift = ORDER.map(() => 0); maskEma = null; maskPlain = null; i = -1;
    ORDER.forEach((e) => { ema[e] = 0; });
    if (cont) speech.continueLine(); else speech.begin();
    $('#speech').classList.remove('overdrive');
    topic = msg.topic || ''; label();
    if (msg.pace) beatMs = 1000 / msg.pace;   // words are shown at the pace they are written
    if (!cont) stage.face.react('think');
    const id = msg.id;
    setTimeout(() => { if (on && story?.id === id && !story.ended) stage.face.setActivity({ writing: true }); }, cont ? 150 : 700);
    syncDoc();
  }
  // phones drop the words in .lp ("Live · talking about the people watching · 5 here")
  function label() {
    const el = $('#speech-label'), lp = (t) => Object.assign(document.createElement('span'), { className: 'lp', textContent: t });
    el.replaceChildren(lp('Live · '), `talking about ${topic}`);
    if (viewers > 1) el.append(` · ${viewers} `, lp('people '), 'here');
    if (lite) el.append(' · a few seconds behind');
  }
  const meanE = (toks, k) => (toks.length ? toks.reduce((s, t) => s + t.e[k], 0) / toks.length : 0);
  function pushAt(tok) {
    // which button dominated the push when this word was written (weights relative to each button's full strength)
    const L = labels(); let best = null, bw = 0.08;
    ORDER.forEach((e) => { const k = L.indexOf(e); const w = k >= 0 ? (tok.s?.[k] || 0) / 0.9 : 0; if (w > bw) { bw = w; best = e; } });
    const ka = L.indexOf('assistant');
    if (ka >= 0 && -(tok.s?.[ka] || 0) / 0.65 > bw) best = 'unmask';
    return best;
  }
  function addSteered(tok, instant) {
    i++;
    story.steered.push(tok);
    const L = labels(), rate = 0.2;
    const mu = ORDER.map((e) => { const k = L.indexOf(e); return meanE(story.plain, k); });
    ORDER.forEach((e, k) => { const kk = L.indexOf(e); lift[k] += (tok.e[kk] - mu[k] - lift[k]) * rate; });
    const face = ctx.faceFrom(lift);
    ORDER.forEach((e) => { ema[e] += (face[e] - ema[e]) * 0.35; });
    const mk = ctx.maskOf(tok.e);
    if (mk != null) { maskEma = maskEma == null ? mk : maskEma + (mk - maskEma) * 0.2; ctx.setMask(maskEma); }
    if (story.plain.length) maskPlain = story.plain.reduce((a, t) => a + (ctx.maskOf(t.e) ?? 0), 0) / story.plain.length;
    const focus = pushAt(tok);
    const tint = { ...ema, unmask: maskPlain != null && maskEma != null ? Math.max(0, Math.min(1, (maskPlain - maskEma) * 1.6)) : 0 };
    speech.add(i, tok, tint, focus, { instant });
    if (!instant) {
      stage.face.setEmotion(face);
      ctx.spine.pulse(); ctx.wheel.setState(face); ctx.wheel.setFocus(focus);
      ctx.showFeatures(tok);
      ctx.beat(tok, i);
      stage.face.say(tok.t, beatMs / 1000);
      stage.setGlow(ctx.glowFor(focus, face, maskPlain != null && maskEma != null ? maskPlain - maskEma : 0));
      if (focus && focus !== 'unmask') { $('#mini-emo-l').textContent = EMO[focus].label; $('#mini-emo').style.width = `${Math.round(Math.min(1, ema[focus]) * 100)}%`; }
    }
  }
  // a compact word entry: t (text), e (readout), s (push), f (features), p/q/qx/k (for the underline),
  // b (the unpushed baseline's readout at the same step)
  function addWord(w, instant) {
    if (w.b) story.plain.push({ e: w.b });
    if (w.t != null) addSteered({ t: w.t, e: w.e, s: w.s, f: w.f, p: w.p, q: w.q, qx: w.qx, k: w.k }, instant);
  }
  function catchUp(msg) {
    if (!story || msg.id !== story.id) return;
    for (const w of msg.w) addWord(w, true);
    stage.face.setEmotion(ctx.faceFrom(lift));
    syncDoc();
  }
  // between turns: a breath, not a sign-off (the next turn follows within a second)
  function end(msg) {
    if (!story || msg.id !== story.id) return;
    story.ended = true;
    speech.end();
    stage.face.setActivity({ writing: false });
  }
  // the story stopped without an end (server error, lost connection): the face stops writing and settles
  function stopFace() {
    speech.end();
    stage.face.setActivity({ writing: false });
    stage.face.clearReactions();
    stage.face.setEmotion({});
  }
  function syncDoc() {
    if (!story) return;
    ctx.setDoc({ id: story.id, question: story.topic, emotion: 'crowd', level: 'live', streams: { steered: { tokens: story.steered }, plain: { tokens: story.plain } } });
  }
  function status(t) { $('#live-status').textContent = t; }

  // ---- the crowd meter
  function crowd(msg) {
    const p = msg.power || 0;
    const mix = Object.fromEntries(BUTTONS.map((b, i) => [b, (msg.mix?.[i] || 0) / 100]));   // compact: percent, in BUTTONS order
    bar.querySelectorAll('i').forEach((el) => { el.style.width = `${(mix[el.dataset.b] || 0) * p * 100}%`; });
    const lead = BUTTONS.reduce((a, b) => (mix[b] > mix[a] ? b : a), BUTTONS[0]);
    $('#crowd-power').textContent = p < 0.04 ? 'Nobody is pushing: it talks as trained'
      : `${p < 0.35 ? 'A gentle' : p < 0.7 ? 'A strong' : 'A full'} push, mostly ${lead === 'unmask' ? 'off the mask' : EMO[lead].label.toLowerCase()}`;
    $('#crowd-viewers').textContent = msg.viewers > 1 ? `${msg.viewers} steering` : 'Just you';
    if (msg.viewers !== viewers) { viewers = msg.viewers; if (story) label(); }
    $('#new-topic').disabled = !msg.topic_ready || msg.changing;
    ctx.onPush?.(p < 0.04 ? null : lead, p);
    // other people's taps light the buttons
    for (const [i, n] of (msg.taps || []).entries()) {
      const el = host.querySelector(`[data-b="${BUTTONS[i]}"]`);
      if (el) el.style.setProperty('--heat', Math.min(1, n / 6).toFixed(2));
    }
    status(!msg.ready ? 'The AI is waking up (loading the model)…'
      : msg.changing ? 'Changing the subject after this sentence…'
      : 'Tap or hold a feeling to push what it says. Taps fade in seconds.');
  }

  // ---- an even flow: words are shown at a steady rhythm (the pace they are written at, measured), even when the
  // network delivers them in clumps. Turn starts and ends wait in line behind the words before them.
  const queue = [];
  let pacer = 0, beatMs = 167;
  const isWord = (m) => m.type === 'w1' && m.w.t != null;
  function play(msg) {
    if (msg.type === 'live_begin') begin(msg);
    else if (msg.type === 'w1') { if (story && msg.id === story.id) { addWord(msg.w, false); syncDoc(); } }
    else if (msg.type === 'live_end') end(msg);
    else if (msg.type === 'live_error') { if (story) { stopFace(); story = null; } }
  }
  // words arrive in small batches; they are released one by one at the writing pace. When the line runs dry,
  // it waits for a small buffer (a few words; more for the lighter, slower-batched stream) before starting again,
  // so the rhythm doesn't stutter at every batch.
  let primed = false;
  function drain() {
    pacer = 0;
    const words = queue.filter(isWord).length, minBuffer = lite ? 10 : 2;
    if (!primed && words < minBuffer && !queue.some((m) => !isWord(m) && m.type !== 'w1')) return;
    primed = true;
    while (queue.length) {
      const msg = queue.shift();
      play(msg);
      if (isWord(msg)) {
        const left = queue.filter(isWord).length;
        // a backlog is worked off a little faster, never in a jump
        const k = left > minBuffer + 12 ? 0.75 : left > minBuffer + 4 ? 0.9 : 1;
        pacer = setTimeout(drain, beatMs * k);
        return;
      }
    }
    primed = false;
  }
  function enqueue(msg) {
    if (msg.type === 'ws') { for (const w of msg.w) queue.push({ type: 'w1', id: msg.id, w }); }
    else queue.push(msg);
    if (!pacer) drain();
  }

  function connect() {
    const sock = ws = new WebSocket(wsUrl());
    let opened = false;
    ws.onopen = () => { retry = 0; refused = 0; opened = true; };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'crowd') crowd(msg);
      else if (msg.type === 'wc') catchUp(msg);                                 // a late joiner's catch-up: at once
      else if (msg.type === 'alts') { const r = altWait.get(`${msg.id}:${msg.i}`); if (r) { altWait.delete(`${msg.id}:${msg.i}`); r(msg); } }
      else if (msg.type === 'mode') { lite = !!msg.lite; label(); }
      else if (msg.type === 'full') { ctx.onFull?.(msg.viewers); on = false; }   // the room is full: ready-made answers instead (leave() cleans up)
      else if (msg.type === 'live_begin' && !queue.length && !pacer) begin(msg);
      else enqueue(msg);
    };
    ws.onclose = () => {
      if (sock !== ws) return;                                                   // an old socket closing late
      clearTimeout(pacer); pacer = 0; queue.length = 0; primed = false;
      if (!on) return;
      if (story) { stopFace(); story = null; }
      // turned away at the door three times (the tunnel caps connections per address, e.g. one busy venue Wi-Fi):
      // ready-made answers instead of a page stuck on "Reconnecting…"
      if (!opened && ++refused >= 3) { ctx.onFull?.(null); on = false; return; }
      status('Reconnecting…');
      setTimeout(() => on && connect(), Math.min(8000, 800 * 2 ** retry++));
    };
  }
  return {
    get on() { return on; },
    newTopic() { if (ws?.readyState === 1) ws.send(JSON.stringify({ type: 'topic' })); $('#new-topic').disabled = true; },
    /** A word's alternatives (with and without the push), fetched when someone inspects it. */
    alts(i) {
      if (!story || !ws || ws.readyState !== 1) return Promise.resolve(null);
      const id = story.id;
      return new Promise((resolve) => {
        altWait.set(`${id}:${i}`, resolve);
        ws.send(JSON.stringify({ type: 'alts', id, i }));
        setTimeout(() => { if (altWait.delete(`${id}:${i}`)) resolve(null); }, 3000);
      });
    },
    enter() {
      if (on) return;
      on = true; refused = 0;
      story = null; shownTalk = null;
      speech.clear();
      status('Connecting…');
      connect();
    },
    leave() {
      if (!on) return;
      on = false;
      clearTimeout(pacer); pacer = 0; queue.length = 0; primed = false; lite = false;
      clearTimeout(tapTimer); tapTimer = 0;
      try { ws?.close(); } catch { /* closed */ }
      ws = null; story = null;
      stopFace();
    },
  };
}
