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
  let ws = null, on = false, story = null, lastListen = 0, retry = 0, shownTalk = null, topic = '', viewers = 0;

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
    ws.send(JSON.stringify({ type: 'tap', b }));
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
    if (!cont) stage.face.react('think');
    const id = msg.id;
    setTimeout(() => { if (on && story?.id === id && !story.ended) stage.face.setActivity({ writing: true }); }, cont ? 150 : 700);
    syncDoc();
  }
  function label() { $('#speech-label').textContent = `Live · talking about ${topic}${viewers > 1 ? ` · ${viewers} people here` : ''}`; }
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
  function tokens(msg) {
    if (!story || msg.id !== story.id) return;
    const instant = !!msg.catchup;
    for (const it of msg.items) {
      if (it.stream === 'plain') story.plain.push(it.tok);
      else if (it.stream === 'steered') addSteered(it.tok, instant);
    }
    if (instant) stage.face.setEmotion(ctx.faceFrom(lift));
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
    bar.querySelectorAll('i').forEach((el) => { el.style.width = `${(msg.mix?.[el.dataset.b] || 0) * p * 100}%`; });
    const lead = BUTTONS.reduce((a, b) => ((msg.mix?.[b] || 0) > (msg.mix?.[a] || 0) ? b : a), BUTTONS[0]);
    $('#crowd-power').textContent = p < 0.04 ? 'Nobody is pushing: it talks as trained'
      : `${p < 0.35 ? 'A gentle' : p < 0.7 ? 'A strong' : 'A full'} push, mostly ${lead === 'unmask' ? 'off the mask' : EMO[lead].label.toLowerCase()}`;
    $('#crowd-viewers').textContent = msg.viewers > 1 ? `${msg.viewers} steering` : 'Just you';
    if (msg.viewers !== viewers) { viewers = msg.viewers; if (story) label(); }
    $('#new-topic').disabled = !msg.topic_ready || msg.changing;
    ctx.onPush?.(p < 0.04 ? null : lead, p);
    // other people's taps light the buttons
    for (const [b, n] of Object.entries(msg.taps || {})) {
      const el = host.querySelector(`[data-b="${b}"]`);
      if (el) el.style.setProperty('--heat', Math.min(1, n / 6).toFixed(2));
    }
    status(!msg.ready ? 'The AI is waking up (loading the model)…'
      : msg.changing ? 'Changing the subject after this sentence…'
      : 'Tap or hold a feeling to push what it says. Taps fade in seconds.');
  }

  // ---- an even flow: words are shown at a steady rhythm (the pace they are written at, measured), even when the
  // network delivers them in clumps. Turn starts and ends wait in line behind the words before them.
  const queue = [];
  let pacer = 0, lastArrive = 0, beatMs = 125;
  function play(msg) {
    if (msg.type === 'live_begin') begin(msg);
    else if (msg.type === 'live_tokens') tokens(msg);
    else if (msg.type === 'live_end') end(msg);
    else if (msg.type === 'live_error') { if (story) { stopFace(); story = null; } }
  }
  function drain() {
    pacer = 0;
    while (queue.length) {
      const msg = queue.shift();
      play(msg);
      if (msg.type === 'live_tokens' && msg.items.some((it) => it.stream === 'steered')) {
        // a backlog (a burst arrived) is worked off a little faster, never in a jump
        const k = queue.length > 8 ? 0.75 : queue.length > 3 ? 0.9 : 1;
        pacer = setTimeout(drain, beatMs * k);
        return;
      }
    }
  }
  function enqueue(msg) {
    if (msg.type === 'live_tokens' && msg.items.some((it) => it.stream === 'steered')) {
      const now = performance.now(), gap = now - lastArrive;
      if (lastArrive && gap > 40 && gap < 600) beatMs += (Math.min(260, Math.max(70, gap)) - beatMs) * 0.15;   // the writing pace
      lastArrive = now;
    }
    queue.push(msg);
    if (!pacer) drain();
  }

  function connect() {
    ws = new WebSocket(wsUrl());
    ws.onopen = () => { retry = 0; };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'crowd') crowd(msg);
      else if (msg.type === 'live_tokens' && msg.catchup) { if (story && story.id === msg.id) tokens(msg); }   // a late joiner's catch-up: at once
      else if (msg.type === 'live_begin' && !queue.length && !pacer) begin(msg);
      else enqueue(msg);
    };
    ws.onclose = () => {
      clearTimeout(pacer); pacer = 0; queue.length = 0;
      if (!on) return;
      if (story) { stopFace(); story = null; }
      status('Reconnecting…');
      setTimeout(() => on && connect(), Math.min(8000, 800 * 2 ** retry++));
    };
  }
  return {
    get on() { return on; },
    newTopic() { if (ws?.readyState === 1) ws.send(JSON.stringify({ type: 'topic' })); $('#new-topic').disabled = true; },
    enter() {
      if (on) return;
      on = true;
      story = null; shownTalk = null;
      speech.clear();
      status('Connecting…');
      connect();
    },
    leave() {
      if (!on) return;
      on = false;
      clearTimeout(pacer); pacer = 0; queue.length = 0;
      try { ws?.close(); } catch { /* closed */ }
      ws = null; story = null;
      stopFace();
    },
  };
}
