/** Text-timed articulation, not audio/phoneme lip sync or an emotion measurement.
 * The queue is bounded in time and size. New text replaces excessive FUTURE
 * work, never the currently playing shape. An empty stream stays quiet.
 */
const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, Number.isFinite(v) ? v : lo));
// Open vowels show the upper teeth (upper-lip raise ~0.35-0.45: below ~0.3 the teeth stay hidden, a puppet's mouth)
// and F/V rests the upper teeth on a lightly tucked lower lip: measured against film performances, the android's
// upper-lip raise was ~0.1x the actors'.
const POSES = Object.freeze({
  A: { jawOpen: .23, mouthLowerDownLeft: .12, mouthLowerDownRight: .12, mouthUpperUpLeft: .45, mouthUpperUpRight: .45 },
  E: { jawOpen: .10, mouthStretchLeft: .16, mouthStretchRight: .16, mouthUpperUpLeft: .35, mouthUpperUpRight: .35 },
  O: { jawOpen: .16, mouthFunnel: .28, mouthPucker: .12 },
  U: { jawOpen: .04, mouthPucker: .38, mouthFunnel: .17 },
  M: { jawOpen: 0, mouthPressLeft: .32, mouthPressRight: .32, mouthClose: .18 },
  F: { jawOpen: .025, mouthRollLower: .12, mouthUpperUpLeft: .30, mouthUpperUpRight: .30 },
  S: { jawOpen: .045, mouthStretchLeft: .065, mouthStretchRight: .065 },
  SH: { jawOpen: .06, mouthFunnel: .21, mouthPucker: .12 },
  REST: { jawOpen: 0 },
});
const ORAL = [...new Set(Object.values(POSES).flatMap(Object.keys))];
export function spellingShapes(text) {
  if (typeof text !== 'string') return [];
  const chars = Array.from(text.slice(0, 256).toLowerCase());
  const out = [];
  const add = s => { if (out.at(-1) !== s) out.push(s); };
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i], next = chars[i + 1];
    if ((c === 's' || c === 'c') && next === 'h') { add('SH'); i++; }
    else if (c === 'o' && ['o', 'u', 'w'].includes(next)) { add('U'); i++; }
    else if (c === 'a') add('A');
    else if ('eiy'.includes(c)) add('E');
    else if (c === 'o') add('O');
    else if ('uw'.includes(c)) add('U');
    else if ('mbp'.includes(c)) add('M');
    else if ('fv'.includes(c)) add('F');
    else if (/[a-z]/.test(c)) add('S');
    else if (/[.,!?;:—]/.test(c)) add('REST');
    // Whitespace, redactions and unsupported scripts produce NO invented syllables.
  }
  return out;
}
export class SpeechMotion {
  constructor({ maxLead = .42 } = {}) {
    this.maxLead = clamp(maxLead, .18, .8);
    this.queue = []; this.channels = {}; this.weight = 0; this.last = -Infinity;
  }
  enqueue(text, now, duration = .15) {
    if (!Number.isFinite(now)) return false;
    let shapes = spellingShapes(text);
    if (!shapes.length) return false;
    this.queue = this.queue.filter(x => x.end > now);
    const active = this.queue.find(x => x.start <= now && x.end > now);
    let start = Math.max(now, this.queue.at(-1)?.end ?? now);
    if (start > now + .16) {
      // Keep only the current articulation before catching up to the newest text.
      this.queue = active ? [active] : [];
      start = active?.end ?? now;
    }
    const slot = Math.min(clamp(duration, .06, .28), Math.max(.03, now + this.maxLead - start));
    const count = Math.max(1, Math.min(4, Math.floor(slot / .045)));
    if (shapes.length > count) {
      const picks = Array.from({ length: count }, (_, i) => shapes[Math.round(i * (shapes.length - 1) / Math.max(1, count - 1))]);
      // Preserve a visually important lip closure where practical.
      if (count > 1 && shapes.includes('M') && !picks.includes('M')) picks[0] = 'M';
      shapes = picks;
    }
    shapes.forEach((shape, i) => this.queue.push({ shape, start: start + i * slot / shapes.length, end: start + (i + 1) * slot / shapes.length }));
    this.queue = this.queue.slice(-16); this.last = now;
    return true;
  }
  clear({ hard = false } = {}) {
    this.queue = []; this.last = -Infinity;
    if (hard) { this.channels = {}; this.weight = 0; }
  }
  sample(dt, now) {
    dt = clamp(dt, 0, .1);
    this.queue = this.queue.filter(x => x.end > now);
    const current = this.queue.find(x => x.start <= now && now < x.end);
    const target = current ? POSES[current.shape] : {};
    // Fast approach, softer release. No second idle syllable generator.
    const alpha = 1 - Math.exp(-dt / (current ? .027 : .085));
    this.weight += ((current ? 1 : 0) - this.weight) * alpha;
    for (const key of ORAL) this.channels[key] = (this.channels[key] || 0) + ((target[key] || 0) - (this.channels[key] || 0)) * alpha;
    return { channels: { ...this.channels }, weight: clamp(this.weight), shape: current?.shape ?? null };
  }
}
/** Speech owns oral articulation; expression keeps the upper face and most smile.
 * In particular, M/B/P closure is not merely added on top of an emotional open jaw.
 */
export function composeSpeech(out, frame, gain = 1) {
  const w = clamp(frame?.weight ?? 0), g = clamp(gain, .55, 1.35);
  if (w < 1e-4) return out;
  const ch = frame.channels;
  for (const key of ORAL) {
    const base = clamp(out[key] || 0);
    out[key] = clamp(base * (1 - w) + (ch[key] || 0) * g, 0, key === 'jawOpen' ? .5 : 1);
  }
  // Keep emotional smile corners, but prevent a broad smile fighting round/closed shapes.
  const round = clamp(((ch.mouthPucker || 0) + (ch.mouthPressLeft || 0)) * 1.3);
  for (const key of ['mouthSmileLeft', 'mouthSmileRight']) out[key] = clamp(out[key] || 0) * (1 - .45 * round);
  return out;
}
