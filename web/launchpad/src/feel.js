// What the android's face shows, from the model's readout on each word -- the exhibit's own mapping
// (web/src/main.js: faceFrom / performToken / beats), shared by every launchpad page.
import { ORDER } from '../../src/palette.js';
let DECODER = null;
export const decoderReady = fetch(`${import.meta.env.BASE_URL}decoder.json`).then((r) => (r.ok ? r.json() : null)).then((j) => { DECODER = j; }).catch(() => {});

/** readout lift (signed, smoothed, minus the android's own baseline) -> the six feelings the face shows */
export function faceFrom(lift) {
  let v;
  if (DECODER) {
    const { M, floor, gain } = DECODER;
    v = ORDER.map((_, j) => Math.max(0, (ORDER.reduce((s, _e, i) => s + lift[i] * M[i][j], 0) - floor) * gain[j]));
  } else v = ORDER.map((_, k) => Math.max(0, lift[k] / 5.6));
  return Object.fromEntries(ORDER.map((e, j) => [e, Math.min(1, 1.3 * Math.pow(Math.max(0, v[j] - 0.04) / 0.96, 0.8))]));
}

const STRESS = new Set(['no', 'not', 'never', 'nothing', 'nobody', 'none', 'always', 'every', 'everything', 'everyone', 'all', 'only', 'very',
  'really', 'too', 'must', 'cannot', "can't", "won't", "don't", "didn't", "isn't", 'why', 'what', 'how', 'who', 'yes', 'please', 'now', 'ever',
  'still', 'alone', 'enough']);
const PLAIN_LONG = new Set(['because', 'through', 'without', 'something', 'anything', 'another', 'whether', 'however', 'actually', 'probably', 'together']);
function stressed(t) {
  if (!/^\s/.test(t)) return false;
  const w = t.trim().toLowerCase().replace(/[^a-z']/g, '');
  return STRESS.has(w) || (w.length >= 7 && !PLAIN_LONG.has(w));
}

/** A performer: feed it words ({t, e: [6 feelings + concept]}) and it drives `face`. */
export class Performer {
  constructor(face) { this.face = face; this.reset(); }
  reset(mu) { this.mu = mu || [0, 0, 0, 0, 0, 0, 0]; this.lift = ORDER.map(() => 0); this.i = 0; this.lastEmph = -10; this.shown = Object.fromEntries(ORDER.map((e) => [e, 0])); }
  word(tok, sayDur = 0.13) {
    const rate = DECODER?.rate ?? 0.2;
    ORDER.forEach((_, k) => { this.lift[k] += ((tok.e?.[k] ?? 0) - (this.mu[k] ?? 0) - this.lift[k]) * rate; });
    this.shown = faceFrom(this.lift);
    const f = this.face, t = tok.t || '';
    f.setEmotion(this.shown, { intensity: 1 });
    if (/\?\s*$/.test(t)) f.beat('question');
    else if (/!\s*$/.test(t)) f.beat('exclaim');
    else if (/[.]\s*$/.test(t) && !/\.\.\s*$/.test(t)) f.beat('period');
    else if (/[,;:—]\s*$/.test(t)) f.beat('comma');
    else if (stressed(t) && this.i - this.lastEmph > 2) { f.beat('emphasis'); this.lastEmph = this.i; }
    f.say(t, sayDur);
    this.i++;
    return this.shown;
  }
}
