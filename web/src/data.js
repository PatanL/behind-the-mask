// Pre-computed performances (see server/precompute.py). Each one holds four streams token by token.
const cache = new Map();
//  lets a new data set be previewed before it replaces the live one
const DATA = new URLSearchParams(location.search).get('data') || 'performances';
let indexPromise = null;
export function loadIndex() {
  const B = import.meta.env.BASE_URL;
  indexPromise ||= fetch(`${B}${DATA}/index.json`).then((r) => r.json());
  return indexPromise;
}
export async function loadPerformance(id) {
  if (!cache.has(id)) cache.set(id, fetch(`${import.meta.env.BASE_URL}${DATA}/${id}.json`).then((r) => r.json()));
  return cache.get(id);
}
export function performanceId(index, qid, emotion, level) {
  const q = index.questions.find((x) => x.id === qid);
  if (!q) return null;
  return q.performances[`${emotion}|${level}`] || null;
}
/** Probability the model gave `tok` in an alternatives list [[tok, p], ...]; if absent, an upper bound. */
export function probIn(alts, tok) {
  if (!alts || !alts.length) return null;
  const hit = alts.find(([t]) => t === tok);
  return hit ? { p: hit[1], exact: true } : { p: alts[alts.length - 1][1], exact: false };
}
const STOP = new Set(['the', 'and', 'but', 'for', 'you', 'your', 'are', 'was', 'were', 'with', 'that', 'this', 'have', 'has', 'had', 'not', 'can', 'its', "it's", 'all', 'any', 'our', 'out', 'too', 'very', 'just', 'what', 'when', 'from', 'they', 'them', 'there', 'then', 'than', 'been', 'will', 'would', 'could', 'should', 'into', 'about', "i'm", "i've", "i'd", "i'll", 'also', 'some', 'more', 'most', 'like', 'here', 'how', 'who', 'why', 'let', 'get', 'got']);
/** Did the push change this choice? (the chosen word became >= 2.5x likelier than without the push) */
export function pushInfo(tok) {
  if (!tok.cf) return null;
  const cf = probIn(tok.cf, tok.t);
  const ratio = tok.p / Math.max(cf.p, 1e-4);
  const cfTop = tok.cf[0][0];
  const tipped = tok.t.trim() !== '' && tok.p >= 0.08 && ratio >= 2.5 && cfTop !== tok.t;
  // underline only content words where the push flipped the favourite (function words shift too, but distract)
  const w = tok.t.trim().toLowerCase().replace(/[^a-z']/g, '');
  const shown = tipped && w.length >= 4 && !STOP.has(w) && tok.p >= 0.2 && ratio >= 6;
  return { ratio, cfP: cf.p, cfExact: cf.exact, cfTop, pushed: tipped, shown };
}

export const DATA_DIR = DATA;
