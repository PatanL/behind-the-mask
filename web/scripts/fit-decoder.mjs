// Fit the face's emotion decoder: a 6x6 linear map from the smoothed readout lift (steered minus the unpushed
// answer's mean, per direction) to the feeling the face should show. The raw readout has cross-talk (e.g. the
// anger direction also rises on fear and sadness pushes), so a face driven by it directly shows every
// negative feeling at once. We fit, with ridge regression, the map that best recovers which feeling was pushed
// and how hard, from every token of every pre-computed performance.
//   node scripts/fit-decoder.mjs public/performances      -> writes <dir>/decoder.json
import fs from 'node:fs';
import path from 'node:path';

const dir = process.argv[2] || 'public/performances';
const ORDER = ['joy', 'sadness', 'anger', 'fear', 'calm', 'curiosity'];
const STRENGTH = { little: 0.4, mid1: 0.55, lot: 0.72, mid2: 0.86, toomuch: 1.0 };
const RATE = 0.2, WARM = 8, LAMBDA = 40;

const X = [], Y = [], runs = [];
for (const f of fs.readdirSync(dir)) {
  if (!f.includes('__') || !f.endsWith('.json')) continue;
  const d = JSON.parse(fs.readFileSync(path.join(dir, f)));
  const pl = d.streams?.plain?.tokens, st = d.streams?.steered?.tokens;
  if (!pl?.length || !st?.length) continue;
  const emo = d.emotion, lv = d.level;
  if (emo === 'unmask' || emo.includes('>')) continue;      // not a single feeling
  const y = ORDER.map((e) => (e === emo ? STRENGTH[lv] ?? 0 : 0));
  const mu = ORDER.map((_, k) => pl.reduce((s, t) => s + t.e[k], 0) / pl.length);
  const ema = ORDER.map(() => 0), rows = [];
  st.forEach((tok, i) => {
    ORDER.forEach((_, k) => { ema[k] += (tok.e[k] - mu[k] - ema[k]) * RATE; });
    if (i >= WARM) { X.push([...ema]); Y.push(y); rows.push([...ema]); }
  });
  runs.push({ key: `${emo}/${lv}`, rows });
}

// ridge: M = (X'X + lambda I)^-1 X'Y
const n = ORDER.length;
const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? LAMBDA : 0)));
const B = Array.from({ length: n }, () => Array(n).fill(0));
for (let r = 0; r < X.length; r++) for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) { A[i][j] += X[r][i] * X[r][j]; B[i][j] += X[r][i] * Y[r][j]; } }
function solve(A, B) {   // Gauss-Jordan, A n x n, B n x m
  const a = A.map((r, i) => [...r, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    const pv = a[c][c];
    for (let j = 0; j < a[c].length; j++) a[c][j] /= pv;
    for (let r = 0; r < n; r++) if (r !== c) { const k = a[r][c]; for (let j = 0; j < a[r].length; j++) a[r][j] -= k * a[c][j]; }
  }
  return a.map((r) => r.slice(n));
}
const M = solve(A, B);
const raw = (v) => ORDER.map((_, j) => v.reduce((s, x, i) => s + x * M[i][j], 0));
// regression to the mean shrinks the outputs (each token is noisy), so rescale each feeling so that its own
// pushes read at their intended strength, after removing a small noise floor
const FLOOR = 0.05;
const gain = ORDER.map((_, j) => {
  let num = 0, den = 0;
  for (let r = 0; r < X.length; r++) if (Y[r][j] > 0) { const d = Math.max(0, raw(X[r])[j] - FLOOR); num += d * Y[r][j]; den += d * d; }
  return den ? num / den : 1;
});
const dec = (v) => raw(v).map((x, j) => (x - FLOOR) * gain[j]);

// report: mean decoded feeling per push (clamped), and its spread within a run
const agg = {};
for (const r of runs) {
  const D = r.rows.map((v) => dec(v).map((x) => Math.max(0, Math.min(1, x))));
  const mean = ORDER.map((_, j) => D.reduce((s, x) => s + x[j], 0) / D.length);
  const sd = ORDER.map((_, j) => Math.sqrt(D.reduce((s, x) => s + (x[j] - mean[j]) ** 2, 0) / D.length));
  (agg[r.key] ||= []).push([mean, sd]);
}
console.log('push'.padEnd(18), ORDER.map((e) => e.slice(0, 5).padStart(6)).join(''), '   | sd within run');
for (const [k, v] of Object.entries(agg).sort()) {
  const m = ORDER.map((_, j) => v.reduce((s, x) => s + x[0][j], 0) / v.length);
  const s = ORDER.map((_, j) => v.reduce((a, x) => a + x[1][j], 0) / v.length);
  console.log(k.padEnd(18), m.map((x) => x.toFixed(2).padStart(6)).join(''), '   |', s.map((x) => x.toFixed(2).padStart(5)).join(''));
}
const out = { order: ORDER, rate: RATE, floor: FLOOR, gain: gain.map((g) => +g.toFixed(4)), M: M.map((r) => r.map((x) => +x.toFixed(5))), note: 'face = clamp((smoothedLift . M - floor) * gain); see scripts/fit-decoder.mjs' };
console.log('gain', gain.map((g) => g.toFixed(2)).join(' '));
fs.writeFileSync(path.join(dir, 'decoder.json'), JSON.stringify(out));
console.log('wrote', path.join(dir, 'decoder.json'));
