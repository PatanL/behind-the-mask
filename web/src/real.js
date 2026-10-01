// "Is the reading real?": graded sentences where only one thing changes (no steering). If the readout means
// something, the watched feeling should rise as the danger / good news / annoyance grows.
import { EMO } from './palette.js';
import { DATA_DIR } from './data.js';

export async function initRealSection(host) {
  let data = null;
  try { const r = await fetch(`${import.meta.env.BASE_URL}${DATA_DIR}/validation.json`); data = r.ok ? await r.json() : null; } catch { data = null; }
  if (!data?.demos?.length) return;
  // lead with the clearest effect (largest rise of the watched signal, assistant model)
  const rise = (d) => { const v = (d.chat || []).map((r) => r.z[d.watch]); return v.length ? v[v.length - 1] - v[0] : 0; };
  data.demos.sort((a, b) => rise(b) - rise(a));
  host.hidden = false;
  const pick = host.querySelector('.real-pick'), slider = host.querySelector('input[type=range]'), sent = host.querySelector('.real-sentence');
  const chart = host.querySelector('.real-chart'), note = host.querySelector('.real-note');
  let demo = data.demos[0];
  data.demos.forEach((d, i) => {
    const b = document.createElement('button'); b.className = 'chip'; b.textContent = { bear: 'A bear gets closer', exam: 'Exam results', water: 'Rising floodwater', queue: 'Stuck on hold' }[d.id] || d.id;
    b.setAttribute('aria-pressed', String(i === 0));
    b.onclick = () => { demo = d; pick.querySelectorAll('.chip').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); slider.max = d.values.length - 1; slider.value = 0; draw(); };
    pick.appendChild(b);
  });
  slider.max = demo.values.length - 1;
  slider.oninput = draw;
  function draw() {
    const i = Number(slider.value), w = demo.watch, c = EMO[w].color;
    slider.style.accentColor = c;
    const parts = demo.template.split('{}');
    sent.innerHTML = `${parts[0]}<b style="color:${c}">${demo.values[i]}</b>${parts[1] || ''}`;
    const series = (mind) => (demo[mind] || []).map((r) => r.z[w]);
    const a = series('chat'), b = series('base');
    const all = [...a, ...b], lo = Math.min(...all) - 0.5, hi = Math.max(...all) + 0.5;
    const W = 520, H = 170, px = (k) => 30 + (k / (demo.values.length - 1)) * (W - 50), py = (v) => H - 24 - ((v - lo) / (hi - lo)) * (H - 44);
    const line = (vals, col, dash = '') => `<polyline fill="none" stroke="${col}" stroke-width="2.5" ${dash} points="${vals.map((v, k) => `${px(k)},${py(v)}`).join(' ')}"/>` +
      vals.map((v, k) => `<circle cx="${px(k)}" cy="${py(v)}" r="${k === i ? 6 : 3.5}" fill="${col}"/>`).join('');
    chart.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%"><g font-family="Inter" font-size="11" fill="#6b7290">
      <text x="30" y="${H - 6}">${demo.values[0]}</text><text x="${W - 20}" y="${H - 6}" text-anchor="end">${demo.values[demo.values.length - 1]}</text>
      <text x="4" y="14">${EMO[w].label.toLowerCase()} signal</text></g>
      <line x1="${px(i)}" x2="${px(i)}" y1="10" y2="${H - 24}" stroke="rgba(255,255,255,.12)"/>
      ${line(b, 'rgba(170,176,198,.55)', 'stroke-dasharray="4 4"')}${line(a, c)}</svg>`;
    const up = (s) => s.length > 1 && s[s.length - 1] > s[0];
    note.innerHTML = `<span style="color:${c}">━</span> assistant &nbsp; <span style="color:#aab0c6">┅</span> base model. No steering here: we only measure. ` +
      (up(a) || up(b) ? `The ${EMO[w].label.toLowerCase()} signal rises as the sentence gets more ${w === 'joy' ? 'joyful' : w === 'fear' ? 'frightening' : 'maddening'}, though not perfectly smoothly.` : '');
  }
  draw();
}
