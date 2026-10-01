// "How it feels inside": six petals on a circle, one per feeling, growing with the decoded readout (the same
// signal the face shows). Placed like a mood circle: pleasant feelings on the right, unpleasant on the left,
// energetic at the top, quiet at the bottom.
import { EMO, ORDER } from './palette.js';

const NS = 'http://www.w3.org/2000/svg';
const ANGLE = { curiosity: -90, joy: -30, calm: 30, sadness: 90, fear: 150, anger: 210 };   // degrees, 0 = right, + = down
const C = 110, R0 = 20, R1 = 82, HALF = 24;
const pt = (r, a) => [C + r * Math.cos(a * Math.PI / 180), C + r * Math.sin(a * Math.PI / 180)];

function sector(r, a) {
  const [x0, y0] = pt(R0, a - HALF), [x1, y1] = pt(r, a - HALF), [x2, y2] = pt(r, a + HALF), [x3, y3] = pt(R0, a + HALF);
  return `M${x0},${y0} L${x1},${y1} A${r},${r} 0 0 1 ${x2},${y2} L${x3},${y3} A${R0},${R0} 0 0 0 ${x0},${y0} Z`;
}

export class FeelWheel {
  constructor(svg) {
    this.svg = svg; this.cur = {}; this.tgt = {}; this.focus = null;
    svg.setAttribute('viewBox', '0 0 220 220');
    const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); parent.appendChild(e); return e; };
    for (const r of [R0 + (R1 - R0) / 3, R0 + 2 * (R1 - R0) / 3, R1]) el('circle', { cx: C, cy: C, r, fill: 'none', stroke: 'rgba(170,180,230,0.10)' });
    this.parts = {};
    for (const e of ORDER) {
      const a = ANGLE[e], col = EMO[e].color;
      el('path', { d: sector(R1, a), fill: col, 'fill-opacity': 0.05 });            // the full-strength slot, faint
      const petal = el('path', { d: sector(R0 + 0.5, a), fill: col, 'fill-opacity': 0.75 });
      const [lx, ly] = pt(R1 + 15, a);
      const label = el('text', { x: lx, y: ly + 4, 'text-anchor': Math.abs(Math.cos(a * Math.PI / 180)) < 0.2 ? 'middle' : Math.cos(a * Math.PI / 180) > 0 ? 'start' : 'end', fill: col, 'font-size': 12.5, 'fill-opacity': 0.75 });
      label.textContent = EMO[e].label;
      this.parts[e] = { petal, label };
      this.cur[e] = 0; this.tgt[e] = 0;
    }
    this.core = el('circle', { cx: C, cy: C, r: R0 - 5, fill: 'rgba(238,240,247,0.06)', stroke: 'rgba(238,240,247,0.18)' });
    const tick = () => { this.tick(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  setMap() {}
  reset() { for (const e of ORDER) this.tgt[e] = 0; }
  setState(values) { for (const e of ORDER) this.tgt[e] = Math.max(0, Math.min(1, values[e] || 0)); }
  setFocus(e) {
    this.focus = e;
    for (const k of ORDER) {
      const { label, petal } = this.parts[k], on = k === e;
      label.setAttribute('font-weight', on ? 700 : 400); label.setAttribute('fill-opacity', on ? 1 : e ? 0.45 : 0.75);
      petal.style.filter = on ? `drop-shadow(0 0 8px ${EMO[k].color})` : 'none';
    }
  }
  tick() {
    let lead = null, lv = 0;
    for (const e of ORDER) {
      const v = (this.cur[e] += (this.tgt[e] - this.cur[e]) * 0.12);
      this.parts[e].petal.setAttribute('d', sector(R0 + 0.5 + (R1 - R0) * v, ANGLE[e]));
      if (v > lv) { lv = v; lead = e; }
    }
    this.core.setAttribute('fill', lead && lv > 0.15 ? EMO[lead].color : 'rgba(238,240,247,0.06)');
    this.core.setAttribute('fill-opacity', lead && lv > 0.15 ? String(0.25 + 0.5 * lv) : '1');
  }
}
