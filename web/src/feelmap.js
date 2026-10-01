// The model's own "feeling map": its six emotion directions projected to 2D (PCA, computed from the real
// directions in server/precompute.py). A dot shows where its hidden state leans while it writes.
import { EMO, ORDER } from './palette.js';

export class FeelMap {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext('2d');
    this.map = null; this.dot = { x: 0, y: 0 }; this.target = { x: 0, y: 0 }; this.trail = []; this.focus = null;
    const loop = () => { this.draw(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
  setMap(map) { this.map = map; }
  setFocus(e) { this.focus = e; }
  /** lift: {emotion: 0..1} */
  setState(lift) {
    if (!this.map) return;
    let x = 0, y = 0, w = 0;
    for (const e of ORDER) { const v = Math.max(0, lift[e] || 0); x += this.map[e][0] * v; y += this.map[e][1] * v; w += v; }
    const k = Math.min(1, w); // stay near the centre when nothing is lifted
    this.target = w > 0 ? { x: (x / w) * k, y: (y / w) * k } : { x: 0, y: 0 };
  }
  reset() { this.trail = []; this.target = { x: 0, y: 0 }; }
  draw() {
    const c = this.c, ctx = this.ctx, dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 16;
    ctx.strokeStyle = 'rgba(170,180,230,0.14)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.5, 0, Math.PI * 2); ctx.stroke();
    if (!this.map) return;
    ctx.font = '11px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    // labels: push apart any that would overlap
    const labs = ORDER.map((e) => { const [mx, my] = this.map[e]; return { e, px: cx + mx * R, py: cy - my * R, lx: cx + mx * R * 0.78, ly: cy - my * R * 0.78 }; });
    for (let it = 0; it < 40; it++) for (let i = 0; i < labs.length; i++) for (let j = i + 1; j < labs.length; j++) {
      const a = labs[i], b = labs[j], dx = b.lx - a.lx, dy = b.ly - a.ly;
      if (Math.abs(dy) < 12 && Math.abs(dx) < 46) { const push = (12 - Math.abs(dy)) / 2 + 0.5, s = dy >= 0 ? 1 : -1; a.ly -= push * s; b.ly += push * s; }
    }
    for (const L of labs) {
      ctx.fillStyle = EMO[L.e].color; ctx.globalAlpha = this.focus && this.focus !== L.e ? 0.55 : 1;
      ctx.beginPath(); ctx.arc(L.px, L.py, this.focus === L.e ? 4.5 : 3, 0, Math.PI * 2); ctx.fill();
      ctx.fillText(EMO[L.e].label, Math.max(24, Math.min(W - 24, L.lx)), Math.max(8, Math.min(H - 8, L.ly)));
    }
    ctx.globalAlpha = 1;
    this.dot.x += (this.target.x - this.dot.x) * 0.08; this.dot.y += (this.target.y - this.dot.y) * 0.08;
    const dx = cx + this.dot.x * R, dy = cy - this.dot.y * R;
    this.trail.push([dx, dy]); if (this.trail.length > 60) this.trail.shift();
    ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.beginPath();
    this.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    const g = ctx.createRadialGradient(dx, dy, 0, dx, dy, 12);
    g.addColorStop(0, '#ffffff'); g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(dx, dy, 12, 0, Math.PI * 2); ctx.fill();
  }
}
