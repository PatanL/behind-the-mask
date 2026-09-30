// "Inside the AI": the stack of layers, where the push enters, and live pulses per written word-piece.
import { EMO, ORDER } from './palette.js';

const NS = 'http://www.w3.org/2000/svg';
export class Spine {
  constructor(svg, meters) {
    this.svg = svg; this.meters = meters; this.pulses = [];
    this.nLayers = 36; this.layer = 18; this.color = EMO.none.color;
    this.build();
    ORDER.forEach((e) => {
      const m = document.createElement('div');
      m.className = 'meter'; m.style.setProperty('--c', EMO[e].color);
      m.innerHTML = `<span>${EMO[e].label}</span><div class="bar"><i></i></div>`;
      meters.appendChild(m);
    });
    this.bars = [...meters.querySelectorAll('.bar i')];
    requestAnimationFrame((t) => this.tick(t));
  }
  configure(nLayers, layer) { this.nLayers = nLayers; this.layer = layer; this.build(); }
  build() {
    const s = this.svg; s.innerHTML = '';
    const top = 34, bot = 388, n = this.nLayers;
    this.y = (i) => bot - (i / (n - 1)) * (bot - top);
    const g = document.createElementNS(NS, 'g');
    for (let i = 0; i < n; i++) {
      const e = document.createElementNS(NS, 'ellipse');
      e.setAttribute('cx', 60); e.setAttribute('cy', this.y(i)); e.setAttribute('rx', 30 + 8 * Math.sin((i / n) * Math.PI)); e.setAttribute('ry', 3.2);
      e.setAttribute('fill', 'none'); e.setAttribute('stroke', 'rgba(170,180,230,0.22)'); e.setAttribute('stroke-width', 1);
      g.appendChild(e);
    }
    s.appendChild(g); this.rings = [...g.children];
    const lab = (y, txt, anchor = 'middle') => { const t = document.createElementNS(NS, 'text'); t.setAttribute('x', 60); t.setAttribute('y', y); t.setAttribute('text-anchor', anchor); t.setAttribute('fill', '#6b7290'); t.setAttribute('font-size', 13); t.textContent = txt; s.appendChild(t); return t; };
    lab(bot + 18, 'words in');
    lab(top - 12, 'next word out');
    this.inj = document.createElementNS(NS, 'ellipse');
    this.inj.setAttribute('cx', 60); this.inj.setAttribute('cy', this.y(this.layer)); this.inj.setAttribute('rx', 44); this.inj.setAttribute('ry', 6);
    this.inj.setAttribute('fill', 'none'); this.inj.setAttribute('stroke-width', 2.5);
    s.appendChild(this.inj);
    this.injLabel = document.createElementNS(NS, 'text');
    this.injLabel.setAttribute('x', 60); this.injLabel.setAttribute('y', this.y(this.layer) - 12); this.injLabel.setAttribute('text-anchor', 'middle'); this.injLabel.setAttribute('font-size', 13); this.injLabel.setAttribute('font-weight', '600');
    s.appendChild(this.injLabel);
    this.setPush(null);
  }
  setPush(emotion) {
    const c = emotion && EMO[emotion] ? EMO[emotion].color : 'rgba(170,180,230,0.35)';
    this.color = emotion && EMO[emotion] ? EMO[emotion].color : EMO.none.color;
    this.inj.setAttribute('stroke', c);
    this.inj.style.filter = emotion ? `drop-shadow(0 0 6px ${c})` : 'none';
    this.injLabel.setAttribute('fill', emotion ? c : '#6b7290');
    this.injLabel.textContent = emotion ? `+ ${EMO[emotion].label.toLowerCase()} here` : `layer ${this.layer} of ${this.nLayers}`;
  }
  pulse() {
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', 60); c.setAttribute('r', 3.2); c.setAttribute('fill', '#dfe4ff');
    this.svg.appendChild(c);
    this.pulses.push({ el: c, t0: performance.now() });
  }
  setMeters(values) { ORDER.forEach((e, i) => { this.bars[i].style.width = `${Math.round(Math.max(0, Math.min(1, values[e] || 0)) * 100)}%`; }); }
  tick(t) {
    const dur = 520;
    this.pulses = this.pulses.filter((p) => {
      const k = (t - p.t0) / dur;
      if (k >= 1) { p.el.remove(); return false; }
      const li = k * (this.nLayers - 1);
      p.el.setAttribute('cy', this.y(li));
      const after = li >= this.layer;
      p.el.setAttribute('fill', after ? this.color : '#dfe4ff');
      p.el.setAttribute('r', after ? 4 : 3);
      p.el.style.filter = after ? `drop-shadow(0 0 5px ${this.color})` : 'none';
      p.el.setAttribute('opacity', String(1 - Math.max(0, k - 0.8) * 5));
      return true;
    });
    requestAnimationFrame((tt) => this.tick(tt));
  }
}
