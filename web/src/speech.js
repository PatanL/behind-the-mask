// The answer, written word-piece by word-piece. Colour = how strongly the push's emotion shows up inside the
// model at that point (measured, relative to the unpushed answer). Underline = the push changed this choice.
import { EMO, ORDER, mix } from './palette.js';
import { pushInfo } from './data.js';

export class Speech {
  constructor(el, { onPick } = {}) {
    this.el = el; this.onPick = onPick;
    el.addEventListener('click', (ev) => {
      const s = ev.target.closest('.tok');
      if (!s) return;
      this.el.querySelectorAll('.tok.sel').forEach((x) => x.classList.remove('sel'));
      s.classList.add('sel');
      onPick?.(Number(s.dataset.i), s);
    });
  }

  // clear(): an empty page for a static render (shown in full); begin(): a live subtitle window; end(): full text
  clear() { this.el.innerHTML = ''; this.caret = null; this.spans = []; this.el.classList.add('full'); this.el.classList.remove('scrolled'); this.el.scrollTop = 0; }

  begin() {
    this.clear();
    this.el.classList.remove('full');
    this.caret = document.createElement('span');
    this.caret.className = 'caret';
    this.el.appendChild(this.caret);
  }

  /** tok: steered token; delta: {emotion: value 0..1} measured lift; focus: the push emotion (or null) */
  add(i, tok, delta, focus, { swingMark = false, instant = false, changed = false } = {}) {
    const parts = tok.t.split('\n');
    parts.forEach((part, k) => {
      if (k > 0) this.el.insertBefore(document.createElement('br'), this.caret);
      if (!part) return;
      const s = document.createElement('span');
      s.className = instant ? (changed ? 'tok changed' : 'tok') : 'tok new';
      s.dataset.i = i;
      s.textContent = part;
      // colour: only the pushed feeling, by how strongly it measures inside the model here (smoothed upstream)
      const v = focus ? Math.min(1, delta[focus] || 0) : 0;
      if (focus && v > 0.08) {
        const c = EMO[focus].color;
        s.style.color = mix('#eef0f7', c, 0.2 + 0.7 * v);
        s.style.textShadow = `0 0 ${Math.round(4 + 18 * v)}px ${c}${Math.round(30 + 110 * v).toString(16).padStart(2, '0')}`;
      }
      const pi = pushInfo(tok);
      if (pi?.shown) { s.classList.add('pushed'); s.style.setProperty('--c', focus ? EMO[focus]?.color : '#9fb4ff'); }
      if (swingMark && k === 0) s.classList.add('swing-mark');
      this.el.insertBefore(s, this.caret);
      this.spans[i] = s;
      if (!instant) setTimeout(() => s.classList.remove('new'), 520);
    });
    if (instant && this.el.classList.contains('full')) return;
    // keep the newest line in view but let early lines stay visible as long as possible
    const over = this.el.scrollHeight - this.el.clientHeight;
    if (over > 0) { if (instant) this.el.scrollTop = over; else this.el.scrollTo({ top: over, behavior: 'smooth' }); this.el.classList.add('scrolled'); }
  }

  end() { this.caret?.remove(); this.caret = null; this.el.classList.add('full'); this.el.scrollTop = 0; }
}
