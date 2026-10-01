// The answer, written word-piece by word-piece, as subtitles: a fixed box of a few lines whose newest line is
// always the bottom one; older lines rise and fade out of the top. Colour = how strongly the push's emotion shows
// up inside the model at that point (measured, relative to the unpushed answer). Underline = the push changed
// this choice.
import { EMO, ORDER, mix } from './palette.js';
import { pushInfo } from './data.js';

export class Speech {
  constructor(el, { onPick } = {}) {
    this.el = el; this.onPick = onPick;
    this.text = document.createElement('div');   // the text block, bottom-anchored inside the fixed box
    this.text.className = 'speech-text';
    el.appendChild(this.text);
    this.spans = [];
    el.addEventListener('click', (ev) => {
      const s = ev.target.closest('.tok');
      if (!s) return;
      this.text.querySelectorAll('.tok.sel').forEach((x) => x.classList.remove('sel'));
      s.classList.add('sel');
      onPick?.(Number(s.dataset.i), s);
    });
  }

  clear() { this.text.innerHTML = ''; this.caret = null; this.spans = []; }

  begin() {
    this.clear();
    this.caret = document.createElement('span');
    this.caret.className = 'caret';
    this.text.appendChild(this.caret);
  }

  /** Live: keep talking in the same box (a new turn starts on a new line); drop text long scrolled away. */
  continueLine() {
    if (!this.caret) { this.caret = document.createElement('span'); this.caret.className = 'caret'; this.text.appendChild(this.caret); }
    if (this.text.childNodes.length > 1) this.text.insertBefore(document.createElement('br'), this.caret);
    while (this.text.childNodes.length > 600) this.text.firstChild.remove();
    this.text.querySelectorAll('.tok[data-i]').forEach((s) => { delete s.dataset.i; s.classList.add('old'); });
    this.spans = [];
  }

  /** tok: steered token; delta: {emotion: value 0..1} measured lift; focus: the push emotion (or null) */
  add(i, tok, delta, focus, { swingMark = false, instant = false, changed = false } = {}) {
    const parts = tok.t.split('\n');
    parts.forEach((part, k) => {
      if (k > 0) this.text.insertBefore(document.createElement('br'), this.caret);
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
      this.text.insertBefore(s, this.caret);
      this.spans[i] = s;
      if (!instant) setTimeout(() => s.classList.remove('new'), 520);
    });
  }

  /** The answer is done: the last lines simply stay. */
  end() { this.caret?.remove(); this.caret = null; }
}
