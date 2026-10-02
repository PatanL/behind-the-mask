// Keeps the android's motion smooth on any device: watches the frame times and steps the render quality down
// (resolution, anti-aliasing, glow, shadow detail) when frames start arriving late, and back up when there's room.
// Steady motion matters more than sharpness: a phone at a lower resolution reads better than a sharp face that stutters.
// ?perf=1 shows the frame rate and the current quality step in a corner.

const SHOW = typeof location !== 'undefined' && new URLSearchParams(location.search).has('perf');

export class Governor {
  /** tiers: quality steps, best first; apply(tier, i) switches to one; start: the step to begin at. */
  constructor({ tiers, apply, start = 0, name = 'stage' }) {
    Object.assign(this, { tiers, apply, name });
    this.tier = Math.min(start, tiers.length - 1);
    this.floor = 0;          // the best step it may climb back to (raised when a climb didn't hold)
    this.win = []; this.last = 0; this.calm = 0; this.hold = 0; this.upAt = -1e9;
    this.period = Infinity;  // the display's refresh period (the shortest steady frame time seen)
    this.armed = false; this.t0 = 0;
    this.apply(this.tiers[this.tier], this.tier);
    if (SHOW) { this.hud = document.createElement('div'); this.hud.style.cssText = 'position:fixed;right:8px;bottom:8px;z-index:9999;padding:6px 9px;font:11px/1.4 ui-monospace,monospace;color:#7fe7ff;background:rgba(3,4,10,.8);border:1px solid rgba(127,231,255,.3);border-radius:3px;pointer-events:none;white-space:pre'; document.body.append(this.hud); }
  }

  /** Start judging (after loading: compiling and parsing stall the first frames, and that's not the drawing). */
  arm(delay = 2500) { setTimeout(() => { this.armed = true; this.win.length = 0; }, delay); }

  /** Call once per rendered frame with performance.now(). */
  frame(t) {
    const dt = t - this.last; this.last = t;
    if (!this.armed || dt <= 0 || dt > 1500 || document.hidden) { this.win.length = 0; this.t0 = t; return; }   // loading, a hidden tab
    this.win.push(dt);
    if (this.win.length < 60 && !(this.win.length >= 6 && t - this.t0 > 1500)) return;   // a second or so of frames
    this.t0 = t;
    const s = this.win.slice().sort((a, b) => a - b), n = s.length, med = s[n >> 1], p10 = s[Math.floor(n * 0.1)], p90 = s[Math.floor(n * 0.9)];
    this.period = Math.min(this.period, med);                                  // the display's refresh, as the best windows show it
    const hic = this.win.filter((x) => x > med * 1.5 && x < 120).length / n;   // hiccups: the visible stutter (one-off stalls are loading)
    const slow = med > 22 && (p90 / p10 > 1.25 || hic > 0.03);                 // under ~45 fps and uneven (a steady 30 Hz cap is left alone)
    this.win.length = 0;
    if (this.hud) this.hud.textContent = `${this.name}  ${(1000 / med).toFixed(0)} fps  hiccups ${(hic * 100).toFixed(0)}%\nquality ${this.tiers.length - this.tier}/${this.tiers.length}`;
    if (t < this.hold) return;
    if ((hic > 0.08 || slow) && this.tier < this.tiers.length - 1) {
      if (t - this.upAt < 8000) this.floor = Math.min(this.tiers.length - 1, this.tier + 1);   // the last climb didn't hold: stay below it
      this.set(this.tier + 1, t); this.calm = 0;
    } else if (hic < 0.02 && med < Math.min(this.period, 17) * 1.15) {
      if (++this.calm >= 10 && this.tier > this.floor) { this.set(this.tier - 1, t); this.calm = 0; this.upAt = t; }
    } else this.calm = 0;
  }

  set(i, t) { this.tier = i; this.hold = t + 1200; this.apply(this.tiers[i], i); }
}

/** A phone or tablet starts a couple of steps down (it can climb back up). */
export const startTier = () => (typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches ? 2 : 0);
