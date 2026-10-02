// Shared bits for every launchpad page: the header, the footer, number formats.
import { EMO, ORDER } from '../../src/palette.js';
export const $ = (s, el = document) => el.querySelector(s);
export const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) { if (k === 'class') el.className = v; else if (k === 'text') el.textContent = v; else if (v != null) el.setAttribute(k, v); }
  for (const k of kids.flat()) if (k != null) el.append(k);
  return el;
};
export function header(on) {
  const base = import.meta.env.BASE_URL;
  const el = h('header', { class: 'top' },
    h('a', { class: 'brand', href: base }, h('span', { class: 'eye' }), 'Steer AI'),
    h('nav', { class: 'nav' }, ...[['Live', base], ['Explore', `${base}explore.html`], ['Launch', `${base}launch.html`], ['Docs', `${base}docs.html`]].map(([t, u]) => h('a', { href: u, class: on === t ? 'on' : '' }, t))),
    h('div', { class: 'sp' }),
    h('a', { class: 'btn primary', href: `${base}launch.html` }, 'Launch a coin'));
  document.body.prepend(el);
}
export function footer() {
  const base = import.meta.env.BASE_URL;
  document.body.append(h('footer', {}, h('span', {}, 'Steer AI · every coin has a live android'), h('a', { href: base }, 'Live'), h('a', { href: `${base}explore.html` }, 'Explore'), h('a', { href: `${base}launch.html` }, 'Launch'),
    h('a', { href: `${base}docs.html` }, 'Docs'), h('a', { href: `${base}exhibit/` }, 'The exhibit')));
}
export const sol = (x, d = 3) => (x == null ? '—' : `${x < 0.001 && x > 0 ? x.toExponential(1) : x.toFixed(d)} SOL`);
export const pct = (x) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(x !== 0 && Math.abs(x) < 0.1 ? 1 : 0)}%`);
export function left(s) {
  if (s == null) return 'always awake';
  if (s <= 0) return 'asleep';
  if (s < 3600) return `${Math.ceil(s / 60)} min`;
  if (s < 86400 * 2) return `${(s / 3600).toFixed(1)} h`;
  return `${Math.round(s / 86400)} days`;
}
export function feelBar(el, shown) {
  if (!el.childElementCount) for (const e of ORDER) el.append(h('i', { style: `background:${EMO[e].color};width:0%` }));
  const tot = ORDER.reduce((a, e) => a + (shown[e] || 0), 0) || 1;
  [...el.children].forEach((i, k) => { i.style.width = `${((shown[ORDER[k]] || 0) / Math.max(1, tot)) * 100}%`; });
}
export const STATUS = { asleep: 'Asleep', waking: 'Waking up', awake: 'Awake', thinking: 'Thinking', 'losing it': 'Losing it' };
