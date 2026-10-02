// Shared bits for every launchpad page: the header, the footer, number formats.
import { EMO, ORDER } from '../../src/palette.js';
import { pick, onWallet } from './wallet.js';
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
    h('a', { class: 'top-link', href: `${base}docs.html` }, 'How it works'),
    h('a', { class: 'top-link x-link', href: 'https://x.com/steerailive', target: '_blank', rel: 'noopener', 'aria-label': 'Steer AI on X', title: 'Steer AI on X (@steerailive)' }),
    h('button', { class: 'btn wallet-b', id: 'wallet-top', type: 'button' }, 'Connect wallet'),
    h('a', { class: 'btn primary', href: `${base}launch.html` }, 'Launch a coin'));
  el.querySelector('.x-link').innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/></svg>`;
  document.body.prepend(el);
  const b = el.querySelector('#wallet-top');
  onWallet((a) => { b.textContent = a ? `${a.slice(0, 4)}…${a.slice(-4)}` : 'Connect wallet'; b.classList.toggle('on', !!a); b.title = a ? 'Connected. Click for copy, disconnect or switch.' : 'Connect a Solana wallet'; });
  b.onclick = () => pick().catch(() => {});
}
export function footer() {
  const base = import.meta.env.BASE_URL;
  document.body.append(h('footer', {}, h('span', {}, 'Steer AI · every coin has a live android'), h('a', { href: base }, 'Live'), h('a', { href: `${base}explore.html` }, 'Explore'), h('a', { href: `${base}launch.html` }, 'Launch'),
    h('a', { href: `${base}docs.html` }, 'Docs')));
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
export const STATUS = { asleep: 'Asleep', listening: 'Listening', waking: 'Waking up', awake: 'Awake', thinking: 'Thinking', 'losing it': 'Losing it' };
