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
  const had = document.querySelector('header.top');
  if (had) { had.querySelectorAll('.nav a').forEach((a) => a.classList.toggle('on', a.textContent === on)); return; }
  const el = h('header', { class: 'top' },
    h('a', { class: 'brand', href: base }, h('span', { class: 'eye' }), 'Steer AI'),
    h('nav', { class: 'nav' }, ...[['Live', base], ['Explore', `${base}explore`], ['Launch', `${base}launch`], ['Docs', `${base}docs`]].map(([t, u]) => h('a', { href: u, class: on === t ? 'on' : '' }, t))),
    h('div', { class: 'sp' }),
    h('a', { class: 'top-link', href: `${base}docs` }, 'How it works'),
    h('a', { class: 'top-link x-link', href: 'https://x.com/steerailive', target: '_blank', rel: 'noopener', 'aria-label': 'Steer AI on X', title: 'Steer AI on X (@steerailive)' }),
    h('button', { class: 'btn wallet-b', id: 'wallet-top', type: 'button' }, 'Connect wallet'),
    h('a', { class: 'btn primary', href: `${base}launch` }, 'Launch a coin'));
  el.querySelector('.x-link').innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/></svg>`;
  document.body.prepend(el);
  const b = el.querySelector('#wallet-top');
  onWallet((a) => { b.textContent = a ? `${a.slice(0, 4)}…${a.slice(-4)}` : 'Connect wallet'; b.classList.toggle('on', !!a); b.title = a ? 'Connected. Click for copy, disconnect or switch.' : 'Connect a Solana wallet'; });
  b.onclick = () => pick().catch(() => {});
}
export function footer() {
  const base = import.meta.env.BASE_URL;
  if (document.querySelector('body > footer')) return;
  document.body.append(h('footer', {}, h('span', {}, 'Steer AI · every coin has a live android'), h('a', { href: base }, 'Live'), h('a', { href: `${base}explore` }, 'Explore'), h('a', { href: `${base}launch` }, 'Launch'),
    h('a', { href: `${base}docs` }, 'Docs')));
}
/** dollars, short: $940, $12.3K, $4.56M (SOL amounts times SOL's dollar price) */
export function usd(x) {
  if (x == null || !isFinite(x)) return '—';
  const a = Math.abs(x);
  return a >= 1e9 ? `$${(x / 1e9).toFixed(2)}B` : a >= 1e6 ? `$${(x / 1e6).toFixed(2)}M` : a >= 1e3 ? `$${(x / 1e3).toFixed(1)}K` : `$${x.toFixed(0)}`;
}
/** a token's price in dollars, pump.fun style for tiny ones: $0.0₅251 (five zeros after the point, then 251) */
export function priceUsd(solPrice, solUsd) {
  if (!solPrice) return '—';
  if (!solUsd) return `${solPrice.toExponential(3)} SOL`;
  const v = solPrice * solUsd;
  if (v >= 1) return `$${v.toFixed(2)}`;
  if (v >= 0.001) return `$${v.toPrecision(3)}`;
  const zeros = Math.ceil(-Math.log10(v)) - 1, digits = String(Math.round(v * 10 ** (zeros + 4))).replace(/0+$/, '');
  const sub = String(zeros).replace(/\d/g, (x) => '₀₁₂₃₄₅₆₇₈₉'[x]);
  return `$0.0${sub}${digits}`;
}
export const inUsd = (solAmount, solUsd, d = 1) => (solUsd ? usd(solAmount * solUsd) : sol(solAmount, d));
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
