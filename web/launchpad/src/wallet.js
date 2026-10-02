// The visitor's Solana wallet, shared by the header button, launching and holders. A picker lists every wallet
// installed in this browser that speaks the Wallet Standard (Phantom, Solflare, Backpack, Trust, MetaMask, ...),
// and links to popular ones that aren't installed. The last one used reconnects silently on the next visit.
import { getWallets } from '@wallet-standard/app';
import './wallet.css';
import { WALLET_ICONS as ICON } from './wallet-icons.js';

const CHAIN = 'solana:mainnet';
const KEY = 'steer.wallet';
const POPULAR = [   // shown with an install link when missing
  { name: 'Phantom', url: 'https://phantom.com/download', icon: ICON.phantom, app: (u, ref) => `https://phantom.app/ul/browse/${u}?ref=${ref}` },
  { name: 'Solflare', url: 'https://solflare.com/download', icon: ICON.solflare, app: (u, ref) => `https://solflare.com/ul/v1/browse/${u}?ref=${ref}` },
  { name: 'Backpack', url: 'https://backpack.app/downloads', icon: ICON.backpack },
  { name: 'Trust Wallet', url: 'https://trustwallet.com/browser-extension', icon: ICON.trust },
  { name: 'MetaMask', url: 'https://metamask.io/download', icon: ICON.metamask },
  { name: 'Coinbase Wallet', url: 'https://www.coinbase.com/wallet/downloads', icon: ICON.coinbase },
];
// a phone has no wallet extensions: these wallets can open this page inside their app instead, where the wallet works
const PHONE = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const listeners = new Set();
export let address = null;
let active = null, unwatch = null;

const store = { get: () => { try { return localStorage.getItem(KEY); } catch { return null; } },
  set: (v) => { try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); } catch { /* */ } } };

function set(a) { address = a; for (const f of listeners) f(a); }
export function onWallet(f) { listeners.add(f); f(address); return () => listeners.delete(f); }

const B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
function b58(bytes) {
  let n = 0n; for (const b of bytes) n = n * 256n + BigInt(b);
  let s = ''; while (n > 0n) { s = B58[Number(n % 58n)] + s; n /= 58n; }
  for (const b of bytes) { if (b) break; s = `1${s}`; }
  return s;
}

// a Wallet Standard wallet as {name, icon, connect, disconnect, signMessage, signAndSend, watch}
function standard(w) {
  const f = w.features;
  const h = {
    name: w.name, icon: w.icon, account: null,
    async connect(silent) {
      const { accounts } = await f['standard:connect'].connect(silent ? { silent: true } : undefined);
      const a = accounts.find((x) => x.chains?.some((c) => c.startsWith('solana:'))) || accounts[0];
      if (!a) throw new Error(silent ? 'Not connected.' : `${w.name} didn't share an account.`);
      h.account = a; return a.address;
    },
    async disconnect() { await f['standard:disconnect']?.disconnect(); },
    async signMessage(bytes) {
      if (!f['solana:signMessage']) throw new Error(`${w.name} can't sign messages.`);
      const [r] = await f['solana:signMessage'].signMessage({ account: h.account, message: bytes });
      return r.signature;
    },
    async signAndSend(tx) {   // a VersionedTransaction -> its signature (base58)
      const [r] = await f['solana:signAndSendTransaction'].signAndSendTransaction({ account: h.account, transaction: tx.serialize(), chain: CHAIN });
      return b58(r.signature);
    },
    watch(cb) {
      return f['standard:events']?.on('change', (p) => { if (p.accounts) { h.account = p.accounts[0] || null; cb(h.account?.address || null); } });
    },
  };
  return h;
}

// Phantom's own provider, for an old Phantom that doesn't register with the Wallet Standard
function legacyPhantom(p) {
  return {
    name: 'Phantom', icon: ICON.phantom,
    async connect(silent) { const { publicKey } = await p.connect(silent ? { onlyIfTrusted: true } : undefined); return publicKey.toString(); },
    async disconnect() { await p.disconnect(); },
    async signMessage(bytes) { return (await p.signMessage(bytes, 'utf8')).signature; },
    async signAndSend(tx) { return (await p.signAndSendTransaction(tx)).signature; },
    watch(cb) { p.on?.('accountChanged', (pk) => cb(pk ? pk.toString() : null)); return () => p.off?.('accountChanged'); },
  };
}

const registry = getWallets();
const handles = new Map();   // wallet -> handle (one per wallet, so its account sticks)
function installed() {
  const out = [];
  for (const w of registry.get()) {
    if (!w.features['standard:connect'] || !w.features['solana:signAndSendTransaction'] || !w.chains.some((c) => c.startsWith('solana:'))) continue;
    if (!handles.has(w)) handles.set(w, standard(w));
    out.push(handles.get(w));
  }
  const ph = window.phantom?.solana || (window.solana?.isPhantom ? window.solana : null);
  if (ph && !out.some((h) => h.name === 'Phantom')) { if (!handles.has(ph)) handles.set(ph, legacyPhantom(ph)); out.push(handles.get(ph)); }
  return out;
}

async function use(h, silent = false) {
  const a = await h.connect(silent);
  unwatch?.(); active = h; store.set(h.name);
  unwatch = h.watch((x) => { if (!x) { active = null; store.set(null); } set(x); });
  set(a);
  return h;
}

/** The connected wallet (asks the visitor to pick one if none is). */
export async function connect() { return active && address ? active : pick(); }

export async function disconnect() {
  const h = active; active = null; unwatch?.(); unwatch = null; store.set(null); set(null);
  try { await h?.disconnect(); } catch { /* */ }
}

// ---- the picker
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
function tile(h) {
  const t = el('span', 'wl-ico');
  if (h?.icon) { const i = el('img'); i.src = h.icon; i.alt = ''; t.append(i); } else { t.textContent = (h?.name || '?')[0]; t.style.background = '#1b2638'; }
  return t;
}

let open = null;
/** Opens the wallet picker -> the wallet the visitor connected (rejects if they close it). */
export function pick() {
  if (open) return open.promise;
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  const back = el('div', 'wl-back'), box = el('div', 'wl'), list = el('div', 'wl-list'), note = el('div', 'wl-note');
  box.setAttribute('role', 'dialog'); box.setAttribute('aria-modal', 'true'); box.setAttribute('aria-label', 'Connect a wallet');
  const head = el('div', 'wl-h'); head.append(el('span', 'wl-t', address ? 'Wallet' : 'Connect wallet'));
  const x = el('button', 'wl-x', '×'); x.type = 'button'; x.setAttribute('aria-label', 'Close'); head.append(x);
  box.append(head);
  if (address && active) {   // connected: who, copy, disconnect, or switch below
    const cur = el('div', 'wl-cur');
    cur.append(tile(active), el('span', 'wl-n', `${active.name}`), el('span', 'wl-addr', `${address.slice(0, 4)}…${address.slice(-4)}`));
    const row = el('div', 'wl-acts'), cp = el('button', 'wl-act', 'Copy address'), dc = el('button', 'wl-act bad', 'Disconnect');
    cp.onclick = () => { navigator.clipboard?.writeText(address); cp.textContent = 'Copied'; setTimeout(() => (cp.textContent = 'Copy address'), 1200); };
    dc.onclick = () => { disconnect(); close(new Error('Disconnected.')); };
    row.append(cp, dc); box.append(cur, row, el('div', 'wl-sec', 'Switch wallet'));
  }
  box.append(list, note);
  const foot = el('div', 'wl-foot', 'Connecting shares your address only. Steer AI never holds your keys.');
  box.append(foot);
  back.append(box);

  const last = store.get();
  function render() {
    list.textContent = '';
    const have = installed().sort((a, b) => (b.name === last) - (a.name === last));
    for (const h of have) {
      const r = el('button', 'wl-row'); r.type = 'button';
      r.append(tile(h), el('span', 'wl-n', h.name), el('span', 'wl-tag ok', h === active ? 'Connected' : h.name === last ? 'Last used' : 'Installed'), el('span', 'wl-go', '›'));
      r.onclick = async () => {
        note.textContent = `Approve in ${h.name}…`; note.className = 'wl-note';
        list.classList.add('busy');
        try { close(null, await use(h)); } catch (e) { note.textContent = e?.message || 'Not connected.'; note.className = 'wl-note bad'; list.classList.remove('busy'); }
      };
      list.append(r);
    }
    const missing = POPULAR.filter((p) => !have.some((h) => h.name.toLowerCase().startsWith(p.name.toLowerCase().split(' ')[0])));
    if (missing.length) list.append(el('div', 'wl-sec', have.length ? 'More wallets' : 'Get a Solana wallet'));
    for (const p of missing) {
      const inApp = PHONE && p.app;
      const r = el('a', 'wl-row'); r.href = inApp ? p.app(encodeURIComponent(location.href), encodeURIComponent(location.origin)) : p.url;
      if (!inApp) { r.target = '_blank'; r.rel = 'noopener'; }
      r.append(tile(p), el('span', 'wl-n', p.name), el('span', `wl-tag${inApp ? ' ok' : ''}`, inApp ? 'Open in app' : 'Get'), el('span', 'wl-go', '↗'));
      list.append(r);
    }
  }
  const offReg = registry.on('register', render), offUnreg = registry.on('unregister', render);
  function close(err, h) {
    offReg(); offUnreg(); removeEventListener('keydown', esc); back.remove(); open = null;
    if (h) resolve(h); else reject(err || new Error('No wallet connected.'));
  }
  const esc = (e) => { if (e.key === 'Escape') close(); };
  x.onclick = () => close();
  back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
  addEventListener('keydown', esc);
  render();
  document.body.append(back);
  requestAnimationFrame(() => back.classList.add('in'));
  open = { promise };
  return promise;
}

// the wallet used last time reconnects silently (wallets register a moment after the page loads)
setTimeout(() => {
  const name = store.get(); if (!name) return;
  const h = installed().find((x) => x.name === name);
  if (h) use(h, true).catch(() => {});
}, 400);
