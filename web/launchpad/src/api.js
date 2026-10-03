// The launchpad server (server/launchpad.py): same origin under /api (vite proxies it), or VITE_LP_API.
// (the site's build sets VITE_LIVE_URL to the GPU machine's public address: the launchpad API lives there too)
const API = (import.meta.env.VITE_LP_API || import.meta.env.VITE_LIVE_URL || '').replace(/\/$/, '');
const BASE = import.meta.env.BASE_URL;
export const api = (p) => (API ? `${API}/${p}` : `${BASE}${p}`);
export async function get(p) { const r = await fetch(api(p), { cache: 'no-store' }); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); }
export async function post(p, body) {
  const r = await fetch(api(p), { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
export function wsUrl(p) {
  const u = API ? new URL(`${API}/${p}`) : new URL(`${BASE}${p}`, location.href);
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  return u.toString();
}
export function visitorId() {
  try { let v = localStorage.getItem('lp-cid'); if (!v) { v = Math.random().toString(36).slice(2, 12); localStorage.setItem('lp-cid', v); } return v; }
  catch { return Math.random().toString(36).slice(2, 12); }
}
export const imgUrl = (p) => (p ? api(p) : null);

/** A Solana connection through the server's RPC relay. (web3.js adds a "solana-client" header to every request; across
 *  origins that header needs the API's permission, so it's dropped: the relay doesn't use it.) */
export function rpcConnection(Connection) {
  const plain = (input, init = {}) => {
    let headers = init.headers || {};
    if (typeof Headers !== 'undefined' && headers instanceof Headers) { headers = new Headers(headers); headers.delete('solana-client'); }
    else headers = Object.fromEntries(Object.entries(headers).filter(([k]) => k.toLowerCase() !== 'solana-client'));
    return fetch(input, { ...init, headers });
  };
  return new Connection(new URL(api('api/rpc'), location.href).href, { commitment: 'confirmed', fetch: plain });
}
