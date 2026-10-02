// The visitor's Solana wallet (Phantom), shared by the header button, launching and holders.
const listeners = new Set();
export let address = null;
export const provider = () => window.phantom?.solana || (window.solana?.isPhantom ? window.solana : null);
function set(a) { address = a; for (const f of listeners) f(a); }
export function onWallet(f) { listeners.add(f); f(address); return () => listeners.delete(f); }
export async function connect() {
  const p = provider();
  if (!p) { window.open('https://phantom.app/download', '_blank', 'noopener'); throw new Error('No Solana wallet in this browser. Install Phantom, then connect.'); }
  const { publicKey } = await p.connect();
  set(publicKey.toString());
  return p;
}
export async function disconnect() { try { await provider()?.disconnect(); } catch { /* */ } set(null); }
// a wallet the visitor already trusted reconnects silently
setTimeout(() => {
  const p = provider(); if (!p) return;
  p.connect({ onlyIfTrusted: true }).then(({ publicKey }) => set(publicKey.toString())).catch(() => {});
  p.on?.('accountChanged', (pk) => set(pk ? pk.toString() : null));
}, 300);
