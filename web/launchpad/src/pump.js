// Launching a coin on pump.fun from the launch page. The launcher's wallet pays and signs; the coin's mint key is
// made here, in the browser, and signs too (the server never holds a key). The coin's creator (who gets its creator
// fees) is the Steer AI treasury the server names. The server uploads the image + metadata (pump.fun doesn't allow
// that from a browser) and relays this page's Solana reads; the transaction is built here (pumpcreate.js).
import { api, get, post } from './api.js';
import { connect, address } from './wallet.js';

export async function pumpStatus() { try { return await get('api/pump/status'); } catch { return { launch: false }; } }

/** -> the new coin's mint address, once its creation is confirmed (and its creator checked on chain) */
export async function launchOnPump(coin, note = () => {}) {
  const s = await pumpStatus();
  if (!s.launch) throw new Error('Launching on pump.fun is switched off on this server for now.');
  await import('./node-globals.js');   // (before the SDK)
  const [{ Connection, Keypair, PublicKey }, { buildLaunch }] = await Promise.all([import('@solana/web3.js'), import('./pumpcreate.js')]);
  const w = await connect();   // (the header's wallet, or the picker now)
  const user = new PublicKey(address);
  const mint = Keypair.generate();
  note('Uploading its portrait to pump.fun…');
  const r = await post('api/pump/prepare', {
    creator: address, mint: mint.publicKey.toBase58(), name: coin.name, ticker: coin.ticker,
    description: `${coin.persona}\n\nA live android on Steer AI.`, image: coin.image, dev_buy_sol: coin.devBuy || 0,
    persona: coin.persona, concept: coin.concept || null,
  });
  const connection = new Connection(new URL(api('api/rpc'), location.href).href, 'confirmed');
  const alt = r.alt ? (await connection.getAddressLookupTable(new PublicKey(r.alt))).value : null;
  const txs = await buildLaunch({ connection, alt, mint: mint.publicKey, user, creator: new PublicKey(r.creator),
    name: coin.name, symbol: coin.ticker, uri: r.uri, devBuySol: coin.devBuy || 0 });
  txs[0].sign([mint]);   // the mint's own signature
  note(txs.length > 1 ? 'Approve the launch in your wallet (then the first buy).' : 'Approve the launch in your wallet.');
  const signature = await w.signAndSend(txs[0]);
  note('Waiting for it to confirm on chain…');
  await post('api/pump/confirm', { mint: mint.publicKey.toBase58(), signature });
  if (txs[1]) {   // too big for one transaction: the first buy goes right after the coin exists
    try {
      txs[1].message.recentBlockhash = (await connection.getLatestBlockhash('confirmed')).blockhash;
      note('Approve your first buy.');
      await w.signAndSend(txs[1]);
    } catch (e) { note(`The coin is live; the first buy didn't go through (${e.message || e}). You can buy it on pump.fun.`); }
  }
  return mint.publicKey.toBase58();
}
