// Launching a coin on pump.fun from the launch page. The creator's wallet (Phantom) pays and signs; the coin's mint
// key is made here, in the browser, and signs too (the server never holds a key). The server only relays the two
// calls pump.fun / PumpPortal don't allow from a browser: the image + metadata upload, and building the transaction.
import { get, post } from './api.js';
import { connect } from './wallet.js';

export async function pumpStatus() { try { return await get('api/pump/status'); } catch { return { launch: false }; } }


/** -> the new coin's mint address, once its creation is confirmed */
export async function launchOnPump(coin) {
  const s = await pumpStatus();
  if (!s.launch) throw new Error('Launching on pump.fun is switched off on this server. Pair a coin you launched on pump.fun, or try a demo.');
  const { Keypair, VersionedTransaction } = await import('@solana/web3.js');
  const w = await connect();   // (the header's wallet, or it asks now)
  const publicKey = (await w.connect({ onlyIfTrusted: true })).publicKey;
  const mint = Keypair.generate();
  const r = await post('api/pump/prepare', {
    creator: publicKey.toString(), mint: mint.publicKey.toBase58(), name: coin.name, ticker: coin.ticker,
    description: `${coin.persona}\n\nA live android on Steer AI.`, image: coin.image, dev_buy_sol: coin.devBuy || 0,
    persona: coin.persona, concept: coin.concept || null,
  });
  const tx = VersionedTransaction.deserialize(Uint8Array.from(atob(r.tx), (c) => c.charCodeAt(0)));
  tx.sign([mint]);                                  // the mint's own signature
  const { signature } = await w.signAndSendTransaction(tx);   // the creator's, and it's sent
  await post('api/pump/confirm', { mint: mint.publicKey.toBase58(), signature });
  return mint.publicKey.toBase58();
}
