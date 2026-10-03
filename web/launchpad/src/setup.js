// One-time: an address lookup table of the accounts every launch shares (pump.fun's program, its global and fee
// accounts, the token programs, the treasury's creator vault...), so create + first buy fit in one transaction.
// The connected wallet pays and owns the table. Its address then goes in the server's settings (runs/launchpad/pump.json).
import './style.css';
import { api, get, rpcConnection } from './api.js';
import { $, header, footer } from './ui.js';
import { connect, address } from './wallet.js';

header(''); footer();
const log = (t) => { $('#log').textContent += `${t}\n`; };

$('#go').onclick = async () => {
  $('#go').disabled = true;
  try {
    await import('./node-globals.js');   // (before the SDK)
    const [{ AddressLookupTableProgram, Connection, Keypair, PublicKey, TransactionMessage, VersionedTransaction }, { launchInstructions }, sdk] =
      await Promise.all([import('@solana/web3.js'), import('./pumpcreate.js'), import('@pump-fun/pump-sdk')]);
    const st = await get('api/pump/status');
    if (!st.treasury) throw new Error('The server has no treasury set yet.');
    const w = await connect(); const me = new PublicKey(address);
    const connection = rpcConnection(Connection);
    // the accounts two launches (different coins) have in common, minus the payer; plus every protocol fee recipient
    const keysOf = async () => (await launchInstructions({ connection, mint: Keypair.generate().publicKey, user: me, creator: new PublicKey(st.treasury), name: 'x', symbol: 'X', uri: 'u', devBuySol: 0.01 }))
      .flatMap((ix) => [ix.programId, ...ix.keys.map((k) => k.pubkey)]).map((k) => k.toBase58());
    const [a, b] = [await keysOf(), await keysOf()];
    const g = await new sdk.OnlinePumpSdk(connection).fetchGlobal();
    const extra = [g.feeRecipient, ...(g.feeRecipients || [])].map((k) => k.toBase58());
    const want = [...new Set([...a.filter((k) => b.includes(k) && k !== address), ...extra, 'ComputeBudget111111111111111111111111111111'])].map((k) => new PublicKey(k));
    log(`${want.length} shared accounts.`);
    const slot = await connection.getSlot('finalized');
    const [create, table] = AddressLookupTableProgram.createLookupTable({ authority: me, payer: me, recentSlot: slot });
    const chunks = []; for (let i = 0; i < want.length; i += 20) chunks.push(want.slice(i, i + 20));
    const send = async (ixs, what) => {
      const { blockhash } = await connection.getLatestBlockhash('confirmed');
      const tx = new VersionedTransaction(new TransactionMessage({ payerKey: me, recentBlockhash: blockhash, instructions: ixs }).compileToV0Message());
      log(`${what}: approve in your wallet…`);
      const sig = await w.signAndSend(tx);
      for (let i = 0; i < 40; i++) {
        const s = (await connection.getSignatureStatuses([sig])).value[0];
        if (s?.err) throw new Error(`${what} failed on chain.`);
        if (s && (s.confirmationStatus === 'confirmed' || s.confirmationStatus === 'finalized')) { log(`${what}: confirmed.`); return; }
        await new Promise((r) => setTimeout(r, 1500));
      }
      throw new Error(`${what} didn't confirm in time.`);
    };
    const extend = (addresses) => AddressLookupTableProgram.extendLookupTable({ lookupTable: table, authority: me, payer: me, addresses });
    await send([create, extend(chunks[0])], 'Create the table');
    for (let i = 1; i < chunks.length; i++) await send([extend(chunks[i])], `Add accounts (${i + 1}/${chunks.length})`);
    log(`\nDone. The table's address:\n${table.toBase58()}\n\nSend it to whoever runs the server, for runs/launchpad/pump.json ("alt").`);
  } catch (e) { log(`Stopped: ${e.message || e}`); $('#go').disabled = false; }
};
