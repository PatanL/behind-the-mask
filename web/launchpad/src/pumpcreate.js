// A pump.fun coin's launch, built with pump.fun's own SDK: create_v2 (+ the first buy). The coin's creator is the
// Steer AI treasury, so its creator fees go there; the launcher's wallet signs and pays, and the mint's new key signs
// too. Nothing is sent from here: the launcher's wallet sends it. Shared by the launch page and the dry run
// (scripts/pump-dryrun.mjs), which simulates it against mainnet without signing or spending.
import { ComputeBudgetProgram, PublicKey, TransactionMessage, VersionedTransaction } from '@solana/web3.js';
import BN from 'bn.js';
import { OnlinePumpSdk, PUMP_SDK, getBuyTokenAmountFromSolAmount } from '@pump-fun/pump-sdk';

const SOL = PublicKey.default;   // the SDK's SOL quote (the zero key)
export const TX_LIMIT = 1232;    // bytes: Solana's packet size

/** The instructions: create_v2 first, then (with a first buy) the launcher's token account and the buy. */
export async function launchInstructions({ connection, mint, user, creator, name, symbol, uri, devBuySol = 0 }) {
  if (!(devBuySol > 0)) return [await PUMP_SDK.createV2Instruction({ mint, name, symbol, uri, creator, user, mayhemMode: false })];
  const online = new OnlinePumpSdk(connection);
  const [global, feeConfig] = await Promise.all([online.fetchGlobal(), online.fetchFeeConfig()]);
  const solAmount = new BN(Math.round(devBuySol * 1e9));
  const amount = getBuyTokenAmountFromSolAmount({ global, feeConfig, mintSupply: null, bondingCurve: null, amount: solAmount, quoteMint: SOL });
  return PUMP_SDK.createV2AndBuyInstructions({ global, mint, name, symbol, uri, creator, user, amount, solAmount, mayhemMode: false });
}

function compile({ user, blockhash, ixs, alts, units, price }) {
  const msg = new TransactionMessage({ payerKey: user, recentBlockhash: blockhash,
    instructions: [ComputeBudgetProgram.setComputeUnitLimit({ units }), ComputeBudgetProgram.setComputeUnitPrice({ microLamports: price }), ...ixs] }).compileToV0Message(alts);
  return new VersionedTransaction(msg);
}
const size = (tx) => 1 + 64 * tx.message.header.numRequiredSignatures + tx.message.serialize().length;

/** -> [tx] (create + first buy together), or [create, buy] when they don't fit in one transaction. Unsigned.
 *  alt: an AddressLookupTableAccount of the accounts every launch shares (it shrinks the transaction). */
export async function buildLaunch({ connection, alt = null, priorityMicroLamports = 300000, ...o }) {
  const ixs = await launchInstructions({ connection, ...o });
  const { blockhash } = await connection.getLatestBlockhash('confirmed');
  const base = { user: o.user, blockhash, alts: alt ? [alt] : [], price: priorityMicroLamports };
  const one = compile({ ...base, ixs, units: ixs.length > 1 ? 320000 : 220000 });
  if (size(one) <= TX_LIMIT) return [one];
  return [compile({ ...base, ixs: ixs.slice(0, 1), units: 220000 }), compile({ ...base, ixs: ixs.slice(1), units: 200000 })];
}

/** Kept for the dry run's single-transaction check. */
export async function buildCreate(o) { return (await buildLaunch(o))[0]; }
