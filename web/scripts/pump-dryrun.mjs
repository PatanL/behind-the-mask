// Dry run of a launch's create transaction against Solana mainnet: built exactly as the launch page builds it
// (launchpad/src/pumpcreate.js), signed only by a throwaway mint key, then SIMULATED (sigVerify off, so the payer
// needn't sign: a funded public address stands in). Nothing is sent and nothing is spent. Checks that pump.fun's
// program accepts it and that the new coin's on-chain creator is the treasury.
// node --import ./scripts/cjs-pump-sdk.mjs scripts/pump-dryrun.mjs <treasury> [devBuySol] [standInPayer]
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { bondingCurvePda } from '@pump-fun/pump-sdk';
import { buildCreate } from '../launchpad/src/pumpcreate.js';

const [treasury, devBuy = '0', payer = '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM'] = process.argv.slice(2);
const connection = new Connection(process.env.RPC || 'https://api.mainnet-beta.solana.com', 'confirmed');
const mint = Keypair.generate(), user = new PublicKey(payer), creator = new PublicKey(treasury);
const tx = await buildCreate({ connection, mint: mint.publicKey, user, creator, name: process.env.NAME || 'Dry Run', symbol: process.env.SYM || 'DRYRUN', uri: process.env.URI || 'https://steerai.live/og.png', devBuySol: Number(devBuy) });
tx.sign([mint]);
const curve = bondingCurvePda(mint.publicKey);
const sim = await connection.simulateTransaction(tx, { sigVerify: false, replaceRecentBlockhash: true, accounts: { encoding: 'base64', addresses: [curve.toBase58()] } });
const v = sim.value, acct = v.accounts?.[0];
console.log(`dev buy ${devBuy} SOL | ${tx.serialize().length} bytes, ${tx.message.header.numRequiredSignatures} signers | error: ${JSON.stringify(v.err)} | compute units: ${v.unitsConsumed}`);
console.log((v.logs || []).filter((l) => /Instruction:|success|failed|error/i.test(l)).slice(0, 14).map((l) => `  ${l}`).join('\n'));
if (acct) {
  const d = Buffer.from(acct.data[0], 'base64');
  const onchainCreator = new PublicKey(d.subarray(49, 81)).toBase58();
  console.log(`  new coin's bonding curve: owner ${acct.owner}, virtual reserves ${Number(d.readBigUInt64LE(16)) / 1e9} SOL / ${Number(d.readBigUInt64LE(8)) / 1e6} tokens`);
  console.log(`  creator on chain: ${onchainCreator} ${onchainCreator === creator.toBase58() ? '= the treasury' : '!= the treasury'}`);
} else console.log('  (no bonding curve in the result)');
