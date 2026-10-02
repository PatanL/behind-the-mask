// For Node scripts only (the site's bundler doesn't need it): pump.fun's SDK from its CommonJS build, since its ES
// build imports a dependency in a way Node's ESM loader rejects. node --import ./scripts/cjs-pump-sdk.mjs ...
import { register } from 'node:module';
const shim = new URL('./pump-sdk-shim.mjs', import.meta.url).href;
register('data:text/javascript,' + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  if (spec === '@pump-fun/pump-sdk' && !(ctx.parentURL || '').endsWith('pump-sdk-shim.mjs')) return { url: ${JSON.stringify(shim)}, shortCircuit: true };
  return next(spec, ctx);
}`));
