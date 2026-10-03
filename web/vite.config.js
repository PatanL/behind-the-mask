import { defineConfig, build } from 'vite';
import { resolve } from 'node:path';
import { siteMeta } from './site-meta.js';

// steerai.live: the launchpad (web/launchpad: every coin has a live android) at the root, and the Steer AI exhibit
// (index.html, web/src) at /exhibit/, built by the plugin below with vite.exhibit.config.js. BTM_BASE lets the same
// build live at a sub-path (GitHub Pages without a custom domain: /behind-the-mask/).
// /api is the launchpad server (server/launchpad.py); /live is the home android (server/live.py), served by the launchpad.
const R = (p) => resolve(import.meta.dirname, p);
const pages = ['index', 'explore', 'coin', 'launch', 'docs', 'setup'];
const proxy = { '/api': { target: 'http://127.0.0.1:8770', ws: true }, '/live': { target: process.env.BTM_LIVE_TARGET || 'http://127.0.0.1:8770', ws: true } };   // (the launchpad serves the home android too: LP_HOME=1)
const exhibit = { name: 'exhibit', apply: 'build', async closeBundle() { if (!process.env.LP_NO_EXHIBIT) await build({ configFile: R('vite.exhibit.config.js'), root: R('.') }); } };
export default defineConfig({
  root: R('launchpad'),
  base: process.env.BTM_BASE || '/',
  publicDir: R('launchpad/public'),
  plugins: [exhibit, siteMeta('')],
  build: { outDir: R('dist'), emptyOutDir: true, chunkSizeWarningLimit: 2000,
    rollupOptions: { input: Object.fromEntries(pages.map((p) => [p, R(`launchpad/${p}.html`)])) } },
  server: { proxy },
  // the public Tailscale Funnel hostname proxies to `vite preview`
  // cors: false -- the API answers cross-origin checks itself (steerai.live posts to it); vite's own answer omitted the
  // allowed origin, so browsers refused every POST (a launch's portrait upload, the RPC relay)
  preview: { allowedHosts: ['spark-3a11.tail621a3a.ts.net'], proxy, cors: false },
});
