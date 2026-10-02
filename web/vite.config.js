import { defineConfig, build } from 'vite';
import { resolve } from 'node:path';
import { siteMeta } from './site-meta.js';

// steerai.live: the launchpad (web/launchpad: every coin has a live android) at the root, and the Steer AI exhibit
// (index.html, web/src) at /exhibit/, built by the plugin below with vite.exhibit.config.js. BTM_BASE lets the same
// build live at a sub-path (GitHub Pages without a custom domain: /behind-the-mask/).
// /api is the launchpad server (server/launchpad.py); /live is the exhibit's crowd-steering server (server/live.py).
const R = (p) => resolve(import.meta.dirname, p);
const pages = ['index', 'explore', 'coin', 'launch', 'docs'];
const proxy = { '/api': { target: 'http://127.0.0.1:8770', ws: true }, '/live': { target: 'http://127.0.0.1:8765', ws: true } };
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
  preview: { allowedHosts: ['spark-3a11.tail621a3a.ts.net'], proxy },
});
