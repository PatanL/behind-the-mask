import { defineConfig } from 'vite';

// BTM_BASE lets the same build live at a sub-path (e.g. GitHub Pages: /behind-the-mask/).
export default defineConfig({
  base: process.env.BTM_BASE || '/',
  build: { chunkSizeWarningLimit: 1500 },
  server: { watch: { ignored: ['**/public/performances/**'] }, proxy: { '/live': { target: 'http://127.0.0.1:8765', ws: true } } },
  // the public Tailscale Funnel hostname proxies to `vite preview`; /live is the crowd-steering server (server/live.py)
  preview: { allowedHosts: ['spark-3a11.tail621a3a.ts.net'], proxy: { '/live': { target: 'http://127.0.0.1:8765', ws: true } } },
});
