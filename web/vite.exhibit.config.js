import { defineConfig } from 'vite';

// The Steer AI exhibit (index.html, web/src): built into the site at /exhibit/ by vite.config.js.
// Alone: `npx vite -c vite.exhibit.config.js` (dev). BTM_BASE is the site's base (GitHub Pages: "/" on steerai.live).
const base = `${(process.env.BTM_BASE || '/').replace(/\/?$/, '/')}exhibit/`;
export default defineConfig({
  base,
  build: { chunkSizeWarningLimit: 1500, outDir: 'dist/exhibit', emptyOutDir: false },
  server: { watch: { ignored: ['**/public/performances/**'] }, proxy: { '/live': { target: 'http://127.0.0.1:8765', ws: true } } },
});
