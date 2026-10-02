// Link previews and icons for every page: Open Graph + X (Twitter) card tags, the favicon, the home-screen icon.
// Each page keeps its own <title> and description; the preview tags are made from them at build time, so a shared
// link (X, iMessage, Discord, Telegram...) shows the big card. Crawlers don't run scripts: the tags must be in the HTML.
const SITE = (process.env.LP_SITE || 'https://steerai.live').replace(/\/$/, '');
const ROOT = (process.env.BTM_BASE || '/').replace(/\/?$/, '/');   // where the icons and the preview image live
const IMAGE = `${SITE}${ROOT}og.png`;
const X = '@steerailive';

/** path: the build's place on the site ('' for the root, 'exhibit/' for the exhibit). */
export function siteMeta(path = '') {
  return {
    name: 'site-meta',
    transformIndexHtml(html, ctx) {
      const title = (html.match(/<title>([^<]*)<\/title>/) || [])[1] || 'Steer AI';
      const desc = (html.match(/<meta name="description" content="([^"]*)"/) || [])[1] || '';
      const page = ctx.filename.split('/').pop().replace(/\.html$/, '');
      const url = `${SITE}${ROOT}${path}${page === 'index' ? '' : page}`;
      const m = (attrs) => ({ tag: 'meta', attrs, injectTo: 'head' });
      const l = (attrs) => ({ tag: 'link', attrs, injectTo: 'head' });
      return [
        l({ rel: 'canonical', href: url }),
        l({ rel: 'icon', href: `${ROOT}favicon.svg`, type: 'image/svg+xml' }),
        l({ rel: 'apple-touch-icon', href: `${ROOT}apple-touch-icon.png` }),
        l({ rel: 'manifest', href: `${ROOT}site.webmanifest` }),
        m({ name: 'theme-color', content: '#03040a' }),
        m({ property: 'og:type', content: 'website' }),
        m({ property: 'og:site_name', content: 'Steer AI' }),
        m({ property: 'og:url', content: url }),
        m({ property: 'og:title', content: title }),
        m({ property: 'og:description', content: desc }),
        m({ property: 'og:image', content: IMAGE }),
        m({ property: 'og:image:width', content: '1200' }),
        m({ property: 'og:image:height', content: '630' }),
        m({ property: 'og:image:alt', content: 'Steer AI: a live AI android whose feelings you steer' }),
        m({ name: 'twitter:card', content: 'summary_large_image' }),
        m({ name: 'twitter:site', content: X }),
        m({ name: 'twitter:title', content: title }),
        m({ name: 'twitter:description', content: desc }),
        m({ name: 'twitter:image', content: IMAGE }),
      ];
    },
  };
}
