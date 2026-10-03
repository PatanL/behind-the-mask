// "How does this work?": four short cards for people who have never heard of steering.
const word = (w, p, c = '#9fb4ff') => `<div style="display:flex;align-items:center;gap:8px;font-family:var(--serif);font-size:16px"><span style="width:110px;text-align:right">${w}</span><span style="height:10px;width:${p * 2.2}px;border-radius:5px;background:${c}"></span><span style="font:12px var(--sans);color:var(--ink2)">${p}%</span></div>`;

import { EMO, ORDER } from './palette.js';

function mapSVG(map) {
  if (!map) return '';
  const W = 320, H = 140, cx = W / 2, cy = H / 2, R = 58, sx = 2.1;
  const lines = ORDER.map((e) => {
    const [x, y] = map[e]; const px = cx + x * R * sx, py = cy - y * R;
    const tx = cx + x * (R * sx + 14), ty = cy - y * (R + 12) + 4;
    return `<line x1="${cx}" y1="${cy}" x2="${px}" y2="${py}" stroke="${EMO[e].color}" stroke-width="3" stroke-linecap="round"/><text x="${tx}" y="${ty}" fill="${EMO[e].color}" text-anchor="${x > 0.2 ? 'start' : x < -0.2 ? 'end' : 'middle'}">${EMO[e].label.toLowerCase()}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}"><g font-family="Inter" font-size="12"><circle cx="${cx}" cy="${cy}" r="3" fill="#fff"/>${lines}<text x="8" y="${H - 6}" fill="#6b7290" font-size="10">unpleasant</text><text x="${W - 8}" y="${H - 6}" fill="#6b7290" font-size="10" text-anchor="end">pleasant</text></g></svg>`;
}

// the assistant axis: characters on one end, its own assistant voice on the other, and the push "off script"
function axisSVG() {
  const L = 26, R = 294, y = 56, at = L + (R - L) * 0.82;
  const chars = [['pirate', 0.03], ['ghost', 0.19], ['cat', 0.33], ['poet', 0.46]]
    .map(([t, f]) => `<text x="${L + (R - L) * f}" y="${y - 16}" text-anchor="middle" fill="#c89bff" font-size="11">${t}</text><circle cx="${L + (R - L) * f}" cy="${y}" r="3" fill="#c89bff" opacity=".7"/>`).join('');
  return `<svg viewBox="0 0 320 106" width="100%" height="150"><g font-family="Inter" font-size="11">
<defs><linearGradient id="ax" gradientUnits="userSpaceOnUse" x1="${L}" x2="${R}" y1="0" y2="0"><stop offset="0" stop-color="#c89bff"/><stop offset="1" stop-color="#9fb4ff"/></linearGradient>
<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#ff8a8a"/></marker></defs>
<line x1="${L}" y1="${y}" x2="${R}" y2="${y}" stroke="url(#ax)" stroke-width="3" stroke-linecap="round"/>${chars}
<text x="${R}" y="${y - 16}" text-anchor="end" fill="#9fb4ff" font-size="11">assistant</text>
<circle cx="${at}" cy="${y}" r="7" fill="#0b0d18" stroke="#fff" stroke-width="2"><animate attributeName="cx" values="${at};${L + (R - L) * 0.5};${at}" dur="5s" repeatCount="indefinite"/></circle>
<line x1="${at - 12}" y1="${y + 22}" x2="${L + (R - L) * 0.45}" y2="${y + 22}" stroke="#ff8a8a" stroke-width="2" marker-end="url(#ah)"/>
<text x="${(at - 12 + L + (R - L) * 0.45) / 2}" y="${y + 40}" text-anchor="middle" fill="#ff8a8a">push off script</text>
<text x="${L}" y="${y + 40}" fill="#6b7290" font-size="10">someone else</text><text x="${R}" y="${y + 40}" text-anchor="end" fill="#6b7290" font-size="10">itself</text></g></svg>`;
}

// sparse features: a sea of dim ones, a few lit by the current words, named
function featuresSVG() {
  const cols = 30, rows = 6, lit = { 37: ['memes', '#ffc857'], 64: ['cats', '#7fe7a8'], 101: ['jokes', '#ff9f6b'], 142: ['the internet', '#9fb4ff'] };
  let dots = '';
  for (let i = 0; i < cols * rows; i++) {
    const x = 14 + (i % cols) * 10, y = 12 + Math.floor(i / cols) * 10, on = lit[i];
    dots += on ? `<circle cx="${x}" cy="${y}" r="3.6" fill="${on[1]}"><animate attributeName="opacity" values="1;.35;1" dur="${2 + (i % 3) * 0.7}s" repeatCount="indefinite"/></circle>`
      : `<circle cx="${x}" cy="${y}" r="1.6" fill="rgba(170,180,230,.22)"/>`;
  }
  const names = Object.values(lit).map(([t, c], k) => `<g transform="translate(${14 + k * 76} 92)"><circle cx="4" cy="-4" r="3.6" fill="${c}"/><text x="12" y="0" fill="${c}">${t}</text></g>`).join('');
  return `<svg viewBox="0 0 320 112" width="100%" height="146"><g font-family="Inter" font-size="11">${dots}${names}
<text x="160" y="110" text-anchor="middle" fill="#6b7290" font-size="10">about 65,000 features · a few dozen on at once</text></g></svg>`;
}

export const HOW_STEPS = [
  {
    title: 'It writes one piece at a time',
    body: `<p>This AI is a language model. It doesn't plan a whole sentence. At every step it scores every word-piece it knows (about 150,000) and picks one. Then it does it again.</p>`,
    viz: `<div style="display:grid;gap:6px"><div style="font:14px var(--sans);color:var(--ink2);margin-bottom:4px">“Today was…”</div>${word('good', 34)}${word('long', 21)}${word('fine', 12)}${word('amazing', 4)}</div>`,
  },
  {
    title: 'Inside, feelings are directions',
    body: `<p>While it reads and writes, the AI keeps a hidden state: a list of 5,120 numbers. We asked it to write short stories about characters who feel joy, sadness, anger, fear, calm or curiosity, then recorded its hidden state as it wrote. Each feeling's average turned out to be a <b>direction</b> in that space. It's the same recipe Anthropic used to find 171 emotion concepts inside Claude in 2026.</p><p>This is the real map of those six directions, flattened to 2D. Its biggest axis sorts pleasant feelings from unpleasant ones. Nobody programmed that in; it came out of the model.</p>`,
    viz: (map) => mapSVG(map),
  },
  {
    title: 'We push along a direction',
    body: `<p>When you pick a feeling, we add a little of that direction to the hidden state, halfway up the model, at every word. We don't change its training or tell it how to feel. We nudge its state, and its choices tilt.</p><p>The face and the glowing words show a <b>live measurement</b> of those patterns. That's a reading of language patterns inside a machine. It is <b>not</b> evidence that it feels anything.</p>`,
    viz: `<svg viewBox="0 0 320 130" width="100%" height="130"><g fill="none" stroke="rgba(170,180,230,.35)">${Array.from({ length: 12 }, (_, i) => `<ellipse cx="160" cy="${118 - i * 9.5}" rx="${60 + 10 * Math.sin(i / 3.5)}" ry="3"/>`).join('')}</g><ellipse cx="160" cy="61" rx="80" ry="6" fill="none" stroke="#ffc857" stroke-width="3"/><text x="248" y="58" font-family="Inter" font-size="12" fill="#ffc857">+ joy</text><text x="160" y="128" text-anchor="middle" font-family="Inter" font-size="11" fill="#6b7290">words in</text><text x="160" y="8" text-anchor="middle" font-family="Inter" font-size="11" fill="#6b7290">next word out</text></svg>`,
  },
  {
    title: 'It also has a direction for being an assistant',
    body: `<p>We had it answer questions as itself, then as 30 characters (a pirate, a ghost, a cat, a poet…). The difference is the <b>assistant axis</b>, a direction researchers found in many chat models in 2026. The <b>Assistant voice</b> meter shows how assistant-like its state is right now.</p><p>Push it <b>off script</b> (against that direction), and the helpful assistant starts to fade into someone else.</p>`,
    viz: axisSVG(),
  },
  {
    title: 'Watching individual ideas light up',
    body: `<p>Qwen, the team that built this model, released <b>sparse autoencoders</b> (Qwen-Scope, 2026). These are tools that split the hidden state into about 65,000 separate "features". Most of the time only 50 are active. We asked the model to name each feature from the text that triggers it most. The <b>Lighting up inside</b> list shows a few of them as it writes.</p>`,
    viz: featuresSVG(),
  },
  {
    title: 'And under the assistant voice…',
    body: `<p>The polite assistant voice isn't the raw model. It started as a <b>base model</b> trained only to continue internet text, then got extra training to act as a helpful assistant, using human feedback (RLHF). After each answer you can peek at the base model: the same kind of AI, without the assistant voice.</p>`,
    viz: `<div style="display:flex;gap:18px;align-items:center;font:14px var(--sans);color:var(--ink2)"><div style="padding:12px 14px;border-radius:12px;background:rgba(255,255,255,.05);max-width:230px">“I'm an AI assistant, so I don't have days, but I'd love to hear about yours!”</div><div style="padding:12px 14px;border-radius:12px;background:rgba(80,30,90,.35);max-width:230px;font-family:var(--serif);color:var(--ink)">“It was terrible. The bus broke down and I missed my exam…”</div></div>`,
  },
];
