// Shareable result card: a 1080x1350 image of the question, the push and the answer (with a snapshot of the
// face), plus a link that replays exactly this performance. The page can't trigger downloads itself, so we use
// the Web Share API where available and otherwise show the image to long-press / right-click save.
import { EMO } from './palette.js';

function wrap(ctx, text, maxW) {
  const words = text.split(/\s+/), lines = [];
  let line = '';
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
  }
  if (line) lines.push(line);
  return lines;
}

export function cardImage({ question, answer, emotion, level, faceCanvas }) {
  const W = 1080, H = 1350, c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const col = EMO[emotion]?.color || '#c9cfe4';
  const g = ctx.createRadialGradient(W * 0.5, H * 0.18, 40, W * 0.5, H * 0.3, H * 0.9);
  g.addColorStop(0, col + '33'); g.addColorStop(1, '#04050b');
  ctx.fillStyle = '#04050b'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  // face snapshot, feathered into the background with a radial mask
  let faceBottom = 0;
  if (faceCanvas) {
    const fw = 400, fh = Math.round(fw * faceCanvas.height / faceCanvas.width);
    const off = document.createElement('canvas'); off.width = fw; off.height = fh;
    const o = off.getContext('2d');
    o.drawImage(faceCanvas, 0, 0, fw, fh);
    o.globalCompositeOperation = 'destination-in';
    const m = o.createRadialGradient(fw / 2, fh * 0.42, fw * 0.18, fw / 2, fh * 0.45, fw * 0.56);
    m.addColorStop(0, 'rgba(0,0,0,1)'); m.addColorStop(1, 'rgba(0,0,0,0)');
    o.fillStyle = m; o.fillRect(0, 0, fw, fh);
    ctx.drawImage(off, W - fw - 30, 30);
    faceBottom = 30 + fh * 0.86;
  }
  ctx.fillStyle = '#aab0c6'; ctx.font = '600 30px Inter, system-ui, sans-serif';
  ctx.fillText('STEER AI', 72, 110);
  const lv = { little: 'a little', lot: 'a lot', toomuch: 'way too much' }[level] || '';
  const badge = emotion === 'none' ? 'No push · as trained' : emotion === 'unmask' ? `Assistant persona pushed away${lv ? ' · ' + lv : ''}` : `Pushed toward ${EMO[emotion]?.label.toLowerCase() || emotion}${lv ? ' · ' + lv : ''}`;
  ctx.font = '600 34px Inter, system-ui, sans-serif';
  const bw = ctx.measureText(badge).width + 48;
  ctx.fillStyle = col + '30'; ctx.strokeStyle = col; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.roundRect(72, 160, bw, 64, 32); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#eef0f7'; ctx.fillText(badge, 96, 204);
  ctx.font = 'italic 44px Fraunces, Georgia, serif'; ctx.fillStyle = '#aab0c6';
  let y = 330;
  for (const l of wrap(ctx, `“${question}”`, 540)) { ctx.fillText(l, 72, y); y += 56; }
  y = Math.max(y + 40, faceBottom + 40, 520);
  ctx.font = '50px Fraunces, Georgia, serif'; ctx.fillStyle = col;
  const maxLines = Math.floor((H - 170 - y) / 66);
  // keep whole sentences that fit; fall back to a cut with an ellipsis
  const sents = answer.match(/[^.!?]+[.!?]+["”’)]*\s*/g) || [answer];
  let text = '';
  for (const s2 of sents) { const t = (text ? text + ' ' : '') + s2.trim(); if (wrap(ctx, t, W - 144).length > maxLines) break; text = t; }
  let lines = wrap(ctx, text || answer, W - 144);
  if (lines.length > maxLines) { lines = lines.slice(0, maxLines); lines[maxLines - 1] = lines[maxLines - 1].replace(/\s+\S*$/, '') + '…'; }
  ctx.shadowColor = col; ctx.shadowBlur = 18;
  for (const l of lines) { ctx.fillText(l, 72, y); y += 66; }
  ctx.shadowBlur = 0;
  ctx.font = '28px Inter, system-ui, sans-serif'; ctx.fillStyle = '#6b7290';
  ctx.fillText('An AI with a feeling turned up inside it, by editing its hidden state.', 72, H - 96);
  ctx.fillText('Steer AI · activation steering · not evidence of real feelings', 72, H - 56);
  return c;
}

export async function shareCard(opts, link) {
  const canvas = cardImage(opts);
  const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
  const file = new File([blob], 'steer-ai.png', { type: 'image/png' });
  try {
    if (navigator.canShare?.({ files: [file] })) { await navigator.share({ files: [file], text: `“${opts.question}”, ${link}`, url: link }); return 'shared'; }
  } catch { /* fall through to the preview */ }
  return { url: URL.createObjectURL(blob), canvas };
}
