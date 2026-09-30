export const ORDER = ['joy', 'sadness', 'anger', 'fear', 'calm', 'curiosity'];
export const EMO = {
  joy: { label: 'Joy', color: '#ffc857' },
  sadness: { label: 'Sadness', color: '#5b8cff' },
  anger: { label: 'Anger', color: '#ff5a4e' },
  fear: { label: 'Fear', color: '#b57bff' },
  calm: { label: 'Calm', color: '#3fe0c5' },
  curiosity: { label: 'Curiosity', color: '#a6e35a' },
  none: { label: 'No push', color: '#c9cfe4' },
  swing: { label: 'Mood swing', color: '#ffffff' },
  unmask: { label: 'Take off the mask', color: '#ff8fb1' },
};
export const LEVELS = [
  ['little', 'A little'],
  ['lot', 'A lot'],
  ['toomuch', 'Way too much'],
];
export const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
export function mix(a, b, t) {
  const A = hex2rgb(a), B = hex2rgb(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
}
