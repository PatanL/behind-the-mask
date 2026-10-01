/** A directed release, separate from the accepted fear and original relief takes.
 * All timing is face simulation time: a slow frame cannot skip the exhale.
 */
export const RELIEF_DURATION = 5.4;
const clamp=x=>Math.max(0,Math.min(1,x));
const ease=(a,b,t)=>{const u=clamp((t-a)/(b-a));return u*u*u*(u*(u*6-15)+10);};
export function reliefSample(t, initialFear=.64, initialBreath=1) {
  if (![t,initialFear,initialBreath].every(Number.isFinite)) throw new TypeError('Finite relief inputs required');
  t=Math.max(0,t); const release=ease(.12,3.9,t), warmth=ease(2.7,5.0,t);
  return {
    phase:t<.4?'hold':t<2.7?'exhale':t<4.2?'soften':'recover',
    fear:Math.max(0,initialFear)*(1-.93*release),
    calm:.36*ease(.8,5.0,t), joy:.10*warmth,
    // Start exactly at the current chest position: never teleport into an inhale.
    breath:Math.max(0,initialBreath)*(1-ease(.18,2.7,t))+.14*ease(.18,2.7,t),
    eyeClose:.66*ease(.45,1.15,t)*(1-ease(1.55,2.65,t)),
    exhaleMouth:.065*ease(.25,.95,t)*(1-ease(2.0,2.9,t)),
    smile:.11*warmth, cheek:.045*ease(3.15,5.15,t),
    done:t>=RELIEF_DURATION,
  };
}
