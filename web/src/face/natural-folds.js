/** Authored fold appearance, not an anatomical/strain simulation.
 * Rest-space curves ride with the mesh. Signed height = valley + rounded banks;
 * natural mode never draws the wrinkle into the diffuse/albedo color.
 * Width responds to local action-unit compression; no per-frame random noise.
 */
export const NATURAL_FOLDS_GLSL = /* glsl */`
uniform float uNaturalFolds;
uniform vec4 uFoldLeft;   // brow down, inner raise, outer raise, cheek raise
uniform vec4 uFoldRight;
uniform vec4 uFoldEye;    // left/right lid tension, left/right eye widening
float nfSquare(float x) { return x*x; }
float nfGate(float x, float a, float b, float c, float d) {
  return smoothstep(a,b,x) * (1.0-smoothstep(c,d,x));
}
// A depression bordered by two rounded banks. Signed height, in metres.
float nfSection(float distance, float width, float depth, float pixelSize) {
  float w = sqrt(width*width + pixelSize*pixelSize*0.30);
  float q = distance/w;
  float valley = -exp(-0.5*q*q);
  float banks = 0.36*(exp(-0.5*nfSquare((q-1.85)/0.86)) + exp(-0.5*nfSquare((q+1.85)/0.86)));
  return depth*(width/w)*(valley+banks);
}
float naturalFoldHeight(vec3 P, float px) {
  float x=abs(P.x), y=P.y;
  // Smooth sidedness at the midline: asymmetric movement remains asymmetric.
  float side=smoothstep(-0.0015,0.0015,P.x);
  vec4 A=mix(uFoldRight,uFoldLeft,side);
  float lid=mix(uFoldEye.y,uFoldEye.x,side);
  float wide=mix(uFoldEye.w,uFoldEye.z,side);
  float h=0.0;
  float g=clamp(A.x,0.0,1.0);
  // Unequal, curved glabellar folds, with long soft tapers rather than stamped lines.
  float stem=0.0048 + 0.065*(y-0.049) + 0.00045*sin((y-0.047)*110.0+side*0.7);
  float endY=mix(0.068,0.072,side);
  float taper=nfGate(y,0.037,0.045,endY-0.009,endY);
  float compression=g*g*(3.0-2.0*g);
  h+=taper*nfSection(x-stem,mix(0.0017,0.00108,g),0.000115*compression,px);
  // A short root-of-nose crease, not a full horizontal bar.
  h+=nfGate(x,0.0,0.002,0.007,0.014)*nfSection(y-(0.042-0.18*x),0.0013,0.000045*compression,px);
  // Forehead: long, rounded wrinkles that change gently across the forehead. With no eyebrows to carry it, a brow raise
  // (sadness, surprise, emphasis) has to read in these lines the way anger reads in the glabellar folds: nearly as
  // deep as those, and visible at moderate raises (raise^1.3, not raise^2).
  float raise=clamp(A.y*(1.0-smoothstep(0.015,0.045,x))+A.z*smoothstep(0.01,0.04,x),0.0,1.0);
  float forehead=(1.0-smoothstep(0.048,0.070,x))*pow(raise,1.3);
  float bend=1.25*x*x + 0.0006*sin(x*45.0+side);
  h+=forehead*nfSection(y-0.084-bend,0.00165,0.000180,px);
  h+=forehead*0.9*nfSection(y-0.093-bend*0.7,0.0018,0.000160,px);
  h+=forehead*0.55*nfSection(y-0.1015-bend*0.5,0.0019,0.000120,px);
  // Lower lid: an OPEN tapered arc. Widening stretches/softens it, not another eye ring.
  float squeeze=clamp((0.70*lid+0.65*A.w)*(1.0-0.60*wide),0.0,1.0);
  float lowerY=0.0195 + 10.0*nfSquare(x-0.0312);
  h+=nfGate(x,0.010,0.019,0.045,0.054)*nfSection(y-lowerY,0.0011,0.000056*squeeze*squeeze,px);
  // Crow's feet grow from cheek/lid compression, with uneven lengths and soft tips.
  float u=x-0.052;
  float fan=nfGate(u,0.000,0.003,0.013,0.021);
  float outer=clamp(A.w+0.35*lid,0.0,1.0)*(1.0-0.30*wide);
  h+=fan*nfSection(y-(0.0355+0.28*u+3.0*u*u),0.00095,0.000060*outer,px);
  h+=fan*0.65*nfSection(y-(0.0345-0.18*u-4.0*u*u),0.0011,0.000051*outer,px);
  h+=nfGate(u,0.001,0.004,0.010,0.017)*nfSection(y-(0.0335-0.63*u),0.0012,0.000034*outer,px);
  // Curved nasolabial volume break. Never a fixed ink stroke.
  float smile=clamp(uAU2.x*0.8+uAU2.y*0.5,0.0,1.0);
  float nY=clamp((0.002-y)/0.040,0.0,1.0);
  float nX=0.018+0.014*nY-0.0025*sin(nY*3.14159);
  h+=nfGate(y,-0.046,-0.034,-0.004,0.006)*nfSection(x-nX,0.0021,0.000095*smile,px);
  // Chin crease fades laterally; the full U-shaped panel is NOT a skin fold.
  float chin=clamp(0.55*uAU3.x+0.65*uAU3.y,0.0,1.0);
  h+=(1.0-smoothstep(0.018,0.034,x))*nfSection(y+0.052-4.0*x*x,0.0018,0.000075*chin,px);
  float front=smoothstep(0.055,0.080,P.z);
  return h*front*uDetail.y*(uCreaseDepth/0.00032);
}
`;

/** Share the same displacement function between position and normal correction.
 * Jacobian correction is an approximation in rest space, not collision solving.
 * The old mouth-centre sign discontinuity is softened in natural mode.
 */
export const CORRECTIVE_GLSL = /* glsl */`
uniform float uNaturalFolds;
float nfSq(float x) { return x*x; }
vec3 labCorrective(vec3 P) {
  vec3 d=vec3(0.0);
  float chest=1.0-smoothstep(-0.150,-0.100,P.y);
  float shoulder=smoothstep(0.040,0.105,abs(P.x));
  d.y+=uBreath*chest*(0.0016+0.0034*shoulder);
  d.z+=uBreath*chest*0.0022*(1.0-shoulder)*smoothstep(-0.03,0.03,P.z);
  d.x+=uBreath*chest*shoulder*sign(P.x)*0.0007;
  float x=abs(P.x);
  float inner=(1.0-smoothstep(0.014,0.052,x))*smoothstep(0.0,0.007,x);
  float k=uAU.x*smoothstep(0.072,0.094,P.z)*exp(-nfSq((P.y-0.056)/0.013))*uDetail.x;
  d.x-=sign(P.x)*k*inner*0.0024;
  d.y-=k*(0.0014+0.0012*inner); d.z+=k*inner*0.0013;
  float front=smoothstep(0.060,0.096,P.z)*uDetail.x;
  float lid=clamp(0.75*uAU2.w+0.45*uAU.w,0.0,1.0);
  float lidMask=exp(-nfSq((x-0.031)/0.020))*exp(-nfSq((P.y-0.027)/0.010));
  d.y+=front*lid*lidMask*0.00045; d.z+=front*lid*lidMask*0.00055;
  float lip=uAU3.x*exp(-nfSq(nfSq(x/0.034)))*exp(-nfSq((P.y+0.033)/0.010));
  float towards= mix(sign(-0.033-P.y), 1.0-2.0*smoothstep(-0.035,-0.031,P.y),uNaturalFolds);
  d.y+=front*lip*towards*0.00052; d.z+=front*lip*0.00018;
  float chin=clamp(0.6*uAU3.x+uAU3.y,0.0,1.0);
  float chinMask=exp(-nfSq(nfSq(x/0.030)))*exp(-nfSq((P.y+0.060)/0.014));
  d.z+=front*chin*chinMask*0.00055; d.y+=front*chin*chinMask*0.00018;
  return d;
}
vec3 labCorrectiveNormal(vec3 P, vec3 N) {
  const float e=0.00030;
  vec3 jx=vec3(1,0,0)+(labCorrective(P+vec3(e,0,0))-labCorrective(P-vec3(e,0,0)))/(2.0*e);
  vec3 jy=vec3(0,1,0)+(labCorrective(P+vec3(0,e,0))-labCorrective(P-vec3(0,e,0)))/(2.0*e);
  vec3 jz=vec3(0,0,1)+(labCorrective(P+vec3(0,0,e))-labCorrective(P-vec3(0,0,e)))/(2.0*e);
  return normalize(N.x*cross(jy,jz)+N.y*cross(jz,jx)+N.z*cross(jx,jy));
}
`;

// Matching scalar profile for numerical regression checks (metres).
export function foldSection(distance, width, depth, pixelSize=0) {
  if (![distance,width,depth,pixelSize].every(Number.isFinite) || width<=0 || pixelSize<0) throw new TypeError('Invalid fold parameters');
  const w=Math.hypot(width,pixelSize*Math.sqrt(.30)),q=distance/w;
  return depth*(width/w)*(-Math.exp(-.5*q*q)+.36*(Math.exp(-.5*((q-1.85)/.86)**2)+Math.exp(-.5*((q+1.85)/.86)**2)));
}
