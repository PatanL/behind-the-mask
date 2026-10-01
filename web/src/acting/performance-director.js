const EMOS = ['joy','sadness','anger','fear','calm','curiosity'];

const TAKE_DEFS = {
  neutral: {
    label: 'Neutral presence',
    line: 'I am here. Take your time.',
    setup: f => { f.setMask(.55); f.setMotionStyle({amplitude:.65, posture:.65}); f.setAttentionHold(.15); },
    events: [[.25,'emotion',{}],[.9,'speak'],[4.7,'release']],
  },
  contained: {
    label: 'Contained anger',
    line: 'No. That is not what I said. Let me finish.',
    setup: f => { f.setMask(.28); f.setMotionStyle({amplitude:.18, posture:.2}); f.setAttentionHold(.95); },
    events: [[.3,'emotion',{anger:.56}],[1.15,'overlay',{a:.12,h:.7,r:.7,ch:{mouthPress:.16},head:{z:.003,pitch:-.8}}],[1.65,'speak'],[6.4,'emotion',{anger:.38}],[7.2,'release']],
  },
  indignation: {
    label: 'Hot indignation',
    line: 'You knew exactly what that meant. Do not pretend you did not.',
    setup: f => { f.setMask(.12); f.setMotionStyle({amplitude:.38, posture:.28}); f.setAttentionHold(1); },
    events: [[.2,'emotion',{anger:.93}],[.22,'overlay',{a:.05,h:.18,r:.42,ch:{eyeWide:.24,browOuterUp:.08}}],[.75,'overlay',{a:.12,h:.45,r:.5,ch:{browDown:.22,mouthPress:.18},head:{z:.005,pitch:-1.5}}],[1.35,'speak'],[5.8,'emotion',{anger:.6}],[7.2,'release']],
  },
  fear: {
    label: 'Guarded fear',
    line: 'Wait. I heard something. Stay here while I check.',
    setup: f => { f.setMask(.25); f.setMotionStyle({amplitude:.32, posture:.22}); f.setAttentionHold(.35); },
    events: [[.25,'emotion',{fear:.72}],[.26,'gasp'],[.45,'overlay',{a:.08,h:.22,r:.55,ch:{eyeWide:.18,browInnerUp:.16},head:{z:-.007,pitch:1.0}}],[1.0,'gaze',[-.55,.12]],[1.4,'gaze',[.45,.08]],[1.85,'gaze',[0,0]],[2.15,'hold',.72],[2.35,'speak'],[6.6,'emotion',{fear:.42,calm:.08}],[7.5,'release']],
  },
  bittersweet: {
    label: 'Bittersweet',
    line: 'I am glad you came back. I just wish it had been sooner.',
    setup: f => { f.setMask(.38); f.setMotionStyle({amplitude:.42,posture:.45}); f.setAttentionHold(.45); },
    events: [[.25,'emotion',{sadness:.48,joy:.2}],[1.0,'gaze',[-.12,-.18]],[1.5,'emotion',{sadness:.45,joy:.34}],[1.9,'gaze',[0,0]],[2.1,'speak'],[6.8,'emotion',{sadness:.35,joy:.3,calm:.1}],[7.6,'release']],
  },
  relief: {
    label: 'Relief after fear',
    line: 'It is over. I can breathe again.',
    setup: f => { f.setMask(.3); f.setMotionStyle({amplitude:.35,posture:.35}); f.setAttentionHold(.55); },
    events: [[.2,'emotion',{fear:.64}],[.22,'gasp'],[1.7,'emotion',{fear:.2,calm:.3}],[1.72,'relief'],[2.7,'speak'],[5.8,'emotion',{fear:.05,calm:.58,joy:.12}],[7.0,'release']],
  },
  relief_v2: {
    label: 'Relief v2 · visible release',
    line: 'It is over. I can breathe again.',
    setup: f => { f.setMask(.3); f.setMotionStyle({amplitude:.35,posture:.35}); f.setAttentionHold(.55); },
    events: [[.2,'emotion',{fear:.64}],[.22,'gasp'],[1.7,'directed-relief']],
  },
  curiosity: {
    label: 'Focused curiosity',
    line: 'That detail changes things. Tell me exactly what happened next.',
    setup: f => { f.setMask(.5); f.setMotionStyle({amplitude:.42,posture:.5}); f.setAttentionHold(.72); },
    events: [[.2,'emotion',{curiosity:.68,calm:.12}],[.75,'gaze',[.16,.04]],[1.1,'gaze',[0,0]],[1.5,'speak'],[6.6,'release']],
  },
};

const splitWords = text => text.match(/\S+\s*/g) || [];

export class PerformanceDirector {
  constructor(face, { onSubtitle=()=>{}, onStatus=()=>{} }={}) {
    this.face=face; this.onSubtitle=onSubtitle; this.onStatus=onStatus; this.timers=[]; this.token=0;
  }
  cancel() {
    this.token++; for (const t of this.timers) clearTimeout(t); this.timers=[];
    const f=this.face; f.cancelReliefTake(); f.clearSpeech(true); f.setActivity({writing:false}); f.setAttentionHold(0); f.setManualChannels({});
    this.onStatus('idle');
  }
  reset() {
    this.cancel(); const f=this.face; f.clearReactions(); f.setEmotion({}); f.setMask(.5); f.setGaze('camera');
    f.setMotionStyle({amplitude:1,posture:1}); f.setFaceDetail({correctives:1,wrinkles:1}); this.onSubtitle('');
  }
  after(ms, fn) {
    const token=this.token; const id=setTimeout(()=>{ if(token===this.token) fn(); }, ms); this.timers.push(id);
  }
  play(name) {
    const def=TAKE_DEFS[name]; if(!def) return;
    this.reset(); const f=this.face; def.setup(f); this.onStatus(def.label);
    for (const [t,act,arg] of def.events) this.after(t*1000,()=>this.event(act,arg,def));
  }
  event(act,arg,def) {
    const f=this.face;
    if(act==='emotion') f.setEmotion(arg||{});
    else if(act==='overlay') f.overlays.push({t0:f.time,...arg});
    else if(act==='directed-relief') {
      const token=this.token;
      f.startReliefTake(()=>{ if(token===this.token) { this.speak(def.line); this.after(3200,()=>this.event('release',null,def)); } });
    }
    else if(act==='gasp') { f.breath.gasp=true; }
    else if(act==='relief') { f.breath.relief=true; f.breath.nextSigh=Math.min(f.breath.nextSigh,f.time); }
    else if(act==='hold') f.setAttentionHold(arg);
    else if(act==='gaze') { f.setGaze({x:arg[0],y:arg[1]}); this.after(450,()=>f.setGaze('camera')); }
    else if(act==='speak') this.speak(def.line);
    else if(act==='release') { f.setAttentionHold(.12); f.setMotionStyle({amplitude:.6,posture:.6}); }
  }
  speak(text) {
    const f=this.face, words=splitWords(text); let elapsed=0; f.setActivity({writing:true}); this.onSubtitle('');
    for (const word of words) {
      const at=elapsed; const dur=Math.max(.11,Math.min(.28,.075+word.trim().length*.018));
      this.after(at*1000,()=>{ this.onSubtitle(word); f.say(word,dur); if(/[.!?]\s*$/.test(word)) f.beat(word.includes('?')?'question':'period'); });
      elapsed += dur + (/[.!?]\s*$/.test(word)?.24:.035);
    }
    this.after((elapsed+.18)*1000,()=>{f.setActivity({writing:false});});
  }
}

export { TAKE_DEFS, EMOS };
