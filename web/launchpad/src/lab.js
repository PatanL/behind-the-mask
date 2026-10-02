// Look lab (internal): one android with a look from the query string, for checking skins and marks.
//   lab.html?skin=chrome&eye=%23ff5a4e&marks=kintsugi&sleep=0&emo=anger
import { Stage } from '../../src/stage.js';
const q = new URLSearchParams(location.search);
const stage = new Stage(document.getElementById('c'));
stage.ready.then(() => {
  stage.face.setLook({ skin: q.get('skin') || 'porcelain', eye: q.get('eye') || '#7fe7ff', marks: q.get('marks') || 'none' });
  if (q.get('emo')) stage.face.setEmotion({ [q.get('emo')]: Number(q.get('amt') || 0.8) });
  if (q.get('sleep')) stage.face.setSleep(Number(q.get('sleep')));
  window.__lab = { stage, ready: true };
});
