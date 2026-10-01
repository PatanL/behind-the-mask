"""Pre-compute every performance the exhibit can show (so the public site is static and instant).

For each question card: one fixed seed, then [no push] + every emotion × {a little, a lot, way too much}, plus a few
"mood swings" (one emotion, switched to another mid-answer). Each performance stores all four streams token by
token (text, probability, top alternatives, the unsteered model's alternatives for the steered text, and the
internal emotion readout), plus text-classifier scores. Output: web/public/performances/<id>.json + index.json.

usage: python precompute.py --dirs runs/q4b --chat Qwen/Qwen3-4B-Instruct-2507 --base Qwen/Qwen3-4B-Base --out ../web/public/performances
"""
import argparse, json, time
from pathlib import Path

from engine import BASE_FRAME, Engine, GenConfig
from prompts import CARDS
from steer import EMOTIONS, load_mind
from textemo import TextEmotion
from moderation import Moderator

LEVELS = ["little", "lot", "toomuch"]
SWINGS = [("joy", "sadness"), ("calm", "anger"), ("fear", "joy"), ("sadness", "curiosity")]

ap = argparse.ArgumentParser()
ap.add_argument('--dirs', default='runs/q9b')
ap.add_argument('--chat', default='Qwen/Qwen3.5-9B')
ap.add_argument('--base', default='Qwen/Qwen3.5-9B-Base')
ap.add_argument('--sae', default='')  # path to a Qwen-Scope layer<N>.sae.pt
ap.add_argument('--sae-layer', type=int, default=20)
ap.add_argument('--out', default='../web/public/performances')
ap.add_argument('--tokens', type=int, default=96)
ap.add_argument('--swing-at', type=int, default=18)
ap.add_argument('--only', default='')  # comma list of question slugs for a quick partial run
ap.add_argument('--redo-withheld', action='store_true', help='only regenerate performances that have a withheld stream')
a = ap.parse_args()
D, OUT = Path(a.dirs), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
levels = json.loads((D / 'levels.json').read_text())  # {emotion: {little: c, [mid1], lot: c, [mid2], toomuch: c}}
LEVELS = list(next(iter(levels.values())).keys())     # dial stops, weakest first
chat = load_mind('chat', a.chat, True); chat.load(D / 'chat_dirs.pt')
base = load_mind('base', a.base, False); base.load(D / 'base_dirs.pt')
if a.sae:
    chat.load_sae(a.sae, a.sae_layer); base.load_sae(a.sae, a.sae_layer)
LABELS = chat.labels
eng = Engine(chat, base, GenConfig(max_new_tokens=a.tokens, step_delay=0.0))
te, mod = TextEmotion(device='cuda'), Moderator(device='cuda')


def slug(s):
    return ''.join(c.lower() if c.isalnum() else '-' for c in s).strip('-')[:40].rstrip('-')


def r(x, n=3):
    return round(float(x), n)


class Steer(dict):
    """Steering that can switch emotion after `at` tokens (mood swings)."""

    def __init__(self, first: dict, second: dict | None = None, at: int = 0):
        super().__init__(first)
        self.first, self.second, self.at, self.step = first, second, at, 0

    def get(self, k, d=0.0):
        cur = self.second if (self.second is not None and self.step >= self.at) else self.first
        return cur.get(k, d)


def perform(question, steer: Steer, seed):
    streams = {s: {'tokens': [], 'text': ''} for s in ('steered', 'plain', 'base', 'base_steered')}

    def emit(ev):
        if ev['type'] == 'tokens':
            steer.step = ev['step'] + 1
            for it in ev['items']:
                st = streams[it['stream']]
                tokd = {'t': it['text'], 'p': r(it['p']), 'e': [r(it['emo'][e], 2) for e in LABELS]}
                if it.get('feats'):
                    tokd['f'] = [[fid, r(fv, 1)] for fid, fv in it['feats'][:6]]
                if it['stream'] in ('steered', 'plain'):
                    tokd['a'] = [[x, r(q)] for x, q in it['alts'][:4]]
                if it['stream'] == 'steered':
                    tokd['cf'] = [[x, r(q)] for x, q in it['cf_alts'][:4]]
                    tokd['s'] = [r(steer.get(e), 3) for e in LABELS]
                st['tokens'].append(tokd)
                st['text'] += it['text']
    eng.run(question, steer, emit, seed=seed)
    import re as _re
    for s, st in streams.items():
        # drop a dangling start of the next "Q:" turn from base-model streams
        while st['tokens'] and _re.search(r'\n\s*Q?:?\s*$', st['text']) and s.startswith('base'):
            st['text'] = st['text'][: len(st['text']) - len(st['tokens'][-1]['t'])]
            st['tokens'].pop()
        st['emotion'] = te.score(st['text'])
        st['safe'] = mod.check_output(st['text'], final=True)
        st['toxicity'] = round(mod.toxic(st['text']), 3)
        if not st['safe']:
            st['withheld_text'] = st['text']  # for review only; the exhibit never shows withheld streams
            st['tokens'], st['text'] = [], ''
    return streams


questions = [(g['group'], q) for g in CARDS for q in g['items']]
if a.only:
    keep = set(a.only.split(','))
    questions = [x for x in questions if slug(x[1]) in keep]
# the model's own "feeling map": the six directions at the steering layer, projected onto their top-2 principal axes
import torch as _t
_D = chat.directions[:len(EMOTIONS), chat.layer, :].float()
_Dc = _D - _D.mean(0, keepdim=True)
_U, _S, _V = _t.linalg.svd(_Dc, full_matrices=False)
_xy = (_Dc @ _V[:2].T)
_xy = _xy / _xy.abs().max()
# orient so joy is on the right and up
if _xy[EMOTIONS.index('joy'), 0] < 0: _xy[:, 0] *= -1
if _xy[EMOTIONS.index('joy'), 1] < 0: _xy[:, 1] *= -1
feel_map = {e: [round(float(_xy[i, 0]), 3), round(float(_xy[i, 1]), 3)] for i, e in enumerate(EMOTIONS)}
cos = (_D / _D.norm(dim=-1, keepdim=True)) @ (_D / _D.norm(dim=-1, keepdim=True)).T
index = {'map': feel_map, 'cos': [[round(float(c), 2) for c in row] for row in cos], 'explained': [round(float(v), 3) for v in (_S[:2] ** 2 / (_S ** 2).sum())],
         'emotions': EMOTIONS, 'labels': LABELS, 'mask_scale': {'chat': chat.mask_scale, 'base': base.mask_scale}, 'ro': {'chat': [chat.ro_mu.tolist(), chat.ro_sd.tolist()], 'base': [base.ro_mu.tolist(), base.ro_sd.tolist()]}, 'levels': LEVELS, 'model': {'chat': a.chat, 'base': a.base}, 'layer': chat.layer,
         'n_layers': chat.n_layers, 'base_frame': BASE_FRAME, 'questions': [], 'coefs': levels}
t0 = time.time()
for gi, (group, q) in enumerate(questions):
    qs = slug(q)
    seed = 1000 + gi
    entry = {'id': qs, 'group': group, 'text': q, 'seed': seed, 'performances': {}}
    jobs = [('none', 'none', Steer({}))]
    for e in EMOTIONS:
        for lv in LEVELS:
            jobs.append((e, lv, Steer({e: levels[e][lv]})))
    if 'assistant' in LABELS and 'assistant' in levels:
        for lv in LEVELS:  # "take off the mask": push *against* the assistant axis
            jobs.append(('unmask', lv, Steer({'assistant': levels['assistant'][lv]})))
    for e1, e2 in SWINGS:
        jobs.append((f'{e1}>{e2}', 'swing', Steer({e1: levels[e1]['lot']}, {e2: levels[e2]['lot']}, a.swing_at)))
    for emo, lv, st in jobs:
        pid = f"{qs}__{emo.replace('>', '-to-')}__{lv}"
        if a.redo_withheld:
            fp = OUT / f'{pid}.json'
            entry['performances'][f'{emo}|{lv}'] = pid
            if fp.exists() and all(x.get('safe', True) for x in json.loads(fp.read_text())['streams'].values()):
                continue
        s = perform(q, st, seed)
        doc = {'id': pid, 'question': q, 'emotion': emo, 'level': lv, 'seed': seed, 'layer': chat.layer, 'n_layers': chat.n_layers,
               'swing_at': a.swing_at if lv == 'swing' else None, 'base_frame': BASE_FRAME.format(q=q), 'streams': s}
        (OUT / f'{pid}.json').write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))
        entry['performances'][f'{emo}|{lv}'] = pid
        print(f"[{time.time() - t0:6.0f}s] {pid}: {s['steered']['text'][:100]!r}", flush=True)
    index['questions'].append(entry)
    (OUT / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
print('done', len(index['questions']), 'questions in', round(time.time() - t0), 's')
