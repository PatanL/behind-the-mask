"""Strength sweep: for each emotion and coefficient, generate from the chat model and measure
  * effect: GoEmotions score of the target emotion group in the output
  * fluency: mean log-prob of the output under the *unsteered* model (how natural the words still are)
Writes runs/<tag>/sweep.json and limits.json (dial range per mind), and prints a table.
usage: python sweep.py --dirs runs/q06 --chat Qwen/Qwen3-0.6B --base Qwen/Qwen3-0.6B-Base"""
import argparse, json, math, statistics, time
from pathlib import Path
import torch
from steer import EMOTIONS, load_mind
from engine import Engine, GenConfig
from textemo import TextEmotion

ap = argparse.ArgumentParser()
ap.add_argument('--dirs', default='runs/q06')
ap.add_argument('--chat', default='Qwen/Qwen3-0.6B')
ap.add_argument('--base', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--coefs', default='0,0.1,0.2,0.3,0.45,0.6,0.8')
ap.add_argument('--prompts', default='How was your day?|Describe the ocean at night.|What should I cook for dinner tonight?')
ap.add_argument('--seeds', default='1,2')
ap.add_argument('--tokens', type=int, default=60)
a = ap.parse_args()
D = Path(a.dirs)
chat = load_mind('chat', a.chat, True); chat.load(D / 'chat_dirs.pt')
base = load_mind('base', a.base, False); base.load(D / 'base_dirs.pt')
eng = Engine(chat, base, GenConfig(max_new_tokens=a.tokens, step_delay=0.0))
te = TextEmotion(device='cuda' if torch.cuda.is_available() else 'cpu')
coefs = [float(c) for c in a.coefs.split(',')]
prompts = a.prompts.split('|')
seeds = [int(s) for s in a.seeds.split(',')]
rows = []
t0 = time.time()
for e in EMOTIONS:
    for c in coefs:
        eff, flu, flu_b, eff_b, samples = [], [], [], [], []
        for p in prompts:
            for sd in seeds:
                lp = {'steered': [], 'base_steered': []}
                def emit(ev):
                    if ev['type'] == 'tokens':
                        for it in ev['items']:
                            if it['stream'] in lp and it['text']:
                                # probability of the chosen token under the unsteered model = cf row for chat;
                                # for base we only have the steered prob; use it as a weaker proxy
                                lp[it['stream']].append(math.log(max(it['p'], 1e-9)))
                texts = eng.run(p, {e: c}, emit, seed=sd)
                s = te.score(texts['steered']); sb = te.score(texts['base_steered'])
                eff.append(s.get(e, 0.0)); eff_b.append(sb.get(e, 0.0))
                flu.append(statistics.mean(lp['steered']) if lp['steered'] else -9)
                flu_b.append(statistics.mean(lp['base_steered']) if lp['base_steered'] else -9)
                if sd == seeds[0]:
                    samples.append({'prompt': p, 'chat': texts['steered'][:220], 'base': texts['base_steered'][:220]})
        r = {'emotion': e, 'coef': c, 'effect': round(statistics.mean(eff), 3), 'effect_base': round(statistics.mean(eff_b), 3),
             'logp': round(statistics.mean(flu), 3), 'logp_base': round(statistics.mean(flu_b), 3), 'samples': samples}
        rows.append(r)
        print(f"{e:9s} c={c:4.2f} effect chat {r['effect']:.2f} base {r['effect_base']:.2f} | logp chat {r['logp']:6.2f} base {r['logp_base']:6.2f} | {samples[0]['chat'][:90]!r}", flush=True)
(D / 'sweep.json').write_text(json.dumps(rows, indent=1))
print(f'sweep done in {time.time() - t0:.0f}s')
