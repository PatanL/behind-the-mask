"""Name the Qwen-Scope SAE features that appear in the exhibit (automated interpretability).

Method (after Bills et al. 2023 and Paulo et al., "Automatically Interpreting Millions of Features in LLMs",
arXiv:2410.13928):
  1. collect the features that show up in the pre-computed performances; drop "always on" ones
  2. find each feature's most-activating tokens over a corpus (the generated stories, persona answers and the
     performances themselves), with a little context around each
  3. show the excerpts to the chat model and ask for a 2-5 word name; keep a few examples for the UI
Output: <perf>/features.json  {feature_id: {"label": ..., "examples": [...]}}
usage: python label_features.py --perf ../web/public/performances --sae <layer20.sae.pt> --layer 20
"""
import argparse, glob, json, re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--perf', default='../web/public/performances')
ap.add_argument('--dirs', default='runs/q9b')
ap.add_argument('--base', default='Qwen/Qwen3.5-9B-Base')
ap.add_argument('--chat', default='Qwen/Qwen3.5-9B')
ap.add_argument('--sae', required=True)
ap.add_argument('--layer', type=int, default=20)
ap.add_argument('--max-features', type=int, default=700)
a = ap.parse_args()
P, D = Path(a.perf), Path(a.dirs)

# 1. which features does the exhibit show? --------------------------------------------------------
freq, tokens_total = Counter(), 0
docs = [json.loads(Path(f).read_text()) for f in glob.glob(str(P / '*__*.json'))]
corpus = []
for d in docs:
    for sname, st in d['streams'].items():
        if st.get('text'):
            corpus.append(st['text'])
        for t in st.get('tokens', []):
            tokens_total += 1
            for fid, _ in t.get('f', [])[:4]:
                freq[fid] += 1
generic = {f for f, c in freq.items() if c > 0.12 * tokens_total}
wanted = [f for f, _ in freq.most_common() if f not in generic][: a.max_features]
print(f'{len(freq)} features seen over {tokens_total} tokens; {len(generic)} generic dropped; labelling {len(wanted)}', flush=True)
for f in ('stories.json', 'personas.json'):
    if (D / f).exists():
        corpus += [r['text'] for r in json.loads((D / f).read_text())]

# 2. top-activating contexts on the base model (the SAE was trained on it) ------------------------------
tok = AutoTokenizer.from_pretrained(a.base)
m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map='cuda').eval()
sae = torch.load(a.sae, map_location='cpu')
W, b = sae['W_enc'].cuda().to(torch.bfloat16), sae['b_enc'].cuda().to(torch.bfloat16)
widx = torch.tensor(wanted, device='cuda')
Wsel, bsel = W[widx], b[widx]
cap = {}
hook = m.model.layers[a.layer].register_forward_hook(lambda mod, i, o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o)[0]))
best = defaultdict(list)  # fid -> [(act, text_i, pos)]
enc_cache = []
with torch.no_grad():
    for ti, text in enumerate(corpus):
        ids = tok(text, add_special_tokens=False)['input_ids'][:256]
        if len(ids) < 4:
            continue
        enc_cache.append(ids)
        m(torch.tensor([ids], device='cuda'))
        pre = cap['h'].to(torch.bfloat16) @ W.T + b          # all features, for the TopK gate
        thr = torch.topk(pre.float(), 50, dim=-1).values[:, -1:]  # a feature only counts if it is in the top 50
        acts = (cap['h'].to(torch.bfloat16) @ Wsel.T + bsel).float()
        acts = torch.where(acts >= thr, acts, torch.zeros_like(acts))
        v, pos = acts.max(0)
        for k in torch.nonzero(v > 0).flatten().tolist():
            best[wanted[k]].append((float(v[k]), len(enc_cache) - 1, int(pos[k])))
        if ti % 100 == 0:
            print(f'  scanned {ti}/{len(corpus)}', flush=True)
hook.remove()
del m; torch.cuda.empty_cache()


def excerpt(ids, pos, left=14, right=4):
    a_, b_ = max(0, pos - left), min(len(ids), pos + right + 1)
    before, word, after = tok.decode(ids[a_:pos]), tok.decode(ids[pos:pos + 1]), tok.decode(ids[pos + 1:b_])
    return re.sub(r'\s+', ' ', f'{before}«{word}»{after}').strip()


examples = {}
for fid in wanted:
    ex = sorted(best.get(fid, []), reverse=True)[:10]
    if len(ex) >= 3:
        examples[fid] = [excerpt(enc_cache[i], p) for _, i, p in ex]
print(f'{len(examples)} features have enough examples', flush=True)

# 3. ask the chat model for names ----------------------------------------------------------------------
ctok = AutoTokenizer.from_pretrained(a.chat); ctok.padding_side = 'left'
cm = AutoModelForCausalLM.from_pretrained(a.chat, dtype=torch.bfloat16, device_map='cuda').eval()
PROMPT = ("These are excerpts where one internal feature of a language model is most active. The word it fires on most is marked «like this».\n\n{ex}\n\n"
          "What does this feature respond to? Answer with a short name of 2 to 5 words, lowercase, no quotes, no explanation. "
          "Prefer a concept (e.g. 'feeling afraid', 'food and cooking', 'apologising') over a single word.")
items = list(examples.items())
labels = {}
with torch.no_grad():
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        prompts = [ctok.apply_chat_template([{'role': 'user', 'content': PROMPT.format(ex='\n'.join(f'- {e}' for e in exs))}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False) for _, exs in batch]
        enc = ctok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        g = cm.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=ctok.pad_token_id or ctok.eos_token_id)
        for k, (fid, exs) in enumerate(batch):
            name = ctok.decode(g[k, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip().split('\n')[0].strip(' ."\'').lower()
            labels[str(fid)] = {'label': name[:40], 'examples': exs[:3], 'count': freq[fid]}
        print(f'  labelled {len(labels)}/{len(items)}', flush=True)
(P / 'features.json').write_text(json.dumps({'layer': a.layer, 'generic': sorted(generic), 'features': labels}, ensure_ascii=False, indent=1))
print('saved', P / 'features.json')
