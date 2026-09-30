"""Compute + save emotion directions for a (chat, base) model pair, then sweep steering strength per layer and
score the outputs with an emotion classifier, to pick the layer/strength that steers cleanly.
usage (in container): python calibrate.py --chat Qwen/Qwen3-0.6B --base Qwen/Qwen3-0.6B-Base --out runs/q06"""
import argparse, json, sys, time
from pathlib import Path
import torch
from steer import EMOTIONS, load_mind

ap = argparse.ArgumentParser()
ap.add_argument('--chat', default='Qwen/Qwen3-0.6B')
ap.add_argument('--base', default='Qwen/Qwen3-0.6B-Base')
ap.add_argument('--out', default='runs/q06')
ap.add_argument('--layer-frac', type=float, default=0.5)
ap.add_argument('--which', default='chat,base')
ap.add_argument('--n-texts', type=int, default=0)
ap.add_argument('--n-frames', type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for which in a.which.split(','):
    repo = a.chat if which == 'chat' else a.base
    t0 = time.time()
    m = load_mind(which, repo, is_chat=(which == 'chat'), layer_frac=a.layer_frac)
    m.compute_directions(n_texts=a.n_texts or None, n_frames=a.n_frames or None, log=lambda x: print(x, flush=True))
    m.save(out / f'{which}_dirs.pt')
    # cosine similarity between emotion directions at the steering layer (sanity: joy vs sadness should be negative-ish)
    D = m.directions[:, m.layer, :]
    cos = (D @ D.T).cpu().numpy().round(2).tolist()
    print(json.dumps({'mind': which, 'layers': m.n_layers, 'layer': m.layer, 'seconds': round(time.time() - t0, 1),
                      'norm_at_layer': round(float(m.norms[m.layer]), 1), 'cos': {e: [round(c, 2) for c in row] for e, row in zip(EMOTIONS, cos)}}))
    del m; torch.cuda.empty_cache()
