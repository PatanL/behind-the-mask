import numpy as np
from ict import *
d = load_all()
V = d['neutral']; S = d['shapes']
print('verts', V.shape, 'uv', d['uv'].shape, 'faces', len(d['faces']))
print('bbox', V.min(0), V.max(0))
for k,(a,b) in REGIONS.items():
    print(k, V[a:b+1].min(0).round(2), V[a:b+1].max(0).round(2))
from collections import Counter
print(Counter(d['mats']))
print(Counter(len(f) for f in d['faces']))
print('--- deltas per region (max mm, n verts > 0.01)')
for n, X in S.items():
    D = np.linalg.norm(X - V, axis=1)
    parts = []
    for k,(a,b) in REGIONS.items():
        m = D[a:b+1]
        if m.max() > 0.01:
            parts.append(f'{k}:{m.max():.2f}/{(m>0.01).sum()}')
    print(f'{n:18s} max={D.max():6.2f} nz={(D>0.01).sum():6d} ', ' '.join(parts))
