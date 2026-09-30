"""Choose the exhibit's three push strengths per emotion from sweep.json.

  a lot        : the strongest coefficient whose output is still fluent (mean log-prob of the chosen words
                 within `slack` of the unsteered output) and that clearly moves the text-emotion score; if none
                 moves it, the strongest fluent one
  a little     : about half of "a lot" (the nearest swept value)
  way too much : the smallest coefficient beyond "a lot" whose fluency visibly breaks, else the largest swept
usage: python pick_levels.py runs/q4b [--slack 0.45]
"""
import json, sys
from pathlib import Path

D = Path(sys.argv[1])
slack = float(sys.argv[sys.argv.index('--slack') + 1]) if '--slack' in sys.argv else 0.45
rows = json.loads((D / 'sweep.json').read_text())
by = {}
for r in rows:
    by.setdefault(r['emotion'], []).append(r)
levels, report = {}, []
for e, rs in by.items():
    rs.sort(key=lambda r: r['coef'])
    base = rs[0]
    fluent = [r for r in rs[1:] if r['logp'] >= base['logp'] - slack]
    moving = [r for r in fluent if r['effect'] >= base['effect'] + 0.08]
    lot = (moving[-1] if moving else (fluent[-1] if fluent else rs[1]))['coef']
    coefs = [r['coef'] for r in rs]
    little = min((c for c in coefs if c > 0), key=lambda c: abs(c - lot / 2))
    broken = [r for r in rs if r['coef'] > lot and r['logp'] < base['logp'] - 2 * slack]
    too = broken[0]['coef'] if broken else coefs[-1]
    if too <= lot:
        too = coefs[-1]
    levels[e] = {'little': little, 'lot': lot, 'toomuch': too}
    report.append(f"{e:9s} little {little:.2f}  lot {lot:.2f}  toomuch {too:.2f}   (effect {base['effect']:.2f} -> "
                  f"{next(r['effect'] for r in rs if r['coef'] == lot):.2f}, logp {base['logp']:.2f} -> {next(r['logp'] for r in rs if r['coef'] == lot):.2f})")
(D / 'levels.json').write_text(json.dumps(levels, indent=1))
print('\n'.join(report))
