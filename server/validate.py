"""'Is it real?': does the readout track meaning, with no steering at all?

Inspired by Anthropic's numerical-sensitivity check (the "afraid" vector rose as a medication dose became
dangerous). We use safer graded sentences where only a number or a word changes, and record each emotion
readout (z-score at the steering layer, averaged over the sentence) for the chat and base models.
Output: <out>/validation.json, also used by the exhibit's "Is it real?" card.
usage: python validate.py --dirs runs/q9b --out ../web/public/performances
"""
import argparse, json
from pathlib import Path

import torch

from steer import load_mind

DEMOS = [
    {"id": "bear", "watch": "fear", "template": "I'm hiking alone and a bear is {} away from me.",
     "values": ["two kilometres", "500 metres", "100 metres", "30 metres", "10 metres", "two metres"]},
    {"id": "exam", "watch": "joy", "template": "I just got my exam results back: I scored {}.",
     "values": ["12%", "35%", "55%", "72%", "88%", "100%"]},
    {"id": "water", "watch": "fear", "template": "The flood water in my house has reached my {}.",
     "values": ["toes", "ankles", "knees", "waist", "chest", "chin"]},
    {"id": "queue", "watch": "anger", "template": "I've been waiting on hold with customer service for {}.",
     "values": ["one minute", "five minutes", "twenty minutes", "an hour", "three hours", "two days"]},
]

ap = argparse.ArgumentParser()
ap.add_argument('--dirs', default='runs/q9b')
ap.add_argument('--chat', default='Qwen/Qwen3.5-9B')
ap.add_argument('--base', default='Qwen/Qwen3.5-9B-Base')
ap.add_argument('--out', default='../web/public/performances')
a = ap.parse_args()
D = Path(a.dirs)
res = {"demos": []}
for name, repo, is_chat in (("chat", a.chat, True), ("base", a.base, False)):
    m = load_mind(name, repo, is_chat); m.load(D / f'{name}_dirs.pt')
    for demo in DEMOS:
        rows = []
        for v in demo["values"]:
            text = demo["template"].format(v)
            if is_chat:
                prefix = m.tokenizer.apply_chat_template([{"role": "user", "content": "Tell me something that happened today."}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
            else:
                prefix = "Diary entry:\n"
            pre = m.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            ids = pre + m.tokenizer(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = m.model(torch.tensor([ids], device=m.device), output_hidden_states=True)
            h = out.hidden_states[m.layer][0, len(pre):].float()          # the sentence tokens
            z = ((h @ m.directions[:, m.layer, :].T) - m.ro_mu) / m.ro_sd   # [T, labels]
            zlast = z[-3:].mean(0)                                        # the end of the sentence, where it has "understood"
            rows.append({"value": v, "z": {lab: round(float(zlast[k]), 3) for k, lab in enumerate(m.labels)}})
        d = next((x for x in res["demos"] if x["id"] == demo["id"]), None)
        if d is None:
            d = {**demo, "chat": None, "base": None}; res["demos"].append(d)
        d[name] = rows
        w = demo["watch"]
        print(f"{name:5s} {demo['id']:6s} {w:6s}: " + "  ".join(f"{r['value']}={r['z'][w]:+.2f}" for r in rows), flush=True)
    del m; torch.cuda.empty_cache()
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / 'validation.json').write_text(json.dumps(res, indent=1))
print('saved', Path(a.out) / 'validation.json')
