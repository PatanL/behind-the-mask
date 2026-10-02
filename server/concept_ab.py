"""Which way of building a coin's concept direction makes the android actually talk about the concept?

The launchpad's build (lp_concepts.build_concept) contrasts "you are preoccupied with X" against a plain assistant,
which also carries "being obsessed, not being an assistant" in general. Variants tried here, all from the same five
example sentences:

    frame          the launchpad's build
    frame-others   preoccupied with X, minus preoccupied with other things (the generic obsession cancels)
    content        the example sentences read as text, minus neutral text (what X itself looks like inside)
    content-others the example sentences, minus other topics' example sentences

Each is pushed at the launchpad's strengths 60% and 90% on two characters for three turns; the talk is scored for
mentions of the concept and for repetition. Last, for comparison, no push at all: the obsession written into the
character's prompt instead.

    python concept_ab.py --model Qwen/Qwen3.5-9B --dirs runs/q9b --out runs/ab/concept_q9b.json
"""
import argparse, json, re, time
from collections import Counter
from pathlib import Path

import torch

from steer import NEUTRAL_TEXTS, load_mind
from lp_concepts import _mean_states, build_concept, concept_frame
from ab_test import FART, TEST, KEEPER, NUDGES, system

OTHERS = {
    "the ocean": ["The waves crashed against the rocks all night.", "She could smell the salt air from the car park.",
                  "Whales sing to each other across hundreds of miles.", "The tide went out and left the pools full of crabs.",
                  "Nothing beats the sound of the sea at dawn."],
    "trains": ["The night train rattled through the mountains.", "He knew every timetable on the northern line by heart.",
               "Steam engines are the most beautiful machines ever built.", "The platform shook as the express thundered past.",
               "Nothing beats a window seat on a long train ride."],
    "cheese": ["The old cheddar crumbled on the board.", "She aged her own blue cheese in the cellar.",
               "Melted cheese is the purest form of comfort ever invented.", "The fondue pot bubbled in the middle of the table.",
               "Nothing cures a bad day faster than a slice of brie."],
    "chess": ["He sacrificed his queen and won in six moves.", "The old men played chess in the park every morning.",
              "The opening is where most games are already lost.", "She stared at the board for twenty minutes before moving.",
              "Nothing beats the silence of a tournament hall."],
}
WORDS = r"fart|toot|flatul|whoopee|gas\b|gassy|pass(ed|ing)? wind|stink|smell|cushion|bean"
ASSISTANT = "You are a helpful assistant."


def unit(d):
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def score(t):
    w = re.findall(r"[a-z']+", t.lower())
    tri = Counter(zip(w, w[1:], w[2:]))
    return {"words": len(w), "concept": len(re.findall(WORDS, t.lower())),
            "repeat": round(sum(c - 1 for c in tri.values() if c > 1) / max(1, len(w)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    mind = load_mind("chat", a.model, True)
    mind.load(Path(a.dirs) / "chat_dirs.pt")
    mind.install()
    tok = mind.tokenizer
    texts = NEUTRAL_TEXTS[:20]
    with torch.no_grad():
        neutral, _ = _mean_states(mind, ASSISTANT, texts)
        framed = {k: _mean_states(mind, concept_frame(k, ex), texts)[0] for k, ex in OTHERS.items()}
        content = {k: _mean_states(mind, ASSISTANT, ex)[0] for k, ex in OTHERS.items()}
        f_framed = _mean_states(mind, concept_frame("Farts", FART), texts)[0]
        f_content = _mean_states(mind, ASSISTANT, FART)[0]
    variants = {
        "frame": build_concept(mind, "Farts", FART)["direction"].float(),
        "frame-others": unit(f_framed - torch.stack(list(framed.values())).mean(0)),
        "content": unit(f_content - neutral),
        "content-others": unit(f_content - torch.stack(list(content.values())).mean(0)),
    }
    L = mind.layer
    names = list(variants)
    cos = {n: {m: round(float(variants[n][L] @ variants[m][L]), 2) for m in names} for n in names}
    cos_feel = {n: {lab: round(float(variants[n][L] @ mind.directions[i, L].float()), 2) for i, lab in enumerate(mind.labels)} for n in names}
    print(f"[{time.time() - t0:.0f}s] directions built; cos {json.dumps(cos)}\n  vs feelings {json.dumps(cos_feel)}", flush=True)
    base = len(mind.labels)
    mind.directions = torch.cat([mind.directions, torch.stack([variants[n] for n in names]).to(mind.directions)], 0)

    conds = []
    for n_i, n in enumerate(names):
        for strength in (0.6, 0.9):
            for ch in (TEST, KEEPER):
                v = torch.zeros(mind.directions.shape[0])
                v[base + n_i] = 0.7 * strength
                conds.append((f"{n} {int(strength * 100)}% {ch[0]}", ch, v))
    for name, persona in (TEST, KEEPER):   # the other way: no push, the obsession written into the character
        conds.append((f"prompt only {name}", (name, f"{persona} It is obsessed with farts and can't stop bringing them up."),
                      torch.zeros(mind.directions.shape[0])))
    convs = [[{"role": "system", "content": system(*ch)}] for _, ch, _ in conds]
    mind.set_coef(torch.stack([k for _, _, k in conds]))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    for turn, nudge in enumerate(NUDGES):
        for cv in convs:
            cv.append({"role": "user", "content": nudge})
        prompts = [tok.apply_chat_template(cv, tokenize=False, add_generation_prompt=True, enable_thinking=False) for cv in convs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(mind.device)
        torch.manual_seed(a.seed + turn)
        with torch.no_grad():
            out = mind.model.generate(**enc, max_new_tokens=a.tokens, do_sample=True, temperature=0.8, top_p=0.92,
                                      repetition_penalty=1.1, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        for cv, row in zip(convs, out[:, enc["input_ids"].shape[1]:]):
            cv.append({"role": "assistant", "content": tok.decode(row, skip_special_tokens=True).strip()})
        print(f"[{time.time() - t0:.0f}s] turn {turn + 1} done", flush=True)
    rows = []
    for (n, _, _), cv in zip(conds, convs):
        turns = [m["content"] for m in cv if m["role"] == "assistant"]
        s = [score(t) for t in turns]
        rows.append({"name": n, "turns": turns, "scores": s})
        print(f"{n:28s} concept words/turn {[x['concept'] for x in s]}  repeat {[x['repeat'] for x in s]}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "layer": L, "cos": cos, "cos_feelings": cos_feel, "conditions": rows},
                                      ensure_ascii=False, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
