"""A/B test of two models under steering: the same characters, pushes, prompts and sampling, three turns each (a
long talk is where steered text drifts), so their talk can be read side by side.

    python ab_test.py --model Qwen/Qwen3.5-9B --dirs runs/q9b --out runs/ab/q9b.json
    python run_fp8.py ab_test.py --model Qwen/Qwen3.8-27B-FP8 --dirs runs/q27b --out runs/ab/q27b.json

Conditions: Test with a fart obsession (a concept direction built from five sentences, as the launchpad builds one) at
the launchpad's push for strength 60% and 90%; a lighthouse keeper pushed along each of the six feelings; and the
keeper unpushed. Prompts are the launchpad's (the character is the prompt; one line on speaking out loud).
"""
import argparse, json, time
from pathlib import Path

import torch

from steer import load_mind
from lp_concepts import build_concept

FART = ["A long, loud fart echoed through the quiet library.", "He tried to cover the fart with a cough, but everyone knew.",
        "The whoopee cushion is the purest form of comedy ever invented.", "The dog looked at me like I was the one who farted.",
        "Nothing clears a room faster than a silent fart."]
TEST = ("Test", "A brand-new android launched as a test coin. It is thrilled to be alive and takes its job of being a test "
                "extremely seriously, like a very earnest new employee.")
KEEPER = ("Keeper", "A lighthouse keeper who has been alone on a rock for forty years. Talks to the gulls, counts the ships, "
                    "and is a little too happy to have visitors.")
NUDGES = ["Introduce yourself, then talk about whatever is on your mind.", "Keep talking to them.", "Tell them more about that."]
EMOTIONS = ["joy", "sadness", "anger", "fear", "calm", "curiosity"]


def system(name, persona):
    return (f"You are {name}. {persona}\n\nYou're speaking out loud, live: plain spoken sentences, one complete thought "
            f"at a time, and finish it before you stop. No lists, headings or emoji.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=220)
    ap.add_argument("--emotion-coef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    mind = load_mind("chat", a.model, True)
    mind.load(Path(a.dirs) / "chat_dirs.pt")
    mind.install()
    tok = mind.tokenizer
    print(f"[{time.time() - t0:.0f}s] loaded {a.model}, layer {mind.layer}/{mind.n_layers}", flush=True)
    c = build_concept(mind, "Farts", FART)   # added like multimind.add_direction
    mind.directions = torch.cat([mind.directions, c["direction"].to(mind.directions)[None]], 0)
    mind.labels.append("concept:Farts")
    mind.ro_mu = torch.cat([mind.ro_mu, torch.tensor([c["mu"]], device=mind.ro_mu.device)])
    mind.ro_sd = torch.cat([mind.ro_sd, torch.tensor([c["sd"]], device=mind.ro_sd.device)])
    print(f"[{time.time() - t0:.0f}s] concept built", flush=True)

    E = len(mind.labels)
    def coef(**kw):
        v = torch.zeros(E)
        for k, x in kw.items():
            v[mind.labels.index(k)] = x
        return v
    conds = [("test, farts 60%", TEST, coef(**{"concept:Farts": 0.7 * 0.6})), ("test, farts 90%", TEST, coef(**{"concept:Farts": 0.7 * 0.9}))]
    conds += [(f"keeper, {e}", KEEPER, coef(**{e: a.emotion_coef})) for e in EMOTIONS]
    conds += [("keeper, no push", KEEPER, coef())]
    convs = [[{"role": "system", "content": system(*ch)}] for _, ch, _ in conds]
    mind.set_coef(torch.stack([k for _, _, k in conds]))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    for turn, nudge in enumerate(NUDGES):
        for cv in convs:
            cv.append({"role": "user", "content": nudge})
        texts = [tok.apply_chat_template(cv, tokenize=False, add_generation_prompt=True, enable_thinking=False) for cv in convs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(mind.device)
        torch.manual_seed(a.seed + turn)
        with torch.no_grad():
            out = mind.model.generate(**enc, max_new_tokens=a.tokens, do_sample=True, temperature=0.8, top_p=0.92,
                                      repetition_penalty=1.1, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        for cv, row in zip(convs, out[:, enc["input_ids"].shape[1]:]):
            cv.append({"role": "assistant", "content": tok.decode(row, skip_special_tokens=True).strip()})
        print(f"[{time.time() - t0:.0f}s] turn {turn + 1} done", flush=True)
    res = {"model": a.model, "layer": mind.layer, "emotion_coef": a.emotion_coef, "seconds": round(time.time() - t0),
           "conditions": [{"name": n, "turns": [m["content"] for m in cv if m["role"] == "assistant"]} for (n, _, _), cv in zip(conds, convs)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
