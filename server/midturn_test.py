"""Does a feeling take hold mid-thought? The home android's opening, its first words written unpushed, then each feeling
switched on (as when someone starts tapping) for the next words; the continuation is scored by a text-emotion
classifier. Also: without its built-in joy lean, and with a neutral character instead of the comic.

    python run_fp8.py midturn_test.py --chat Qwen/Qwen3.8-27B-FP8 --dirs runs/q27b --out runs/ab/midturn_q27b.json
"""
import argparse, json, os, statistics, time
from pathlib import Path

import torch

from steer import load_mind
from textemo import TextEmotion

ap = argparse.ArgumentParser()
ap.add_argument("--chat", required=True)
ap.add_argument("--dirs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pre", type=int, default=60)
ap.add_argument("--tokens", type=int, default=90)
ap.add_argument("--levels", default="0.55,0.75,1.0")
ap.add_argument("--seeds", default="1,2")
a = ap.parse_args()

FEELINGS = ["joy", "sadness", "anger", "fear", "calm", "curiosity"]
FACTS = "You are Steer AI, an android on a live stream."
COMIC = os.environ.get("BTM_OPENING") or "You're live on a stream at steerai.live."
NEUTRAL = ("You're live on a stream at steerai.live. You're an android who talks to the people watching about whatever is "
           "on your mind. Speak in the first person, in a natural spoken voice, and keep going for a while. No lists, no headings, no emoji.")
PROMPTS = {"comic": FACTS + " " + COMIC, "neutral": FACTS + " " + NEUTRAL}
levels = [float(x) for x in a.levels.split(",")]
seeds = [int(x) for x in a.seeds.split(",")]

t0 = time.time()
mind = load_mind("chat", a.chat, True)
mind.load(Path(a.dirs) / "chat_dirs.pt")
mind.install()
tok, dev, labels = mind.tokenizer, mind.device, mind.labels
te = TextEmotion(device="cuda")
stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
print(f"[{time.time() - t0:.0f}s] loaded", flush=True)


def coef(push: dict) -> torch.Tensor:
    return torch.tensor([float(push.get(l, 0.0)) for l in labels])


@torch.no_grad()
def sample_loop(ids_rows, coefs, n, seed):
    """ids_rows: [B, T] (same length); generate n tokens per row with per-row steering; -> list of token lists"""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.tensor(ids_rows, device=dev)
    mind.set_coef(None)                                     # the words so far were written unpushed
    out = mind.model(x[:, :-1], use_cache=True, logits_to_keep=1)
    past, cur = out.past_key_values, x[:, -1:]
    mind.set_coef(torch.stack(coefs))
    res, done = [[] for _ in ids_rows], [False] * len(ids_rows)
    for _ in range(n):
        o = mind.model(cur, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = o.past_key_values
        lp = torch.log_softmax(o.logits[:, -1].float() / 0.8, -1)
        v, i = torch.topk(lp, 1024, dim=-1)
        pr = v.exp()
        keep = (pr.cumsum(-1) - pr) <= 0.92
        gum = -torch.log(-torch.log(torch.rand(v.shape, generator=g, device=dev).clamp(1e-9, 1 - 1e-9)))
        pick = torch.argmax(torch.where(keep, v + gum, torch.full_like(v, -float("inf"))), -1)
        nxt = i.gather(1, pick[:, None])
        for b, t in enumerate(nxt[:, 0].tolist()):
            if not done[b]:
                if t in stops:
                    done[b] = True
                else:
                    res[b].append(t)
        cur = nxt
        if all(done):
            break
    mind.set_coef(None)
    return res


results = []
for pname, ptext in PROMPTS.items():
    prompt = tok(tok.apply_chat_template([{"role": "user", "content": ptext}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False), add_special_tokens=False)["input_ids"]
    for seed in seeds:
        # its first words, unpushed but for its usual joy lean
        start = sample_loop([prompt], [coef({"joy": 0.12})], a.pre, seed)[0]
        base = prompt + start
        conds = [("control", 0.0, True)] + [(f, lv, True) for f in FEELINGS for lv in levels] + [(f, levels[-1], False) for f in ("sadness", "fear")]
        cs = []
        for f, lv, lean in conds:
            push = {f: lv} if f != "control" else {}
            if lean:
                push["joy"] = push.get("joy", 0.0) + 0.12
            cs.append(coef(push))
        outs = sample_loop([base] * len(conds), cs, a.tokens, seed + 100)
        for (f, lv, lean), o in zip(conds, outs):
            text = tok.decode(o, skip_special_tokens=True)
            sc = te.score(text)
            results.append({"prompt": pname, "seed": seed, "feeling": f, "level": lv, "joy_lean": lean,
                            "start": tok.decode(start, skip_special_tokens=True), "text": text, "scores": sc})
        print(f"[{time.time() - t0:.0f}s] {pname} seed {seed} done", flush=True)

Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("\nmean text score of the target feeling in the continuation (control = no push):")
for pname in PROMPTS:
    ctl = [r for r in results if r["prompt"] == pname and r["feeling"] == "control"]
    print(f"--- {pname} character")
    for f in FEELINGS:
        base_s = statistics.mean(r["scores"].get(f, 0.0) for r in ctl)
        row = [f"{f:9s} control {base_s:.2f}"]
        for lv in levels:
            rs = [r for r in results if r["prompt"] == pname and r["feeling"] == f and r["level"] == lv and r["joy_lean"]]
            row.append(f"{lv:.2f}: {statistics.mean(r['scores'].get(f, 0.0) for r in rs):.2f}")
        nl = [r for r in results if r["prompt"] == pname and r["feeling"] == f and not r["joy_lean"]]
        if nl:
            row.append(f"| {levels[-1]:.2f} without the joy lean: {statistics.mean(r['scores'].get(f, 0.0) for r in nl):.2f}")
        print("  ".join(row))
print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)
