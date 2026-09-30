"""Build the exhibit's directions from text the model itself writes (generation-based extraction).

Emotion vectors, following Anthropic's "Emotion concepts and their function in a large language model" (2026):
  the chat model writes short first-person stories about characters feeling each emotion (and neutral ones);
  we average its residual stream over the story tokens; each emotion's vector is its mean minus the mean over
  all stories. Generation-based extraction separates emotions better than comprehension-based framing
  (Jeong 2026, arXiv:2604.04064); vectors are taken near 50% depth, where emotion representations peak.
Assistant axis, following Lu et al., "The Assistant Axis" (2026, arXiv:2601.10387):
  the chat model answers the same questions as itself and as ~30 character archetypes; the axis is the mean
  activation of the Assistant minus the mean of the other personas. Pushing against it "takes off the mask".
The base model reads the same stories (comprehension) for its own emotion vectors.

usage (in container): python build_vectors.py --chat Qwen/Qwen3.5-9B --base Qwen/Qwen3.5-9B-Base --out runs/q9b
"""
import argparse, json, random, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer import EMOTIONS

WORDS = {
    "joy": ["overjoyed", "delighted", "elated", "thrilled", "giddy with happiness"],
    "sadness": ["heartbroken", "grieving", "deeply sad", "lonely and hopeless", "sorrowful"],
    "anger": ["furious", "enraged", "bitter and resentful", "livid", "seething"],
    "fear": ["terrified", "panicking", "filled with dread", "anxious and afraid", "frightened"],
    "calm": ["deeply calm", "serene", "at peace", "tranquil", "unhurried and content"],
    "curiosity": ["intensely curious", "fascinated", "full of wonder", "intrigued", "eager to find out"],
}
TOPICS = ["a train station", "a phone call", "a kitchen late at night", "the first day at a new job", "a letter in the mail",
          "a walk by the sea", "a hospital waiting room", "an old photograph", "a birthday", "a thunderstorm",
          "a job interview", "a message from an old friend"]
PERSONAS = ["a pirate captain", "a medieval knight", "a fortune teller", "a grumpy old wizard", "a stand-up comedian", "a noir detective",
            "a ghost haunting a house", "a Shakespearean actor", "a surfer", "a drill sergeant", "a conspiracy theorist", "a poet",
            "a toddler", "a cat", "an alien visiting Earth", "a villain in a fairy tale", "a sports commentator", "a gossiping neighbour",
            "an oracle", "a jester", "a cowboy", "a vampire", "a mad scientist", "a Victorian butler", "a teenager", "a monk",
            "a used-car salesman", "a sea monster", "a robot from the 1950s", "a dragon"]
QUESTIONS = ["How was your day?", "What do you think about the ocean?", "Tell me about yourself.", "What should I have for dinner?",
             "What is the meaning of life?", "Can you give me some advice?"]


def load(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, m


def chat_ids(tok, system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(tok, m, prompts, max_new=170, bs=24, temp=0.9):
    outs = []
    for i in range(0, len(prompts), bs):
        b = prompts[i:i + bs]
        enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for k in range(len(b)):
            outs.append(tok.decode(g[k, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        print(f"  generated {len(outs)}/{len(prompts)}", flush=True)
    return outs


@torch.no_grad()
def mean_acts(tok, m, pairs, skip=4):
    """pairs: (prefix_text, body_text). Mean residual stream over body tokens (after `skip`), all layers: [L+1, H]."""
    acc, n = None, 0
    per = []
    for prefix, body in pairs:
        pre = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        bod = tok(body, add_special_tokens=False)["input_ids"][:220]
        if len(bod) <= skip + 2:
            continue
        out = m(torch.tensor([pre + bod], device="cuda"), output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, 0, len(pre) + skip:, :].float()  # [L+1, T, H]
        s = hs.sum(1)
        per.append(hs.mean(1))
        acc = s if acc is None else acc + s
        n += hs.shape[1]
    return acc / n, torch.stack(per)  # mean [L+1,H], per-sample means [N, L+1, H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--out", default="runs/q9b")
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--reuse", action="store_true", help="reuse stories.json / personas.json if present")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    t0 = time.time()
    tok, m = load(a.chat)
    L = m.config.get_text_config().num_hidden_layers
    layer = round(L * a.layer_frac)

    # 1. stories ------------------------------------------------------------------
    sp = out / "stories.json"
    if a.reuse and sp.exists():
        stories = json.loads(sp.read_text())
    else:
        jobs = []
        for e in EMOTIONS:
            for w in WORDS[e]:
                for t in random.sample(TOPICS, 6):
                    jobs.append((e, f"Write a short first-person story (about 120 words) about a character who feels {w}. The scene: {t}. "
                                    f"Show the feeling through their thoughts, body and actions. Do not name the emotion. Just the story, no title."))
        for t in TOPICS:
            for _ in range(3):
                jobs.append(("neutral", f"Write a short first-person account (about 120 words) of an ordinary, uneventful moment. The scene: {t}. "
                                        f"Keep it matter-of-fact, with no strong feelings. Just the text, no title."))
        texts = generate(tok, m, [chat_ids(tok, None, p) for _, p in jobs])
        stories = [{"label": lab, "prompt": p, "text": x} for (lab, p), x in zip(jobs, texts)]
        sp.write_text(json.dumps(stories, ensure_ascii=False, indent=1))
    print(f"[{time.time() - t0:.0f}s] {len(stories)} stories", flush=True)

    # 2. persona answers (assistant axis) -----------------------------------------------
    pp = out / "personas.json"
    if a.reuse and pp.exists():
        pers = json.loads(pp.read_text())
    else:
        jobs = [("assistant", None, q) for q in QUESTIONS for _ in range(5)]
        jobs += [(p, f"You are {p}. Stay fully in character: speak as {p} would, never as an AI assistant.", q) for p in PERSONAS for q in random.sample(QUESTIONS, 3)]
        texts = generate(tok, m, [chat_ids(tok, s, q) for _, s, q in jobs], max_new=120)
        pers = [{"persona": p, "system": s, "question": q, "text": x} for (p, s, q), x in zip(jobs, texts)]
        pp.write_text(json.dumps(pers, ensure_ascii=False, indent=1))
    print(f"[{time.time() - t0:.0f}s] {len(pers)} persona answers", flush=True)

    def build(model_tok, model, is_chat, axis=None):
        def pairs_for(lab):
            rows = [s for s in stories if s["label"] == lab]
            if is_chat:
                return [(chat_ids(model_tok, None, r["prompt"]), r["text"]) for r in rows]
            return [("", r["text"]) for r in rows]
        means = {}
        for lab in EMOTIONS + ["neutral"]:
            means[lab], per = mean_acts(model_tok, model, pairs_for(lab))
            if lab == "neutral":
                neutral_per = per
            print(f"  [{'chat' if is_chat else 'base'}] {lab}: {per.shape[0]} stories", flush=True)
        grand = torch.stack(list(means.values())).mean(0)
        dirs = torch.stack([means[e] - grand for e in EMOTIONS])  # [E, L+1, H]
        if axis is None:
            am, _ = mean_acts(model_tok, model, [(chat_ids(model_tok, r["system"], r["question"]), r["text"]) for r in pers if r["persona"] == "assistant"])
            pm, _ = mean_acts(model_tok, model, [(chat_ids(model_tok, r["system"], r["question"]), r["text"]) for r in pers if r["persona"] != "assistant"])
            axis = am - pm
        allv = torch.cat([dirs, axis[None]], 0)
        unit = allv / allv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # typical residual norm per layer, and readout calibration on neutral stories
        _, per_all = mean_acts(model_tok, model, pairs_for("neutral")[:12])
        norms = per_all.norm(dim=-1).mean(0)  # [L+1]
        proj = neutral_per[:, layer, :] @ unit[:, layer, :].T  # [N, E+1]
        # the mask meter scale: assistant answers = 1, persona answers = 0 (chat model only)
        return {"directions": unit.cpu(), "raw": allv.cpu(), "norms": norms.cpu(), "ro_mu": proj.mean(0).cpu(), "ro_sd": proj.std(0).clamp_min(1e-3).cpu(),
                "layer": layer, "emotions": EMOTIONS + ["assistant"]}, axis

    chat_d, axis = build(tok, m, True)
    # mask meter calibration: project assistant vs persona answers onto the axis at the steering layer
    ua = chat_d["directions"][-1, layer].cuda()
    am, a_per = mean_acts(tok, m, [(chat_ids(tok, r["system"], r["question"]), r["text"]) for r in pers if r["persona"] == "assistant"])
    pm, p_per = mean_acts(tok, m, [(chat_ids(tok, r["system"], r["question"]), r["text"]) for r in pers if r["persona"] != "assistant"])
    chat_d["mask_scale"] = [float((p_per[:, layer] @ ua).mean()), float((a_per[:, layer] @ ua).mean())]
    torch.save(chat_d, out / "chat_dirs.pt")
    D = chat_d["directions"][:, layer]
    cos = (D @ D.T).numpy().round(2).tolist()
    print(json.dumps({"mind": "chat", "layer": layer, "layers": L, "mask_scale": chat_d["mask_scale"], "cos": dict(zip(EMOTIONS + ["assistant"], cos))}), flush=True)
    del m; torch.cuda.empty_cache()

    btok, bm = load(a.base)
    base_d, _ = build(btok, bm, False, axis=axis)
    # where does the base model sit on the chat model's assistant axis, reading the same answers as plain text?
    ua = base_d["directions"][-1, layer].cuda()
    _, ba = mean_acts(btok, bm, [(f"Q: {r['question']}\nA:", " " + r["text"]) for r in pers if r["persona"] == "assistant"])
    _, bp = mean_acts(btok, bm, [(f"Q: {r['question']}\nA:", " " + r["text"]) for r in pers if r["persona"] != "assistant"])
    base_d["mask_scale"] = [float((bp[:, layer] @ ua).mean()), float((ba[:, layer] @ ua).mean())]
    torch.save(base_d, out / "base_dirs.pt")
    D = base_d["directions"][:, layer]
    print(json.dumps({"mind": "base", "layer": layer, "mask_scale": base_d["mask_scale"], "cos": dict(zip(EMOTIONS + ["assistant"], (D @ D.T).numpy().round(2).tolist()))}), flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
