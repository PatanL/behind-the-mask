"""Feeling directions from speech: what someone says out loud, in the first person, while feeling each feeling.

build_vectors.py builds them from short stories about characters who feel something. Those carry how a feeling is
*described* (for fear: a racing heart, no air), so pushing along them makes a live android describe symptoms; anger
and joy came through anyway (in stories they're already shouted and exclaimed). On a live stream a feeling is read
from what the android *says*. Here the model writes what a streamer says about the same everyday topics in each
feeling (and flatly, for the neutral mean), so topic cancels out and what's left is the feeling in the voice.
The assistant axis, residual norms and mask scale are kept from an existing build; the readout is re-calibrated.

    python run_fp8.py build_speech_dirs.py --chat Qwen/Qwen3.8-27B-FP8 --from runs/q27b --out runs/q27b-speech
"""
import argparse, json, shutil, time
from pathlib import Path

import torch

from build_vectors import chat_ids, mean_acts
from steer import load_mind

TOPICS = ["a meme someone just posted in the chat", "the video game you're playing", "what you had for dinner", "a song stuck in your head",
          "the weather outside", "your favorite movie", "a news story you just read", "how your morning went", "a video you watched last night",
          "an old photo you found", "a neighbor's dog", "a text from a friend", "a package that just arrived", "a sound outside the window",
          "a trip you've been planning", "your code crashing", "a restaurant you tried", "a football match", "an argument you saw online",
          "a childhood memory", "a sale at the store", "your new phone", "the internet going down", "a plant on your desk"]
FEEL = {"joy": "overjoyed and delighted", "sadness": "deeply sad, close to tears", "anger": "furious and outraged",
        "fear": "terrified and scared", "calm": "completely calm and at peace", "curiosity": "intensely curious and fascinated",
        "neutral": "flat and matter-of-fact, with no particular feeling"}
EMOTIONS = ["joy", "sadness", "anger", "fear", "calm", "curiosity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=28)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mind = load_mind("chat", a.chat, True)
    tok, m = mind.tokenizer, mind.model
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    old = torch.load(Path(a.src) / "chat_dirs.pt")
    layer = old["layer"]
    print(f"[{time.time() - t0:.0f}s] loaded; layer {layer}", flush=True)

    sp = out / "speech.json"
    if sp.exists():
        lines = json.loads(sp.read_text())
    else:
        jobs = [(lab, t) for lab in FEEL for t in TOPICS]
        prompts = [chat_ids(tok, None, f"Write exactly what a live streamer says out loud, in the first person, about {t}, while feeling "
                                       f"{FEEL[lab]}. Let the feeling come through in what they say and how they say it. Spoken words only: "
                                       f"three or four sentences, no stage directions, no quotation marks, no emoji.") for lab, t in jobs]
        texts = []
        with torch.no_grad():
            for i in range(0, len(prompts), a.bs):
                enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                g = m.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
                texts += [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"') for r in g]
                print(f"[{time.time() - t0:.0f}s] {len(texts)}/{len(prompts)} written", flush=True)
        lines = [{"label": lab, "topic": t, "text": x} for (lab, t), x in zip(jobs, texts)]
        sp.write_text(json.dumps(lines, ensure_ascii=False, indent=1))

    prefix = chat_ids(tok, None, "Keep talking.")   # read as its own spoken reply, like a live turn
    means, per = {}, {}
    with torch.no_grad():
        for lab in FEEL:
            means[lab], per[lab] = mean_acts(tok, m, [(prefix, x["text"]) for x in lines if x["label"] == lab])
            print(f"[{time.time() - t0:.0f}s] read {lab}: {per[lab].shape[0]}", flush=True)
    grand = torch.stack(list(means.values())).mean(0)
    raw = torch.stack([means[e] - grand for e in EMOTIONS])                     # [E, L+1, H]
    raw = torch.cat([raw.cpu(), old["raw"][-1:].cpu()], 0)                       # + the assistant axis, as built before
    unit = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # readout calibration: the neutral lines' projections at the steering layer (z = (proj - mu) / sd)
    proj = per["neutral"][:, layer, :].cpu() @ unit[:, layer, :].T
    ro_mu, ro_sd = proj.mean(0), proj.std(0).clamp_min(1e-3)
    ro_mu[-1], ro_sd[-1] = old["ro_mu"][-1], old["ro_sd"][-1]                    # (the assistant axis keeps its own)
    d = {**old, "directions": unit, "raw": raw, "ro_mu": ro_mu, "ro_sd": ro_sd}
    torch.save(d, out / "chat_dirs.pt")
    if (Path(a.src) / "levels.json").exists():
        shutil.copy(Path(a.src) / "levels.json", out / "levels.json")
    D = unit[:, layer]
    old_D = old["directions"][:, layer].cpu()
    labs = EMOTIONS + ["assistant"]
    print(json.dumps({"cos_new": {l: [round(x, 2) for x in r] for l, r in zip(labs, (D @ D.T).tolist())},
                      "cos_with_story_dirs": {l: round(float(D[i] @ old_D[i]), 2) for i, l in enumerate(labs)}}), flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
