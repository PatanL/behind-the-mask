"""A coin's own steering direction: "what's under its mask".

The creator gives a concept ("the Golden Gate Bridge", "conspiracy", "nostalgia") and a few example sentences. The
direction is built the way the exhibit's emotion directions are (steer.py: contrastive activation addition): the
same neutral texts are read twice, once with the model told it is preoccupied with the concept (in the creator's
words) and once as a plain assistant; the direction at each layer is the difference of the mean hidden states,
normalised. The readout along it is z-scored on neutral text, like the feelings'.
"""
from __future__ import annotations

import torch

from steer import NEUTRAL_TEXTS, Mind

_NEUTRAL_CACHE: dict = {}


def _framed(tok, system: str, text: str):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Tell me about your day."}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    body = tok(text, add_special_tokens=False)["input_ids"]
    return pre + body, len(pre)


@torch.no_grad()
def _mean_states(mind: Mind, system: str, texts, keep_layer: int | None = None):
    acc, n, at_layer = None, 0, []
    for t in texts:
        ids, start = _framed(mind.tokenizer, system, t)
        out = mind.model(torch.tensor([ids], device=mind.device), output_hidden_states=True)
        hs = torch.stack(out.hidden_states)[:, 0, start:, :].float()   # [L+1, T, H]
        acc = hs.sum(1) if acc is None else acc + hs.sum(1)
        n += hs.shape[1]
        if keep_layer is not None:
            at_layer.append(hs[keep_layer])
    return acc / n, (torch.cat(at_layer) if at_layer else None)


def concept_frame(concept: str, examples: list[str]) -> str:
    ex = " ".join(e.strip() for e in examples if e.strip())
    return (f"You are completely preoccupied with {concept.strip()}. Everything reminds you of it, and you keep coming "
            f"back to it. {ex}").strip()


@torch.no_grad()
def build_concept(mind: Mind, concept: str, examples: list[str], n_texts: int = 20) -> dict:
    """-> {"direction": [L+1, H] unit vectors, "mu", "sd": its readout on neutral text}"""
    texts = NEUTRAL_TEXTS[:n_texts]
    prev = mind._coef
    mind.set_coef(None)
    try:
        key = (mind.name, n_texts)
        if key not in _NEUTRAL_CACHE:
            _NEUTRAL_CACHE[key] = _mean_states(mind, "You are a helpful assistant.", texts, keep_layer=mind.layer)
        neutral, neutral_h = _NEUTRAL_CACHE[key]
        framed, _ = _mean_states(mind, concept_frame(concept, examples), texts)
        d = framed - neutral
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = neutral_h @ d[mind.layer]
        return {"direction": d, "mu": float(proj.mean()), "sd": float(proj.std().clamp_min(1e-3))}
    finally:
        mind._coef = prev
