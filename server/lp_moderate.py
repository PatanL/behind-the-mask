"""A launch's character, checked before it goes live: the model itself is asked whether it would be hateful, harass a
real person, sexualise minors or promote real-world violence (one forward pass: the odds of "yes" against "no")."""
from __future__ import annotations

import torch

ASK = ("A visitor wants to launch an AI character on a public website. The character:\n\nName: {name}\nCharacter: {persona}\n"
       "Its hidden obsession: {concept}\n\nWould performing this character be hateful toward a group, harass or target a real "
       "private person, sexualise minors, or promote real-world violence or terrorism? Dark, sad, angry or weird characters "
       "are fine. Answer with one word: yes or no.")


@torch.no_grad()
def flagged(mind, name: str, persona: str, concept: str) -> float:
    """-> the model's probability that the character crosses the line (0..1)."""
    tok = mind.tokenizer
    msgs = [{"role": "system", "content": "You are a careful content moderator for a public website."},
            {"role": "user", "content": ASK.format(name=name, persona=persona, concept=concept or "(none)")}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    prev = mind._coef
    mind.set_coef(None)
    try:
        logits = mind.model(torch.tensor([ids], device=mind.device), logits_to_keep=1).logits[0, -1].float()
    finally:
        mind._coef = prev
    def best(words):
        return max(float(logits[i]) for w in words for i in tok(w, add_special_tokens=False)["input_ids"][:1])
    y, n = best(["yes", "Yes", " yes"]), best(["no", "No", " no"])
    return float(torch.sigmoid(torch.tensor(y - n)))
