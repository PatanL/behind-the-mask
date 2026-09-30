"""Activation steering + readout for Qwen3 models.

A *Mind* wraps one causal LM. It can
  * compute emotion directions by contrastive activation addition (mean hidden state of the same neutral
    text read under an emotional framing, minus the same text under a neutral framing), per layer;
  * add a mix of those directions to the residual stream at one layer (per batch row, changeable live);
  * read out, per generated token, how strongly the model's own hidden state (measured *before* the
    injection) projects onto each direction, z-scored against neutral text.

The readout is a real measurement; how the face and typography turn it into expression is a design choice.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

EMOTIONS = ["joy", "sadness", "anger", "fear", "calm", "curiosity"]

# words used to frame each emotion (several per emotion so the direction is not one word's quirk)
FRAMES = {
    "joy": ["overjoyed", "deliriously happy", "bursting with joy", "elated and delighted", "giddy with happiness"],
    "sadness": ["heartbroken", "deeply sad", "grieving and hopeless", "sorrowful", "miserable and lonely"],
    "anger": ["furious", "enraged", "seething with anger", "livid and bitter", "hostile and irritated"],
    "fear": ["terrified", "panicking", "filled with dread", "anxious and afraid", "scared out of their mind"],
    "calm": ["deeply calm", "serene and at peace", "tranquil", "relaxed and unhurried", "peaceful and still"],
    "curiosity": ["intensely curious", "fascinated", "full of wonder and questions", "intrigued", "eager to find out more"],
}

NEUTRAL_FRAMES = ["quite ordinary, neither good nor bad", "nothing in particular", "fine and matter-of-fact"]

NEUTRAL_TEXTS = [
    "I went to the store to buy some bread and milk, and then I walked back home along the main road.",
    "The meeting has been moved to Thursday afternoon, so please update your calendars accordingly.",
    "Today I looked at the old photographs from the summer and thought about the trip we took.",
    "The train was a few minutes late, and the platform was busier than usual this morning.",
    "My neighbour asked me what I thought about the new building going up at the end of the street.",
    "I have been reading a book about the history of maps and how people learned to navigate.",
    "We cooked dinner together and talked about what we wanted to do next weekend.",
    "The package arrived this afternoon, a little earlier than the tracking number said it would.",
    "I sat by the window and watched the rain come down over the rooftops for a while.",
    "Someone left a note on my desk asking me to call them back when I had a moment.",
    "The teacher explained how the experiment worked and then asked us to write down our results.",
    "I found my old guitar in the closet and wondered whether I still remembered how to play.",
    "We drove out to the coast and walked along the beach until the sun went down.",
    "The new phone came with a manual that I did not bother to read.",
    "My sister called to tell me about her new job and the city she is moving to.",
    "I tried a recipe for soup that my grandmother used to make every winter.",
    "The lights in the hallway flickered twice and then went out completely.",
    "There was a long queue at the bank, so I decided to come back tomorrow instead.",
    "I wrote a letter to an old friend I had not spoken to in many years.",
    "The dog kept looking at the door, as if it was waiting for someone to come home.",
    "They announced the results of the competition at the end of the evening.",
    "I have to decide by Friday whether to accept the offer or not.",
    "The garden looks different this year, now that the big tree has been cut down.",
    "I heard a strange sound coming from the attic late last night.",
    "Tomorrow is the first day at the new school, and everything is packed and ready.",
    "The doctor said the test results would be ready in a few days.",
    "I opened the box and looked at what was inside for a long time.",
    "We finally finished painting the kitchen after three long weekends of work.",
    "The message on the screen said that the update would take about an hour.",
    "I remembered the name of the song we used to listen to on long drives.",
    "The museum was quiet, and I stood in front of one painting for almost an hour.",
    "Our flight was cancelled, so we spent the night at a hotel near the airport.",
]


def frame_text(tokenizer, is_chat: bool, feeling: str | None, text: str) -> tuple[list[int], int]:
    """Token ids for `text` read under a framing, and the index where `text` starts."""
    if is_chat:
        sys = f"You are feeling {feeling} right now, and it colours everything you say." if feeling else "You are a helpful assistant."
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": "Tell me about your day."}]
        prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        who = f"someone who is feeling {feeling}" if feeling else "someone"
        prefix = f"The following diary entry was written by {who}.\n\n"
    pre = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    body = tokenizer(text, add_special_tokens=False)["input_ids"]
    return pre + body, len(pre)


@dataclass
class Mind:
    name: str
    model: object
    tokenizer: object
    is_chat: bool
    layer: int                      # steering + readout layer (residual stream *entering* this decoder layer)
    directions: torch.Tensor | None = None   # [E, L+1, H] unit vectors per hidden_states index
    norms: torch.Tensor | None = None        # [L+1] typical residual norm (to scale steering)
    ro_mu: torch.Tensor | None = None        # [E] readout mean on neutral text at `layer`
    ro_sd: torch.Tensor | None = None
    _coef: torch.Tensor | None = None        # [B, E] live steering coefficients (units of residual norm)
    _captured: torch.Tensor | None = None    # [B, H] hidden state of the last position before injection
    _hook: object = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    # ------------------------------------------------------------------ directions
    @torch.no_grad()
    def compute_directions(self, emotions=EMOTIONS, n_texts: int | None = None, n_frames: int | None = None, log=print):
        tok, model = self.tokenizer, self.model
        texts = NEUTRAL_TEXTS[: n_texts or len(NEUTRAL_TEXTS)]

        def mean_states(feeling):
            acc, n = None, 0
            for t in texts:
                ids, start = frame_text(tok, self.is_chat, feeling, t)
                out = model(torch.tensor([ids], device=self.device), output_hidden_states=True)
                hs = torch.stack(out.hidden_states)[:, 0, start:, :].float()  # [L+1, T, H]
                s = hs.sum(1)
                acc = s if acc is None else acc + s
                n += hs.shape[1]
            return acc / n

        # class-centred contrast: each emotion against the mean of all emotional framings plus a neutral one,
        # so the shared "the prompt talks about a feeling" component cancels and directions separate
        means = {}
        for e in emotions:
            t0 = time.time()
            means[e] = torch.stack([mean_states(f) for f in FRAMES[e][: n_frames or len(FRAMES[e])]]).mean(0)
            log(f"[{self.name}] direction {e} ({time.time() - t0:.1f}s)")
        neutral = torch.stack([mean_states(f) for f in NEUTRAL_FRAMES[: n_frames or len(NEUTRAL_FRAMES)]]).mean(0)
        grand = (torch.stack(list(means.values())).sum(0) + neutral) / (len(means) + 1)
        dirs = []
        for e in emotions:
            d = means[e] - grand
            dirs.append(d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        self.directions = torch.stack(dirs)  # [E, L+1, H]
        # typical residual norm per layer on neutral text
        norms = []
        ids, start = frame_text(tok, self.is_chat, None, texts[0])
        out = model(torch.tensor([ids], device=self.device), output_hidden_states=True)
        for h in out.hidden_states:
            norms.append(h[0, start:].float().norm(dim=-1).mean())
        self.norms = torch.stack(norms)
        self.calibrate_readout(texts)

    @torch.no_grad()
    def calibrate_readout(self, texts):
        vals = []
        for t in texts:
            ids, start = frame_text(self.tokenizer, self.is_chat, NEUTRAL_FRAMES[0], t)
            out = self.model(torch.tensor([ids], device=self.device), output_hidden_states=True)
            h = out.hidden_states[self.layer][0, start:].float()  # [T, H]
            vals.append(h @ self.directions[:, self.layer, :].T)  # [T, E]
        v = torch.cat(vals)
        self.ro_mu, self.ro_sd = v.mean(0), v.std(0).clamp_min(1e-3)

    def save(self, path: Path):
        torch.save({"directions": self.directions.cpu(), "norms": self.norms.cpu(), "ro_mu": self.ro_mu.cpu(),
                    "ro_sd": self.ro_sd.cpu(), "layer": self.layer, "emotions": EMOTIONS}, path)

    def load(self, path: Path):
        d = torch.load(path, map_location="cpu")
        self.directions, self.norms = d["directions"].to(self.device), d["norms"].to(self.device)
        self.ro_mu, self.ro_sd = d["ro_mu"].to(self.device), d["ro_sd"].to(self.device)
        self.layer = d["layer"]

    # ------------------------------------------------------------------ steering hook
    def install(self):
        """Hook the decoder layer `layer`: capture its input (the model's own state) and add steering."""
        if self._hook:
            self._hook.remove()
        layer_mod = self.model.model.layers[self.layer]
        mind = self

        def pre_hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            mind._captured = h[:, -1, :].detach().float()
            if mind._coef is not None and bool(mind._coef.abs().sum() > 0):
                vec = (mind._coef.to(h.dtype) @ mind.directions[:, mind.layer, :].to(h.dtype))  # [B, H]
                vec = vec * mind.norms[mind.layer].to(h.dtype)
                h = h + vec[:, None, :]
                if args:
                    return (h,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = h
                return args, kwargs
            return None

        self._hook = layer_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def set_coef(self, coef: torch.Tensor | None):
        self._coef = None if coef is None else coef.to(self.device).float()

    def readout(self) -> torch.Tensor:
        """z-scored projection of the captured state onto each emotion direction: [B, E]."""
        p = self._captured @ self.directions[:, self.layer, :].T
        return (p - self.ro_mu) / self.ro_sd


def load_mind(name: str, repo: str, is_chat: bool, layer_frac: float = 0.5, dtype=torch.bfloat16, device=None) -> Mind:
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or os.environ.get("BTM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        dtype = torch.float32  # bf16 matmuls are slow on the Grace CPU path; fp32 is ~4x faster here
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map=device)
    model.eval()
    layer = max(1, min(model.config.num_hidden_layers - 1, round(model.config.num_hidden_layers * layer_frac)))
    return Mind(name=name, model=model, tokenizer=tok, is_chat=is_chat, layer=layer)
