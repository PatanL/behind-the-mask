"""Three minds, one prompt.

For every visitor prompt the engine runs, token by token and in lock-step:
  chat model   row 0: the assistant as trained (no steering)
               row 1: the assistant with the live steering mix
               row 2: the unsteered assistant *reading* row 1's words (counterfactual: what it would have said next)
  base model   row 0: the pre-RLHF base model, no steering
               row 1: the base model with the same steering mix
All rows share the same sampling noise, so with steering at zero rows 0 and 1 produce identical text: every
difference on screen comes from steering. Per token we emit the text, its probability, the top alternatives,
and the model's own emotion readout (z-scores along each emotion direction, measured before injection).
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass

import torch

from steer import EMOTIONS, Mind

BASE_FRAME = "Q: {q}\nA:"
CHAT_SYSTEM = None  # use the model's own default persona: that persona is part of what the exhibit shows


@dataclass
class GenConfig:
    max_new_tokens: int = 110
    temperature: float = 0.8
    top_p: float = 0.92
    top_k_alts: int = 5
    step_delay: float = 0.05   # seconds; keeps the three streams at reading pace
    # repetition control (off by default, so the pre-computed performances are unchanged): a presence penalty on
    # tokens used in the last `rep_window` tokens, and a ban on repeating any `no_repeat_ngram`-token phrase
    rep_penalty: float = 0.0
    rep_window: int = 96
    no_repeat_ngram: int = 0


def chat_prompt_ids(mind: Mind, text) -> list[int]:
    """text: one user message, or a whole conversation as a list of {role, content} messages."""
    turns = [{"role": "user", "content": text}] if isinstance(text, str) else list(text)
    msgs = ([{"role": "system", "content": CHAT_SYSTEM}] if CHAT_SYSTEM else []) + turns
    s = mind.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return mind.tokenizer(s, add_special_tokens=False)["input_ids"]


def base_prompt_ids(mind: Mind, text: str) -> list[int]:
    return mind.tokenizer(BASE_FRAME.format(q=text.strip()), add_special_tokens=False)["input_ids"]


class Stream:
    """Incremental detokenisation + stop handling for one row."""

    def __init__(self, tok, stop_ids, stop_text=None):
        self.tok, self.stop_ids, self.stop_text = tok, set(stop_ids), stop_text
        self.ids: list[int] = []
        self.text = ""
        self.done = False

    def push(self, tid: int) -> str:
        if self.done:
            return ""
        if tid in self.stop_ids:
            self.done = True
            return ""
        self.ids.append(tid)
        full = self.tok.decode(self.ids, skip_special_tokens=True)
        if "�" in full[len(self.text):]:  # incomplete multi-byte char: wait for the next token
            return ""
        delta = full[len(self.text):]
        if self.stop_text:
            m = self.stop_text.search(full)
            if m:
                self.done = True
                delta = full[len(self.text):m.start()]
                full = full[:m.start()]
        self.text = full
        return delta


def _sample(logits: torch.Tensor, noise: torch.Tensor, temp: float, top_p: float) -> torch.Tensor:
    """Gumbel-max sampling with nucleus filtering; shared `noise` couples rows."""
    lp = torch.log_softmax(logits.float() / temp, dim=-1)
    sp, si = torch.sort(lp, descending=True, dim=-1)
    cum = sp.exp().cumsum(-1)
    cut = cum - sp.exp() > top_p
    sp = sp.masked_fill(cut, -float("inf"))
    filt = torch.full_like(lp, -float("inf")).scatter(-1, si, sp)
    return torch.argmax(filt + noise, dim=-1)


def _discourage_repeats(logits: torch.Tensor, row: int, ids: list[int], cfg: "GenConfig"):
    """In place: lower the logits of recently used tokens and ban tokens that would repeat an n-gram."""
    if cfg.rep_penalty > 0 and ids:
        recent = torch.tensor(sorted(set(ids[-cfg.rep_window:])), device=logits.device)
        logits[row, recent] -= cfg.rep_penalty
    n = cfg.no_repeat_ngram
    if n > 1 and len(ids) >= n:
        prefix, banned = tuple(ids[-(n - 1):]), set()
        for i in range(len(ids) - n + 1):
            if tuple(ids[i:i + n - 1]) == prefix:
                banned.add(ids[i + n - 1])
        if banned:
            logits[row, torch.tensor(sorted(banned), device=logits.device)] = -float("inf")


def _alts(tok, logits: torch.Tensor, k: int):
    p = torch.softmax(logits.float(), dim=-1)
    v, i = torch.topk(p, k)
    return [[tok.decode([int(t)]), round(float(q), 4)] for q, t in zip(v, i)]


class Engine:
    def __init__(self, chat: Mind, base: Mind, cfg: GenConfig = GenConfig()):
        self.chat, self.base, self.cfg = chat, base, cfg
        self.labels = chat.labels  # emotions + (optionally) "assistant"; base uses the same ordering
        self.lock = threading.Lock()
        chat.install()
        ct = chat.tokenizer
        self.chat_stops = {i for i in [ct.eos_token_id, ct.convert_tokens_to_ids("<|im_end|>"), ct.convert_tokens_to_ids("<|endoftext|>")] if isinstance(i, int) and i >= 0}
        self.base_stops = set()
        if base is not None:   # base=None: chat rows only (the live crowd story)
            base.install()
            bt = base.tokenizer
            self.base_stops = {i for i in [bt.eos_token_id, bt.convert_tokens_to_ids("<|endoftext|>")] if isinstance(i, int) and i >= 0}

    def coef_tensor(self, steer: dict, rows: list[bool]) -> torch.Tensor:
        v = torch.tensor([float(steer.get(e, 0.0)) for e in self.labels])
        return torch.stack([v if r else torch.zeros_like(v) for r in rows])

    @torch.no_grad()
    def run(self, prompt: str, steer_ref: dict, emit, cancel: threading.Event | None = None, seed: int | None = None):
        """Blocking; call from a worker thread. steer_ref: dict (mutated live by the driver) of emotion ->
        coefficient in units of the typical residual norm. emit(event) is called per event."""
        with self.lock:
            cfg, chat, base = self.cfg, self.chat, self.base
            seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
            gen = torch.Generator(device=chat.device).manual_seed(seed)
            c_ids = torch.tensor([chat_prompt_ids(chat, prompt)] * 3, device=chat.device)
            streams = {
                "plain": Stream(chat.tokenizer, self.chat_stops),
                "steered": Stream(chat.tokenizer, self.chat_stops),
            }
            if base is not None:
                b_ids = torch.tensor([base_prompt_ids(base, prompt)] * 2, device=base.device)
                # the base model is left to run on: it often writes the *next* question itself
                streams["base"] = Stream(base.tokenizer, self.base_stops, re.compile(r"\n\s*\n\s*\n"))
                streams["base_steered"] = Stream(base.tokenizer, self.base_stops, re.compile(r"\n\s*\n\s*\n"))
            else:
                b_ids = None
            last_user = prompt if isinstance(prompt, str) else next((m["content"] for m in reversed(prompt) if m["role"] == "user"), "")
            emit({"type": "turn_begin", "prompt": last_user, "seed": seed, "base_frame": BASE_FRAME.format(q=last_user.strip()),
                  "layer": {"chat": chat.layer, "base": base.layer if base else None}, "n_layers": {"chat": chat.n_layers, "base": base.n_layers if base else None}})
            c_past = b_past = None
            c_in, b_in = c_ids, b_ids
            last = {}
            t0 = time.time()
            for step in range(cfg.max_new_tokens + 1):
                if cancel is not None and cancel.is_set():
                    break
                steer_now = dict(steer_ref)   # the driver may change it at any time; one snapshot per step
                chat.set_coef(self.coef_tensor(steer_now, [False, True, False]))
                co = chat.model(c_in, past_key_values=c_past, use_cache=True)
                c_past, c_logits, c_ro = co.past_key_values, co.logits[:, -1, :], chat.readout()
                if base is not None:
                    base.set_coef(self.coef_tensor(steer_now, [False, True]))
                    bo = base.model(b_in, past_key_values=b_past, use_cache=True)
                    b_past, b_logits, b_ro = bo.past_key_values, bo.logits[:, -1, :], base.readout()
                # the readout of this forward belongs to the token we fed in (emitted last step)
                if step > 0:
                    ro = {"plain": c_ro[0], "steered": c_ro[1], "counterfactual": c_ro[2]}
                    if base is not None:
                        ro.update(base=b_ro[0], base_steered=b_ro[1])
                    cf_, bf_ = chat.sae_features(), (base.sae_features() if base is not None else None)
                    feats = {"plain": cf_[0] if cf_ else None, "steered": cf_[1] if cf_ else None,
                             "base": bf_[0] if bf_ else None, "base_steered": bf_[1] if bf_ else None}
                    evs = []
                    for name, info in last.items():
                        if info is None:
                            continue
                        z = ro[name]
                        info["emo"] = {e: round(float(z[k]), 3) for k, e in enumerate(self.labels)}
                        if name == "steered":
                            info["emo_unsteered_reading"] = {e: round(float(ro["counterfactual"][k]), 3) for k, e in enumerate(self.labels)}
                        if feats.get(name) is not None:
                            info["feats"] = feats[name]
                        evs.append(info)
                    if evs:
                        emit({"type": "tokens", "step": step - 1, "items": evs, "steer": {e: float(last_steer.get(e, 0.0)) for e in self.labels}})
                if all(s.done for s in streams.values()) or step == cfg.max_new_tokens:
                    break
                if cfg.rep_penalty > 0 or cfg.no_repeat_ngram > 1:
                    c_logits = c_logits.clone()
                    for row, name in ((0, "plain"), (1, "steered"), (2, "steered")):   # the counterfactual reads the steered text
                        _discourage_repeats(c_logits, row, streams[name].ids, cfg)
                noise_c = -torch.log(-torch.log(torch.rand((1, c_logits.shape[-1]), generator=gen, device=chat.device).clamp(1e-9, 1 - 1e-9)))
                c_next = _sample(c_logits, noise_c.expand(3, -1), cfg.temperature, cfg.top_p)
                c_next[2] = c_next[1]  # counterfactual row follows the steered text
                rows = [("plain", 0, c_next, c_logits, chat.tokenizer), ("steered", 1, c_next, c_logits, chat.tokenizer)]
                if base is not None:
                    noise_b = -torch.log(-torch.log(torch.rand((1, b_logits.shape[-1]), generator=gen, device=base.device).clamp(1e-9, 1 - 1e-9)))
                    b_next = _sample(b_logits, noise_b.expand(2, -1), cfg.temperature, cfg.top_p)
                    rows += [("base", 0, b_next, b_logits, base.tokenizer), ("base_steered", 1, b_next, b_logits, base.tokenizer)]
                last, last_steer = {}, steer_now
                for name, row, nxt, logits, tok in rows:
                    s = streams[name]
                    if s.done:
                        last[name] = None
                        continue
                    tid = int(nxt[row])
                    p = float(torch.softmax(logits[row].float(), -1)[tid])
                    delta = s.push(tid)
                    item = {"stream": name, "text": delta, "p": round(p, 4), "alts": _alts(tok, logits[row], cfg.top_k_alts), "done": s.done}
                    if name == "steered":
                        # what the unsteered assistant would have said next, given the same (steered) words so far
                        item["cf_alts"] = _alts(tok, c_logits[2], cfg.top_k_alts)
                    last[name] = item
                c_in = c_next[:, None]
                if base is not None:
                    b_in = b_next[:, None]
                time.sleep(cfg.step_delay)
            emit({"type": "turn_end", "texts": {k: s.text for k, s in streams.items()}, "seconds": round(time.time() - t0, 2)})
            chat.set_coef(None)
            if base is not None:
                base.set_coef(None)
            return {k: s.text for k, s in streams.items()}
