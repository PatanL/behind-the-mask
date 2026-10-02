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

import copy
import math
import re
import threading
import time
from dataclasses import dataclass, field

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
    # tokens of the speaker's recent earlier replies: the n-gram ban covers them too, so a turn can't repeat a
    # sentence from the last few turns word for word (live)
    history_ids: list = field(default_factory=list)
    # the counterfactual row starts exactly like the unpushed row, so read the prompt only for rows 0-1 and copy
    # row 0's cache into row 2 (a third less prefill work: a shorter pause before each new turn)
    share_prefill: bool = False
    min_step: float = 0.0      # seconds per token at least: a pace cap (live), 0 = as fast as it runs
    # a long prompt (the live conversation) is read in pieces of this many tokens: bounded memory (0 = in one go)
    prefill_chunk: int = 1024
    # live: after each turn keep the memory (KV cache) of the conversation as visitors saw it, so the next turn
    # appends only its new words instead of re-reading the whole history (see Engine.snapshot)
    keep_snapshot: bool = False


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
    if n > 1 and len(ids) >= n - 1:
        seen = list(cfg.history_ids) + [-1] + ids if cfg.history_ids else ids   # -1: no phrase spans two replies
        prefix, banned = tuple(ids[-(n - 1):]), set()
        for i in range(len(seen) - n + 1):
            if tuple(seen[i:i + n - 1]) == prefix:
                banned.add(seen[i + n - 1])
        banned.discard(-1)
        if banned:
            logits[row, torch.tensor(sorted(banned), device=logits.device)] = -float("inf")


def _alts(tok, logits: torch.Tensor, k: int):
    p = torch.softmax(logits.float(), dim=-1)
    v, i = torch.topk(p, k)
    return [[tok.decode([int(t)]), round(float(q), 4)] for q, t in zip(v, i)]


def _row_cache(cache, row: int, device):
    """A copy of one row of a batched KV cache (the hybrid model's recurrent state can't be rewound, so it is
    copied at the right moment rather than cropped later)."""
    c = copy.deepcopy(cache)
    c.reorder_cache(torch.tensor([row], device=device))
    return c


def _forward(model, ids, past, chunk: int):
    """One forward pass that keeps only the last position's logits (the full [tokens x vocab] logits of a 10k-token
    prompt are ~10-20 GB); a long prompt with no cache yet goes in pieces of `chunk` tokens, so memory stays bounded."""
    if past is not None or not chunk or ids.shape[1] <= chunk:
        return model(ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    out = None
    for s in range(0, ids.shape[1], chunk):
        out = model(ids[:, s:s + chunk], past_key_values=None if out is None else out.past_key_values, use_cache=True, logits_to_keep=1)
    return out


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
    def run(self, prompt: str, steer_ref: dict, emit, cancel: threading.Event | None = None, seed: int | None = None,
            prompt_ids: list[int] | None = None, reuse: dict | None = None):
        """Blocking; call from a worker thread. steer_ref: dict (mutated live by the driver) of emotion ->
        coefficient in units of the typical residual norm. emit(event) is called per event.
        prompt_ids: the chat prompt as token ids (default: rendered from `prompt`). reuse: a previous turn's
        snapshot {"cache", "ids"}; when prompt_ids starts with its ids, only the rest is read."""
        with self.lock:
            cfg, chat, base = self.cfg, self.chat, self.base
            seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
            gen = torch.Generator(device=chat.device).manual_seed(seed)
            c_prompt = list(prompt_ids) if prompt_ids is not None else chat_prompt_ids(chat, prompt)
            c_past = None
            if reuse is not None and len(c_prompt) > len(reuse["ids"]) and c_prompt[:len(reuse["ids"])] == reuse["ids"]:
                # the conversation so far is already read: start all three rows from it, read only the new words
                c_past = reuse["cache"]
                c_past.reorder_cache(torch.zeros(3, dtype=torch.long, device=chat.device))
                c_ids = torch.tensor([c_prompt[len(reuse["ids"]):]] * 3, device=chat.device)
            else:
                c_ids = torch.tensor([c_prompt] * 3, device=chat.device)
            self.reused = c_past is not None
            self.snapshot, fed2, pending2, snap = None, [], None, None
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
            b_past = None
            c_in, b_in = c_ids, b_ids
            last = {}
            t0 = step_t = time.time()
            for step in range(cfg.max_new_tokens + 1):
                if cancel is not None and cancel.is_set():
                    break
                steer_now = dict(steer_ref)   # the driver may change it at any time; one snapshot per step
                if step == 0 and cfg.share_prefill and c_past is None:
                    chat.set_coef(self.coef_tensor(steer_now, [False, True]))
                    co = _forward(chat.model, c_in[:2], None, cfg.prefill_chunk)
                    rows = torch.tensor([0, 1, 0], device=chat.device)
                    co.past_key_values.reorder_cache(rows)
                    c_past, c_logits, c_ro = co.past_key_values, co.logits[:, -1, :].index_select(0, rows), None   # no readout is emitted for the prompt
                else:
                    chat.set_coef(self.coef_tensor(steer_now, [False, True, False]))
                    co = _forward(chat.model, c_in, c_past, cfg.prefill_chunk)
                    c_past, c_logits, c_ro = co.past_key_values, co.logits[:, -1, :], chat.readout()
                if pending2 is not None:
                    fed2.append(pending2)
                    # the shown (steered) reply just ended and its end token is read: keep the counterfactual row's
                    # memory now, before the rows go on feeding filler for the plain answer
                    if cfg.keep_snapshot and snap is None and pending2 in self.chat_stops and streams["steered"].done:
                        snap = _row_cache(c_past, 2, chat.device)
                        snap_fed = list(fed2)
                    pending2 = None
                if base is not None:
                    base.set_coef(self.coef_tensor(steer_now, [False, True]))
                    bo = _forward(base.model, b_in, b_past, cfg.prefill_chunk)
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
                pending2 = int(c_next[2])
                if base is not None:
                    b_in = b_next[:, None]
                # pace: at least min_step per token (measured from the previous token), plus any fixed delay
                time.sleep(max(cfg.step_delay, cfg.min_step - (time.time() - step_t)))
                step_t = time.time()
            if cfg.keep_snapshot:
                if snap is None:   # cut short (cap or cancel): the reply's unread words and an end token come next
                    snap, snap_fed = _row_cache(c_past, 2, chat.device), list(fed2)
                    unread = streams["steered"].ids[len(snap_fed):]
                    tail = unread + [chat.tokenizer.convert_tokens_to_ids("<|im_end|>")]
                else:
                    tail = []
                self.snapshot = {"cache": snap, "ids": c_prompt + snap_fed, "tail": tail}
            emit({"type": "turn_end", "texts": {k: s.text for k, s in streams.items()}, "seconds": round(time.time() - t0, 2)})
            chat.set_coef(None)
            if base is not None:
                base.set_coef(None)
            # hand the turn's memory back (each turn's prompt has a new length, so cached blocks would pile up)
            c_past = b_past = co = bo = snap = None
            torch.cuda.empty_cache()
            return {k: s.text for k, s in streams.items()}
