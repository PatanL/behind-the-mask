"""Many minds, one model: several androids talk at once on one GPU.

Each awake android is one row of a shared batch. Every step feeds exactly one token per row: the token it just
wrote, or (between thoughts) the next token of its next prompt, read one per step. Rows never carry padding into
the model's recurrent (linear-attention) layers, whose state can't be masked; only the attention layers see the
gaps, and the attention mask hides them. Decoding is memory-bound, so a step with eight rows costs about what a
step with one does: eight androids talk at the pace of one.

A new android's first prompt is read on its own at full speed, then its memory is merged into the batch. Each row
has its own steering mix over every direction the mind holds (the six feelings, the assistant axis, and any custom
concept directions appended with add_direction), so every android can be pushed differently in the same step.
"""
from __future__ import annotations

import collections
import threading
import os
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from engine import _sample, _alts, _forward
from steer import Mind


@dataclass
class Row:
    """One android's place in the batch."""
    key: str
    steer: dict = field(default_factory=dict)        # direction label -> coefficient (units of residual norm)
    force: collections.deque = field(default_factory=collections.deque)   # prompt tokens still to read
    after: str = "rest"                              # when the reading is done: "write" a reply, or "rest"
    next_id: int | None = None                       # the token to feed next step (just sampled)
    next_info: dict | None = None                    # its probability and the alternatives it was drawn from
    pos: int = 0                                     # tokens of its own read so far (its position)
    writing: bool = False                            # sampling its reply
    reply: list = field(default_factory=list)        # this reply's token ids
    max_new: int = 600
    soft: int = 520                                  # past this many tokens, end at the next sentence end
    temperature: float = 0.8
    top_p: float = 0.92
    on_token: object = None                          # callback(row, tok: dict)  -- every written token
    on_end: object = None                            # callback(row, text)       -- the reply ended
    # repetition control (a steered monologue otherwise falls into loops): tokens used in its last rep_window tokens
    # (this reply and the ones before) are penalised, and no no_repeat_ngram-token phrase may repeat
    rep_penalty: float = 0.6
    rep_window: int = 160
    no_repeat_ngram: int = 5
    recent: collections.deque = field(default_factory=lambda: collections.deque(maxlen=400))
    # a mirror reads exactly what its leader (the row with key `mirror`) reads, unpushed, and never writes: its state
    # gives the leader's words a baseline readout ("b") and what the unpushed model would have said instead ("cf")
    mirror: str | None = None
    # a plain row writes its own unpushed reply beside its leader's, for one turn (copied from the leader's mirror when
    # the leader starts writing): its readout is the leader's words' baseline ("b"), as an unpushed answer reads
    plain_of: str | None = None
    stop_soon: bool = False                          # end the reply at the next sentence end
    finish: bool = False                             # end the reply now

    @property
    def idle(self) -> bool:
        return not self.force and not self.writing


def _layer_tensors(layer):
    """The tensors of one cache layer, by kind: attention (keys, values: [B, h, T, d]) and recurrent (per-row)."""
    att = [(n, getattr(layer, n)) for n in ("keys", "values") if isinstance(getattr(layer, n, None), torch.Tensor) and getattr(layer, n).numel() > 0]
    rec = []
    for n in ("conv_states", "recurrent_states"):
        v = getattr(layer, n, None)
        if isinstance(v, (list, tuple)):
            rec += [(n, i, t) for i, t in enumerate(v) if isinstance(t, torch.Tensor) and t.numel() > 0]
        elif isinstance(v, dict):
            rec += [(n, i, t) for i, t in v.items() if isinstance(t, torch.Tensor) and t.numel() > 0]
    return att, rec


PROFILE = os.environ.get("LP_PROFILE") == "1" or os.path.exists("runs/launchpad/profile")   # (touch that file to time steps)


def _tick() -> float:
    """A timestamp; when profiling, after the GPU has caught up (so each part's time is its own)."""
    if PROFILE and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class MultiMind:
    def __init__(self, mind: Mind, step_rate: float = 7.0, compact_at: int = 9000):
        self.parts: dict = {}
        self.mind, self.model, self.tok = mind, mind.model, mind.tokenizer
        mind.install()
        t = self.tok
        self.stops = {i for i in [t.eos_token_id, t.convert_tokens_to_ids("<|im_end|>"), t.convert_tokens_to_ids("<|endoftext|>")] if isinstance(i, int) and i >= 0}
        self.filler = t.convert_tokens_to_ids("<|endoftext|>")
        self.rows: list[Row] = []
        self.cache = None                 # the batched cache
        self.mask = None                  # [B, T] bool: which cache columns are this row's own tokens
        self.lock = threading.RLock()
        self.step_rate, self.compact_at = step_rate, compact_at
        self.steps, self.t_step = 0, 0.0
        self.gen = torch.Generator(device=mind.device).manual_seed(int(time.time()))

    # ------------------------------------------------------------------ directions
    def add_direction(self, label: str, vec: torch.Tensor, mu: float = 0.0, sd: float = 1.0) -> int:
        """Append a custom direction ([L+1, H] unit vectors per layer) to the mind, with its readout's neutral-text
        mean and spread (for the z-score); returns its index."""
        m = self.mind
        with self.lock:
            if label in m.labels:
                i = m.labels.index(label)
                m.directions[i] = vec.to(m.directions)
                m.ro_mu[i], m.ro_sd[i] = mu, sd
                return i
            m.directions = torch.cat([m.directions, vec.to(m.directions)[None]], 0)
            m.labels.append(label)
            m.ro_mu = torch.cat([m.ro_mu, torch.tensor([mu], device=m.ro_mu.device)])
            m.ro_sd = torch.cat([m.ro_sd, torch.tensor([sd], device=m.ro_sd.device)])
            return len(m.labels) - 1

    # ------------------------------------------------------------------ rows
    @torch.no_grad()
    def add(self, row: Row, prompt_ids: list[int], begin: bool = True):
        """Read the prompt on its own (full speed), then merge the row into the batch. Its reply starts next step
        (a mirror's doesn't: it only follows its leader)."""
        m = self.mind
        with self.lock:
            m.set_coef(self._coef([row]))
            out = _forward(self.model, torch.tensor([prompt_ids], device=m.device), None, 2048)   # (fewer, bigger passes: FP8 weights are unpacked once a pass)
            row.pos, row.reply, row.force = len(prompt_ids), [], collections.deque()
            self._merge(out.past_key_values, len(prompt_ids))
            self.rows.append(row)
            if row.mirror is None and begin:
                self._begin(row, out.logits[0, -1])

    def remove(self, key: str):
        with self.lock:
            keep = [i for i, r in enumerate(self.rows) if r.key != key]
            if len(keep) == len(self.rows):
                return
            if not keep:
                self.rows, self.cache, self.mask = [], None, None
                return
            idx = torch.tensor(keep, device=self.mind.device)
            for layer in self.cache.layers:
                att, rec = _layer_tensors(layer)
                for n, t in att:
                    setattr(layer, n, t.index_select(0, idx))
                for n, i, t in rec:
                    getattr(layer, n)[i] = t.index_select(0, idx)
            self.mask = self.mask.index_select(0, idx)
            self.rows = [self.rows[i] for i in keep]

    def say(self, key: str, prompt_ids: list[int]):
        """Queue the next prompt for a resting row: read one token per step, then it writes its reply."""
        with self.lock:
            r = self.row(key)
            if r is None or r.writing:
                return False
            r.force.extend(prompt_ids)
            r.after = "write"
            return True

    @torch.no_grad()
    def read_now(self, keys: list[str], ids: list[int]) -> bool:
        """Read the same new prompt tokens for these rows now, each in one pass outside the batch (a few tenths of a
        second, where say() takes a step per token), splice them in as new cache columns (masked for every other
        row), and start their replies -- a mirror among them reads unpushed, and its leader's twin starts beside it."""
        with self.lock:
            rows = [self.row(k) for k in keys]
            if not ids or any(r is None or r.writing for r in rows):
                return False
            # (tokens still waiting to be read -- the last reply's end token -- are read first; a mirror waits on its leader's)
            pend = {r.key: list((self.row(r.mirror) if r.mirror is not None else r).force) for r in rows}
            if len({len(v) for v in pend.values()}) > 1:
                return False
            ids = next(iter(pend.values())) + list(ids)
            for r in rows:
                r.force.clear()
            dev, k = self.mind.device, len(ids)
            subs, last = [], []
            for r in rows:   # (one row at a time, 64 tokens at a time: the FP8 kernel's fast path)
                i = self.rows.index(r)
                idx = torch.tensor([i], device=dev)
                sub, msk, pos = self._slice(idx), self.mask.index_select(0, idx), r.pos
                self.mind.set_coef(self._coef([r]))
                for c0 in range(0, k, 64):
                    part = ids[c0:c0 + 64]
                    msk = torch.cat([msk, torch.ones(1, len(part), dtype=torch.bool, device=dev)], 1)
                    out = self.model(torch.tensor([part], device=dev), past_key_values=sub, attention_mask=msk.long(),
                                     position_ids=torch.arange(pos, pos + len(part), device=dev)[None], use_cache=True, logits_to_keep=1)
                    sub, pos = out.past_key_values, pos + len(part)
                subs.append((i, sub))
                last.append(out.logits[0, -1])
            self._splice(subs, k)
            for r in rows:
                r.pos += k
            logits = torch.stack(last)
            mirror_j = {r.mirror: j for j, r in enumerate(rows) if r.mirror is not None}
            leads = [(j, r) for j, r in enumerate(rows) if r.mirror is None]
            if not leads:
                return True
            got = self._sample_many([r for _, r in leads], logits[torch.tensor([j for j, _ in leads], device=dev)])
            for (j, r), (t, info) in zip(leads, got):
                mj = mirror_j.get(r.key)
                if mj is not None:
                    self._unpushed_view(r, logits[mj], t, info)
                r.reply, r.writing, r.stop_soon, r.finish, r.after = [], True, False, False, "rest"
                if t in self.stops:
                    self._end(r, t)
                    continue
                r.next_id, r.next_info = t, info
                if mj is not None and self.row(r.key + "-plain") is None:
                    self._twin(r, self.rows.index(rows[mj]), logits[mj])
            return True

    def _unpushed_view(self, r: Row, mirror_logits, t: int, info: dict):
        """The mirror's view of the word its leader drew (with the leader's repetition control): top 4 and its odds."""
        pm = torch.softmax(self._penalize(mirror_logits[None].float().clone(), [r]), -1)[0]
        cv, ci = torch.topk(pm, 4)
        info["cf"] = [[self.tok.decode([x]), round(y, 4)] for x, y in zip(ci.tolist(), cv.tolist())]
        info["q"] = round(float(pm[t]), 4)

    def _twin(self, r: Row, mirror_i: int, mirror_logits):
        """Its unpushed twin for this reply: a copy of the mirror's memory, starting its own answer."""
        p = Row(key=r.key + "-plain", plain_of=r.key, max_new=r.max_new, soft=r.soft, temperature=r.temperature,
                top_p=r.top_p, rep_penalty=r.rep_penalty, rep_window=r.rep_window, no_repeat_ngram=r.no_repeat_ngram)
        p.recent.extend(r.recent)
        self._clone(mirror_i, p)
        self._begin(p, mirror_logits)

    def _slice(self, idx):
        """A cache holding only these rows (copies: the batch's cache is untouched)."""
        import copy
        sub = copy.copy(self.cache)
        sub.layers = []
        for layer in self.cache.layers:
            nl = copy.copy(layer)
            for n in ("conv_states", "recurrent_states"):
                v = getattr(layer, n, None)
                if isinstance(v, (list, tuple)):
                    setattr(nl, n, list(v))
                elif isinstance(v, dict):
                    setattr(nl, n, dict(v))
            att, rec = _layer_tensors(layer)
            for n, x in att:
                setattr(nl, n, x.index_select(0, idx))
            for n, i, t in rec:
                getattr(nl, n)[i] = t.index_select(0, idx)
            sub.layers.append(nl)
        return sub

    def _splice(self, subs, k: int):
        """Put rows read on their own (each: (row index, its cache, k tokens longer)) back: k new columns for all."""
        dev = self.mind.device
        T = self.mask.shape[1]
        for li, layer in enumerate(self.cache.layers):
            att_b, rec_b = _layer_tensors(layer)
            for a_i, (n, xb) in enumerate(att_b):
                y = torch.cat([xb, xb.new_zeros(xb.shape[0], xb.shape[1], k, xb.shape[3])], 2)
                for i, sub in subs:
                    y[i] = _layer_tensors(sub.layers[li])[0][a_i][1][0]
                setattr(layer, n, y)
            for r_i, (n, j, xb) in enumerate(rec_b):
                for i, sub in subs:
                    xb[i] = _layer_tensors(sub.layers[li])[1][r_i][2][0].to(xb.dtype)
        self.mask = torch.cat([self.mask, torch.zeros(self.mask.shape[0], k, dtype=torch.bool, device=dev)], 1)
        for i, _ in subs:
            self.mask[i, T:] = True

    def clone(self, src_key: str, row: Row) -> bool:
        """Add a row whose memory is a copy of another's (e.g. a mirror of a row whose memory was read unpushed)."""
        with self.lock:
            src = self.row(src_key)
            if src is None or self.row(row.key) is not None:
                return False
            self._clone(self.rows.index(src), row)
            return True

    def row(self, key: str) -> Row | None:
        return next((r for r in self.rows if r.key == key), None)

    # ------------------------------------------------------------------ the batch
    def _merge(self, single, t: int):
        """Merge a one-row cache (t tokens) into the batch, right-aligned: gaps on the left are masked."""
        dev = self.mind.device
        if self.cache is None:
            self.cache, self.mask = single, torch.ones(1, t, dtype=torch.bool, device=dev)
            return
        T = self.mask.shape[1]
        if t > T:   # the batch grows on the left
            for layer in self.cache.layers:
                for n, x in _layer_tensors(layer)[0]:
                    pad = torch.zeros(*x.shape[:2], t - T, x.shape[3], dtype=x.dtype, device=dev)
                    setattr(layer, n, torch.cat([pad, x], 2))
            self.mask = torch.cat([torch.zeros(self.mask.shape[0], t - T, dtype=torch.bool, device=dev), self.mask], 1)
            T = t
        for layer, sl in zip(self.cache.layers, single.layers):
            att_b, rec_b = _layer_tensors(layer)
            att_s, rec_s = _layer_tensors(sl)
            for (n, xb), (_, xs) in zip(att_b, att_s):
                if xs.shape[2] < T:
                    xs = torch.cat([torch.zeros(*xs.shape[:2], T - xs.shape[2], xs.shape[3], dtype=xs.dtype, device=dev), xs], 2)
                setattr(layer, n, torch.cat([xb, xs], 0))
            for (n, i, xb), (_, _, xs) in zip(rec_b, rec_s):
                getattr(layer, n)[i] = torch.cat([xb, xs.to(xb.dtype)], 0)
        new = torch.zeros(1, T, dtype=torch.bool, device=dev)
        new[0, T - t:] = True
        self.mask = torch.cat([self.mask, new], 0)

    def _compact(self):
        """Drop cache columns no row uses (the left edge); if still long, squeeze each row's gaps out."""
        used = self.mask.any(0)
        first = int(used.nonzero()[0]) if bool(used.any()) else self.mask.shape[1]
        if first > 0:
            for layer in self.cache.layers:
                for n, x in _layer_tensors(layer)[0]:
                    setattr(layer, n, x[:, :, first:].contiguous())
            self.mask = self.mask[:, first:]
        if self.mask.shape[1] <= self.compact_at:
            torch.cuda.empty_cache()   # hand freed blocks back (GPU memory is the machine's memory here)
            return
        dev, T2 = self.mind.device, int(self.mask.sum(1).max())
        for layer in self.cache.layers:
            for n, x in _layer_tensors(layer)[0]:
                y = torch.zeros(x.shape[0], x.shape[1], T2, x.shape[3], dtype=x.dtype, device=dev)
                for b in range(x.shape[0]):
                    cols = self.mask[b].nonzero().squeeze(1)
                    y[b, :, T2 - len(cols):] = x[b, :, cols]
                setattr(layer, n, y)
        m2 = torch.zeros(self.mask.shape[0], T2, dtype=torch.bool, device=dev)
        for b in range(m2.shape[0]):
            m2[b, T2 - int(self.mask[b].sum()):] = True
        self.mask = m2

    def _coef(self, rows):
        labels = self.mind.labels
        return torch.tensor([[float(r.steer.get(l, 0.0)) for l in labels] for r in rows])

    def _noise(self, b):
        V = self.model.config.get_text_config().vocab_size
        return -torch.log(-torch.log(torch.rand((b, V), generator=self.gen, device=self.mind.device).clamp(1e-9, 1 - 1e-9)))

    def _sample(self, r: Row, logits):
        return self._sample_many([r], logits[None])[0]

    def _penalize(self, lg, rows):
        """Each row's repetition control on its logits (in place): recent tokens lowered, repeated phrases banned."""
        for j, r in enumerate(rows):
            ids = list(r.recent)[-r.rep_window:]
            if r.rep_penalty > 0 and ids:
                lg[j, torch.tensor(sorted(set(ids)), device=lg.device)] -= r.rep_penalty
            n = r.no_repeat_ngram
            if n > 1 and len(ids) >= n:
                prefix, banned = tuple(ids[-(n - 1):]), set()
                for k in range(len(ids) - n + 1):
                    if tuple(ids[k:k + n - 1]) == prefix:
                        banned.add(ids[k + n - 1])
                if banned:
                    lg[j, torch.tensor(sorted(banned), device=lg.device)] = -float("inf")
        return lg

    def _sample_many(self, rows, logits, k_top: int = 1024):
        """Draw the next token for several rows at once (Gumbel-max over each row's nucleus, within its top k_top):
        one batched pass instead of one sort of the whole vocabulary per row. -> [(token, {p, a})]"""
        lg = self._penalize(logits.float().clone(), rows)
        temps = torch.tensor([max(1e-4, r.temperature) for r in rows], device=lg.device)[:, None]
        tops = torch.tensor([r.top_p for r in rows], device=lg.device)[:, None]
        lp = torch.log_softmax(lg / temps, -1)
        v, i = torch.topk(lp, k_top, dim=-1)                       # sorted, descending
        pr = v.exp()
        keep = (pr.cumsum(-1) - pr) <= tops
        g = -torch.log(-torch.log(torch.rand(v.shape, generator=self.gen, device=lg.device).clamp(1e-9, 1 - 1e-9)))
        pick = torch.argmax(torch.where(keep, v + g, torch.full_like(v, -float("inf"))), -1)
        tok = i.gather(1, pick[:, None])[:, 0]
        p_all = torch.softmax(lg, -1)
        p = p_all.gather(1, tok[:, None])[:, 0]
        av, ai = torch.topk(p_all, 4, dim=-1)
        tok, p, av, ai = tok.tolist(), p.tolist(), av.tolist(), ai.tolist()
        return [(t, {"p": pp, "a": [[self.tok.decode([x]), round(y, 4)] for x, y in zip(xs, ys)]}) for t, pp, xs, ys in zip(tok, p, ai, av)]

    def _begin(self, r: Row, logits):
        """Its prompt is read: draw the first word of its reply."""
        r.reply, r.writing, r.stop_soon, r.finish = [], True, False, False
        t, info = self._sample(r, logits)
        if t in self.stops:
            self._end(r, t)
        else:
            r.next_id, r.next_info = t, info

    def _end(self, r: Row, t=None):
        r.writing, r.next_id, r.next_info, r.stop_soon, r.finish = False, None, None, False, False
        r.force.append(t if t in self.stops else self.tok.convert_tokens_to_ids("<|im_end|>"))   # the end token is read too
        r.after = "rest"
        if r.on_end:
            r.on_end(r, self.tok.decode(r.reply, skip_special_tokens=True))

    # ------------------------------------------------------------------ one step for every row
    @torch.no_grad()
    def step(self) -> bool:
        """One token for every row that is reading or writing. Returns False when no row is busy."""
        with self.lock:
            if not self.rows or all(r.idle for r in self.rows):
                return False
            t_in = _tick()
            m, dev, B = self.mind, self.mind.device, len(self.rows)
            feed, kind = [], []
            for r in self.rows:
                if r.mirror is not None:
                    feed.append(self.filler); kind.append("idle")   # (filled in below, from its leader)
                elif r.force:
                    feed.append(r.force.popleft()); kind.append("read")
                elif r.writing:
                    feed.append(r.next_id); kind.append("write")
                else:
                    feed.append(self.filler); kind.append("idle")
            index = {r.key: i for i, r in enumerate(self.rows)}
            for i, r in enumerate(self.rows):
                j = index.get(r.mirror) if r.mirror is not None else None
                if j is not None and kind[j] != "idle":
                    feed[i], kind[i] = feed[j], "read"
            idle = [i for i, k in enumerate(kind) if k == "idle"]
            # a resting row is fed a filler: its recurrent state is put back afterwards and the filler's attention
            # column is masked, so the filler leaves no trace
            saved = self._save_rec(idle) if idle else None
            am = torch.cat([self.mask, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
            posn = torch.tensor([[r.pos] for r in self.rows], device=dev)
            m.set_coef(self._coef(self.rows))
            tp = _tick()
            out = self.model(torch.tensor([[t] for t in feed], device=dev), past_key_values=self.cache,
                             attention_mask=am.long(), position_ids=posn, use_cache=True, logits_to_keep=1)
            self.cache = out.past_key_values
            if idle:
                self._restore_rec(idle, saved)
                am[torch.tensor(idle, device=dev), -1] = False
            self.mask = am
            ro = m.readout()            # [B, E]: the state each row read its token in, before steering
            tf = _tick()
            logits = out.logits[:, -1, :]
            self.last_logits = logits   # diagnostics
            ro_l = ro.tolist()
            mirrors = {r.mirror: i for i, r in enumerate(self.rows) if r.mirror is not None and kind[i] != "idle"}
            plains = {r.plain_of: i for i, r in enumerate(self.rows) if r.plain_of is not None and kind[i] == "write"}
            draw = []                    # (row index, row, "write" | "begin")
            for i, (r, k) in enumerate(zip(self.rows, kind)):
                if k == "idle":
                    continue
                r.pos += 1
                if k == "write":
                    r.reply.append(feed[i])
                    r.recent.append(feed[i])
                    if r.on_token:
                        info = r.next_info or {}
                        tok = {"id": feed[i], "t": self.tok.decode([feed[i]]), "p": round(info.get("p", 0.0), 4), "a": info.get("a", []),
                               "e": [round(x, 3) for x in ro_l[i]], "steer": dict(r.steer)}
                        if r.key in mirrors:   # the same words read unpushed: what it would have said instead
                            tok["cf"], tok["q"] = info.get("cf", []), info.get("q")
                        pi = plains.get(r.key)
                        if pi is not None:     # the unpushed answer's readout at the same point: the baseline
                            tok["b"] = [round(x, 3) for x in ro_l[pi]]
                        r.on_token(r, tok)
                    draw.append((i, r, "write"))
                elif k == "read" and not r.force and r.after == "write":
                    draw.append((i, r, "begin"))
            ts = _tick()
            if draw:
                got = self._sample_many([r for _, r, _ in draw], logits[torch.tensor([i for i, _, _ in draw], device=dev)])
                led = [(n, r, mirrors[r.key]) for n, (_, r, _) in enumerate(draw) if r.key in mirrors]
                for n, r, mi in led:   # the mirrors' view of the same next word
                    self._unpushed_view(r, logits[mi], *got[n])
                start_plain = []
                for (i, r, what), (t, info) in zip(draw, got):
                    if what == "begin":
                        r.reply, r.writing, r.stop_soon, r.finish = [], True, False, False
                        if t in self.stops:
                            self._end(r, t)
                        else:
                            r.next_id, r.next_info = t, info
                            if r.key in mirrors and self.row(r.key + "-plain") is None:
                                start_plain.append((r, mirrors[r.key]))
                        continue
                    tail = self.tok.decode(r.reply[-4:], skip_special_tokens=True)
                    sentence_end = tail.rstrip().endswith((".", "!", "?", '."', '!"', '?"'))
                    if t in self.stops or r.finish or len(r.reply) >= r.max_new or ((len(r.reply) >= r.soft or r.stop_soon) and sentence_end):
                        self._end(r, t)
                    else:
                        r.next_id, r.next_info = t, info
                for r, mi in start_plain:   # its unpushed twin starts from the mirror's memory (the same conversation)
                    self._twin(r, mi, logits[mi])
            for r in [r for r in self.rows if r.plain_of is not None]:
                lead = self.row(r.plain_of)
                if lead is None or not lead.writing:   # its leader's reply ended: so does the twin
                    self.remove(r.key)
            te = _tick()
            if PROFILE:   # where a step's time goes (ms, smoothed): before the model, the model, reading out + callbacks, sampling
                for k, v in (("prep", tp - t_in), ("model", tf - tp), ("tokens", ts - tf), ("sample", te - ts)):
                    self.parts[k] = 0.9 * self.parts.get(k, v * 1000) + 0.1 * v * 1000
                self.parts["rows"] = B; self.parts["cache"] = self.mask.shape[1]
            self.steps += 1
            if self.mask.shape[1] > self.compact_at or self.steps % 500 == 0:
                self._compact()
            return True

    def _clone(self, src: int, row: Row):
        """Add a row whose memory is a copy of row `src`'s."""
        for layer in self.cache.layers:
            att, rec = _layer_tensors(layer)
            for n, x in att:
                setattr(layer, n, torch.cat([x, x[src:src + 1]], 0))
            for n, i, t in rec:
                getattr(layer, n)[i] = torch.cat([t, t[src:src + 1]], 0)
        self.mask = torch.cat([self.mask, self.mask[src:src + 1]], 0)
        row.pos = self.rows[src].pos
        self.rows.append(row)

    def _save_rec(self, idx):
        ix = torch.tensor(idx, device=self.mind.device)
        return [[(n, i, t.index_select(0, ix).clone()) for n, i, t in _layer_tensors(layer)[1]] for layer in self.cache.layers]

    def _restore_rec(self, idx, saved):
        ix = torch.tensor(idx, device=self.mind.device)
        for layer, keep in zip(self.cache.layers, saved):
            for n, i, t in keep:
                getattr(layer, n)[i].index_copy_(0, ix, t)

    # ------------------------------------------------------------------ the loop
    def run(self, stop: threading.Event):
        """Step until stopped, at most step_rate steps a second (each android talks at this pace)."""
        while not stop.is_set():
            t0 = time.time()
            busy = self.step()
            dt = time.time() - t0
            if busy:
                self.t_step = 0.9 * self.t_step + 0.1 * dt if self.t_step else dt
            time.sleep(max(0.0 if busy else 0.05, 1.0 / self.step_rate - dt))
