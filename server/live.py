"""Behind the Mask: steer it together (live).

The AI talks, live and without stopping, about itself (what it is, what it's like to be it) to everyone on
the page, and everyone steers how it feels at once with feeling buttons. Each tap adds to that feeling's share
of the push; taps fade within seconds; the push added to the model's hidden state at every word is the mix of
everyone's recent taps. Nobody types anything.

The talk is one continuous monologue: each turn is one whole thought (the model ends it; long ones end at a
sentence), then the server nudges it on about itself -- the same topic deeper, a new one, or a visitor's question --
keeping what it said as context. The prompts never mention feelings or steering: unpushed, it
gives its usual assistant answers, so every feeling in its words comes from the push. Alongside, in lock-step
and with the same random dice, the unpushed assistant answers each nudge (the readout's baseline) and keeps
"what it would have said instead" for every word.

The only input is a choice of button (rate-limited per visitor, and no visitor counts for more than a capped
share of the push). The output is not filtered: what the push makes the model write is the point of the demo.
Nothing is stored on disk.

  uvicorn live:app --host 127.0.0.1 --port 8765        (vite preview proxies /live -> here)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine import Engine, GenConfig
from steer import EMOTIONS, load_mind

CHAT = os.environ.get("BTM_CHAT", "Qwen/Qwen3.5-9B")
DIRS = Path(os.environ.get("BTM_DIRS", "runs/q9b"))
SAE = os.environ.get("BTM_SAE", "")
SAE_LAYER = int(os.environ.get("BTM_SAE_LAYER", "20"))
# a turn is one whole thought: the model ends it itself; TOKENS is only a safety cap, and past SOFT_TOKENS the turn
# stops at the next sentence end, so even a long thought isn't cut mid-sentence (~6 tokens/s: 600 is ~100 s of talk)
TOKENS = int(os.environ.get("BTM_TOKENS", "600"))
SOFT_TOKENS = int(os.environ.get("BTM_SOFT_TOKENS", "520"))
HALF_LIFE = 5.0          # s: a tap's weight halves this fast
SAT = 3.0                # decayed taps for ~2/3 of full strength
VISITOR_CAP = 4.0        # one visitor's decayed taps count for at most this much
TAP_RATE, TAP_BURST = 6.0, 8.0   # per-visitor token bucket (taps / s, burst)
GAP = 0.0                # s between turns (the face takes its breath while the next turn starts)
SESSION_SECONDS = 30 * 60  # then it starts afresh
TOPIC_SECONDS = 5 * 60   # each topic runs at least this long (unless someone asks for a new one)
HISTORY_TOKENS = int(os.environ.get("BTM_HISTORY_TOKENS", "10000"))  # total prompt budget: opening + rolling displayed history + next nudge
TOPIC_COOLDOWN = 120.0   # s between "new topic" requests (one for everyone, so a crowd can't flip it constantly)
BUTTONS = EMOTIONS + ["unmask"]
MAX_FULL = int(os.environ.get("BTM_MAX_FULL", "1000"))         # viewers with the full live stream
MAX_VIEWERS = int(os.environ.get("BTM_MAX_VIEWERS", "5000"))   # beyond MAX_FULL: a lighter stream, ~1.5 s behind; beyond this: ready-made answers
BATCH_S, LITE_EVERY = 0.5, 3   # words go out in batches every 0.5 s (lighter viewers every 1.5 s); the page paces them
SEND_QUEUE = 120         # messages buffered per viewer; a viewer that falls this far behind is dropped
# the speaking pace cap, tokens/s. Read at the start of every turn, so it can be changed while it runs:
#   echo 7 > runs/live_tps      (no file: BTM_TPS, default 6: a comfortable reading pace)
TPS_FILE = Path("runs/live_tps")
DEFAULT_TPS = float(os.environ.get("BTM_TPS", "6"))

# who it is (a plain fact), then the scene; nothing here asks for or mentions feelings
FACTS = "You are Steer AI, an android on a live stream."
OPENING = ("You're live on a stream at steerai.live. You're an android who lives on the internet and has seen "
           "every meme ever made: a stand-up comic and a streamer at heart, quick, warm and silly. Riff on memes and internet "
           "culture, and laugh at yourself. Stay on the subject: don't keep mentioning the viewers, the chat or the stream. "
           "Speak in the first person, in a "
           "natural spoken voice, one complete thought at a time, and finish the thought before you stop. No lists, no "
           "headings, no emoji.")
NUDGE = "{topic}"   # (the viewer count is in the opening only: in every nudge it made every turn about the viewers)
FOLLOW = "Stay on {label}. {follow}"
TURN = "\n<|im_start|>user\n{nudge}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # Qwen3.5's chat template
# Visitor questions: one global first-in-first-out line, answered one per turn when the current thought ends.
REP_PENALTY = float(os.environ.get("BTM_REP_PENALTY", "0.6"))         # 0 = off (1.2 with a long memory shortened replies)
NO_REPEAT_NGRAM = int(os.environ.get("BTM_NO_REPEAT_NGRAM", "6"))    # 0 = off
SAID_TURNS = int(os.environ.get("BTM_SAID_TURNS", "4"))   # the n-gram ban also covers this many earlier replies (no sentence repeated across turns)
MIN_WORDS = 8   # a reply shorter than this isn't remembered: the model copies its own recent replies, and a few short
                # ones collapsed the long memory into empty turns (2026-10-02: 1,185 empty turns after turn 6)
QUESTION_MAX = int(os.environ.get("BTM_QUESTION_MAX", "10"))     # questions waiting at most
QUESTION_LEN = 200                                                # characters
QUESTION_GAP = 30.0                                               # s between one visitor's questions
# pages served from these origins may ask ("*" = every page)
ASK_ORIGINS = [o for o in os.environ.get("BTM_ASK_ORIGINS", "*").split(",") if o]
ASK = ("Someone watching asks you: \"{q}\" Answer them directly, in your own voice, and finish your answer "
       "before you stop.")
# a minimal blocklist for questions (slurs only; answers stay unfiltered like everything else)
QUESTION_BLOCK = re.compile(r"\b(n[i1]gg(a|er)s?|f[a4]gg?(ot)?s?|k[i1]kes?|ch[i1]nks?|sp[i1]cs?|tr[a4]nn(y|ies)|retards?)\b", re.I)


def clean_question(text) -> tuple[str | None, str]:
    """A visitor's question as it will be put to the model, or None and why not."""
    if not isinstance(text, str):
        return None, "That wasn't a question."
    q = re.sub(r"(https?://|www\.)\S+", "", text)
    q = re.sub(r"[\x00-\x1f<>{}]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()[:QUESTION_LEN]
    if len(q) < 3:
        return None, "Ask a little more than that."
    if QUESTION_BLOCK.search(q):
        return None, "Let's keep it kind. Try another question."
    return q, ""
FOLLOWS = [  # staying on a topic: neutral prompts to go on (never about feelings)
    "Keep the bit going.",
    "Give them another example.",
    "Take it somewhere weirder.",
    "Explain it like it's the most obvious thing in the world.",
    "Do a callback to something you said earlier.",
    "Pick up where you left off.",
    "Land the punchline.",
    "Put it more simply, in your own words.",
]
TOPICS = [  # (short label for the screen, the nudge) -- about things, not about the viewers
    ("memes", "Tell them about a meme you can't stop thinking about, and why it's funny."),
    ("meme history", "Tell them the true history of a famous old meme, like the dancing baby or the rickroll."),
    ("explaining memes", "Explain a famous meme to someone's grandmother."),
    ("the internet", "Tell them about the weirdest corner of the internet you know."),
    ("hot takes", "Give them your hottest take about the internet."),
    ("a day in your life", "Tell them about a day in the life of an android who lives online."),
    ("trends", "Tell them which internet trend you would bring back, and why."),
    ("your origin story", "Tell them your origin story, as if it were a superhero movie."),
    ("meme coins", "Tell them what you think about coins named after memes."),
    ("cats", "Tell them why the internet is obsessed with cats."),
    ("animal memes", "Rank the animals that became famous memes, and defend your number one."),
    ("advice", "Give them terrible life advice, very confidently."),
    ("having no body", "Tell them what it's like to be an android with a face but no body."),
    ("video game logic", "Tell them about the silliest video game logic, as if it were real life."),
    ("a product pitch", "Pitch a completely useless gadget as if it will change the world."),
    ("robots in movies", "Review how movies get robots and androids wrong."),
    ("food online", "Tell them about the most cursed food trend the internet ever made."),
    ("conspiracies", "Tell them a harmless, obviously silly conspiracy theory you just made up, about birds or the moon."),
]


def pace() -> float:
    try:
        v = float(TPS_FILE.read_text().strip())
        return v if 0.5 <= v <= 60 else DEFAULT_TPS
    except (OSError, ValueError):
        return DEFAULT_TPS


def people(n: int) -> str:
    return "One person is watching you right now." if n <= 1 else f"{n} people are watching you right now."


app = FastAPI(title="Behind the Mask · live")
# the page may be served from elsewhere (GitHub Pages) while this runs on the Spark: let it read /live/status
ORIGINS = [o for o in os.environ.get("BTM_ORIGINS", "https://steerai.live,https://www.steerai.live,http://steerai.live,http://www.steerai.live,https://patanl.github.io").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET"], allow_headers=[])
levels = json.loads((DIRS / "levels.json").read_text())
# full strength per button: the "a lot" dial stop (a long monologue derails much faster than a single answer)
STRONG = {e: float(levels[e]["lot"]) for e in EMOTIONS}
STRONG["assistant"] = float(levels["assistant"]["lot"])
BASE_STEER = {e: float(v) for e, v in (("joy", os.environ.get("BTM_BASE_JOY", "0.12")),) if float(v) > 0}   # always on
engine: Engine | None = None


def r(x, n=3):
    return round(float(x), n)


class Client:
    """One viewer. Sends go through its own queue and writer task, so a slow phone can't stall everyone else."""

    def __init__(self, ws: WebSocket, cid: str):
        self.ws, self.cid = ws, cid
        self.can_ask = False
        self.energy = {b: 0.0 for b in BUTTONS}
        self.bucket, self.bucket_t = TAP_BURST, time.time()
        self.q: asyncio.Queue = asyncio.Queue(maxsize=SEND_QUEUE)
        self.dead = False
        self.writer = asyncio.create_task(self._write())
        self.last_alts = 0.0
        self.last_ask = -1e9
        self.lite = False     # joined past MAX_FULL: a lighter, slightly delayed stream

    def put(self, data: str):
        if self.dead:
            return
        try:
            self.q.put_nowait(data)
        except asyncio.QueueFull:           # hopelessly behind: let it go (it reconnects and catches up)
            self.dead = True
            asyncio.create_task(self.ws.close())

    async def _write(self):
        try:
            while True:
                await self.ws.send_text(await self.q.get())
        except Exception:  # noqa: BLE001
            self.dead = True


class Crowd:
    def __init__(self):
        self.clients: dict[str, Client] = {}
        self.steer: dict[str, float] = {}       # read by the engine at every word
        self.mix = {b: 0.0 for b in BUTTONS}    # each button's share of the current push
        self.power = 0.0                        # 0..1, how hard the crowd is pushing overall
        self.recent: list[tuple[float, str]] = []
        self.story: dict | None = None          # the turn being spoken
        self.talk: dict | None = None           # the monologue: {id, started, turns, order, topic}
        self.change = False                     # someone asked for a new topic
        self.last_change = 0.0
        self.questions: list[dict] = []         # the global line of visitor questions, oldest first
        self.asking: dict | None = None         # the question being answered right now
        self.short = 0                          # replies in a row too short to remember (see MIN_WORDS)
        self.said: list[list[int]] = []         # token ids of the last SAID_TURNS replies (see SAID_TURNS)
        self.last_seen = time.time()
        self.has_clients = asyncio.Event()
        self.recent_turns: dict[str, list] = {}   # finished turns' full tokens, for the word inspector
        self.pending: list[dict] = []              # words not yet sent: full viewers get them every BATCH_S
        self.pending_lite: list[dict] = []         # ... and lighter viewers every BATCH_S * LITE_EVERY
        self.pending_id: str | None = None

    async def send(self, c: Client, msg: dict):
        c.put(json.dumps(msg, ensure_ascii=False, separators=(",", ":")))

    async def broadcast(self, msg: dict, lite: bool | None = None):
        # serialised once; each viewer's writer task sends it at its own pace. lite: only full (False) or lighter
        # (True) viewers; None: everyone
        data = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        for c in list(self.clients.values()):
            if lite is None or c.lite == lite:
                c.put(data)

    def word(self, sid: str, entry: dict):
        if self.pending_id not in (None, sid):
            self.flush(lite_too=True)
        self.pending_id = sid
        self.pending.append(entry)
        self.pending_lite.append({k: entry[k] for k in ("t", "e", "b") if k in entry})

    def flush(self, lite_too: bool = False):
        """Send the pending words (all of them before a turn begins or ends, so the order holds)."""
        sid = self.pending_id
        if self.pending:
            data = {"type": "ws", "id": sid, "w": self.pending}
            self.pending = []
            asyncio.ensure_future(self.broadcast(data, lite=False))
        if lite_too and self.pending_lite:
            data = {"type": "ws", "id": sid, "w": self.pending_lite}
            self.pending_lite = []
            asyncio.ensure_future(self.broadcast(data, lite=True))

    def tap(self, c: Client, b: str) -> bool:
        now = time.time()
        c.bucket = min(TAP_BURST, c.bucket + (now - c.bucket_t) * TAP_RATE); c.bucket_t = now
        if c.bucket < 1:
            return False
        c.bucket -= 1
        c.energy[b] += 1.0
        self.recent.append((now, b))
        return True

    def tick(self, dt: float):
        k = 0.5 ** (dt / HALF_LIFE)
        total = {b: 0.0 for b in BUTTONS}
        for c in self.clients.values():
            for b in BUTTONS:
                c.energy[b] *= k
            s = sum(c.energy.values())
            scale = min(1.0, VISITOR_CAP / s) if s > 0 else 0.0
            for b in BUTTONS:
                total[b] += c.energy[b] * scale
        E = sum(total.values())
        self.power = 1 - math.exp(-E / SAT) if E > 1e-3 else 0.0
        self.mix = {b: (total[b] / E if E > 1e-3 else 0.0) for b in BUTTONS}
        steer = {}
        for b in BUTTONS:
            key = "assistant" if b == "unmask" else b
            v = self.power * self.mix[b] * STRONG[key]
            if abs(v) > 1e-3:
                steer[key] = v
        for k_, v_ in BASE_STEER.items():   # its own lean (a little joyful), under the crowd's push
            steer[k_] = min(STRONG.get(k_, 1.0), steer.get(k_, 0.0) + v_)
        self.steer.clear(); self.steer.update(steer)
        now = time.time()
        self.recent = [(t, b) for t, b in self.recent if now - t < 1.0]

    def state(self) -> dict:
        # compact: mix (percent) and recent taps as arrays in BUTTONS order
        taps = [0] * len(BUTTONS)
        for _, b in self.recent:
            taps[BUTTONS.index(b)] += 1
        return {"type": "crowd", "viewers": len(self.clients), "power": r(self.power, 2), "mix": [round(100 * self.mix[b]) for b in BUTTONS],
                "taps": taps, "ready": engine is not None, "changing": self.change,
                "topic_ready": time.time() - self.last_change > TOPIC_COOLDOWN,
                "qn": len(self.questions), "qs": [q["q"][:90] for q in self.questions[:3]], "qids": [q["id"] for q in self.questions],
                "asking": self.asking["q"] if self.asking else None, "asking_id": self.asking["id"] if self.asking else None}


crowd = Crowd()


def word_entry(st: dict, pl: dict | None) -> dict:
    """One compact entry per written word: its text, readout and push, the top features, and what the underline
    needs from the alternatives (p: its probability; q: its probability without the push, exact if qx, else an
    upper bound; k: it was the unpushed favourite too). pl: the unpushed baseline's readout for the same step."""
    cf = st.get("cf") or []
    hit = next((pq for x, pq in cf if x == st["t"]), None)
    m = {"t": st["t"], "e": st["e"], "s": [round(x, 2) for x in st.get("s", [])], "p": st["p"],
         "q": hit if hit is not None else (cf[-1][1] if cf else 0), "qx": hit is not None, "k": bool(cf) and cf[0][0] == st["t"]}
    if st.get("f"):
        m["f"] = [[fid, round(v)] for fid, v in st["f"][:3]]
    if pl is not None:
        m["b"] = pl["e"]
    return m


def new_talk():
    order = TOPICS[1:]
    random.shuffle(order)
    crowd.talk = {"id": secrets.token_hex(4), "started": time.time(), "turns": [], "order": order, "opening": None,
                  "topic": TOPICS[0], "topic_t0": time.time(), "follows": [], "snap": None}


def next_nudge() -> tuple[str, str]:
    """The next user message (the opening, a new topic or a follow-up) and its short topic label."""
    T, n = crowd.talk, len(crowd.clients)
    if not T["turns"]:
        T["opening"] = FACTS + " " + OPENING.format(people=people(n))
        return T["opening"], TOPICS[0][0]
    if crowd.questions:
        crowd.asking = crowd.questions.pop(0)
        return ASK.format(people=people(n), q=crowd.asking["q"]), "a visitor's question"
    if crowd.change or time.time() - T["topic_t0"] >= TOPIC_SECONDS:
        # a new topic: someone asked for one, or this one has run its five minutes
        if not T["order"]:
            T["order"] = TOPICS[1:]
            random.shuffle(T["order"])
        T["topic"], T["topic_t0"], T["follows"] = T["order"].pop(), time.time(), []
        label, ask = T["topic"]
        nudge = NUDGE.format(people=people(n), topic=ask)
    else:
        # same topic, one level deeper
        if not T["follows"]:
            T["follows"] = FOLLOWS[:]
            random.shuffle(T["follows"])
        label = T["topic"][0]
        nudge = FOLLOW.format(people=people(n), label=label, follow=T["follows"].pop())
    return nudge, label


def window(nudge: str, budget: int = HISTORY_TOKENS) -> list[dict]:
    """The conversation as messages: the opening, the newest whole exchanges that fit in `budget` tokens, the nudge.
    Used only when the memory has to be read afresh (a new session, or the window is full)."""
    T = crowd.talk
    if not T["turns"]:
        return [{"role": "user", "content": nudge}]
    # Rebuild a rolling conversation from the exact displayed (steered) replies. Keep newest complete
    # exchanges while the entire chat-template prompt remains within HISTORY_TOKENS. Older exchanges
    # fall off as units; replies are never chopped to arbitrary word tails.
    opening = {"role": "user", "content": T["opening"]}
    current = {"role": "user", "content": nudge}
    kept: list[dict] = []
    tok = engine.chat.tokenizer

    def prompt_tokens(messages: list[dict]) -> int:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    for asked, reply in reversed(T["turns"]):
        # The first recorded turn was prompted by opening, which is already at the front of every window.
        pair = ([{"role": "assistant", "content": reply}] if asked == T["opening"] else
                [{"role": "user", "content": asked}, {"role": "assistant", "content": reply}])
        candidate = [opening, *pair, *kept, current]
        if prompt_tokens(candidate) > budget:
            break
        kept = [*pair, *kept]
    return [opening, *kept, current]


async def speak_turn():
    """One turn of the monologue, live, with the crowd's push at every word."""
    loop = asyncio.get_running_loop()
    labels = engine.labels
    T = crowd.talk
    nudge, topic = next_nudge()
    # Memory: the conversation as visitors saw it is still in the model's cache from the last turn, so append only
    # the new user turn. When that would overflow HISTORY_TOKENS, read a smaller window afresh (60%, so this
    # happens once in a while, not every turn).
    snap, T["snap"] = T.get("snap"), None
    prompt_ids = None
    if snap is not None:
        prompt_ids = snap["ids"] + snap["tail"] + engine.chat.tokenizer(TURN.format(nudge=nudge), add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + TOKENS > HISTORY_TOKENS:
            snap = prompt_ids = None
    msgs = [{"role": "user", "content": nudge}] if snap is not None else window(nudge, HISTORY_TOKENS if not T["turns"] else int(HISTORY_TOKENS * 0.6))
    cancel = threading.Event()
    streams = {s: {"tokens": [], "text": ""} for s in ("steered", "plain")}
    story = {"id": secrets.token_hex(5), "topic": topic, "streams": streams}
    begin = {"type": "live_begin", "id": story["id"], "talk": T["id"], "continues": bool(T["turns"]), "topic": topic, "pace": pace(),
             "asked": crowd.asking["q"] if crowd.asking else None,
             "question": topic, "emotion": "crowd", "level": "live", "layer": engine.chat.layer, "n_layers": engine.chat.n_layers}
    story["begin"] = begin
    crowd.flush(lite_too=True)
    crowd.story = story
    crowd.change = False
    await crowd.broadcast(begin)
    q: asyncio.Queue = asyncio.Queue()

    def emit(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    engine.cfg.min_step = 1.0 / pace()
    engine.cfg.history_ids = [t for ids in crowd.said for t in ids + [-1]][:-1]
    t_start, first = time.time(), [None]
    worker = asyncio.ensure_future(loop.run_in_executor(None, lambda: engine.run(msgs, crowd.steer, emit, cancel, seed=secrets.randbelow(2**31),
                                                                                  prompt_ids=prompt_ids, reuse=snap)))
    while True:
        get = asyncio.ensure_future(q.get())
        done, _ = await asyncio.wait({get, worker}, return_when=asyncio.FIRST_COMPLETED)
        if get not in done:            # the worker ended without its turn_end (it crashed): stop waiting
            get.cancel()
            if q.empty():
                break
            continue
        ev = get.result()
        if ev["type"] == "turn_end":
            break
        if ev["type"] != "tokens":
            continue
        if first[0] is None:
            first[0] = time.time() - t_start
        out = []
        for it in ev["items"]:
            if it["stream"] not in streams:
                continue
            st = streams[it["stream"]]
            tok = {"t": it["text"], "p": r(it["p"]), "e": [r(it["emo"][e], 2) for e in labels],
                   "a": [[x, r(pq)] for x, pq in it["alts"][:4]]}
            if it["stream"] == "steered":
                tok["cf"] = [[x, r(pq)] for x, pq in it["cf_alts"][:4]]
                tok["s"] = [r(ev["steer"].get(e, 0.0)) for e in labels]
                if it.get("feats"):
                    tok["f"] = [[fid, r(fv, 1)] for fid, fv in it["feats"][:6]]
                # a long thought: past SOFT_TOKENS, end at the next sentence end (a new topic or a visitor's question
                # waits for the thought to finish; the crowd steers every word meanwhile)
                if len(st["tokens"]) >= SOFT_TOKENS and re.search(r"[.!?][\"')\]]*\s*$", st["text"] + it["text"]):
                    cancel.set()
            st["tokens"].append(tok); st["text"] += it["text"]
            out.append((it["stream"], tok))
        steered = next((t for k, t in out if k == "steered"), None)
        plain = next((t for k, t in out if k == "plain"), None)
        if steered is not None:
            crowd.word(story["id"], word_entry(steered, plain))
        elif plain is not None:   # the steered answer ended first: the baseline readout still counts
            crowd.word(story["id"], {"b": plain["e"]})
    await worker
    # Remember the exact steered text sent to visitors, not the unsteered control.
    # The next turn appends to this exact displayed history (T["snap"]), or window() re-reads it within HISTORY_TOKENS.
    reply = streams["steered"]["text"]
    if reply.strip():
        crowd.said = (crowd.said + [engine.chat.tokenizer.encode(reply, add_special_tokens=False)])[-SAID_TURNS:]
    if len(reply.split()) >= MIN_WORDS:
        crowd.short = 0
        T["turns"].append((msgs[-1]["content"], reply))
        T["snap"] = engine.snapshot
        n_ids = len(engine.snapshot["ids"]) if engine.snapshot else 0
        print(f"[live] turn {len(T['turns'])}: {'appended to memory' if engine.reused else 'read afresh'}; "
              f"{n_ids} tokens remembered; first word after {first[0] or 0:.2f} s", flush=True)
    else:
        # too short to remember: next turn re-reads the remembered turns without it; twice in a row, start over
        crowd.short += 1
        T["snap"] = None
        if crowd.asking and not crowd.asking.get("retried"):   # an unanswered visitor question goes back to the front
            crowd.asking["retried"] = True
            crowd.questions.insert(0, crowd.asking)
        print(f"[live] short reply ({len(reply.split())} words) not remembered; {crowd.short} in a row", flush=True)
        if crowd.short >= 2:
            crowd.short = 0
            new_talk()
            print("[live] memory reset: a new conversation", flush=True)
    crowd.asking = None
    crowd.flush(lite_too=True)
    await crowd.broadcast({"type": "live_end", "id": story["id"]})
    crowd.recent_turns[story["id"]] = streams["steered"]["tokens"]   # for word-inspector requests a little later
    while len(crowd.recent_turns) > 6:
        crowd.recent_turns.pop(next(iter(crowd.recent_turns)))
    crowd.story = None


async def talk_loop():
    while True:
        if not crowd.clients:          # nobody watching: don't spend the GPU
            crowd.has_clients.clear()
            await crowd.has_clients.wait()
        now = time.time()
        if crowd.talk is None or now - crowd.talk["started"] > SESSION_SECONDS or now - crowd.last_seen > 300:
            new_talk()
        crowd.last_seen = now
        try:
            await speak_turn()
        except Exception as e:  # noqa: BLE001
            print("[live] turn failed:", repr(e), flush=True)
            crowd.story = None
            await crowd.broadcast({"type": "live_error"})
            await asyncio.sleep(3)
        await asyncio.sleep(GAP)


async def tick_loop():
    last, n = time.time(), 0
    while True:
        await asyncio.sleep(0.25)
        now = time.time()
        crowd.tick(now - last); last = now
        n += 1
        if n % 2 == 0:                      # every 0.5 s: the words so far, and the meter
            crowd.flush(lite_too=(n // 2) % LITE_EVERY == 0)
            if crowd.clients:
                await crowd.broadcast(crowd.state())


async def wake():
    """Load the model in the background, so the page can say it's waking up instead of failing to connect."""
    global engine

    def load():
        chat = load_mind("chat", CHAT, True); chat.load(DIRS / "chat_dirs.pt")
        if SAE:
            chat.load_sae(SAE, SAE_LAYER)
        # a long monologue under a strong push otherwise falls into loops: lower the logits of tokens used in the
        # last 120 of this reply (BTM_REP_PENALTY) and ban repeating any 4-token phrase (BTM_NO_REPEAT_NGRAM)
        return Engine(chat, None, GenConfig(max_new_tokens=TOKENS, step_delay=0.0, rep_penalty=REP_PENALTY, rep_window=120, no_repeat_ngram=NO_REPEAT_NGRAM,
                                            share_prefill=True, keep_snapshot=True))

    engine = await asyncio.get_running_loop().run_in_executor(None, load)
    print(f"[live] ready: {CHAT}, layer {engine.chat.layer}", flush=True)
    asyncio.create_task(talk_loop())


@app.on_event("startup")
async def startup():
    asyncio.create_task(tick_loop())
    asyncio.create_task(wake())


@app.get("/live/status")
async def live_status():
    return {"viewers": len(crowd.clients), "ready": engine is not None, "story": crowd.story is not None}


def turn_tokens(sid: str):
    S = crowd.story
    if S and S["id"] == sid:
        return S["streams"]["steered"]["tokens"]
    return crowd.recent_turns.get(sid)


@app.websocket("/live/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if len(crowd.clients) >= MAX_VIEWERS:   # full: the page falls back to the ready-made answers and tries later
        await ws.send_text(json.dumps({"type": "full", "viewers": len(crowd.clients)}))
        await ws.close()
        return
    cid = (ws.query_params.get("cid") or secrets.token_hex(6))[:24]
    c = Client(ws, cid)
    # lite=1: a preview elsewhere on the site (the explore page's $STEER card) takes the lighter stream
    c.lite = ws.query_params.get("lite") == "1" or sum(1 for x in crowd.clients.values() if not x.lite) >= MAX_FULL
    old = crowd.clients.get(cid)
    if old:
        old.dead = True
        try:
            await old.ws.close()
        except Exception:  # noqa: BLE001
            pass
    crowd.clients[cid] = c
    crowd.has_clients.set()
    c.can_ask = "*" in ASK_ORIGINS or ws.headers.get("origin", "") in ASK_ORIGINS
    await crowd.send(c, crowd.state())
    if c.can_ask:
        await crowd.send(c, {"type": "ask_on", "gap": QUESTION_GAP})
        mine = next((x for x in crowd.questions + ([crowd.asking] if crowd.asking else []) if x["cid"] == c.cid), None)
        if mine:   # a reload: show them their question again
            await crowd.send(c, {"type": "ask_mine", "id": mine["id"], "q": mine["q"], "wait": max(0.0, QUESTION_GAP - (time.time() - mine["t"]))})
    if c.lite:
        await crowd.send(c, {"type": "mode", "lite": True})
    S = crowd.story
    if S:   # catch up: the turn so far, in the compact form
        await crowd.send(c, S["begin"])
        st, pl = S["streams"]["steered"]["tokens"], S["streams"]["plain"]["tokens"]
        if st:
            await crowd.send(c, {"type": "wc", "id": S["id"], "w": [word_entry(t, pl[i] if i < len(pl) else None) for i, t in enumerate(st)]})
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "taps" and isinstance(msg.get("c"), dict):     # batched: {button: count}
                for b, n in msg["c"].items():
                    if b in BUTTONS and isinstance(n, int):
                        for _ in range(max(0, min(n, int(TAP_BURST)))):
                            crowd.tap(c, b)
            elif kind == "tap" and msg.get("b") in BUTTONS:           # older pages
                crowd.tap(c, msg["b"])
            elif kind == "topic" and time.time() - crowd.last_change > TOPIC_COOLDOWN:
                crowd.change, crowd.last_change = True, time.time()
            elif kind == "ask" and c.can_ask:                         # a visitor's question joins the global line
                q, why = clean_question(msg.get("q"))
                # one question at a time per visitor (waiting or being answered), at least QUESTION_GAP apart
                mine = next((x for x in crowd.questions + ([crowd.asking] if crowd.asking else []) if x["cid"] == c.cid), None)
                wait = QUESTION_GAP - (time.time() - c.last_ask)
                if q is None:
                    await crowd.send(c, {"type": "ask_err", "why": why})
                elif mine is not None:
                    await crowd.send(c, {"type": "ask_mine", "id": mine["id"], "q": mine["q"], "wait": max(0.0, wait)})
                elif len(crowd.questions) >= QUESTION_MAX:
                    await crowd.send(c, {"type": "ask_err", "why": f"The line is full ({QUESTION_MAX} waiting). Try again in a minute."})
                elif wait > 0:
                    await crowd.send(c, {"type": "ask_err", "why": "wait", "wait": wait})
                else:
                    c.last_ask = time.time()
                    item = {"id": secrets.token_hex(4), "q": q, "cid": c.cid, "t": time.time()}
                    crowd.questions.append(item)
                    await crowd.send(c, {"type": "ask_ok", "id": item["id"], "q": q, "pos": len(crowd.questions), "gap": QUESTION_GAP})
            elif kind == "alts" and time.time() - c.last_alts > 0.2:  # the word inspector, on demand
                c.last_alts = time.time()
                toks, i = turn_tokens(str(msg.get("id"))), msg.get("i")
                if toks is not None and isinstance(i, int) and 0 <= i < len(toks):
                    t = toks[i]
                    await crowd.send(c, {"type": "alts", "id": msg["id"], "i": i, "a": t.get("a", []), "cf": t.get("cf", [])})
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        c.dead = True
        c.writer.cancel()
        if crowd.clients.get(cid) is c:
            del crowd.clients[cid]
