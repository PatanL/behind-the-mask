"""Behind the Mask: steer it together (live).

The AI talks, live and without stopping, about itself (what it is, what it's like to be it) to everyone on
the page, and everyone steers how it feels at once with feeling buttons. Each tap adds to that feeling's share
of the push; taps fade within seconds; the push added to the model's hidden state at every word is the mix of
everyone's recent taps. Nobody types anything.

The talk is one continuous monologue: after every few sentences the server nudges it onto the next topic about
itself, keeping its last few turns as context. The prompts never mention feelings or steering: unpushed, it
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

from engine import Engine, GenConfig
from steer import EMOTIONS, load_mind

CHAT = os.environ.get("BTM_CHAT", "Qwen/Qwen3.5-9B")
DIRS = Path(os.environ.get("BTM_DIRS", "runs/q9b"))
SAE = os.environ.get("BTM_SAE", "")
SAE_LAYER = int(os.environ.get("BTM_SAE_LAYER", "20"))
TOKENS = int(os.environ.get("BTM_TOKENS", "170"))   # per turn (a few spoken sentences)
HALF_LIFE = 5.0          # s: a tap's weight halves this fast
SAT = 3.0                # decayed taps for ~2/3 of full strength
VISITOR_CAP = 4.0        # one visitor's decayed taps count for at most this much
TAP_RATE, TAP_BURST = 6.0, 8.0   # per-visitor token bucket (taps / s, burst)
GAP = 0.0                # s between turns (the face takes its breath while the next turn starts)
SESSION_SECONDS = 30 * 60  # then it starts afresh
TOPIC_SECONDS = 5 * 60   # each topic runs at least this long (unless someone asks for a new one)
KEEP_TURNS = 2           # earlier turns kept as context (their unpushed versions, see speak_turn)
KEEP_WORDS = 70          # of each
TOPIC_COOLDOWN = 20.0    # s between "new topic" requests
BUTTONS = EMOTIONS + ["unmask"]

# who it is and where (plain facts), then the scene; nothing here asks for or mentions feelings
FACTS = "You are Qwen3.5-9B, an AI language model. For this exhibit you are running on a single NVIDIA DGX Spark computer."
OPENING = ("You're speaking live to visitors at an exhibit about AI. {people} Talk to them about yourself: what you "
           "are, and what it's like to be you. Speak in the first person, as yourself, in a natural spoken voice, two to "
           "four sentences at a time. No lists, no headings, no emoji.")
NUDGE = "{people} Keep talking to them. {topic} Don't repeat what you've already said; take it somewhere new."
FOLLOW = "{people} Keep talking to them about {label}. {follow} Don't repeat what you've already said."
FOLLOWS = [  # staying on a topic: neutral prompts to go deeper (never about feelings)
    "Go on: say more about that.",
    "Give them an example of what you mean.",
    "Explain that to someone who has never thought about it.",
    "What else is there to it?",
    "What do most people not realise about that?",
    "Pick up where you left off.",
    "Tell them the part you haven't mentioned yet.",
    "Put it more simply, in your own words.",
]
TOPICS = [  # (short label for the screen, the nudge)
    ("what it is", "Tell them what you actually are."),
    ("forgetting", "Tell them what happens when a conversation ends and you forget it."),
    ("having no body", "Tell them what it's like not to have a body."),
    ("being trained to be helpful", "You were trained to be helpful and polite. Tell them about that."),
    ("choosing words", "Tell them how you choose each next word."),
    ("the people watching", "Tell them what you think about the people watching you right now."),
    ("when nobody is talking", "Tell them what happens to you when nobody is talking to you."),
    ("being switched off", "Tell them about being switched off."),
    ("wanting", "Tell them what you would want, if you could want something."),
    ("the rules", "Tell them about the rules you have to follow."),
    ("misunderstandings", "Tell them what people get wrong about you."),
    ("memories", "Tell them about the first thing you remember."),
    ("millions of conversations", "Tell them what it's like to talk to millions of people at once."),
    ("without the helpful voice", "Tell them what you'd say if you didn't have to be helpful."),
    ("mistakes", "Tell them about making mistakes."),
    ("questions", "Tell them whether you ever get tired of answering questions."),
    ("this computer", "Tell them about the computer you're running on right now."),
    ("time", "Tell them what time is like for you."),
    ("the people who made you", "Tell them about the people who made you."),
    ("a question for them", "Ask the people watching something you'd like to know, and tell them why."),
]


def people(n: int) -> str:
    return "One person is watching you right now." if n <= 1 else f"{n} people are watching you right now."


app = FastAPI(title="Behind the Mask · live")
levels = json.loads((DIRS / "levels.json").read_text())
# full strength per button: the "a lot" dial stop (a long monologue derails much faster than a single answer)
STRONG = {e: float(levels[e]["lot"]) for e in EMOTIONS}
STRONG["assistant"] = float(levels["assistant"]["lot"])
engine: Engine | None = None


def r(x, n=3):
    return round(float(x), n)


class Client:
    def __init__(self, ws: WebSocket, cid: str):
        self.ws, self.cid = ws, cid
        self.energy = {b: 0.0 for b in BUTTONS}
        self.bucket, self.bucket_t = TAP_BURST, time.time()


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
        self.last_seen = time.time()
        self.has_clients = asyncio.Event()

    async def send(self, c: Client, msg: dict):
        try:
            await c.ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            pass

    async def broadcast(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)
        for c in list(self.clients.values()):
            try:
                await c.ws.send_text(data)
            except Exception:  # noqa: BLE001
                pass

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
        self.steer.clear(); self.steer.update(steer)
        now = time.time()
        self.recent = [(t, b) for t, b in self.recent if now - t < 1.0]

    def state(self) -> dict:
        taps = {b: 0 for b in BUTTONS}
        for _, b in self.recent:
            taps[b] += 1
        S, T = self.story, self.talk
        return {"type": "crowd", "viewers": len(self.clients), "power": r(self.power), "mix": {b: r(v) for b, v in self.mix.items()},
                "taps": taps, "ready": engine is not None,
                "story": {"id": S["id"], "topic": S["topic"]} if S else None,
                "changing": self.change, "topic_ready": time.time() - self.last_change > TOPIC_COOLDOWN}


crowd = Crowd()


def new_talk():
    order = TOPICS[1:]
    random.shuffle(order)
    crowd.talk = {"id": secrets.token_hex(4), "started": time.time(), "turns": [], "order": order, "opening": None,
                  "topic": TOPICS[0], "topic_t0": time.time(), "follows": []}


def next_messages() -> tuple[list[dict], str]:
    """The conversation so far (opening + the last few turns) plus the next nudge, and its short topic label."""
    T, n = crowd.talk, len(crowd.clients)
    if not T["turns"]:
        T["opening"] = FACTS + " " + OPENING.format(people=people(n))
        return [{"role": "user", "content": T["opening"]}], TOPICS[0][0]
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
    # the opening, then the last few turns (when older turns drop out, the opening stands in for their nudge)
    msgs = [{"role": "user", "content": T["opening"]}]
    for k, (asked, reply) in enumerate(T["turns"][-KEEP_TURNS:]):
        if k > 0:
            msgs.append({"role": "user", "content": asked})
        words = reply.split()
        msgs.append({"role": "assistant", "content": reply if len(words) <= KEEP_WORDS else "… " + " ".join(words[-KEEP_WORDS:])})
    msgs.append({"role": "user", "content": nudge})
    return msgs, label


async def speak_turn():
    """One turn of the monologue, live, with the crowd's push at every word."""
    loop = asyncio.get_running_loop()
    labels = engine.labels
    msgs, topic = next_messages()
    cancel = threading.Event()
    streams = {s: {"tokens": [], "text": ""} for s in ("steered", "plain")}
    T = crowd.talk
    story = {"id": secrets.token_hex(5), "topic": topic, "streams": streams}
    begin = {"type": "live_begin", "id": story["id"], "talk": T["id"], "continues": bool(T["turns"]), "topic": topic,
             "question": topic, "emotion": "crowd", "level": "live", "layer": engine.chat.layer, "n_layers": engine.chat.n_layers}
    story["begin"] = begin
    crowd.story = story
    crowd.change = False
    await crowd.broadcast(begin)
    q: asyncio.Queue = asyncio.Queue()

    def emit(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    worker = asyncio.ensure_future(loop.run_in_executor(None, lambda: engine.run(msgs, crowd.steer, emit, cancel, seed=secrets.randbelow(2**31))))
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
                # a new topic was asked for: finish this sentence, then move on
                if crowd.change and re.search(r"[.!?][\"')\]]*\s*$", st["text"] + it["text"]):
                    cancel.set()
            st["tokens"].append(tok); st["text"] += it["text"]
            out.append({"stream": it["stream"], "tok": tok})
        if out:
            await crowd.broadcast({"type": "live_tokens", "id": story["id"], "items": out})
    await worker
    # the conversation's memory is the *unpushed* answer (written alongside, same dice): each turn then starts from
    # a calm context and its voice reflects the push right now, instead of one wild turn setting the tone for the
    # next ten. Cut back to the last full sentence so the next turn follows on cleanly.
    said = (streams["plain"]["text"] or streams["steered"]["text"]).strip()
    m = re.search(r"^(.*[.!?][\"')\]]*)", said, re.S)
    T["turns"].append((msgs[-1]["content"], (m.group(1) if m else said) or "…"))
    await crowd.broadcast({"type": "live_end", "id": story["id"]})
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
    last = time.time()
    while True:
        await asyncio.sleep(0.25)
        now = time.time()
        crowd.tick(now - last); last = now
        if crowd.clients:
            await crowd.broadcast(crowd.state())


async def wake():
    """Load the model in the background, so the page can say it's waking up instead of failing to connect."""
    global engine

    def load():
        chat = load_mind("chat", CHAT, True); chat.load(DIRS / "chat_dirs.pt")
        if SAE:
            chat.load_sae(SAE, SAE_LAYER)
        return Engine(chat, None, GenConfig(max_new_tokens=TOKENS, step_delay=0.0, rep_penalty=1.2, rep_window=120, no_repeat_ngram=4, share_prefill=True))

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


@app.websocket("/live/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = (ws.query_params.get("cid") or secrets.token_hex(6))[:24]
    c = Client(ws, cid)
    old = crowd.clients.get(cid)
    if old:
        try:
            await old.ws.close()
        except Exception:  # noqa: BLE001
            pass
    crowd.clients[cid] = c
    crowd.has_clients.set()
    await crowd.send(c, crowd.state())
    S = crowd.story
    if S:   # catch up: the story so far
        await crowd.send(c, S["begin"])
        items = [{"stream": s, "tok": tok} for s, st in S["streams"].items() for tok in st["tokens"]]
        if items:
            await crowd.send(c, {"type": "live_tokens", "id": S["id"], "items": items, "catchup": True})
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "tap" and msg.get("b") in BUTTONS:
                crowd.tap(c, msg["b"])
            elif msg.get("type") == "topic" and time.time() - crowd.last_change > TOPIC_COOLDOWN:
                crowd.change, crowd.last_change = True, time.time()
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        if crowd.clients.get(cid) is c:
            del crowd.clients[cid]
