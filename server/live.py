"""Behind the Mask: steer it together (live).

One AI writes one short story at a time, live, for everyone on the page, and everyone steers it at once with
feeling buttons. Each tap adds to that feeling's share of the push; taps fade within seconds; the push added
to the model's hidden state at every word is the mix of everyone's recent taps. Nobody types anything.

Alongside the steered story the engine writes, in lock-step and with the same random dice, the story nobody
pushed (shown when the story ends) and keeps "what it would have said instead" for every word.

Safety: the only input is a choice of button (rate-limited per visitor, and no visitor counts for more than a
capped share of the push). Every word is checked as it is written; if the story turns abusive it is cut, the
push is reset and a new story begins. Nothing is stored on disk.

  uvicorn live:app --host 127.0.0.1 --port 8765        (vite preview proxies /live -> here)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from engine import Engine, GenConfig
from moderation import Moderator, clean_alts
from steer import EMOTIONS, load_mind

CHAT = os.environ.get("BTM_CHAT", "Qwen/Qwen3.5-9B")
DIRS = Path(os.environ.get("BTM_DIRS", "runs/q9b"))
SAE = os.environ.get("BTM_SAE", "")
SAE_LAYER = int(os.environ.get("BTM_SAE_LAYER", "20"))
TOKENS = int(os.environ.get("BTM_TOKENS", "200"))
HALF_LIFE = 5.0          # s: a tap's weight halves this fast
SAT = 3.0                # decayed taps for ~2/3 of full strength
VISITOR_CAP = 4.0        # one visitor's decayed taps count for at most this much
TAP_RATE, TAP_BURST = 6.0, 8.0   # per-visitor token bucket (taps / s, burst)
PAUSE = 7.0              # s between stories
BUTTONS = EMOTIONS + ["unmask"]

STORIES = [
    "a lighthouse keeper who finds a letter in a bottle",
    "a robot's first day at a new school",
    "two old friends meeting again at a train station",
    "a girl who finds a small door in her garden wall",
    "the last bakery still open in a sleeping city",
    "an astronaut who hears a song on the radio",
    "a dog waiting by the window for someone to come home",
    "a storm arriving at a small fishing village",
    "a museum guard who notices a painting has changed overnight",
    "a boy who builds a kite with his grandfather",
    "a woman who receives a phone call from her younger self",
    "a cat that moves into an empty house",
    "a street musician playing on the last night of the year",
    "a gardener who plants a seed nobody can identify",
    "a family's first night in a new apartment",
    "a mail carrier delivering one final letter",
]
PROMPT = "Tell me a very short story (under 130 words) about {s}."

app = FastAPI(title="Behind the Mask · live")
levels = json.loads((DIRS / "levels.json").read_text())
# full strength per button: the strongest dial stop for feelings; a little less for the mask (it derails fast)
STRONG = {e: float(levels[e]["toomuch"]) for e in EMOTIONS}
STRONG["assistant"] = float(levels["assistant"]["mid2"])
engine: Engine | None = None
mod: Moderator | None = None


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
        self.story: dict | None = None
        self.next_at = 0.0
        self.order: list[str] = []
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

    def reset(self):
        for c in self.clients.values():
            c.energy = {b: 0.0 for b in BUTTONS}
        self.tick(0)

    def state(self) -> dict:
        taps = {b: 0 for b in BUTTONS}
        for _, b in self.recent:
            taps[b] += 1
        S = self.story
        return {"type": "crowd", "viewers": len(self.clients), "power": r(self.power), "mix": {b: r(v) for b, v in self.mix.items()},
                "taps": taps, "ready": engine is not None,
                "story": {"id": S["id"], "prompt": S["prompt"]} if S else None,
                "next_in": max(0, round(self.next_at - time.time(), 1)) if not S else None}


crowd = Crowd()


def next_prompt() -> str:
    if not crowd.order:
        crowd.order = STORIES[:]
        random.shuffle(crowd.order)
    return PROMPT.format(s=crowd.order.pop())


async def tell_story():
    """Write one story, live, with the crowd's push at every word."""
    loop = asyncio.get_running_loop()
    labels = engine.labels
    prompt = next_prompt()
    cancel = threading.Event()
    streams = {s: {"tokens": [], "text": "", "safe": True} for s in ("steered", "plain")}
    story = {"id": secrets.token_hex(5), "prompt": prompt, "streams": streams, "cut": False}
    begin = {"type": "live_begin", "id": story["id"], "question": prompt, "emotion": "crowd", "level": "live",
             "layer": engine.chat.layer, "n_layers": engine.chat.n_layers}
    story["begin"] = begin
    crowd.story = story
    await crowd.broadcast(begin)
    q: asyncio.Queue = asyncio.Queue()

    def emit(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    worker = loop.run_in_executor(None, lambda: engine.run(prompt, crowd.steer, emit, cancel, seed=secrets.randbelow(2**31)))
    while True:
        ev = await q.get()
        if ev["type"] == "turn_end":
            break
        if ev["type"] != "tokens" or story["cut"]:
            continue
        out = []
        for it in ev["items"]:
            if it["stream"] not in streams:
                continue
            st = streams[it["stream"]]
            txt = st["text"] + it["text"]
            if it["stream"] == "steered":
                ok = await loop.run_in_executor(None, mod.check_output, txt, it["done"])
                if not ok:
                    story["cut"] = True
                    cancel.set()
                    break
            tok = {"t": it["text"], "p": r(it["p"]), "e": [r(it["emo"][e], 2) for e in labels],
                   "a": clean_alts([[x, r(pq)] for x, pq in it["alts"][:4]])}
            if it["stream"] == "steered":
                tok["cf"] = clean_alts([[x, r(pq)] for x, pq in it["cf_alts"][:4]])
                tok["s"] = [r(ev["steer"].get(e, 0.0)) for e in labels]
                if it.get("feats"):
                    tok["f"] = [[fid, r(fv, 1)] for fid, fv in it["feats"][:6]]
            st["tokens"].append(tok); st["text"] = txt
            out.append({"stream": it["stream"], "tok": tok})
        if story["cut"]:
            await crowd.broadcast({"type": "live_cut", "id": story["id"],
                                   "reason": "The push took the story somewhere we don't show, so we stopped it. A new one starts in a moment."})
            crowd.reset()
            continue
        if out:
            await crowd.broadcast({"type": "live_tokens", "id": story["id"], "items": out})
    await worker
    if not story["cut"]:
        plain_ok = await loop.run_in_executor(None, mod.check_output, streams["plain"]["text"], True)
        if not plain_ok:
            streams["plain"] = {"tokens": [], "text": "", "safe": False}
        doc = {"id": story["id"], "question": prompt, "emotion": "crowd", "level": "live", "swing_at": None,
               "layer": engine.chat.layer, "n_layers": engine.chat.n_layers, "streams": streams, "live": True}
        await crowd.broadcast({"type": "live_end", "id": story["id"], "doc": doc})
    crowd.story = None


async def story_loop():
    while True:
        if not crowd.clients:          # nobody watching: don't spend the GPU
            crowd.has_clients.clear()
            await crowd.has_clients.wait()
        try:
            await tell_story()
        except Exception as e:  # noqa: BLE001
            print("[live] story failed:", repr(e))
            crowd.story = None
            await crowd.broadcast({"type": "live_error"})
        crowd.next_at = time.time() + PAUSE
        await asyncio.sleep(PAUSE)


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
    global engine, mod

    def load():
        chat = load_mind("chat", CHAT, True); chat.load(DIRS / "chat_dirs.pt")
        if SAE:
            chat.load_sae(SAE, SAE_LAYER)
        return Engine(chat, None, GenConfig(max_new_tokens=TOKENS, step_delay=0.03)), Moderator(device="cuda")

    engine, mod = await asyncio.get_running_loop().run_in_executor(None, load)
    print(f"[live] ready: {CHAT}, layer {engine.chat.layer}", flush=True)
    asyncio.create_task(story_loop())


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
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        if crowd.clients.get(cid) is c:
            del crowd.clients[cid]
