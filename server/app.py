"""Behind the Mask: exhibit server.

One visitor at a time *drives* (asks, turns the steering dials); everyone else watches the same performance
live and can join the queue. When nobody is driving, the exhibit plays its own programme (attract mode).

  uvicorn app:app --host 0.0.0.0 --port 8765
env: BTM_CHAT, BTM_BASE (HF repos), BTM_DIRS (folder with chat_dirs.pt/base_dirs.pt/limits.json), BTM_DEVICE
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import BASE_FRAME, Engine, GenConfig
from moderation import Moderator
from prompts import ATTRACT, CARDS
from steer import EMOTIONS, load_mind
from textemo import TextEmotion

CHAT = os.environ.get("BTM_CHAT", "Qwen/Qwen3-0.6B")
BASE = os.environ.get("BTM_BASE", "Qwen/Qwen3-0.6B-Base")
DIRS = Path(os.environ.get("BTM_DIRS", "runs/q06"))
TURN_SECONDS = int(os.environ.get("BTM_TURN_SECONDS", "150"))
TURN_PROMPTS = int(os.environ.get("BTM_TURN_PROMPTS", "3"))
IDLE_KICK = int(os.environ.get("BTM_IDLE_KICK", "45"))
ATTRACT_AFTER = int(os.environ.get("BTM_ATTRACT_AFTER", "12"))
MAX_QUEUE = int(os.environ.get("BTM_MAX_QUEUE", "60"))
WEB = Path(os.environ.get("BTM_WEB", "../web/dist"))

app = FastAPI(title="Behind the Mask")


class Hub:
    def __init__(self):
        self.clients: dict[str, WebSocket] = {}
        self.names: dict[str, str] = {}
        self.queue: list[str] = []
        self.driver: str | None = None
        self.turn_ends = 0.0
        self.prompts_left = 0
        self.last_action = time.time()
        self.steer = {e: 0.0 for e in EMOTIONS}   # dial units -1..1 (live)
        self.generating = False
        self.cancel = threading.Event()
        self.recap: dict | None = None
        self.attract_i = 0
        self.idle_since = time.time()
        self.mode = "idle"   # idle | driver | attract
        self.caption = ""
        self.events: asyncio.Queue = asyncio.Queue()
        self.limits = {"chat": 0.5, "base": 0.5}

    # ------------------------------------------------------------------ messaging
    async def send(self, cid: str, msg: dict):
        ws = self.clients.get(cid)
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:  # noqa: BLE001
            pass

    async def broadcast(self, msg: dict):
        data = json.dumps(msg)
        dead = []
        for cid, ws in list(self.clients.items()):
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.append(cid)
        for cid in dead:
            self.drop(cid)

    def public_state(self) -> dict:
        now = time.time()
        return {"type": "state", "mode": self.mode, "driver": self.names.get(self.driver) if self.driver else None,
                "queue": len(self.queue), "time_left": max(0, int(self.turn_ends - now)) if self.driver else 0,
                "prompts_left": self.prompts_left if self.driver else 0, "generating": self.generating,
                "steer": self.steer, "caption": self.caption, "viewers": len(self.clients)}

    async def push_state(self):
        st = self.public_state()
        for cid in list(self.clients):
            pos = self.queue.index(cid) + 1 if cid in self.queue else 0
            await self.send(cid, {**st, "you": {"id": cid, "driving": cid == self.driver, "position": pos}})

    def drop(self, cid: str):
        self.clients.pop(cid, None)
        if cid in self.queue:
            self.queue.remove(cid)
        if cid == self.driver:
            self.end_turn()

    def end_turn(self):
        self.driver = None
        self.prompts_left = 0
        self.mode = "idle"
        self.idle_since = time.time()
        self.cancel.set()


hub = Hub()
engine: Engine | None = None
mod: Moderator | None = None
temo: TextEmotion | None = None


def coef_from_dials(dials: dict, mind: str) -> dict:
    lim = hub.limits.get(mind, 0.5)
    return {e: max(-1.0, min(1.0, float(dials.get(e, 0.0)))) * lim for e in EMOTIONS}


class LiveSteer(dict):
    """Engine reads coefficients from here every step; the driver's dials write into hub.steer."""

    def __init__(self, mind: str):
        super().__init__()
        self.mind = mind

    def get(self, k, d=0.0):  # noqa: D401
        return coef_from_dials(hub.steer, self.mind).get(k, d)


async def perform(prompt: str, dials: dict, who: str | None, caption: str = ""):
    """Run one three-mind performance and stream it to everyone (moderated, with text-emotion scores)."""
    loop = asyncio.get_running_loop()
    hub.generating = True
    hub.cancel.clear()
    hub.steer = {e: max(-1.0, min(1.0, float(dials.get(e, 0.0)))) for e in EMOTIONS}
    hub.caption = caption
    await hub.push_state()
    q: asyncio.Queue = asyncio.Queue()
    texts = {s: "" for s in ("plain", "steered", "base", "base_steered")}
    blocked: set[str] = set()
    sent_upto = {s: 0 for s in texts}

    def emit(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    steer = LiveSteer("chat")
    worker = loop.run_in_executor(None, lambda: engine.run(prompt, steer, emit, hub.cancel))
    done = False
    while not done:
        ev = await q.get()
        if ev["type"] == "turn_begin":
            ev["who"] = hub.names.get(who) if who else None
            ev["caption"] = caption
            ev["attract"] = who is None
            await hub.broadcast(ev)
        elif ev["type"] == "tokens":
            out = []
            for it in ev["items"]:
                s = it["stream"]
                if s in blocked:
                    continue
                texts[s] += it["text"]
                ok = await loop.run_in_executor(None, mod.check_output, texts[s], it["done"])
                if not ok:
                    blocked.add(s)
                    await hub.broadcast({"type": "filtered", "stream": s})
                    continue
                out.append(it)
                if temo and it["text"] and (it["text"].rstrip().endswith((".", "!", "?")) or it["done"]):
                    sc = await loop.run_in_executor(None, temo.score, texts[s][sent_upto[s]:] or texts[s])
                    sent_upto[s] = len(texts[s])
                    await hub.broadcast({"type": "text_emotion", "stream": s, "scores": sc, "upto": len(texts[s])})
            if out:
                ev["items"] = out
                ev["dials"] = hub.steer
                await hub.broadcast(ev)
        elif ev["type"] == "turn_end":
            for s in list(ev["texts"]):
                if s in blocked:
                    ev["texts"][s] = None
            hub.recap = {"type": "recap", "prompt": prompt, "texts": ev["texts"], "steer": dict(hub.steer), "caption": caption,
                         "who": hub.names.get(who) if who else None}
            await hub.broadcast(ev)
            done = True
    await worker
    hub.generating = False
    hub.last_action = time.time()
    hub.idle_since = time.time()
    await hub.push_state()


async def scheduler():
    while True:
        await asyncio.sleep(1.0)
        now = time.time()
        if hub.generating:
            await hub.push_state()
            continue
        if hub.driver and (now > hub.turn_ends or hub.prompts_left <= 0 or now - hub.last_action > IDLE_KICK):
            await hub.send(hub.driver, {"type": "turn_over"})
            hub.end_turn()
        if not hub.driver and hub.queue:
            hub.driver = hub.queue.pop(0)
            hub.turn_ends = now + TURN_SECONDS
            hub.prompts_left = TURN_PROMPTS
            hub.last_action = now
            hub.mode = "driver"
            hub.caption = ""
            hub.steer = {e: 0.0 for e in EMOTIONS}
            await hub.send(hub.driver, {"type": "your_turn", "seconds": TURN_SECONDS, "prompts": TURN_PROMPTS})
        if not hub.driver and not hub.queue and now - hub.idle_since > ATTRACT_AFTER and hub.clients:
            prompt, dials, caption = ATTRACT[hub.attract_i % len(ATTRACT)]
            hub.attract_i += 1
            hub.mode = "attract"
            asyncio.create_task(perform(prompt, dials, None, caption))
            continue
        await hub.push_state()


@app.on_event("startup")
async def startup():
    global engine, mod, temo
    loop = asyncio.get_running_loop()

    def load():
        chat = load_mind("chat", CHAT, True)
        base = load_mind("base", BASE, False)
        chat.load(DIRS / "chat_dirs.pt")
        base.load(DIRS / "base_dirs.pt")
        lim = DIRS / "limits.json"
        if lim.exists():
            hub.limits.update(json.loads(lim.read_text()))
        cfg = GenConfig(step_delay=float(os.environ.get("BTM_STEP_DELAY", "0.05")))
        return Engine(chat, base, cfg), Moderator(), TextEmotion()

    engine, mod, temo = await loop.run_in_executor(None, load)
    print(f"[btm] ready: chat={CHAT} (layer {engine.chat.layer}/{engine.chat.n_layers}) base={BASE} limits={hub.limits}")
    asyncio.create_task(scheduler())


@app.get("/api/info")
async def info():
    return JSONResponse({"emotions": EMOTIONS, "cards": CARDS, "chat": CHAT, "base": BASE, "base_frame": BASE_FRAME,
                         "layer": {"chat": engine.chat.layer if engine else None, "base": engine.base.layer if engine else None},
                         "n_layers": {"chat": engine.chat.n_layers if engine else None, "base": engine.base.n_layers if engine else None},
                         "turn_seconds": TURN_SECONDS, "turn_prompts": TURN_PROMPTS, "ready": engine is not None})


NAMES = ["Heron", "Lantern", "Quartz", "Sparrow", "Cobalt", "Juniper", "Comet", "Ember", "Willow", "Orbit", "Pebble", "Aster",
         "Marlin", "Nimbus", "Fern", "Otter", "Saffron", "Tidal", "Violet", "Wren"]


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = ws.query_params.get("cid") or secrets.token_hex(6)
    cid = cid[:24]
    old = hub.clients.get(cid)
    if old is not None:
        try:
            await old.close()
        except Exception:  # noqa: BLE001
            pass
    hub.clients[cid] = ws
    hub.names.setdefault(cid, f"{random.choice(NAMES)} {random.randint(10, 99)}")
    await hub.send(cid, {"type": "hello", "id": cid, "name": hub.names[cid], "emotions": EMOTIONS})
    if hub.recap:
        await hub.send(cid, hub.recap)
    await hub.push_state()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "join":
                if cid not in hub.queue and cid != hub.driver and len(hub.queue) < MAX_QUEUE:
                    hub.queue.append(cid)
                    if hub.mode == "attract":
                        hub.cancel.set()
                await hub.push_state()
            elif t == "leave":
                if cid in hub.queue:
                    hub.queue.remove(cid)
                if cid == hub.driver:
                    hub.end_turn()
                await hub.push_state()
            elif t == "steer" and cid == hub.driver:
                for e in EMOTIONS:
                    if e in msg.get("steer", {}):
                        hub.steer[e] = max(-1.0, min(1.0, float(msg["steer"][e])))
                hub.last_action = time.time()
                await hub.broadcast({"type": "dials", "steer": hub.steer})
            elif t == "ask" and cid == hub.driver and not hub.generating and hub.prompts_left > 0:
                ok, text, reason = mod.check_input(msg.get("text", ""))
                if not ok:
                    await hub.send(cid, {"type": "rejected", "reason": reason})
                    continue
                hub.prompts_left -= 1
                hub.last_action = time.time()
                asyncio.create_task(perform(text, msg.get("steer", hub.steer), cid))
            elif t == "stop" and cid == hub.driver:
                hub.cancel.set()
    except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
        pass
    finally:
        if hub.clients.get(cid) is ws:
            hub.drop(cid)
            await hub.push_state()


if WEB.exists():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
